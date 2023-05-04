import numpy as np
import os
import os.path as osp
import cv2
import math
import matplotlib.pyplot as plt
import copy

from utils.data import expand_topology_joints
from Motion import InverseKinematics as IK
from Motion import Animation
from Motion import BVH
from utils.data import Joint, Edge
from utils.data import calc_bone_lengths
from utils.data import edge_rot_dict_from_edge_motion_data, anim_from_edge_rot_dict
from utils.data import to_list_4D, un_normalize
from motion_class import StaticData, DynamicData, anim_from_static


def pose2im_all(all_peaks, H=512, W=512):
    limbSeq = [[1, 2], [2, 3], [3, 4],                       # right arm
               [1, 5], [5, 6], [6, 7],                       # left arm
               [8, 9], [9, 10], [10, 11],                    # right leg
               [8, 12], [12, 13], [13, 14],                  # left leg
               [1, 0],                                       # head/neck
               [1, 8],                                       # body,
               ]

    limb_colors = [[0, 60, 255], [0, 120, 255], [0, 180, 255],
                    [170, 255, 0], [85, 255, 0], [0, 255, 0],
                    [255, 170, 0], [255, 85, 0], [255, 0, 0],
                    [100, 0, 255], [150, 0, 255], [200, 0, 255],
                    [255, 0, 255],
                    [100, 100, 100],
                   ]

    joint_colors = [[255, 0, 255], [0, 0, 255], [0, 60, 255], [0, 120, 255], [0, 180, 255],
                    [180, 255, 0], [120, 255, 0], [60, 255, 0], [0, 0, 255],
                    [170, 255, 0], [85, 255, 0], [0, 255, 0],
                    [255, 170, 0], [255, 85, 0], [255, 0, 0],
                    ]

    image = pose2im(all_peaks, limbSeq, limb_colors, joint_colors, H, W)
    return image


def stretch(data, H, W):
    """ Stretch the skeletons proportionally to each other """

    # locate body center in 0,0,0
    data -= data[:, 8:9, :, :]

    # find min/max for each motion and each frame and each axis (that is, across all joints)
    mins = data.min(1)
    maxs = data.max(1)
    diffs = maxs-mins

    scale_from = diffs.max()
    scale_to = min(H, W) - 1

    data -= mins[:, np.newaxis]
    assert data.min() == 0
    data /= scale_from
    assert data.max() == 1
    data *= scale_to
    assert data.max() == scale_to

    return data


def pose2im(all_peaks, limbSeq, limb_colors, joint_colors, H, W, _circle=True, _limb=True, imtype=np.uint8):
    canvas = np.zeros(shape=(H, W, 3))
    canvas.fill(255)

    if _circle:
        for i in range(len(joint_colors)):
            cv2.circle(canvas, (int(all_peaks[i][0]), int(all_peaks[i][1])), 2, joint_colors[i], thickness=2)

    if _limb:
        stickwidth = 2

        for i in range(len(limbSeq)):
            limb = limbSeq[i]
            cur_canvas = canvas.copy()
            point1_index = limb[0]
            point2_index = limb[1]

            if len(all_peaks[point1_index]) > 0 and len(all_peaks[point2_index]) > 0:
                point1 = all_peaks[point1_index][0:2]
                point2 = all_peaks[point2_index][0:2]
                X = [point1[1], point2[1]]
                Y = [point1[0], point2[0]]
                mX = np.mean(X)
                mY = np.mean(Y)
                # cv2.line()
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, limb_colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas.astype(imtype)


def motion2fig_1_motion_3_angles(data, H=512, W=512):
    diffs = [data[0, :, i, 0].max() - data[0, :, i, 0].min() for i in range(3)]
    scale = max(diffs)
    n_frames = data.shape[-1]
    n_samples = 5 # how many samples to take from the whole video
    idx = np.linspace(0, n_frames-1, n_samples).round().astype(int)

    fig, axes = plt.subplots(3, n_samples)
    for sample_idx, i in enumerate(idx):
        for angle_idx, d in enumerate([data[0,:,:2,i], data[0,:,::2,i], data[0,:,1:,i]]):
            img = pose2im_all(d, scale, H, W)
            # axes[angle_idx, i].subplot(3, n_samples, 1)
            axes[angle_idx, sample_idx].axis('off')
            axes[angle_idx, sample_idx].imshow(img[::-1,:]) # image y axis is inverted
    return fig


def motion2fig(static: StaticData, data,  normalisation_data,
               height=512, width=512, n_sampled_motions=5, n_sampled_frames=5):

    dynamics = [DynamicData(motion, static) for motion in data[:n_sampled_motions]]

    n_sampled_motions = min(n_sampled_motions, data.shape[0], 10)
    sampled_frames = np.linspace(0, dynamics[0].n_frames-1, n_sampled_frames).round().astype(int)

    # data shape: n_samples x n_joints x n_features x n_frames
    assert not isinstance(data, list) and not isinstance(data[0], dict)

    data = data * normalisation_data['std'] + normalisation_data['mean']  # TODO: looks like a quicker way to normalise a batch of samples.
    for dynamic in dynamics:
        dynamic.normalise(normalisation_data['mean'][:, :, :, 0], normalisation_data['std'][:, :, :, 0])
        dynamic.sample_frames(sampled_frames)

    # sampled_data = data[:n_sampled_motions][:, :, :, sampled_frames]
    # sampled_data = to_list_4D(sampled_data)
    #
    # # edge_rot_dict_general = normalisation_data
    # sampled_edge_rot_dict_general = copy.deepcopy(edge_rot_dict_general)
    # sampled_edge_rot_dict_general['rot_edge_no_root'] = sampled_edge_rot_dict_general['rot_edge_no_root'][sampled_frames]
    # sampled_edge_rot_dict_general['pos_root'] = sampled_edge_rot_dict_general['pos_root'][sampled_frames]
    # sampled_edge_rot_dict_general['rot_root'] = sampled_edge_rot_dict_general['rot_root'][sampled_frames]

    # edge_rots_dict, _, _ = edge_rot_dict_from_edge_motion_data(sampled_data, edge_rot_dict_general=sampled_edge_rot_dict_general)

    # one_anim, names = anim_from_edge_rot_dict(sampled_edge_rot_dict_general)
    anim, names = anim_from_static(static, dynamics[0])
    one_anim_shape = anim.shape

    # joints = np.zeros((n_sampled_motions, dynamics[0].n_frames, dynamic.n_joints, 3))
    joints = np.zeros((n_sampled_motions,) + one_anim_shape + (3,))  # TODO: Change
    for idx, dynamic in enumerate(dynamics):
        anim, _ = anim_from_static(static, dynamic)
        joints[idx] = Animation.positions_global(anim)

    figure_joints = ['Head', 'Neck', 'RightArm', 'RightForeArm', 'RightHand', 'LeftArm',
                     'LeftForeArm', 'LeftHand', 'Hips', 'RightUpLeg', 'RightLeg',
                     'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot']
    figure_indexes = [list(names).index(joint) for joint in figure_joints]

    data = joints[:, :, figure_indexes, :2]  # b x T x J x 4
    data = data.transpose(0, 2, 3, 1)  # samples x frames x joints x features ==> samples x joints x features x frames
    # data = data[:, :, :2, :]  # use the xy projection

    data = stretch(data, height, width)

    fig, axes = plt.subplots(n_sampled_motions, n_sampled_frames)
    if axes.ndim == 1:  # if there is only one motion
        axes = axes[np.newaxis, :]
    max_w, max_h = np.ceil(data.max(axis=(0,1,3))).astype(int)
    for motion_idx in np.arange(n_sampled_motions):
        for frame_idx in np.arange(n_sampled_frames):
            skeleton = data[motion_idx,:,:,frame_idx]
            img = pose2im_all(skeleton, max_h, max_w)
            axes[motion_idx, frame_idx].axis('off')
            try:
                axes[motion_idx, frame_idx].imshow(img[::-1,:]) # image y axis is inverted
            except:
                pass # in some configurations the image cannot be shown
    return fig


# def motion2bvh(motion_data, bvh_file_path, parents=None, type=None, entity='Joint', normalisation_data=None, static=None):
#     assert entity in ['Joint', 'Edge']
#     if entity == 'Joint':
#         motion2bvh_loc(motion_data, bvh_file_path, parents, type)
#     else:
#     motion2bvh_rot(motion_data, bvh_file_path, normalisation_data=normalisation_data, static=static)


def motion2bvh_rot(motion_data, bvh_file_path, normalisation_data, static):

    if isinstance(motion_data, dict):
        # input is of type edge_rot_dict (e.g., read from GT file)
        motion_data = [motion_data]
        frame_mults = [1]
        is_sub_motion = False
    else:
        # input is at the format of an output of the network
        motion_data = to_list_4D(motion_data)  # add batch dimension and list dimention 1
        motion_data = un_normalize(motion_data,
                                   mean=normalisation_data['mean'],
                                   std=normalisation_data['std'])
        # edge_rot_dicts, frame_mults, is_sub_motion = edge_rot_dict_from_edge_motion_data(motion_data, type=type,
        #                                                                              edge_rot_dict_general=edge_rot_dict_general)

    # from this point input is a list of edge_rot_dicts
    # for i, (edge_rot_dict, frame_mult) in enumerate(zip(edge_rot_dicts, frame_mults)):
    for idx, motion in enumerate(motion_data):
        # anim, names = anim_from_edge_rot_dict(edge_rot_dict, root_name='Hips')
        dynamic = DynamicData(motion[0], static)  # TODO: We need to normalise the motion as here.
        anim, names = anim_from_static(static, dynamic)

        # if is_sub_motion:  # TODO: What about a sub motion?
        #     suffix = f'_{dynamic.n_frames}x{dynamic.n_joints}'
        # elif type == 'edit':
        #     suffix = f'_{idx}'
        # else:
        #     suffix = ''
        # bvh_sub_file_path = bvh_file_path.replace('.bvh', suffix + '.bvh')

        bvh_file_dir = osp.split(bvh_file_path)[0]
        os.makedirs(bvh_file_dir, exist_ok=True)

        BVH.save(bvh_file_path, anim, names)

        # if 'contact' in edge_rot_dict and edge_rot_dict['contact'] is not None:
        #     np.save(bvh_sub_file_path + '.contact.npy', edge_rot_dict['contact'])

# old function to save figures from Joint model.
def motion2bvh_loc(motion_data, bvh_file_path, parents=None, type=None):
    if isinstance(motion_data, list): # saving sub pyramid motions
        bl_full = calc_bone_lengths(motion_data[-1], parents=Joint.parents_list[-1])
        for i, sub_motion in enumerate(motion_data):
            is_openpose = (sub_motion.shape == motion_data[-1].shape)
            sub_motion = sub_motion[0]  # drop batch dim
            if type == 'sample' or type =='interp-mix-pyramid':  # displaying sub motions
                suffix = '_' + str(sub_motion.shape[0]) + 'x' + str(sub_motion.shape[2])
                sub_bvh_file_path = bvh_file_path.replace('.bvh', suffix+'.bvh')

                # multiply bone length such that bone lengths in all levels are comparable
                n_joints = sub_motion.shape[0]
                if n_joints==1:
                    continue
                bl_sub_motion = calc_bone_lengths(sub_motion[np.newaxis], parents=parents[i])
                bl_mult = bl_full['mean'].mean() / bl_sub_motion['mean'].mean()
                sub_motion = sub_motion * bl_mult

                # repeat frames so the frame number of the sub_motion would be the same as the final one,
                # in order to make the visualization synchronized
                frame_mult = int(motion_data[-1].shape[-1] / sub_motion.shape[-1])
                sub_motion = sub_motion.repeat(frame_mult, axis=2)
                one_motion2bvh(sub_motion, sub_bvh_file_path, parents=parents[i], is_openpose=is_openpose)
            elif type == 'edit':
                sub_bvh_file_path = bvh_file_path.replace('.bvh', '_'+'{:02d}'.format(i)+'.bvh')
                motion2bvh_loc(motion_data, bvh_file_path, parents, type)
            else:
                raise('unsupported type for list manipulation')
    else:
        if motion_data.ndim == 4:
            assert motion_data.shape[0] == 1
            motion_data = motion_data[0]
        one_motion2bvh(motion_data, bvh_file_path, parents=parents[-1], is_openpose=True)


def one_motion2bvh(one_motion_data, bvh_file_path, parents, is_openpose=True, names = None, expand = True):

    # support non-skel-aware motions with 16 joints
    if one_motion_data.shape[0] == 16:
        one_motion_data = one_motion_data[:15]

    one_motion_data = one_motion_data.transpose(2, 0, 1)  # joint, axis, frame  -->   frame, joint, axis

    if expand:
        one_motion_data, parents, names = expand_topology_joints(one_motion_data, is_openpose, parents, names)
    anim, sorted_order, _ = IK.animation_from_positions(one_motion_data, parents)
    bvh_file_dir = osp.split(bvh_file_path)[0]
    if not osp.exists(bvh_file_dir):
        os.makedirs(bvh_file_dir)
    BVH.save(bvh_file_path, anim, names[sorted_order])

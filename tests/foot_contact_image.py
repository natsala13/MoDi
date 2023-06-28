import torch
import pytest
import random
import attrdict
import numpy as np
import matplotlib.pyplot as plt


from Motion import BVH
from utils.data import motion_from_raw
from utils.foot import get_foot_location, get_foot_velo
from utils.visualization import motion2fig
from utils.data import anim_from_edge_rot_dict
from motion_class import StaticData, DynamicData

JASPER_DB = 'data/edge_rot_data.npy'
MAW_DB = 'data/edge_rot_maw.npy'
MAW_DEBUG = 'tmp/edge_rot_maw_debug.npy'
MIN_FLOAT_VALUE = 1e-9

DEBUG_MOTION_INDEX = 2

@pytest.fixture(scope='module')
def motion_db():
    return np.load(MAW_DB, allow_pickle=True)


@pytest.fixture(scope='module')
def motion_raw(motion_db):
    edge_rot_data = np.stack([motion['rot_edge_no_root'] for motion in motion_db])
    root_rot_data = np.stack([motion['rot_root'] for motion in motion_db])
    motion_data = np.concatenate([root_rot_data[:, :, np.newaxis], edge_rot_data], axis=2)
    motion_data = motion_data.transpose(0, 2, 3, 1)

    return motion_data


@pytest.fixture(scope='module')
def mean_joints(motion_raw):
    mean = motion_raw.mean(axis=(0, 3))
    mean = mean[np.newaxis, :, :, np.newaxis]

    return mean


@pytest.fixture(scope='module')
def std_joints(motion_raw):
    std = motion_raw.std(axis=(0, 3))
    std = std[np.newaxis, :, :, np.newaxis]
    std[np.where(std < MIN_FLOAT_VALUE)] = MIN_FLOAT_VALUE
    # motion_data = (motion_data - mean_joints) / std_joints

    return std


@pytest.fixture(scope='module')
def normalisation_data():
    return {'mean': np.zeros((1, 4, 32, 1)),
            'std': np.ones((1, 4, 32, 1))}


@pytest.fixture(scope='module')
def static(motion_db):
    offsets = np.concatenate([motion_db[0]['offset_root'][np.newaxis, :], motion_db[0]['offsets_no_root']])

    return StaticData(parents=motion_db[0]['parents_with_root'],
                      offsets=offsets,
                      names=motion_db[0]['names_with_root'],
                      n_channels=4,
                      enable_global_position=True,
                      enable_foot_contact=True)


@pytest.fixture(scope='module')
def fake_args():
    return attrdict.AttrDict({'entity': False, 'rotation_repr': None, 'glob_pos': True,
                              'use_velocity': False, 'normalize': False, 'foot': True, 'axis_up': 1})


@pytest.fixture(scope='module')
def motion_with_preprocess(motion_db, static, fake_args):
    motion_data, _, _, _ = motion_from_raw(fake_args, motion_db, static)

    return motion_data


@pytest.fixture(scope='module')
def random_samples(motion_db):
    random.seed(3)  # seed 17 - [4276, 3392, 2485, 2995, 2372]
    samples = [random.randint(0, len(motion_db)) for _ in range(5)]
    print(f'Using random samples - {samples} out of {len(motion_db)}')
    return [101, 4306, 3983, 4308, 4309]
    return samples


@pytest.fixture(scope='module')
def dynamics(static, motion_with_preprocess, random_samples, fake_args):
    dynamic_motion = DynamicData(torch.tensor(motion_with_preprocess[random_samples]).transpose(1, 2),
                                 static, use_velocity=fake_args.use_velocity)

    return dynamic_motion


@pytest.fixture(scope='module')
def debug_motion(random_samples):
    return random_samples[DEBUG_MOTION_INDEX]


@pytest.fixture(scope='module')
def dynamics_debug(static, motion_with_preprocess, debug_motion):
    debug_dynamic = DynamicData(torch.tensor(motion_with_preprocess[debug_motion]).transpose(0, 1), static)
    res = DynamicData(torch.stack([debug_dynamic.sample_frames(range(5 * i, 5 * i+5)).motion for i in range(5)]), static)

    return res


def test_foot_height_graph(motion_with_preprocess, static, normalisation_data, random_samples):
    motion_torch = torch.from_numpy(motion_with_preprocess).transpose(1, 2)[..., :-2, :]

    foot_location, foot_indexes, offsets = get_foot_location(motion_torch, static, normalisation_data,
                                                             use_global_position=True, use_velocity=True)

    debug_sample = random_samples[DEBUG_MOTION_INDEX]

    foot_velocity = (foot_location[:, 2:] - foot_location[:, 0:-2]).pow(2).sum(axis=-1).sqrt()

    shin_len = offsets[0, foot_indexes].pow(2).sum(axis=-1).sqrt()
    velo_thresh = 0.07 * shin_len[0].numpy()
    # ipdb foot.py:70
    my_foot = foot_location[debug_sample,...,1]
    plt.plot(my_foot[:, 0], c='red')
    plt.plot(my_foot[:, 1], c='blue')

    plt.figure()
    my_vel = foot_velocity[debug_sample]
    plt.plot(my_vel[:, 0], c='red')
    plt.plot(my_vel[:, 1], c='blue')

    plt.plot(np.ones_like(my_vel[:, 0]) * velo_thresh, c='purple', marker='.')

    plt.show()

    pass


# def test_bvh_from_edge_rot():
#     motion = np.load('tmp/maw_debug_after.npy', allow_pickle=True).item()
#     static, dynamic = DynamicData.init_from_edge_rot_dict(motion)
#     anim, names = dynamic.anim_from_static()
#
#     BVH.save(f'./tmp/maw_debug_after.bvh', anim, names)


def test_save_bvh_debug(motion_db, debug_motion):
    motion = motion_db[debug_motion]
    anim, names = anim_from_edge_rot_dict(motion)

    print(f'saving motion {debug_motion}...')
    BVH.save(f'./tmp/debug_motion_edge_{debug_motion}.bvh', anim, names)


def test_save_bvh_from_static_debug(static, dynamics, random_samples):
    dynamic = dynamics[DEBUG_MOTION_INDEX]
    anim, names = dynamic.anim_from_static()

    print(f'saving motion {random_samples[DEBUG_MOTION_INDEX]}...')
    BVH.save(f'./tmp/debug_motion_static_{random_samples[DEBUG_MOTION_INDEX]}.bvh', anim, names)


def test_plot_dynamics_debug(dynamics_debug, static):
    motion2fig(static, dynamics_debug)
    plt.show()


def test_plot_dynamics_all(dynamics, static):
    motion2fig(static, dynamics)
    plt.show()
    # fig_path = osp.join(images_output_folder, 'fake_motion_{}.png'.format(i))
    # fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    # plt.close()  # close figure


def test_foot_contact_loss_plot(motion_with_preprocess, debug_motion, static, normalisation_data, fake_args):
    motion = torch.from_numpy(motion_with_preprocess).transpose(1, 2)  # [debug_motion]
    label_idx = motion.shape[2] - static.foot_number

    velo = get_foot_velo(motion[:, :, :label_idx], static, normalisation_data, True, fake_args.use_velocity)

    predicted_foot_contact = motion[:, 0, label_idx:]
    predicted_foot_contact = torch.sigmoid((predicted_foot_contact - 0.5) * 2 * 6)
    final_loss = (predicted_foot_contact[..., 1:] * velo)

    # loss_all_dataset = final_loss.mean(dim=2).mean(dim=1).sort()[0]
    # plt.hist(loss_all_dataset, bins=20)
    # plt.title('loss value for dataset')

    plt.figure()
    plt.plot(final_loss[debug_motion].transpose(0, 1))
    plt.title('foor contact loss v2 for debug motion')
    plt.show()

    print(f'LOSS - {final_loss}')

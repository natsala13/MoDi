import torch
import pytest
import random
import attrdict
import numpy as np
import matplotlib.pyplot as plt


from Motion import BVH
from utils.data import motion_from_raw
from utils.visualization import motion2fig
from utils.data import anim_from_edge_rot_dict
from motion_class import StaticData, DynamicData, anim_from_static

JASPER_DB = 'data/edge_rot_data.npy'
MAW_DB = 'data/edge_rot_maw.npy'
MAW_DEBUG = 'tmp/edge_rot_maw_debug.npy'
MIN_FLOAT_VALUE = 1e-9


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
                              'use_velocity': None, 'normalize': False, 'foot': True, 'axis_up': 1})


@pytest.fixture(scope='module')
def motion_with_preprocess(motion_db, static, fake_args):
    motion_data, _, _, _ = motion_from_raw(fake_args, motion_db, static)

    return motion_data


@pytest.fixture(scope='module')
def random_samples(motion_db):
    random.seed(2)  # seed 17 - [4276, 3392, 2485, 2995, 2372]
    samples = [random.randint(0, len(motion_db)) for _ in range(5)]
    print(f'Using random samples - {samples} out of {len(motion_db)}')
    return samples


@pytest.fixture(scope='module')
def dynamics(static, motion_with_preprocess, random_samples):
    dynamic_motion = DynamicData(torch.tensor(motion_with_preprocess[random_samples]).transpose(1, 2), static)

    return dynamic_motion


@pytest.fixture(scope='module')
def debug_motion(random_samples):
    DEBUG_MOTION_INDEX = 1
    return random_samples[DEBUG_MOTION_INDEX]


@pytest.fixture(scope='module')
def dynamics_debug(static, motion_with_preprocess, debug_motion):
    debug_dynamic = DynamicData(torch.tensor(motion_with_preprocess[debug_motion]).transpose(0, 1), static)
    res = DynamicData(torch.stack([debug_dynamic.sample_frames(range(5 * i, 5 * i+5)).motion for i in range(5)]), static)

    return res


def foot_height_graph(dynamics, static):
    # ipdb foot.py:70
    # my_foot = foot_up_location[3392]
    # plt.plot(my_foot[:, 0], c='red')
    # plt.plot(my_foot[:, 1], c='blue')
    #
    # plt.figure()
    # my_vel = foot_velocity[3392]
    # plt.plot(my_vel[:, 0], c='red')
    # plt.plot(my_vel[:, 1], c='blue')
    # plt.plot(np.ones_like(my_vel[:, 0]) * 1.7593, c='purple', marker='.')
    #
    # plt.show()

    pass


def test_bvh_from_edge_rot():
    motion = np.load('tmp/maw_debug_after.npy', allow_pickle=True).item()
    static, dynamic = DynamicData.init_from_edge_rot_dict(motion)
    anim, names = anim_from_static(static, dynamic)

    BVH.save(f'./tmp/maw_debug_after.bvh', anim, names)


def test_save_bvh_debug(motion_db, debug_motion):
    motion = motion_db[debug_motion]
    anim, names = anim_from_edge_rot_dict(motion)

    print(f'saving motion {debug_motion}...')
    BVH.save(f'./tmp/debug_motion_{debug_motion}.bvh', anim, names)


def test_plot_dynamics_debug(dynamics_debug, static):
    motion2fig(static, dynamics_debug)
    plt.show()


def test_plot_dynamics_all(dynamics, static):
    motion2fig(static, dynamics)
    plt.show()
    # fig_path = osp.join(images_output_folder, 'fake_motion_{}.png'.format(i))
    # fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    # plt.close()  # close figure

import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt

from utils.data import Edge, anim_from_edge_rot_dict
from motion_class import StaticData, DynamicData, anim_from_static
from utils.visualization import motion2fig
from evaluate import convert_motions_to_location


FAKE_MOTION = 'debug/fig2img/fake_motion_399.npy'
FAKE_METADATA = 'debug/fig2img/fake_metaata_399.npy'
FAKE_METADATA_PROCESSED = 'debug/fig2img/fake_metadata_after_process.npy'

SAVE_PATH = 'debug/fig2img/out.png'
BVH_PATH = 'debug/fig2img/out.bvh'
DB_PATH = 'data/edge_rot_data.npy'

EVALUATE_MOTION_INPUT = 'debug/evaluate/generated_motion_input.npy'
EVALUATE_MOTION_OUTPUT = 'debug/evaluate/generated_motion_to_location_output.npy'
EVALUATE_EDGE_ROT_DICT = 'debug/evaluate/edge_rot_debug.npy'


@pytest.fixture(autouse=True)
def config():
    Edge.enable_global_position()
    Edge.enable_foot_contact()


@pytest.fixture(scope='session')
def motion():
    return torch.tensor(np.load(FAKE_MOTION))


@pytest.fixture(scope='session')
def metadata():
    return np.load(FAKE_METADATA, allow_pickle=True).item()


@pytest.fixture(scope='session')
def metadata_processed():
    return np.load(FAKE_METADATA_PROCESSED, allow_pickle=True)[0]


@pytest.fixture(scope='session')
def static():
    motion_data_raw = np.load(DB_PATH, allow_pickle=True)
    offsets = np.concatenate([motion_data_raw[0]['offset_root'][np.newaxis, :], motion_data_raw[0]['offsets_no_root']])

    return StaticData(parents=motion_data_raw[0]['parents_with_root'],
                      offsets=offsets,
                      names=motion_data_raw[0]['names_with_root'],
                      n_channels=4,
                      enable_global_position=True,
                      enable_foot_contact=True)


@pytest.fixture(scope='session')
def dynamic(static, motion):
    return DynamicData(motion[0].permute(1, 0, 2), static)


@pytest.fixture(scope='session')
def normalisation_data(metadata):
    import ipdb;ipdb.set_trace()
    return {'std': metadata['std'].transpose(1, 2, 0, 3),
            'mean': metadata['mean'].transpose(1, 2, 0, 3),
            'parents_with_root': metadata['parents_with_root']}


@pytest.fixture(scope='session')
def sampled_dynamic(dynamic, normalisation_data):
    sampled_frames = np.linspace(0, dynamic.n_frames - 1, 5).round().astype(int)
    dynamic.normalise(normalisation_data['mean'][:, :, :, 0], normalisation_data['std'][:, :, :, 0])
    dynamic.sample_frames(sampled_frames)

    return dynamic


@pytest.fixture(scope='session')
def motion_input():
    mat = np.load(EVALUATE_MOTION_INPUT, allow_pickle=True)
    return [mat[i] for i in range(len(mat))]


@pytest.fixture(scope='session')
def motion_golden():
    return np.load(EVALUATE_MOTION_OUTPUT, allow_pickle=True)


@pytest.fixture(scope='session')
def evaluate_edge_rot_dict():
    return np.load(EVALUATE_EDGE_ROT_DICT, allow_pickle=True).item()


def test_rotations(static, sampled_dynamic, metadata_processed, normalisation_data):
    rotations_dict = np.insert(metadata_processed['rot_edge_no_root'], 0, metadata_processed['rot_root'], axis=1)
    rotations_static = sampled_dynamic.edge_rotations.permute(2, 1, 0)

    assert rotations_dict.shape == rotations_static.shape
    assert (rotations_dict == rotations_static).all()


def test_root_location(sampled_dynamic, metadata_processed):
    assert (sampled_dynamic.root_location.transpose(0, 1).numpy() == metadata_processed['pos_root']).all()


def test_anim_from_edge_rot(static, sampled_dynamic, metadata_processed, normalisation_data):

    anim, _ = anim_from_edge_rot_dict(metadata_processed)
    anim_static, _ = anim_from_static(static, sampled_dynamic)

    assert (anim.offsets == anim_static.offsets).all()
    assert (anim.rotations == anim_static.rotations).all()
    assert (anim.positions == anim_static.positions).all()
    assert (anim.orients == anim_static.orients).all()


@pytest.mark.skip
def test_generate_figure_static(static, motion, normalisation_data):
    fig = motion2fig(static, motion, normalisation_data=normalisation_data)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()


def test_evaluate_motion_to_location(motion_input, motion_golden, evaluate_edge_rot_dict):
    generated_motions = convert_motions_to_location(motion_input, evaluate_edge_rot_dict, 'mixamo')
    assert np.all(generated_motions == motion_golden)

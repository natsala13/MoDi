import pytest
import numpy as np

from motion_class import StaticSubLayer
from utils.traits import neighbors_by_distance
from utils.data import neighbors_by_distance_o


@pytest.fixture
def test_data(level):
    return np.load(f'tests/data/neighbors_by_distance/neighbors_level_{level}.npy', allow_pickle=True).item()


@pytest.fixture
def parents(test_data):
    return test_data['input']


@pytest.fixture
def affectors_golden(test_data):
    return test_data['output']


@pytest.fixture
def layer(parents):
    return StaticSubLayer(parents, [], 1, [foot_index[1] for foot_index in parents if isinstance(foot_index, tuple)])


# @pytest.mark.parametrize("level", [4, 6, 13, 19, 34])
# def test_neighbors_by_distance(parents, affectors_golden):
#     affectors = neighbors_by_distance_o(parents, 1)

#     assert affectors == affectors_golden

@pytest.mark.parametrize("level", [4, 6, 13, 19, 34])
def test_neighbors_by_distance(layer, affectors_golden):
    affectors = neighbors_by_distance(layer, 1)

    assert affectors == affectors_golden

import pytest
import matplotlib.pyplot as plt

from motion_class import StaticData

PATH = '/Users/nathansala/tau/code/MoDi/data/edge_rot_data.npy'
BVH_EXAMPLE = 'motion0.bvh'
BVH_GENERATED = 'generated_1304.bvh'
BVH_SALUTE = 'Salute.bvh'


# npy_data = np.load(PATH, allow_pickle=True)


@pytest.fixture(autouse=True)
def static():
    return StaticData.init_from_bvh(BVH_EXAMPLE)


@pytest.fixture(autouse=True)
def parents(static):
    return static.parents_list[-1]


@pytest.fixture(autouse=True)
def degree(static, parents):
    return static._topology_degree(parents)


@pytest.fixture(autouse=True)
def pooling(static, degree, parents):
    return static._calculate_pooling_for_level(parents, degree)


def test_parents(parents):
    assert parents == [-1, 0, 1, 2, 3, 4, 5, 6, 4, 8, 9, 10, 11, 4,
                       13, 14, 15, 16, 0, 18, 19, 20, 0, 22, 23, 24]


def test_degree(degree):
    assert degree == [3, 1, 1, 1, 3, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]


def test_sequences(static, degree, parents):
    seq = static._find_seq(0, degree, parents)
    assert seq == [[0, 1, 2, 3, 4], [4, 5, 6, 7], [4, 8, 9, 10, 11, 12], [4, 13, 14, 15, 16, 17], [0, 18, 19, 20, 21],
                   [0, 22, 23, 24, 25]]


def test_pooling_non_normalised(pooling):
    assert pooling == {(0, 2): [(0, 1), (1, 2)], (2, 4): [(2, 3), (3, 4)], (4, 7): [(4, 5), (5, 6), (6, 7)],
                       (4, 9): [(4, 8), (8, 9)], (9, 12): [(9, 10), (10, 11), (11, 12)],
                       (4, 14): [(4, 13), (13, 14)],
                       (14, 17): [(14, 15), (15, 16), (16, 17)], (0, 19): [(0, 18), (18, 19)],
                       (19, 21): [(19, 20), (20, 21)],
                       (0, 23): [(0, 22), (22, 23)], (23, 25): [(23, 24), (24, 25)]}


def test_pooling_normalised(pooling):
    normalised_pooling = StaticData._normalise_joints(pooling)
    print(normalised_pooling)
    assert normalised_pooling == {(0, 1): [(0, 1), (1, 2)], (1, 2): [(2, 3), (3, 4)], (2, 3): [(4, 5), (5, 6), (6, 7)],
                                  (2, 4): [(4, 8), (8, 9)], (4, 5): [(9, 10), (10, 11), (11, 12)],
                                  (2, 6): [(4, 13), (13, 14)], (6, 7): [(14, 15), (15, 16), (16, 17)],
                                  (0, 8): [(0, 18), (18, 19)], (8, 9): [(19, 20), (20, 21)],
                                  (0, 10): [(0, 22), (22, 23)], (10, 11): [(23, 24), (24, 25)]}


def test_all_parents(static):
    assert static.parents_list == [[-1], [-1, 0, 1], [-1, 0, 1, 1, 1, 0, 0],
                                   [-1, 0, 1, 2, 2, 4, 2, 6, 0, 8, 0, 10],
                                   [-1, 0, 1, 2, 3, 4, 5, 6, 4, 8, 9, 10, 11, 4, 13, 14, 15, 16, 0, 18, 19,
                                    20, 0, 22, 23, 24]]


def test_all_pooling(static):
    assert static.skeletal_pooling_dist_1_edges == [{(-1, 0): [(-1, 0), (0, 1), (1, 2)]},
                                                    {(-1, 0): [(-1, 0)], (0, 1): [(0, 1), (0, 5), (0, 6)],
                                                     (1, 2): [(1, 2), (1, 3), (1, 4)]},
                                                    {(-1, 0): [(-1, 0)], (0, 1): [(0, 1), (1, 2)],
                                                     (1, 2): [(2, 3)], (1, 3): [(2, 4), (4, 5)],
                                                     (1, 4): [(2, 6), (6, 7)], (0, 5): [(0, 8), (8, 9)],
                                                     (0, 6): [(0, 10), (10, 11)]},
                                                    {(-1, 0): [(-1, 0)], (0, 1): [(0, 1), (1, 2)],
                                                     (1, 2): [(2, 3), (3, 4)],
                                                     (2, 3): [(4, 5), (5, 6), (6, 7)],
                                                     (2, 4): [(4, 8), (8, 9)],
                                                     (4, 5): [(9, 10), (10, 11), (11, 12)],
                                                     (2, 6): [(4, 13), (13, 14)],
                                                     (6, 7): [(14, 15), (15, 16), (16, 17)],
                                                     (0, 8): [(0, 18), (18, 19)],
                                                     (8, 9): [(19, 20), (20, 21)],
                                                     (0, 10): [(0, 22), (22, 23)],
                                                     (10, 11): [(23, 24), (24, 25)]}]

# plot_static = StaticData.init_from_bvh(BVH_EXAMPLE)
# for parent in plot_static.parents_list:
#     plot_static.plot(parent)
#     plt.figure()

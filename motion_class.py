# BVH file contains...
#
# rot_edge_no_root - (64, 19, 4)
# parents_with_root - 20
# offsets_no_root - (19, 3)
# rot_root - (64, 4)
# pos_root - (64, 3)
# offset_root - (3,)
# names_with_root - (20,)
# contact - (64, 2, 4)


# edge_rot_dict['parents_with_root']
# [-1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 0, 17, 18]
#
# edge_rot_dict['names_with_root']
# names =['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
#        'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
#        'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
#        'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg',
#        'RightFoot']
""" Design change
* traits class should be used the same - record that each instance needs some parent and pooling list.
* Motion class should replace Edge class - answer to all its needs.
    -> n_joints(entity) = calc over parent's list lengths.
    -> n_frames(entity) = calc over how many parents list there are.
    -> n_channels(entity) = how many parents list exists - take all needed values from constant
    -> n_channels = const - 4
    -> parent_list
    -> skeletal_pooling_dist_1
    -> skeletal_pooling_dist_0
* Most usage
    -> Generator and Discriminator build
        - using pooling list and parent list
        - reading const num channles.
        - reading params as num frames and num joints.
    -> Traits uses entity to call n_edges.
* edge_rot_dict_general usage
    -> foot contact loss -> get foot contact / get foor velo (foot.py)
    -> motion2bvh
    -> motion2fig

*** QUESTIONS
-> for some reason parent list is edges indexes in edge list.
* Why do we save real_motion in the start of the run?
* What to do with mean and std?
* Whats the use of enable foot contact? it adds edges to parent list.
* Where are those conversions between array and tensor happening?

#### TODO:
* Change preprocessing instead of saving that npy db, we want to save tensors.
* The end goal is to train a model using my new construction letting generate a skeleton's model dynamically.
* Root location and feet position
    -> during pooling root position and feet contact remain (almost) independent
    -> during convolution both are neigbhoor of something,... to check.

### TODO: Steps
[v] stop using edges -> move to parents lists. DONE (I still save edges for simplicity...)
[v] update git.
[v] add root position.
[v] add foot location.
[X] pooling for after primal skeleton. - MAKE AUTOMATIC
[v] reverse lists
[v] return offsets.
[v] Create dynamic class.
[X] Neighbors list
[ ] Remove root position and foot contact.

[ ] make training working using motion class.
[ ] make sure that bvh is loaded properly from any bvh including pre process function.
[ ] save bvh correctly?
[ ] make skeleton dist1 and dist0 pooling the same - its just mean/max pooling vs spatial pooling from same list.
[ ] look up pre process edges - makes some preprocessing on all bvh data before using it.
[ ] Maybe change the skeleton traits to hold a static object.
[ ] names of feet are not the same for rvery topology...

[ ] Reducing size of model - look up at 3d -> 2d convolutions in traits.

PREPROCESSING
[ ] list to specify wchich joint we are using out of all ones.

"""

import torch
import numpy as np
from Motion import BVH

import networkx as nx
import matplotlib.pyplot as plt

from Motion.AnimationStructure import children_list


BVH_EXAMPLE = 'tests/motion0.bvh'
BVH_GENERATED = 'tests/generated_1304.bvh'

LEFT_FOOT_NAME = 'LeftFoot'
RIGHT_FOOT_NAME = 'RightFoot'


class EdgePoint(tuple):
    def __new__(cls, a, b):
        return super(EdgePoint, cls).__new__(cls, [a, b])

    def __repr__(self):
        return f'Edge{super(EdgePoint, self).__repr__()}'


class StaticData:
    def __init__(self, parents: [int], offsets: np.array, names: [str], n_channels=4,
                 enable_global_position=False, enable_foot_contact=False, rotation_representation='quaternion'):
        self.offsets = offsets
        self.names = names

        self.parents_list, self.skeletal_pooling_dist_1_edges = self.calculate_all_pooling_levels(parents)  # TODO: Make cached properties.
        self.skeletal_pooling_dist_1 = [{edge[1]: [e[1] for e in pooling[edge]] for edge in pooling}
                                        for pooling in self.skeletal_pooling_dist_1_edges]

        self.skeletal_pooling_dist_0 = [{edge[1]: [pooling[edge][-1][1]] for edge in pooling}
                                        for pooling in self.skeletal_pooling_dist_1_edges]

        self.skeletal_pooling_dist_0_edges = None
        self.edges_list = None  # TODO: I dont think I need those 2 varianles anymore...

        # Configurations
        self.__n_channels = n_channels

        self.enable_global_position_flag = False
        self.enable_foot_contact_flag = False

        if enable_global_position:
            self.enable_global_position()
        if enable_foot_contact:
            self.enable_foot_contact()
        if rotation_representation == 'repr6d':
            self.enable_repr6d()

    @staticmethod
    def str():  # TODO: Understand how to remove that.
        return 'Edge'

    @property
    def parents(self):
        return self.parents_list[-1][:len(self.names)]

    @classmethod
    def init_from_bvh(cls, bvf_filepath: str, enable_global_position=False,
                      enable_foot_contact=False, rotation_representation='quaternion'):
        animation, names, frametime = BVH.load(bvf_filepath)
        return cls(animation.parents, animation.offsets, names,
                   enable_global_position=enable_global_position,
                   enable_foot_contact=enable_foot_contact,
                   rotation_representation=rotation_representation)

    # @property
    # def offsets(self) -> np.ndarray:  # TODO: Should I add also the global position?
    #     raise NotImplementedError

    @property
    def entire_motion(self) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def edge_list(parents: [int]) -> [EdgePoint]:
        return [EdgePoint(dst, src + 1) for src, dst in enumerate(parents[1:])]

    @property
    def n_channels(self) -> int:
        return self.__n_channels

    def enable_repr6d(self):  # TODO: Instead of using this function, just create the Static data with configuration.
        self.__n_channels = 6

    def enable_marker4(self):
        self.__n_channels = 12
    # @n_channels.setter
    # def n_channels(self, val: int) -> None:
    #     self.__n_channels = val

    @property
    def n_edges(self):
        return [len(parents) for parents in self.parents_list]

    def save_to_bvh(self, out_filepath: str) -> None:
        raise NotImplementedError

    def is_global_position_enabled(self):  # TODO: remove
        return self.enable_global_position_flag

    def enable_global_position(self):
        """
        TODO: Fooly understand why is it for...
        add a special entity that would be the global position.
        The entity is appended to the edges list.
        No need to really add it in edges_list and all the other structures that are based on tupples. We add it only
        to the structures that are based on indices.
        Its neighboring edges are the same as the neightbors of root """

        if self.enable_global_position_flag:
            return
        self.enable_global_position_flag = True

        for pooling_list in [self.skeletal_pooling_dist_0, self.skeletal_pooling_dist_1]:
            for pooling_hierarchical_stage in pooling_list:
                n_small_stage = max(pooling_hierarchical_stage.keys()) + 1
                n_large_stage = max(val for edge in pooling_hierarchical_stage.values() for val in edge) + 1
                pooling_hierarchical_stage[n_small_stage] = [n_large_stage]

        for parents in self.parents_list:
            parents.append(-2)

    def is_foot_contact_enabled(self, level=-1):
        # return any([isinstance(parent, tuple) and parent[0] == -3 for parent in cls.parents_list[level]])
        return self.enable_foot_contact_flag

    def foot_indexes(self):
        """Run overs pooling list and calculate foot location at each level"""
        foot_indexes = [i for i, name in enumerate(self.names) if name in [LEFT_FOOT_NAME, RIGHT_FOOT_NAME]]
        all_foot_indexes = [foot_indexes]
        for pooling in self.skeletal_pooling_dist_1[::-1]:
            all_foot_indexes += [[k for k in pooling if any(foot in pooling[k] for foot in all_foot_indexes[-1])]]

        return all_foot_indexes[::-1]

    def enable_foot_contact(self):
        """ add special entities that would be the foot contact labels.
        The entities are appended to the edges list.
        No need to really add them in edges_list and all the other structures that are based on tuples. We add them only
        to the structures that are based on indices.
        Their neighboring edges are the same as the neighbors of the feet """

        if self.enable_foot_contact_flag:
            return

        self.enable_foot_contact_flag = True

        all_foot_indeces = self.foot_indexes()

        for parent, foot_indeces in zip(self.parents_list, all_foot_indeces):
            for foot_index in foot_indeces:
                parent.append((-3, foot_index))

        for pooling_list in [self.skeletal_pooling_dist_0, self.skeletal_pooling_dist_1]:
            for pooling_hierarchical_stage, foot_indeces in zip(pooling_list, all_foot_indeces):  # TODO: Do not update the last one
                for _ in foot_indeces:
                    n_small_stage = max(pooling_hierarchical_stage.keys()) + 1
                    n_large_stage = max(val for edge in pooling_hierarchical_stage.values() for val in edge) + 1
                    pooling_hierarchical_stage[n_small_stage] = [n_large_stage]


        # for hierarchical_stage_idx, (feet_idx, parents) in enumerate(zip(cls.feet_idx_list, cls.parents_list)):
        #     for idx, foot in enumerate(feet_idx):
        #         # new entry's 'parent' would be a tuple (-3, foot)
        #         parents.append((-3, foot))
        #
        #         if hierarchical_stage_idx < cls.n_hierarchical_stages-1:  # update pooling only for stages lower than last
        #             last_idx_this = cls.n_edges[hierarchical_stage_idx] + idx
        #             last_idx_larger = cls.n_edges[hierarchical_stage_idx+1] + idx
        #             for pooling_list in [cls.skeletal_pooling_dist_0, cls.skeletal_pooling_dist_1]:
        #                 # last entry in current hierarchy pools from last entry in larger hierarchy
        #                 pooling_list[hierarchical_stage_idx][last_idx_this] = [last_idx_larger]

    @staticmethod
    def _topology_degree(parents: [int]):
        joints_degree = [0] * len(parents)

        for joint in parents[1:]:
            joints_degree[joint] += 1

        return joints_degree

    @staticmethod
    def _find_seq(index: int, joints_degree: [int], parents: [int]) -> [[int]]:
        """Recursive search to find a list of all straight sequences of a skeleton."""
        if joints_degree[index] == 0:
            return [[index]]

        all_sequences = []
        if joints_degree[index] > 1 and index != 0:
            all_sequences = [[index]]

        children_list = [dst for dst, src in enumerate(parents) if src == index]

        for dst in children_list:
            sequence = StaticData._find_seq(dst, joints_degree, parents)
            sequence[0] = [index] + sequence[0]
            all_sequences += sequence

        return all_sequences

    @staticmethod
    def _find_leaves(index: int, joints_degree: [int], parents: [int]) -> [[int]]:
        """Recursive search to find a list of all leaves and their connected joint in a skeleton rest position"""
        if joints_degree[index] == 0:
            return []

        all_leaves_pool = []
        connected_leaves = []
        # if joints_degree[index] > 1 and index != 0:
        #     all_sequences = [[index]]

        children_list = [dst for dst, src in enumerate(parents) if src == index]

        for dst in children_list:
            leaves = StaticData._find_leaves(dst, joints_degree, parents)
            if leaves:
                all_leaves_pool += leaves
            else:
                connected_leaves += [dst]

        if connected_leaves:
            all_leaves_pool += [[index] + connected_leaves]

        return all_leaves_pool


    @staticmethod
    def _edges_from_joints(joints: [int]):
        return [(src, dst) for src, dst in zip(joints[:-1], joints[1:])]

    @staticmethod
    def _pooling_for_edges_list(edges: [EdgePoint]) -> list:
        """Return a list sublist of edges of length 2."""
        pooling_groups = [edges[i:i + 2] for i in range(0, len(edges), 2)]
        if len(pooling_groups) > 1 and len(pooling_groups[-1]) == 1:  # If we have an odd numbers of edges pull 3 of them in once.
            pooling_groups[-2] += pooling_groups[-1]
            pooling_groups = pooling_groups[:-1]

        return pooling_groups

    @staticmethod
    def flatten_dict(values):
        return {k: sublist[k] for sublist in values for k in sublist}

    @staticmethod
    def _calculate_degree1_pooling(parents: [int], degree: [int]) -> {EdgePoint: [EdgePoint]}:
        """Pooling for complex skeleton by trimming long sequences into smaller ones."""
        all_sequences = StaticData._find_seq(0, degree, parents)
        edges_sequences = [StaticData._edges_from_joints(seq) for seq in all_sequences]

        # TODO: Lets keep it to parent lists - I am not sure I want this.
        pooling = [{(edge[0][0], edge[-1][-1]): edge for edge in StaticData._pooling_for_edges_list(edges)} for edges in
                   edges_sequences]
        pooling = StaticData.flatten_dict(pooling)

        # pooling2 = [{joints[-1]: joints for joints in StaticData._pooling_for_edges_list(joints_sequence[1:])} for joints_sequence in
        #            all_sequences]
        # pooling2 = StaticData.flatten_dict(pooling2)

        return pooling

    @staticmethod
    def _calculate_leaves_pooling(parents: [int], degree: [int]) -> {EdgePoint: [EdgePoint]}:
        # all_leaves = StaticData._find_leaves(0, degree, parents)
        all_sequences = StaticData._find_seq(0, degree, parents)
        edges_sequences = [StaticData._edges_from_joints(seq) for seq in all_sequences]

        all_joints = [joint for joint, d in enumerate(degree) if d > 0]
        pooling = {}

        for joint in all_joints:
            pooling[joint] = [edge[0] for edge in edges_sequences if edge[0][0] == joint]

        return {pooling[k][0]: pooling[k] for k in pooling}


    @staticmethod
    def _calculate_pooling_for_level(parents: [int], degree: [int]) -> {EdgePoint: [EdgePoint]}:
        if any(d == 1 for d in degree):
            return StaticData._calculate_degree1_pooling(parents, degree)
        else:
            return StaticData._calculate_leaves_pooling(parents, degree)

    @staticmethod
    def _normalise_joints(pooling: {EdgePoint: [EdgePoint]}) -> {EdgePoint: [EdgePoint]}:
        max_joint = 0
        joint_to_new_joint: {int: int} = {-1: -1, 0: 0}
        new_edges = {}

        for edge in sorted(pooling, key=lambda x: x[1]):
            if edge[1] > max_joint:
                max_joint += 1
                joint_to_new_joint[edge[1]] = max_joint

            new_joint = tuple(joint_to_new_joint[e] for e in edge)
            new_edges[new_joint] = pooling[edge]

        return new_edges

    @staticmethod
    def _edges_to_parents(edges: [EdgePoint]):
        return [edge[0] for edge in edges]

    def calculate_all_pooling_levels(self, parents0):
        # pooling = self._calculate_pooling_for_level(self.parents)
        all_parents = [list(parents0)]
        all_poolings = []
        degree = StaticData._topology_degree(all_parents[-1])

        while any(d == 1 for d in degree):
            pooling = self._calculate_degree1_pooling(all_parents[-1], degree)
            pooling[(-1, 0)] = [(-1, 0)]  # Add root rotation pooling.

            normalised_pooling = self._normalise_joints(pooling)
            normalised_parents = self._edges_to_parents(normalised_pooling.keys())

            all_parents += [normalised_parents]
            all_poolings += [normalised_pooling]

            degree = StaticData._topology_degree(all_parents[-1])

        # TODO: make pooling after primal skeleton automatic.
        all_parents += [[-1, 0, 1], [-1]]
        all_poolings += [{(-1, 0): [(-1, 0)], (0, 1): [(0, 1), (0, 5), (0, 6)], (1, 2): [(1, 2), (1, 3), (1, 4)]},
                         {(-1, 0): [(-1, 0), (0, 1), (1, 2)]}]

        return all_parents[::-1], all_poolings[::-1]

    def neighbors_by_distance(self, parents: [int], dist=1):
        assert dist in [0, 1], 'distance larger than 1 is not supported yet'

        neighbors = {joint_idx: [joint_idx] for joint_idx in range(len(parents))}
        if dist == 0:  # code should be general to any distance. for now dist==1 is the largest supported
            return neighbors

        # handle non virtual joints
        children = children_list(parents)

        # for joint_idx in range(n_entities):
        for dst, src in enumerate(parents):
            # parent_idx = parents[joint_idx]
            if src not in [-1, -2] and not isinstance(src, tuple):
                # -1 is the parent of root. -2 is the parent of global location, tuple for foot_contact
                neighbors[dst].append(src)  # add entity's parent
            neighbors[dst].extend(children[dst])  # append all entity's children

        # handle global pos virtual joint
        if -2 in parents:  # Global position is enabled..
            root_idx = parents.index(-1)
            glob_pos_idx = parents.index(-2)

            # global position should have same neighbors of root and should become his neighbors' neighbor
            neighbors[glob_pos_idx].extend(neighbors[root_idx])
            for root_neighbor in neighbors[root_idx]:
                # changing the neighbors of root during iteration puts the new neighbor in the iteration
                if root_neighbor != root_idx:
                    neighbors[root_neighbor].append(glob_pos_idx)
            neighbors[root_idx].append(glob_pos_idx)  # finally change root itself

        # handle foot contact label virtual joint
        foot_and_contact_label = [(i, parents[i][1]) for i in range(len(parents)) if
                                  isinstance(parents[i], tuple) and parents[i][0] == -3]

        # 'contact' joint should have same neighbors of related joint and should become his neighbors' neighbor
        for foot_idx, contact_label_idx in foot_and_contact_label:
            neighbors[contact_label_idx].extend(neighbors[foot_idx])
            for foot_neighbor in neighbors[foot_idx]:
                # changing the neighbors of root during iteration puts the new neighbor in the iteration
                if foot_neighbor != foot_idx:
                    neighbors[foot_neighbor].append(contact_label_idx)
            neighbors[foot_idx].append(contact_label_idx)  # finally change foot itself

        return neighbors

    def plot(self, parents):
        graph = nx.Graph()
        graph.add_edges_from(self.edge_list(parents))
        nx.draw_networkx(graph)
        plt.show()


class DynamicData:
    def __init__(self, motion: torch.tensor):
        self.motion = motion

    @classmethod
    def init_from_bvh(cls, bvf_filepath: str):
        animation, _, _ = BVH.load(bvf_filepath)
        return cls(animation.rotations)

    # @property
    # def motion(self) -> torch.tensor:
    #     raise NotImplementedError

    @property
    def edge_rotations(self) -> torch.tensor:
        raise NotImplementedError

    @property
    def foot_contact(self) -> torch.tensor:
        raise NotImplementedError

    @property
    def root_location(self) -> torch.tensor:
        raise NotImplementedError

    @property
    def static(self) -> StaticData:
        raise NotImplementedError

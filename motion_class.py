""" Design change
* traits class should be used the same - record that each instance needs some parent and pooling list.
* Motion class should replace Edge class - answer to all its needs.
    -> n_joints(entity) = calc over parent's list lengths.
    -> n_frames(entity) = calc over how many parents list there are.
    -> n_channels(entity) = how many parents list exists - take all needed values from constant
    -> n_channels = {Edge: 4, Joint: 3}
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

### BVH STRUCTURE
* At bvh file, rotations are stored as the rotation of the joint itself.
* In Modi, rotation is pre processed to be stored as the rotation of the joint the edge is pointing to.
* At preprocessing, two new joints are added to every "central" joint (as Hips) so that different feet can move differently.
* Before saving to BVH, we need to transfer rotation of joints to their parents one.
* This feature of adding two more joints for every "central" one, is not working so good.

*** QUESTIONS
** What is data un normalise? what happem if shape is not equal? should we not normalise the data?
* why do we sort joints? they are the same order as the input.
* What does the class Dynamic should look like?
    * Holding a static?
    * Batch dimension? YES
* How to use ClearML?
* In save2bvh we keep asking is motion data a list, Why?

#### TODO:
* Change preprocessing instead of saving that npy db, we want to save tensors.
* The end goal is to train a model using the new construction letting generate a skeleton's model dynamically.
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
[X] Remove root position and foot contact.
[ ] Dynamic class - use velocity flag
[X] Remove expand topology
[x] generate function exist both in generate and in evaluate.
[v] Stop with default None values.
[v] Foot contact for more than 2 feet - change preprocessing
[v] generate bug.

[v] make training working using motion class.
[ ] make sure that bvh is loaded properly from any bvh including pre process function.
[v] save bvh correctly?
[ ] make skeleton dist1 and dist0 pooling the same - its just mean/max pooling vs spatial pooling from same list.
[ ] look up pre process edges - makes some preprocessing on all bvh data before using it.
[?] Maybe change the skeleton traits to hold a static object.
[ ] names of feet are not the same for every topology...
[ ] Every time we load an image we convert it to float and transpose it - change it inside the loader itself.

[ ] Reducing size of model - look up at 3d -> 2d convolutions in traits.

PREPROCESSING
[ ] list to specify which joint we are using out of all ones.

NOTES
* anim_from_edge_rot_dict apply extend joints on motion
* expand joints is important otherwise we see some inconsistency at plotting

"""
import copy
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from Motion import BVH
from Motion.Animation import Animation
from Motion.Quaternions import Quaternions
from utils.data import expand_topology_edges
from Motion.AnimationStructure import children_list, get_sorted_order


BVH_EXAMPLE = 'tests/motion0.bvh'
BVH_GENERATED = 'tests/generated_1304.bvh'

LEFT_FOOT_NAME = 'LeftFoot'
LEFT_TOE = 'LeftToeBase'
RIGHT_FOOT_NAME = 'RightFoot'
RIGHT_TOE = 'RightToeBase'


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

        self.parents_list, self.skeletal_pooling_dist_1_edges = self.calculate_all_pooling_levels(parents)
        self.skeletal_pooling_dist_1 = [{edge[1]: [e[1] for e in pooling[edge]] for edge in pooling}
                                        for pooling in self.skeletal_pooling_dist_1_edges]

        self.skeletal_pooling_dist_0 = [{edge[1]: [pooling[edge][-1][1]] for edge in pooling}
                                        for pooling in self.skeletal_pooling_dist_1_edges]

        # self.skeletal_pooling_dist_0_edges = None
        # self.edges_list = None  # TODO: I dont think I need those 2 varianles anymore...

        # Configurations
        self.__n_channels = n_channels

        self.enable_global_position = enable_global_position
        self.enable_foot_contact = enable_foot_contact

        if enable_global_position:
            self._enable_global_position()
        if enable_foot_contact:
            self._enable_foot_contact()
        if rotation_representation == 'repr6d':
            self._enable_repr6d()

    @classmethod
    def init_from_bvh(cls, bvf_filepath: str, *args, **kwargs):
        animation, names, frametime = BVH.load(bvf_filepath)
        return cls(animation.parents, animation.offsets, names, *args, **kwargs)

    @classmethod
    def init_from_bvh_data(cls, motion, *args, **kwargs):
        offsets = np.concatenate([motion['offset_root'][np.newaxis, :], motion['offsets_no_root']])
        return cls(motion['parents_with_root'], offsets, motion['names_with_root'], *args, **kwargs)

    @property
    def parents(self):
        return self.parents_list[-1][:len(self.names)]

    @property
    def entire_motion(self) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def edge_list(parents: [int]) -> [EdgePoint]:
        return [EdgePoint(dst, src + 1) for src, dst in enumerate(parents[1:])]

    @property
    def n_channels(self) -> int:
        return self.__n_channels

    def _enable_repr6d(self):  # TODO: Instead of using this function, just create the Static data with configuration.
        self.__n_channels = 6

    def _enable_marker4(self):
        self.__n_channels = 12
    # @n_channels.setter
    # def n_channels(self, val: int) -> None:
    #     self.__n_channels = val

    @property
    def n_edges(self):
        return [len(parents) for parents in self.parents_list]

    def save_to_bvh(self, out_filepath: str) -> None:
        raise NotImplementedError

    def _enable_global_position(self):
        """
        TODO: Maybe try and remove that edge?
        add a special entity that would be the global position.
        The entity is appended to the edges list.
        No need to really add it in edges_list and all the other structures that are based on tupples. We add it only
        to the structures that are based on indices.
        Its neighboring edges are the same as the neightbors of root """
        assert self.parents_list[0][-1] != -2

        for pooling_list in [self.skeletal_pooling_dist_0, self.skeletal_pooling_dist_1]:
            for pooling_hierarchical_stage in pooling_list:
                n_small_stage = max(pooling_hierarchical_stage.keys()) + 1
                n_large_stage = max(val for edge in pooling_hierarchical_stage.values() for val in edge) + 1
                pooling_hierarchical_stage[n_small_stage] = [n_large_stage]

        for parents in self.parents_list:
            parents.append(-2)

    def foot_indexes(self, include_toes=True):
        """Run overs pooling list and calculate foot location at each level"""
        # feet_names = [LEFT_FOOT_NAME, LEFT_TOE, RIGHT_FOOT_NAME, RIGHT_TOE] if include_toes else [LEFT_FOOT_NAME,
        #                                                                                           RIGHT_FOOT_NAME]
        feet_names = [LEFT_FOOT_NAME, RIGHT_FOOT_NAME]

        foot_indexes = [i for i, name in enumerate(self.names) if name in feet_names]
        all_foot_indexes = [foot_indexes]
        for pooling in self.skeletal_pooling_dist_1[::-1]:
            all_foot_indexes += [[k for k in pooling if any(foot in pooling[k] for foot in all_foot_indexes[-1])]]

        return all_foot_indexes[::-1]

    @property
    def foot_number(self):
        return len(self.foot_indexes()[-1])

    def _enable_foot_contact(self):
        """ add special entities that would be the foot contact labels.
        The entities are appended to the edges list.
        No need to really add them in edges_list and all the other structures that are based on tuples. We add them only
        to the structures that are based on indices.
        Their neighboring edges are the same as the neighbors of the feet """
        assert not isinstance(self.parents_list[0][-1], tuple)

        all_foot_indexes = self.foot_indexes()
        for parent, foot_indexes in zip(self.parents_list, all_foot_indexes):
            for foot_index in foot_indexes:
                parent.append((-3, foot_index))

        for pooling_list in [self.skeletal_pooling_dist_0, self.skeletal_pooling_dist_1]:
            for pooling_hierarchical_stage, foot_indexes in zip(pooling_list, all_foot_indexes):
                for _ in foot_indexes:
                    n_small_stage = max(pooling_hierarchical_stage.keys()) + 1
                    n_large_stage = max(val for edge in pooling_hierarchical_stage.values() for val in edge) + 1
                    pooling_hierarchical_stage[n_small_stage] = [n_large_stage]

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
        all_parents = [list(parents0)]
        all_poolings = []
        degree = StaticData._topology_degree(all_parents[-1])

        while len(all_parents[-1]) > 2:
            pooling = self._calculate_pooling_for_level(all_parents[-1], degree)
            pooling[(-1, 0)] = [(-1, 0)]

            normalised_pooling = self._normalise_joints(pooling)
            normalised_parents = self._edges_to_parents(normalised_pooling.keys())

            all_parents += [normalised_parents]
            all_poolings += [normalised_pooling]

            degree = StaticData._topology_degree(all_parents[-1])

        # TODO: make pooling after primal skeleton automatic.
        # all_parents += [[-1, 0, 1], [-1]]
        # all_poolings += [{(-1, 0): [(-1, 0)], (0, 1): [(0, 1), (0, 5), (0, 6)], (1, 2): [(1, 2), (1, 3), (1, 4)]},
        #                  {(-1, 0): [(-1, 0), (0, 1), (1, 2)]}]

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

    def plot(self, parents, show=True):
        graph = nx.Graph()
        graph.add_edges_from(self.edge_list(parents))
        nx.draw_networkx(graph)
        if show:
            plt.show()


class DynamicData:
    def __init__(self, motion: torch.tensor, static: StaticData):
        self.motion = motion.clone()  # Shape is B  x K x J x T = batch x channels x joints x frames
        self.static = static

        self.assert_shape_is_right()

        self.use_velocity = True

    def assert_shape_is_right(self):
        assert self.motion.shape[-2] == len(self.static.parents_list[-1])
        assert self.motion.shape[-3] == self.static.n_channels

        foot_contact_joints = self.static.foot_number if self.static.enable_foot_contact else 0
        global_position_joint = 1 if self.static.enable_global_position else 0
        assert len(self.static.names) + global_position_joint + foot_contact_joints == self.motion.shape[-2]

    @classmethod
    def init_from_bvh(cls, bvf_filepath: str,
                      enable_global_position=False,
                      enable_foot_contact=False,
                      rotation_representation='quaternion'):

        animation, names, frametime = BVH.load(bvf_filepath)

        static = StaticData(animation.parents, animation.offsets, names,
                            enable_global_position=enable_global_position,
                            enable_foot_contact=enable_foot_contact,
                            rotation_representation=rotation_representation)

        return cls(animation.rotations.qs, static)

    @classmethod
    def init_from_bvh_data(cls, motion):
        static = StaticData.init_from_bvh_data(motion)
        rotations = np.concatenate([motion['rot_root'][:, np.newaxis], motion['rot_edge_no_root']], axis=1)

        return cls(torch.tensor(rotations.transpose(2, 1, 0)), static)

    def __iter__(self):
        if self.motion.ndim == 4:
            return (DynamicData(motion, self.static) for motion in self.motion).__iter__()
        elif self.motion.ndim == 3:
            return [DynamicData(self.motion, self.static)].__iter__()

    def __getitem__(self, slice_val):
        # TODO: Maybe make sure that the slice doesnt cut out joints or channels?
        return DynamicData(self.motion[slice_val], self.static)

    @property
    def shape(self):
        return self.motion.shape

    @property
    def n_frames(self):
        return self.motion.shape[-1]

    @property
    def n_channels(self):
        return self.motion.shape[-3]

    @property
    def n_joints(self):
        return len(self.static.names)

    @property
    def edge_rotations(self) -> torch.tensor:
        return self.motion[..., :self.n_joints, :]
        # Return only joints representing motion, maybe having a batch dim

    @property
    def foot_contact(self) -> torch.tensor:
        raise NotImplementedError

    @property
    def root_location(self) -> torch.tensor:
        location = self.motion[..., :3, self.n_joints, :]  # drop the 4th item in the position tensor
        location = np.cumsum(location, axis=1) if self.use_velocity else location

        return location  # K x T

    def normalise(self, mean: torch.tensor, std: torch.tensor):
        return DynamicData(self.motion * std + mean, self.static)

    def sample_frames(self, frames_indexes: [int]):
        return DynamicData(self.motion[..., frames_indexes], self.static)

    def basic_anim(self):
        offsets = self.static.offsets
        offsets[0, :] = 0

        positions = np.repeat(offsets[np.newaxis], self.n_frames, axis=0)
        positions[:, 0] = self.root_location.transpose(0, 1)

        orients = Quaternions.id(self.n_joints)
        rotations = Quaternions(self.edge_rotations.permute(2, 1, 0).numpy())

        if rotations.shape[-1] == 6:  # repr6d
            from Motion.transforms import repr6d2quat
            rotations = repr6d2quat(rotations)

        anim_edges = Animation(rotations, positions, orients, offsets, self.static.parents)

        return anim_edges

    def move_rotation_values_to_parents(self, anim_exp):
        children_all_joints = children_list(anim_exp.parents)
        for idx, children_one_joint in enumerate(children_all_joints[1:]):
            parent_idx = idx + 1
            if len(children_one_joint) > 0:  # not leaf
                assert len(children_one_joint) == 1 or (anim_exp.offsets[children_one_joint] == np.zeros(3)).all() and (
                        anim_exp.rotations[:, children_one_joint] == Quaternions.id((self.n_frames, 1))).all()
                anim_exp.rotations[:, parent_idx] = anim_exp.rotations[:, children_one_joint[0]]
            else:
                anim_exp.rotations[:, parent_idx] = Quaternions.id((self.n_frames))

        return anim_exp

    def anim_from_static(self):
        anim_edges = self.basic_anim()

        # TODO: Why do we need to sort names? isnt it already sorted correctly?
        # sorted_order = get_sorted_order(anim_edges.parents)
        # anim_edges_sorted = anim_edges[:, sorted_order]
        # names_sorted = static.names['names_with_root'][sorted_order]

        anim_exp, _, names_exp, _ = expand_topology_edges(anim_edges, names=self.static.names, nearest_joint_ratio=1)
        anim_exp = self.move_rotation_values_to_parents(anim_exp)

        return anim_exp, names_exp

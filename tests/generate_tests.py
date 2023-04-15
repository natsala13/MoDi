import time
import torch
import pytest
import numpy as np
import os.path as osp

from utils.data import Edge
from models.gan import Generator, Discriminator
from utils.traits import SkeletonAwareConv3DTraits, SkeletonAwareFastConvTraits

N_MLP = 8
LATENTS = 512
TRANCANTION = 1
N_INPLACE_CONVOLUTIONS = 2

DEVICE = 'cuda'
DEFAULT_SEED = 1304
EPSILON = 0.001
CHECKPOINT_PATH = 'checkpoints/MoDi_u20_shab87_clearml_ffc4aa_079999.pt'
DATA_PATH = 'tests/data/'
GOLDEN_PATH = 'samples/golden_smaple_1304.npy'


@pytest.fixture(autouse=True)
def entity():
    Edge.enable_global_position()
    Edge.enable_foot_contact()

    return Edge()


# @pytest.fixture
# def regular_generator(checkpoint, entity, traits_class):
#     return create_generator(checkpoint, entity, SkeletonAwareFastConvTraits)
#
#
# @pytest.fixture
# def regular_generator(checkpoint, entity, traits_class):
#     return create_generator(checkpoint, entity, traits_class)


@pytest.fixture(autouse=True)
def checkpoint():
    return torch.load(CHECKPOINT_PATH)


@pytest.fixture
def mean_latent():
    raise NotImplementedError


@pytest.fixture(autouse=True)
def seed():
    return DEFAULT_SEED


@pytest.fixture
def batch():
    return 1


@pytest.fixture
def style(seed, batch):
    rnd_generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    return torch.randn(batch, LATENTS, device=DEVICE, generator=rnd_generator)


def create_generator(checkpoint, entity, traits_class):
    my_generator = Generator(
        LATENTS, N_MLP, traits_class=traits_class, entity=entity, n_inplace_conv=N_INPLACE_CONVOLUTIONS
    ).to(DEVICE)

    my_generator.load_state_dict(checkpoint["g_ema"])
    return my_generator


def sample(generator, style):
    before = time.time()

    style = [style] if style is not list else style  # If our batch dimension is 1.
    motion, w_latent, _ = generator(
        style, truncation=TRANCANTION, truncation_latent=mean_latent,
        return_sub_motions=False, return_latents=True)

    runtime = time.time() - before

    return motion, runtime


@pytest.fixture
def generator_3d(checkpoint, entity):
    return create_generator(checkpoint, entity, SkeletonAwareConv3DTraits)


@pytest.fixture
def sample_3d(generator_3d, style):
    return sample(generator_3d, style)


@pytest.mark.skip
def test_save_generate_golden(seed, style, checkpoint, entity):
    generator = create_generator(checkpoint, entity, SkeletonAwareConv3DTraits)
    motion, _ = sample(generator, style)

    np.save(osp.join(DATA_PATH, f'motion_{seed}.npy'), motion[0].detach().cpu().numpy())


@pytest.mark.parametrize('traits_class', [SkeletonAwareConv3DTraits, SkeletonAwareFastConvTraits])
def test_compare_golden_and_sampled(checkpoint, style, entity, traits_class):
    golden = np.load('samples/golden_smaple_1304.npy')

    generator = create_generator(checkpoint, entity, traits_class)
    motion, runtime = sample(generator, style)

    # TODO: Use logging
    # print(f'diff using {traits_class} is - {torch.tensor(motion.detach().cpu().numpy() - golden).norm()}')
    # print(f'Runtime for class - {traits_class} is {runtime}')

    assert torch.tensor(motion.detach().cpu().numpy() - golden).norm() < EPSILON


@pytest.mark.parametrize('batch', [6])
def test_compare_between_two_models_batch(checkpoint, entity, style):
    generator_3d = create_generator(checkpoint, entity, SkeletonAwareConv3DTraits)
    generator_fast = create_generator(checkpoint, entity, SkeletonAwareFastConvTraits)

    motion_3d, runtime_3d = sample(generator_3d, style)
    motion_fast, runtime_fast = sample(generator_fast, style)

    diff = (motion_3d.detach().cpu() - motion_fast.detach().cpu()).norm()

    print(f'diff on (batch {len(style)}) between 3d conv and fast one is - {diff}')
    print(f'Runtime for 3d Conv is {runtime_3d}')
    print(f'Runtime for fast Conv is {runtime_fast}')

    assert diff < EPSILON

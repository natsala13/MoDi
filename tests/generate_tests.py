import pytest
import torch
import numpy as np
import os.path as osp

from utils.data import Edge
from models.gan import Generator, Discriminator
from utils.traits import SkeletonAwarePoolTraits, SkeletonAwareConv3DTraits, NonSkeletonAwareTraits

N_MLP = 8
LATENTS = 512
TRANCATION = 1
N_INPLACE_CONVOLUTIONS = 2

DEVICE = 'cuda'
FIXED_SEED = 1304
CHECKPOINT_PATH = 'checkpoints/MoDi_u20_shab87_clearml_ffc4aa_079999.pt'
DATA_PATH = 'tests/data/'
GOLDEN_PATH = 'samples/golden_smaple_1304.npy'


@pytest.fixture(autouse=True)
def entity():
    Edge.enable_global_position()

    Edge.enable_foot_contact()
    # args.axis_up = 1

    return Edge()


@pytest.fixture(autouse=True)
def generator(checkpoint, entity):
    # if hasattr(args, 'n_frames_dataset') and args.n_frames_dataset == 128:
    # SkeletonAwareConv3DTraits.set_num_frames(128)

    generator = Generator(
        LATENTS, N_MLP, traits_class=SkeletonAwareConv3DTraits, entity=entity, n_inplace_conv=N_INPLACE_CONVOLUTIONS
    ).to(DEVICE)

    generator.load_state_dict(checkpoint["g_ema"])

    return generator


@pytest.fixture(autouse=True)
def checkpoint():
    return torch.load(CHECKPOINT_PATH)


@pytest.fixture
def mean_latent():
    raise NotImplementedError


@pytest.fixture(autouse=True)
def seed():
    return FIXED_SEED


@pytest.fixture
def style(seed):
    rnd_generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    return torch.randn(1, LATENTS, device=DEVICE, generator=rnd_generator)


@pytest.fixture
def sample(generator, style):
    motion, w_latent, _ = generator(
        [style], truncation=TRANCATION, truncation_latent=mean_latent,
        return_sub_motions=False, return_latents=True)

    return motion


@pytest.mark.skip
def test_save_generate_golden(seed, sample):
    np.save(osp.join(DATA_PATH, f'motion_{seed}.npy'), sample[0].detach().cpu().numpy())


def test_compare_golden_and_sampled(seed, sample):
    golden = np.load('samples/golden_smaple_1304.npy')

    assert (sample.detach().cpu().numpy() == golden).all()


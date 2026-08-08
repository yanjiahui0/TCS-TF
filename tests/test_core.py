import math

import numpy as np
import torch

from tcstf.conformal import SplitConformalPathSet
from tcstf.data.synthetic import generate_s1_aliasing
from tcstf.losses import energy_score
from tcstf.metrics import payload_reduction, recovery_margin_ratio
from tcstf.models.decoder import ReferenceConsistentDecoder
from tcstf.models.encoder import GatedTrajectoryEncoder, KnownStatistics
from tcstf.models.generator import ConditionalLowRankGaussianMixture
from tcstf.tasks.precaution import PrecautionTaskFamily


def test_s1_marginals_and_dependence():
    _, y, meta = generate_s1_aliasing(100_000, seed=2)
    assert np.max(np.abs(y.mean(axis=0) - 0.5)) < 0.01
    for rho in (-0.9, 0.9):
        mask = meta.rho == rho
        pairs = y[mask].reshape(mask.sum(), -1, 2)
        coincidence = np.mean(pairs[..., 0] == pairs[..., 1])
        assert abs(coincidence - (1 + rho) / 2) < 0.02


def test_theorem_48_minimax_value():
    # max{(1-p)/2, p} is minimized at p=1/3 and equals 1/3.
    p = 1 / 3
    assert math.isclose(max((1 - p) / 2, p), 1 / 3)


def test_reference_consistency_exact():
    torch.manual_seed(0)
    dec = ReferenceConsistentDecoder(eta_dim=3, action_dim=2, x_dim=4, z_dim=5, hidden=(16,))
    eta = torch.randn(7, 3)
    a0 = torch.randn(7, 2)
    x = torch.randn(7, 4)
    z = torch.randn(7, 5)
    out = dec(eta, a0, x, z, a0)
    assert torch.equal(out, torch.zeros_like(out))


def test_encoder_radial_projection_and_dimension():
    enc = GatedTrajectoryEncoder(
        x_dim=3,
        horizon=6,
        channels=2,
        known_stats=KnownStatistics(["mean", "max"]),
        learned_bank_width=7,
        deployed_dim=5,
        hidden=(16,),
        radius=2.0,
    )
    x = torch.randn(11, 3)
    y = torch.randn(11, 6, 2)
    z = enc(x, y)
    assert z.shape == (11, 5)
    assert torch.all(torch.linalg.vector_norm(z, dim=-1) <= 2.0 + 1e-5)


def test_generator_shapes_and_support():
    gen = ConditionalLowRankGaussianMixture(3, 5, hidden=(16,), components=3, low_rank=2, radius=1.5)
    x = torch.randn(8, 3)
    z = gen.sample(x, 7, training_relaxation=True)
    assert z.shape == (8, 7, 5)
    assert torch.all(torch.linalg.vector_norm(z, dim=-1) <= 1.5 + 1e-5)
    gen.eval()
    z2 = gen.sample(x, 9, training_relaxation=False)
    assert z2.shape == (8, 9, 5)


def test_energy_score_finite_and_differentiable():
    samples = torch.randn(5, 6, 4, requires_grad=True)
    obs = torch.randn(5, 4)
    loss = energy_score(samples, obs)
    assert torch.isfinite(loss)
    loss.backward()
    assert samples.grad is not None
    assert torch.isfinite(samples.grad).all()


def test_conformal_quantile_convention():
    cp = SplitConformalPathSet(alpha=0.10)
    q = cp.calibrate_scores(np.arange(1, 10_001))
    k = math.ceil((10_000 + 1) * 0.9)
    assert q == k


def test_margin_ratio_includes_optimization_error():
    k = recovery_margin_ratio(0.1, 0.05, 0.5)
    assert np.isclose(k, 0.5)


def test_payload_reduction_default_setting():
    assert np.isclose(payload_reduction(408, 8), 0.9803921568627451)


def test_precaution_reference_relative_loss_zero():
    task = PrecautionTaskFamily()
    eta = task.sample_params(6, torch.device("cpu"))
    x = torch.randn(6, 4)
    y = torch.randint(0, 2, (6, 24, 1)).float()
    a0 = task.reference_action(eta, x)
    rel = task.relative_loss(eta, a0, x, y)
    assert torch.equal(rel, torch.zeros_like(rel))

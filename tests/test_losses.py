"""
Tests d'intégration du pipeline complet F1-F5 + pertes (Sprint 5, Jour 5).

Usage :
    cd horse-pose-repro
    pytest tests/test_losses.py -v
"""
import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from image_to_skeleton import ImageToSkeleton
from skeleton_to_pose2d import SkeletonToPose2D
from lifting_2d_3d import Lifting2Dto3D
from renderer import GeometricConsistencyLoop
from discriminator import Discriminator
from adversarial_loss import discriminator_loss, generator_adversarial_loss
from geometric_consistency import geometric_consistency_loss
from train import build_models, train_step, render_skeleton_batch, omega_loss_fn

N_KEYPOINTS = 18
BATCH_SIZE = 4


@pytest.fixture
def models():
    return build_models(n_keypoints=N_KEYPOINTS)


@pytest.fixture
def optimizers(models):
    phi, omega, lambda_net, discriminator, geo_loop = models
    generator_params = list(phi.parameters()) + list(omega.parameters()) + list(lambda_net.parameters())
    opt_generator = torch.optim.Adam(generator_params, lr=1e-4)
    opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    return opt_generator, opt_discriminator


@pytest.fixture
def dummy_batch():
    images = torch.randn(BATCH_SIZE, 3, 128, 128)
    prior_poses = torch.rand(BATCH_SIZE, N_KEYPOINTS, 2) * 2 - 1
    return images, prior_poses


# --- Tests unitaires du discriminateur ---

def test_discriminator_output_shape():
    D = Discriminator()
    dummy_skeleton = torch.randn(BATCH_SIZE, 1, 128, 128)
    output = D(dummy_skeleton)
    assert output.shape == (BATCH_SIZE, 1)


def test_discriminator_output_range():
    D = Discriminator()
    dummy_skeleton = torch.randn(BATCH_SIZE, 1, 128, 128)
    output = D(dummy_skeleton)
    assert output.min() >= 0 and output.max() <= 1


# --- Tests unitaires des pertes adversariales ---

def test_adversarial_losses_finite():
    d_real = torch.rand(BATCH_SIZE, 1) * 0.5 + 0.25  # évite les bornes exactes 0/1
    d_fake = torch.rand(BATCH_SIZE, 1) * 0.5 + 0.25

    loss_d = discriminator_loss(d_real, d_fake)
    loss_g = generator_adversarial_loss(d_fake)

    assert not torch.isnan(loss_d)
    assert not torch.isnan(loss_g)
    assert not torch.isinf(loss_d)
    assert not torch.isinf(loss_g)


# --- Tests unitaires de la cohérence géométrique ---

def test_geometric_consistency_loss_finite(models):
    _, _, lambda_net, _, geo_loop = models
    y = torch.randn(BATCH_SIZE, N_KEYPOINTS, 2)
    v = torch.randn(BATCH_SIZE, N_KEYPOINTS, 3)
    loop_results = geo_loop(v)

    losses = geometric_consistency_loss(y, v, loop_results)
    for name, val in losses.items():
        assert not torch.isnan(val), f"{name} est NaN"
        assert not torch.isinf(val), f"{name} est Inf"


def test_geometric_consistency_zero_when_no_rotation():
    """Cas de contrôle : si la 'rotation' est l'identité, L_r3D doit être proche de 0."""
    from renderer import apply_rotation
    v_hat = torch.randn(BATCH_SIZE, N_KEYPOINTS, 3)
    identity = torch.eye(3).unsqueeze(0).expand(BATCH_SIZE, -1, -1)
    v_hat_prime = apply_rotation(v_hat, identity)
    assert torch.allclose(v_hat, v_hat_prime, atol=1e-5)


# --- Test de render_skeleton_batch (fonction beta simplifiée) ---

def test_render_skeleton_batch_shape():
    poses = torch.rand(BATCH_SIZE, N_KEYPOINTS, 2) * 2 - 1
    rendered = render_skeleton_batch(poses)
    assert rendered.shape == (BATCH_SIZE, 1, 128, 128)


def test_render_skeleton_batch_differentiable():
    poses = (torch.rand(BATCH_SIZE, N_KEYPOINTS, 2) * 2 - 1).requires_grad_(True)
    rendered = render_skeleton_batch(poses)
    loss = rendered.sum()
    loss.backward()
    assert poses.grad is not None
    assert not torch.isnan(poses.grad).any()


# --- Test d'intégration : un train_step complet, sur plusieurs itérations ---

def test_train_step_runs_without_error(models, optimizers, dummy_batch):
    images, prior_poses = dummy_batch
    losses = train_step(images, prior_poses, models, optimizers)

    for key in ["loss_d", "loss_g_adv", "loss_gc", "loss_omega", "loss_total"]:
        assert key in losses
        assert losses[key] == losses[key]  # NaN check (NaN != NaN)


def test_train_step_stable_over_multiple_iterations(models, optimizers):
    """Simule quelques itérations pour vérifier l'absence de divergence rapide."""
    for _ in range(5):
        images = torch.randn(BATCH_SIZE, 3, 128, 128)
        prior_poses = torch.rand(BATCH_SIZE, N_KEYPOINTS, 2) * 2 - 1
        losses = train_step(images, prior_poses, models, optimizers)

        for key, val in losses.items():
            assert val == val, f"{key} est devenu NaN pendant l'entraînement"
            assert abs(val) < 1e6, f"{key} explose ({val}) -- divergence probable"


def test_discriminator_weights_change_after_step(models, optimizers, dummy_batch):
    """Vérifie que le discriminateur apprend bien (ses poids changent après un step)."""
    phi, omega, lambda_net, discriminator, geo_loop = models
    images, prior_poses = dummy_batch

    weight_before = discriminator.classifier[2].weight.clone()
    train_step(images, prior_poses, models, optimizers)
    weight_after = discriminator.classifier[2].weight

    assert not torch.allclose(weight_before, weight_after), "Les poids du discriminateur n'ont pas changé"


def test_phi_weights_change_after_step(models, optimizers, dummy_batch):
    """Vérifie que F1 (Phi) apprend bien aussi -- validation finale du mécanisme discuté."""
    phi, omega, lambda_net, discriminator, geo_loop = models
    images, prior_poses = dummy_batch

    weight_before = next(phi.parameters()).clone()
    train_step(images, prior_poses, models, optimizers)
    weight_after = next(phi.parameters())

    assert not torch.allclose(weight_before, weight_after), "Les poids de Phi (F1) n'ont pas changé"

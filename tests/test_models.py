"""
Tests unitaires du pipeline F1-F4 (Sprint 4, Jour 5).

Usage :
    cd horse-pose-repro
    pytest tests/test_models.py -v
"""
import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from image_to_skeleton import ImageToSkeleton
from skeleton_to_pose2d import SkeletonToPose2D
from lifting_2d_3d import Lifting2Dto3D
from renderer import GeometricConsistencyLoop, random_rotation_matrix, apply_rotation, project_to_2d

N_KEYPOINTS = 18
BATCH_SIZE = 4


@pytest.fixture(scope="module")
def phi():
    return ImageToSkeleton()


@pytest.fixture(scope="module")
def omega():
    return SkeletonToPose2D(n_keypoints=N_KEYPOINTS)


@pytest.fixture(scope="module")
def lambda_net():
    return Lifting2Dto3D(n_keypoints=N_KEYPOINTS)


@pytest.fixture(scope="module")
def geo_loop(lambda_net):
    return GeometricConsistencyLoop(lambda_net)


@pytest.fixture
def dummy_image():
    return torch.randn(BATCH_SIZE, 3, 128, 128)


# --- Tests par module isolé ---

def test_phi_output_shape(phi, dummy_image):
    output = phi(dummy_image)
    assert output.shape == (BATCH_SIZE, 1, 128, 128)


def test_phi_output_range(phi, dummy_image):
    output = phi(dummy_image)
    assert output.min() >= 0 and output.max() <= 1


def test_omega_output_shape(omega):
    dummy_skeleton = torch.randn(BATCH_SIZE, 1, 128, 128)
    output = omega(dummy_skeleton)
    assert output.shape == (BATCH_SIZE, N_KEYPOINTS, 2)


def test_lambda_output_shape(lambda_net):
    dummy_pose2d = torch.randn(BATCH_SIZE, N_KEYPOINTS, 2)
    output = lambda_net(dummy_pose2d)
    assert output.shape == (BATCH_SIZE, N_KEYPOINTS, 3)


def test_lambda_reusable(lambda_net):
    """Vérifie que Lambda peut être appelé plusieurs fois avec les mêmes poids (nécessaire pour F4)."""
    pose_a = torch.randn(BATCH_SIZE, N_KEYPOINTS, 2)
    pose_b = torch.randn(BATCH_SIZE, N_KEYPOINTS, 2)
    out_a = lambda_net(pose_a)
    out_b = lambda_net(pose_b)
    assert out_a.shape == out_b.shape == (BATCH_SIZE, N_KEYPOINTS, 3)


# --- Tests géométriques (F4) ---

def test_rotation_matrix_shape():
    R = random_rotation_matrix(BATCH_SIZE)
    assert R.shape == (BATCH_SIZE, 3, 3)


def test_rotation_matrix_orthogonal():
    """Une matrice de rotation valide doit être orthogonale : R @ R^T = identité."""
    R = random_rotation_matrix(BATCH_SIZE)
    identity = torch.eye(3).unsqueeze(0).expand(BATCH_SIZE, -1, -1)
    product = torch.bmm(R, R.transpose(1, 2))
    assert torch.allclose(product, identity, atol=1e-5)


def test_projection_shape():
    v = torch.randn(BATCH_SIZE, N_KEYPOINTS, 3)
    y = project_to_2d(v)
    assert y.shape == (BATCH_SIZE, N_KEYPOINTS, 2)


def test_geometric_loop_output_shapes(geo_loop):
    v = torch.randn(BATCH_SIZE, N_KEYPOINTS, 3)
    results = geo_loop(v)
    assert results["v_hat"].shape == (BATCH_SIZE, N_KEYPOINTS, 3)
    assert results["y_hat"].shape == (BATCH_SIZE, N_KEYPOINTS, 2)
    assert results["v_hat_prime"].shape == (BATCH_SIZE, N_KEYPOINTS, 3)
    assert results["v_prime"].shape == (BATCH_SIZE, N_KEYPOINTS, 3)
    assert results["y_prime"].shape == (BATCH_SIZE, N_KEYPOINTS, 2)


def test_geometric_loop_no_nan(geo_loop):
    v = torch.randn(BATCH_SIZE, N_KEYPOINTS, 3)
    results = geo_loop(v)
    for key, val in results.items():
        assert not torch.isnan(val).any(), f"NaN détecté dans {key}"
        assert not torch.isinf(val).any(), f"Inf détecté dans {key}"


# --- Test d'intégration : chaînage complet F1->F2->F3->F4 ---

def test_full_pipeline_chaining(phi, omega, lambda_net, geo_loop, dummy_image):
    skeleton = phi(dummy_image)
    y = omega(skeleton)
    v = lambda_net(y)
    results = geo_loop(v)

    assert skeleton.shape == (BATCH_SIZE, 1, 128, 128)
    assert y.shape == (BATCH_SIZE, N_KEYPOINTS, 2)
    assert v.shape == (BATCH_SIZE, N_KEYPOINTS, 3)
    assert results["y_prime"].shape == (BATCH_SIZE, N_KEYPOINTS, 2)


def test_full_pipeline_gradient_flows(phi, omega, lambda_net, geo_loop, dummy_image):
    """
    Vérifie que le gradient peut bien remonter jusqu'à Phi (F1) à travers
    toute la chaîne -- validation directe du mécanisme de rétropropagation
    discuté (F1 apprend uniquement via ce chemin).
    """
    dummy_image.requires_grad_(False)  # l'image elle-même n'a pas besoin de gradient

    skeleton = phi(dummy_image)
    y = omega(skeleton)
    v = lambda_net(y)
    results = geo_loop(v)

    loss = results["y_prime"].pow(2).mean()  # perte factice, juste pour tester le flux de gradient
    loss.backward()

    # Vérifie qu'au moins un paramètre de Phi (le tout premier module de la chaîne) a reçu un gradient
    phi_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in phi.parameters())
    assert phi_has_grad, "Le gradient n'atteint pas Phi (F1) -- problème dans le chaînage !"

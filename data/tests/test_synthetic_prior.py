"""
Tests unitaires du prior synthétique (Sprint 3, étape 5.3).

Usage :
    cd horse-pose-repro/data
    pytest tests/test_synthetic_prior.py -v
"""
import os
import pytest
import torch

POSES_FILE = os.path.join(os.path.dirname(__file__), "..", "processed", "synthetic_prior", "synthetic_prior_poses.pt")

EXPECTED_N_POSES = 10000
EXPECTED_N_JOINTS = 18
EXPECTED_N_COORDS = 2


@pytest.fixture(scope="module")
def poses():
    if not os.path.exists(POSES_FILE):
        pytest.skip(f"Fichier introuvable : {POSES_FILE} (lance synthetic_prior.py d'abord)")
    return torch.load(POSES_FILE)


def test_poses_not_empty(poses):
    assert poses.shape[0] > 0, "Le tenseur de poses est vide."


def test_poses_expected_count(poses):
    assert poses.shape[0] == EXPECTED_N_POSES, (
        f"Nombre de poses inattendu : {poses.shape[0]}, attendu {EXPECTED_N_POSES}"
    )


def test_poses_shape(poses):
    assert poses.shape[1:] == (EXPECTED_N_JOINTS, EXPECTED_N_COORDS), (
        f"Forme inattendue : {poses.shape}, attendu (N, {EXPECTED_N_JOINTS}, {EXPECTED_N_COORDS})"
    )


def test_no_nan_or_inf(poses):
    assert not torch.isnan(poses).any(), "Des valeurs NaN sont présentes dans les poses."
    assert not torch.isinf(poses).any(), "Des valeurs Inf sont présentes dans les poses."


def test_poses_roughly_normalized(poses):
    """Les coordonnées doivent être dans une plage raisonnable autour de [-1, 1]."""
    max_abs = poses.abs().max().item()
    assert max_abs <= 1.05, f"Valeur max = {max_abs:.3f}, attendu <= ~1.0 après normalisation."


def test_centroids_near_origin(poses):
    """Après centrage, le centroïde de chaque pose doit être proche de (0,0)."""
    centroids = poses.mean(dim=1)
    dist = torch.norm(centroids, dim=1)
    assert dist.mean().item() < 0.5, (
        f"Distance moyenne des centroïdes à l'origine trop élevée : {dist.mean().item():.3f}"
    )


def test_dataloader_batch(poses):
    """Vérifie que le DataLoader produit des batchs cohérents (formes, absence de NaN)."""
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(poses)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(loader))[0]

    assert batch.shape == (16, EXPECTED_N_JOINTS, EXPECTED_N_COORDS)
    assert not torch.isnan(batch).any()

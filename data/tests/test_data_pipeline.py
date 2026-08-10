"""
Tests unitaires du pipeline de données (Sprint 2, Jour 5, étape 5.2).

Usage :
    cd horse-pose-repro/data
    pytest tests/test_data_pipeline.py -v
"""
import os
import json
import pytest
from PIL import Image
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "processed", "final")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
MASKS_DIR = os.path.join(DATASET_DIR, "masks")
REGISTRY_PATH = os.path.join(DATASET_DIR, "dataset_registry.json")

EXPECTED_SIZE = 128


@pytest.fixture(scope="module")
def registry():
    if not os.path.exists(REGISTRY_PATH):
        pytest.skip(f"Registre introuvable : {REGISTRY_PATH} (lance consolidate_dataset.py d'abord)")
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sample_entries(registry):
    entries = registry["entries"]
    if len(entries) > 30:
        step = len(entries) // 30
        entries = entries[::step][:30]
    return entries


def test_registry_not_empty(registry):
    assert registry["total_pairs"] > 0, "Le dataset consolidé est vide."


def test_no_missing_pairs(registry):
    assert registry["total_missing_pairs"] == 0, (
        f"{registry['total_missing_pairs']} paires image/masque incomplètes détectées."
    )


def test_image_dimensions(sample_entries):
    for entry in sample_entries:
        path = os.path.join(IMAGES_DIR, entry["image"])
        with Image.open(path) as img:
            assert img.size == (EXPECTED_SIZE, EXPECTED_SIZE), (
                f"{entry['image']} : taille {img.size}, attendu {EXPECTED_SIZE}x{EXPECTED_SIZE}"
            )


def test_mask_dimensions(sample_entries):
    for entry in sample_entries:
        path = os.path.join(MASKS_DIR, entry["mask"])
        with Image.open(path) as msk:
            assert msk.size == (EXPECTED_SIZE, EXPECTED_SIZE), (
                f"{entry['mask']} : taille {msk.size}, attendu {EXPECTED_SIZE}x{EXPECTED_SIZE}"
            )


def test_image_not_corrupted(sample_entries):
    for entry in sample_entries:
        path = os.path.join(IMAGES_DIR, entry["image"])
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            pytest.fail(f"{entry['image']} est corrompue : {e}")


def test_mask_not_corrupted(sample_entries):
    for entry in sample_entries:
        path = os.path.join(MASKS_DIR, entry["mask"])
        try:
            with Image.open(path) as msk:
                msk.verify()
        except Exception as e:
            pytest.fail(f"{entry['mask']} est corrompu : {e}")


def test_mask_is_binary(sample_entries):
    for entry in sample_entries:
        path = os.path.join(MASKS_DIR, entry["mask"])
        with Image.open(path) as msk:
            arr = np.array(msk.convert("L"))
            non_binary_ratio = np.mean((arr > 10) & (arr < 245))
            assert non_binary_ratio < 0.05, (
                f"{entry['mask']} : {non_binary_ratio:.1%} de pixels non-binaires, "
                f"masque potentiellement corrompu ou mal converti."
            )


def test_mask_not_empty_or_full(sample_entries):
    for entry in sample_entries:
        path = os.path.join(MASKS_DIR, entry["mask"])
        with Image.open(path) as msk:
            arr = np.array(msk.convert("L")) > 127
            fg_ratio = arr.mean()
            assert 0.01 <= fg_ratio <= 0.90, (
                f"{entry['mask']} : ratio foreground {fg_ratio:.1%}, hors bornes attendues."
            )


def test_image_mask_alignment_shape(sample_entries):
    for entry in sample_entries:
        img_path = os.path.join(IMAGES_DIR, entry["image"])
        mask_path = os.path.join(MASKS_DIR, entry["mask"])
        with Image.open(img_path) as img, Image.open(mask_path) as msk:
            assert img.size == msk.size, (
                f"Désalignement dimensionnel entre {entry['image']} et {entry['mask']}"
            )
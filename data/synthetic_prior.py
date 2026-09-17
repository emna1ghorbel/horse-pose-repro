"""
Chargement du prior synthétique 2D (Sprint 3) — dataset CAD cheval (Mu et al.,
CVPR 2020). Réduit le nuage dense de 3299 points par pose aux 18 points clés
anatomiques officiels, en utilisant les indices exacts définis dans le code
source du dépôt Learning-from-Synthetic-Animals (pose/datasets/synthetic_animal_sp.py),
qui correspondent au schéma d'annotation TigDog.

Ordre des 18 points (cf. README du dépôt officiel, correspondance approximative
à valider visuellement) :
    0: left-eye          6: right-back-hoof   12: right-shoulder
    1: right-eye         7: left-front-knee   13: left-front-elbow
    2: chin              8: right-front-knee  14: right-front-elbow
    3: left-front-hoof   9: left-back-knee    15: left-back-elbow
    4: right-front-hoof 10: right-back-knee   16: right-back-elbow
    5: left-back-hoof   11: left-shoulder     17: point non identifié

Usage :
    python synthetic_prior.py --input-dir ./raw/horse_synthetic/horse_combineds5r5_texture --output-dir ./processed/synthetic_prior
"""
import argparse
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Indices officiels pour le cheval (source : pose/datasets/synthetic_animal_sp.py, dépôt Mu et al.)
HORSE_KEYPOINT_INDICES = np.array([
    1718, 1684, 1271, 1634, 1650, 1643, 1659, 925, 392, 564,
    993, 726, 1585, 1556, 427, 1548, 967, 877
])

JOINT_NAMES = [
    "left_eye", "right_eye", "chin",
    "left_front_hoof", "right_front_hoof", "left_back_hoof", "right_back_hoof",
    "left_front_knee", "right_front_knee", "left_back_knee", "right_back_knee",
    "left_shoulder", "right_shoulder",
    "left_front_elbow", "right_front_elbow", "left_back_elbow", "right_back_elbow",
    "unidentified_point",
]


def extract_18_keypoints(kpts_path: str) -> np.ndarray:
    """Charge un fichier _kpts.npy (3299, 3) et retourne les 18 points sélectionnés (18, 3)."""
    pts = np.load(kpts_path).astype(np.float32)
    pts_18 = pts[HORSE_KEYPOINT_INDICES].copy()
    pts_18[:, 2] = 1.0  # visibilité forcée à 1, comme dans le code officiel (train_with_occlusion)
    return pts_18


def normalize_pose(pts_18: np.ndarray) -> np.ndarray:
    """
    Normalisation simple : centrage sur le centroïde + mise à l'échelle par
    l'étendue max (cf. plan §3.2.1, "normalisation des poses").
    """
    xy = pts_18[:, :2]
    centroid = xy.mean(axis=0)
    xy_centered = xy - centroid
    scale = np.max(np.abs(xy_centered)) + 1e-8
    xy_normalized = xy_centered / scale
    out = pts_18.copy()
    out[:, :2] = xy_normalized
    return out


class SyntheticPosePrior(Dataset):
    """Dataset PyTorch du prior synthétique 2D (poses normalisées, 18 points)."""

    def __init__(self, input_dir: str, normalize: bool = True):
        self.kpts_files = sorted(glob.glob(os.path.join(input_dir, "*kpts.npy")))
        if len(self.kpts_files) == 0:
            raise FileNotFoundError(f"Aucun fichier *_kpts.npy trouvé dans {input_dir}")
        self.normalize = normalize

    def __len__(self):
        return len(self.kpts_files)

    def __getitem__(self, idx):
        pts_18 = extract_18_keypoints(self.kpts_files[idx])
        if self.normalize:
            pts_18 = normalize_pose(pts_18)
        return torch.from_numpy(pts_18[:, :2]).float()  # (18, 2) — x,y seulement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True,
                         help="Dossier contenant les fichiers *_kpts.npy bruts (3299 points)")
    parser.add_argument("--output-dir", type=str, default="./processed/synthetic_prior")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = SyntheticPosePrior(args.input_dir, normalize=True)
    print(f"Dataset chargé : {len(dataset)} poses trouvées.")

    n = args.limit or len(dataset)
    all_poses = torch.stack([dataset[i] for i in range(n)])
    print(f"Poses extraites et normalisées : {all_poses.shape}")  # attendu : (n, 18, 2)

    out_path = os.path.join(args.output_dir, "synthetic_prior_poses.pt")
    torch.save(all_poses, out_path)
    print(f"Sauvegardé : {out_path}")

    # Test du DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    print(f"Test DataLoader : batch shape = {batch.shape} (attendu : [8, 18, 2])")


if __name__ == "__main__":
    main()

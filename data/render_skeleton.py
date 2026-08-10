"""
Fonction de rendu du squelette (fonction beta du papier, §3.2.1) : transforme
un ensemble de 18 points clés en une IMAGE de squelette (lignes reliant les
articulations), pas juste un nuage de points.

Nécessaire pour le discriminateur (F5) : il travaille sur des images de
squelette (comme s = Phi(x)), donc le prior synthétique doit aussi être rendu
en image (w = beta(p)) pour être comparable.

Usage :
    python render_skeleton.py --input-dir ./raw/horse_synthetic/horse_combineds5r5_texture --output-dir ./processed/synthetic_prior/skeleton_samples --n-samples 20
"""
import argparse
import glob
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

HORSE_KEYPOINT_INDICES = np.array([
    1718, 1684, 1271, 1634, 1650, 1643, 1659, 925, 392, 564,
    993, 726, 1585, 1556, 427, 1548, 967, 877
])

JOINT_NAMES = [
    "left_eye", "right_eye", "chin",
    "LF_hoof", "RF_hoof", "LB_hoof", "RB_hoof",
    "LF_knee", "RF_knee", "LB_knee", "RB_knee",
    "L_shoulder", "R_shoulder",
    "LF_elbow", "RF_elbow", "LB_elbow", "RB_elbow",
    "extra",
]

# Topologie du squelette : quelle paire de points est reliée par un "os".
# Approximation anatomique raisonnable (tête -> encolure -> épaules/hanches -> pattes).
# A ajuster si la vérification visuelle (Sprint 3, étape précédente) révèle un
# ordre de joints différent de celui suppose ici.
SKELETON_EDGES =[(10, 11), (0, 1), (8, 9), (3, 4), (12, 14), (13, 15), (5, 6), (12, 13), (0, 2), (3, 8), (16, 17), (11, 17), (7, 16), (9, 15), (5, 10), (7, 12), (2, 12)]


def extract_18_keypoints(kpts_path: str) -> np.ndarray:
    pts = np.load(kpts_path).astype(np.float32)
    return pts[HORSE_KEYPOINT_INDICES]


def normalize_pose(pts_18: np.ndarray) -> np.ndarray:
    xy = pts_18[:, :2]
    centroid = xy.mean(axis=0)
    xy_centered = xy - centroid
    scale = np.max(np.abs(xy_centered)) + 1e-8
    xy_normalized = xy_centered / scale
    out = pts_18.copy()
    out[:, :2] = xy_normalized
    return out


def render_skeleton(pts_18_normalized: np.ndarray, size: int = 128) -> np.ndarray:
    """
    Fonction beta : pose 2D normalisee (coords dans [-1, 1]) -> image de
    squelette (size x size, 1 canal), lignes blanches sur fond noir.
    """
    canvas = np.zeros((size, size), dtype=np.uint8)

    # normalisé [-1,1] -> pixels [0, size] avec marge
    margin = size * 0.15
    xy_pix = (pts_18_normalized[:, :2] + 1) / 2 * (size - 2 * margin) + margin
    xy_pix = xy_pix.astype(int)

    for i, j in SKELETON_EDGES:
        pt1 = tuple(xy_pix[i])
        pt2 = tuple(xy_pix[j])
        cv2.line(canvas, pt1, pt2, color=255, thickness=2)

    for x, y in xy_pix:
        cv2.circle(canvas, (x, y), radius=3, color=255, thickness=-1)

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./processed/synthetic_prior/skeleton_samples")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    kpts_files = sorted(glob.glob(os.path.join(args.input_dir, "*kpts.npy")))

    # Échantillon régulier sur les 10 000 (pas juste les 20 premiers, pour la diversité)
    step = max(1, len(kpts_files) // args.n_samples)
    sample_files = kpts_files[::step][: args.n_samples]

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for i, kpts_path in enumerate(sample_files):
        pts_18 = extract_18_keypoints(kpts_path)
        pts_18_norm = normalize_pose(pts_18)
        skeleton_img = render_skeleton(pts_18_norm, size=args.size)

        out_name = f"skeleton_{i:02d}.png"
        cv2.imwrite(os.path.join(args.output_dir, out_name), skeleton_img)

        axes[i].imshow(skeleton_img, cmap="gray")
        axes[i].set_title(f"#{i}", fontsize=9)
        axes[i].axis("off")

    plt.tight_layout()
    grid_path = os.path.join(args.output_dir, "grid_overview.png")
    plt.savefig(grid_path, dpi=100)
    print(f"{len(sample_files)} squelettes rendus et sauvegardés dans {args.output_dir}")
    print(f"Vue d'ensemble : {grid_path}")


if __name__ == "__main__":
    main()

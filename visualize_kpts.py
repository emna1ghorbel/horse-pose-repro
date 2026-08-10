"""
Visualise les 18 points clés extraits sur l'image synthétique correspondante,
avec le numéro de chaque point affiché — utile pour vérifier visuellement
l'ordre exact des joints (Sprint 3, vérification).

Usage :
    python visualize_prior_keypoints.py --input-dir ./raw/horse_synthetic/horse_combineds5r5_texture --index 0
"""
import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

HORSE_KEYPOINT_INDICES = np.array([
    1718, 1684, 1271, 1634, 1650, 1643, 1659, 925, 392, 564,
    993, 726, 1585, 1556, 427, 1548, 967, 877
])

JOINT_NAMES = [
    "0:left_eye", "1:right_eye", "2:chin",
    "3:LF_hoof", "4:RF_hoof", "5:LB_hoof", "6:RB_hoof",
    "7:LF_knee", "8:RF_knee", "9:LB_knee", "10:RB_knee",
    "11:L_shoulder", "12:R_shoulder",
    "13:LF_elbow", "14:RF_elbow", "15:LB_elbow", "16:RB_elbow",
    "17:extra",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--index", type=int, default=0, help="Index de la pose à visualiser (0-9999)")
    parser.add_argument("--output", type=str, default="keypoints_check.png")
    args = parser.parse_args()

    kpts_files = sorted(glob.glob(os.path.join(args.input_dir, "*kpts.npy")))
    if args.index >= len(kpts_files):
        raise ValueError(f"Index {args.index} hors limite (0-{len(kpts_files)-1})")

    kpts_path = kpts_files[args.index]
    img_path = kpts_path[:-8] + "img.png"  # remplace "kpts.npy" par "img.png"

    pts = np.load(kpts_path).astype(np.float32)
    pts_18 = pts[HORSE_KEYPOINT_INDICES]

    print(f"Fichier : {os.path.basename(kpts_path)}")
    print(f"Nuage brut : {pts.shape}, points extraits : {pts_18.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # --- Image seule ---
    if os.path.exists(img_path):
        img = Image.open(img_path)
        axes[0].imshow(img)
    axes[0].set_title("Image synthétique (sans annotation)")
    axes[0].axis("off")

    # --- Image + 18 points numérotés ---
    if os.path.exists(img_path):
        axes[1].imshow(img)
    axes[1].scatter(pts_18[:, 0], pts_18[:, 1], c="red", s=60, zorder=3, edgecolors="white")
    for i, (x, y) in enumerate(pts_18[:, :2]):
        axes[1].annotate(JOINT_NAMES[i], (x, y), color="yellow", fontsize=8,
                          xytext=(4, 4), textcoords="offset points",
                          bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.6))
    axes[1].set_title("18 points clés extraits (numérotés)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=120)
    print(f"Sauvegardé : {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
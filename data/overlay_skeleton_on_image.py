"""
Superpose le squelette (avec la topologie déduite empiriquement) directement
sur l'image synthétique réelle, pour une pose "calme" (idle/walk plutôt que
jump/death/rear) — beaucoup plus facile à juger visuellement que sur fond noir.

Usage :
    python overlay_skeleton_on_image.py --input-dir ./raw/horse_synthetic/horse_combineds5r5_texture --filter idle --index 0
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

# Topologie déduite empiriquement (arbre couvrant minimal sur la variance des distances)
SKELETON_EDGES = [(10, 11), (0, 1), (8, 9), (3, 4), (12, 14), (13, 15), (5, 6), (12, 13), (0, 2), (3, 8), (16, 17), (11, 17), (7, 16), (9, 15), (5, 10), (7, 12), (2, 12)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--filter", type=str, default="idle",
                         help="Filtre sur le nom de fichier (ex: idle, walk) pour choisir des poses calmes")
    parser.add_argument("--index", type=int, default=0, help="Index parmi les fichiers filtrés")
    parser.add_argument("--output", type=str, default="skeleton_overlay.png")
    args = parser.parse_args()

    all_kpts = sorted(glob.glob(os.path.join(args.input_dir, "*kpts.npy")))
    filtered = [f for f in all_kpts if args.filter.lower() in os.path.basename(f).lower()]

    print(f"Fichiers totaux : {len(all_kpts)}, filtrés sur '{args.filter}' : {len(filtered)}")
    if len(filtered) == 0:
        print("Aucun fichier ne correspond au filtre, utilisation de tous les fichiers.")
        filtered = all_kpts

    kpts_path = filtered[args.index]
    img_path = kpts_path[:-8] + "img.png"

    pts = np.load(kpts_path).astype(np.float32)
    pts_18 = pts[HORSE_KEYPOINT_INDICES]

    print(f"Fichier utilisé : {os.path.basename(kpts_path)}")

    fig, ax = plt.subplots(figsize=(8, 8))
    if os.path.exists(img_path):
        img = Image.open(img_path)
        ax.imshow(img)
    else:
        print(f"ATTENTION : image introuvable {img_path}")

    # Lignes du squelette (topologie déduite)
    for i, j in SKELETON_EDGES:
        x = [pts_18[i, 0], pts_18[j, 0]]
        y = [pts_18[i, 1], pts_18[j, 1]]
        ax.plot(x, y, color="lime", linewidth=2, zorder=2)

    # Points numérotés
    ax.scatter(pts_18[:, 0], pts_18[:, 1], c="red", s=50, zorder=3, edgecolors="white")
    for idx, (x, y) in enumerate(pts_18[:, :2]):
        ax.annotate(str(idx), (x, y), color="yellow", fontsize=10, fontweight="bold",
                    xytext=(3, 3), textcoords="offset points")

    ax.set_title(f"Squelette (topologie déduite) sur : {os.path.basename(kpts_path)}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=130)
    print(f"Sauvegardé : {args.output}")
    plt.show()


if __name__ == "__main__":
    main()

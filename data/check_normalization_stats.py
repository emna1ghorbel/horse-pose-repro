"""
Vérification statistique de la distribution des poses normalisées (Sprint 3,
étape 4.2) — histogrammes des coordonnées x/y pour détecter des anomalies
(valeurs aberrantes, poses mal centrées, échelle incohérente).

Usage :
    python check_normalization_stats.py --poses-file ./processed/synthetic_prior/synthetic_prior_poses.pt
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses-file", type=str, required=True)
    parser.add_argument("--output", type=str, default="normalization_stats.png")
    args = parser.parse_args()

    poses = torch.load(args.poses_file)  # (N, 18, 2)
    print(f"Poses chargées : {poses.shape}")

    # --- Vérifications de base ---
    n_nan = torch.isnan(poses).sum().item()
    n_inf = torch.isinf(poses).sum().item()
    print(f"Valeurs NaN    : {n_nan}")
    print(f"Valeurs Inf    : {n_inf}")
    if n_nan > 0 or n_inf > 0:
        print("ATTENTION : des valeurs invalides ont été détectées !")

    x_all = poses[:, :, 0].flatten().numpy()
    y_all = poses[:, :, 1].flatten().numpy()

    print(f"\nX : min={x_all.min():.3f}, max={x_all.max():.3f}, mean={x_all.mean():.3f}, std={x_all.std():.3f}")
    print(f"Y : min={y_all.min():.3f}, max={y_all.max():.3f}, mean={y_all.mean():.3f}, std={y_all.std():.3f}")

    # Attendu (normalisation centrée + échelle max=1) : valeurs dans [-1, 1],
    # moyenne proche de 0 (pas garanti exactement, dépend de la forme du squelette)
    out_of_range = np.sum((np.abs(x_all) > 1.01) | (np.abs(y_all) > 1.01))
    print(f"\nPoints hors de [-1, 1] : {out_of_range} / {len(x_all)} ({100*out_of_range/len(x_all):.2f}%)")

    # --- Centroïde par pose : doit être proche de (0,0) ---
    centroids = poses.mean(dim=1)  # (N, 2)
    centroid_dist = torch.norm(centroids, dim=1)
    print(f"\nDistance moyenne des centroïdes à l'origine : {centroid_dist.mean():.4f} (attendu proche de 0)")
    print(f"Distance max : {centroid_dist.max():.4f}")

    # --- Histogrammes ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(x_all, bins=50, color="steelblue")
    axes[0, 0].set_title("Distribution des coordonnées X")
    axes[0, 0].axvline(0, color="red", linestyle="--", linewidth=1)

    axes[0, 1].hist(y_all, bins=50, color="darkorange")
    axes[0, 1].set_title("Distribution des coordonnées Y")
    axes[0, 1].axvline(0, color="red", linestyle="--", linewidth=1)

    axes[1, 0].hist(centroid_dist.numpy(), bins=50, color="seagreen")
    axes[1, 0].set_title("Distance des centroïdes à l'origine")

    # Scatter de toutes les positions (par joint, pour voir la cohérence anatomique)
    for j in range(poses.shape[1]):
        axes[1, 1].scatter(poses[:, j, 0], poses[:, j, 1], s=1, alpha=0.1)
    axes[1, 1].set_title("Nuage de tous les points (par joint, superposés)")
    axes[1, 1].set_xlim(-1.2, 1.2)
    axes[1, 1].set_ylim(-1.2, 1.2)
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(args.output, dpi=120)
    print(f"\nGraphiques sauvegardés : {args.output}")


if __name__ == "__main__":
    main()

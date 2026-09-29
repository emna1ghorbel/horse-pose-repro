"""
Inspection visuelle des squelettes prédits -- post-entraînement.

Pipeline complet :
    image  --[Phi]-->  heatmap (1,128,128)
                            |
                       [Omega]
                            |
                        pose 2D (18, 2) en [-1, 1]

Sorties par image (grille 3 colonnes) :
  Col 1 : image originale
  Col 2 : heatmap Phi (colormap "hot")
  Col 3 : image originale + squelette superposé (keypoints + os)

Usage :
    python evaluation/skeleton_inspect.py \
        --checkpoint checkpoints/run19_clean_scratch/latest.pt \
        --data-dir data/processed/final \
        --output-dir eval_outputs/skeleton_inspect \
        --output-name inspect_run19_scratch \
        --n-images 16
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torchvision.transforms as T

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "models"))
sys.path.insert(0, os.path.join(REPO_ROOT, "data"))
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))

from image_to_skeleton import ImageToSkeleton
from skeleton_to_pose2d import SkeletonToPose2D

import json

IMG_SIZE = 128

# Arêtes anatomiques du cheval (18 keypoints)
HORSE_EDGES = [
    (0, 2), (1, 2),               # Tête
    (3, 7), (7, 13), (13, 11),    # Jambe avant gauche
    (4, 8), (8, 14), (14, 12),    # Jambe avant droite
    (5, 9), (9, 15), (15, 17),    # Jambe arrière gauche
    (6, 10), (10, 16), (16, 17),  # Jambe arrière droite
    (11, 12),                      # Épaules
    (11, 17), (12, 17),            # Dos
    (2, 11), (2, 12),              # Cou
]

# Noms des 18 keypoints
KP_NAMES = [
    "Oeil G", "Oeil D", "Nez",
    "Epau G",  "Epau D",
    "Hanche G","Hanche D",
    "Genou FG","Genou FD",
    "Genou AG","Genou AD",
    "Pied FG", "Pied FD",
    "Pied AG", "Pied AD",
    "Sabot FG","Sabot FD",
    "Queue",
]

# Couleurs par groupe anatomique
EDGE_COLORS = {
    (0, 2): "#ff6b6b",  (1, 2): "#ff6b6b",           # tête  : rouge
    (2, 11): "#ffd93d", (2, 12): "#ffd93d",            # cou   : jaune
    (11, 12): "#6bcb77",                               # épaules: vert
    (11, 17): "#6bcb77", (12, 17): "#6bcb77",          # dos   : vert
    (3, 7): "#4d96ff",  (7, 13): "#4d96ff", (13, 11): "#4d96ff",   # J.av.G : bleu
    (4, 8): "#845ec2",  (8, 14): "#845ec2", (14, 12): "#845ec2",   # J.av.D : violet
    (5, 9): "#ff9671",  (9, 15): "#ff9671", (15, 17): "#ff9671",   # J.ar.G : orange
    (6, 10): "#f9f871", (10, 16): "#f9f871",(16, 17): "#f9f871",   # J.ar.D : citron
}


# ─────────────────────────────────────────────────────────
# Chargement modèles
# ─────────────────────────────────────────────────────────
def load_models(ckpt_path: str, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    phi   = ImageToSkeleton()
    omega = SkeletonToPose2D(n_keypoints=18)
    phi.load_state_dict(ckpt["phi"])
    omega.load_state_dict(ckpt["omega"])
    phi.to(device).eval()
    omega.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    print(f"  Phi + Omega chargés (epoch={epoch})")
    return phi, omega


# ─────────────────────────────────────────────────────────
# Chargement données
# ─────────────────────────────────────────────────────────
def load_samples(data_dir: str, n: int, seed: int, device):
    registry_path = os.path.join(data_dir, "dataset_registry.json")
    images_dir    = os.path.join(data_dir, "images")

    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)
    entries = registry["entries"]

    rng   = np.random.RandomState(seed)
    total = len(entries)
    step  = max(1, total // n)
    base  = list(range(0, min(n * step, total), step))[:n]
    idxs  = [min(int(i + rng.randint(0, max(1, step // 2))), total - 1) for i in base]

    img_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    imgs_tensor, imgs_display, fnames = [], [], []
    for idx in idxs:
        e   = entries[idx]
        img = Image.open(os.path.join(images_dir, e["image"])).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        imgs_display.append(np.array(img))
        imgs_tensor.append(img_tf(img))
        fnames.append(e["image"])

    imgs_tensor = torch.stack(imgs_tensor).to(device)
    print(f"  {n} images chargées (seed={seed}).")
    return imgs_tensor, imgs_display, fnames


# ─────────────────────────────────────────────────────────
# Dessin du squelette sur l'image
# ─────────────────────────────────────────────────────────
def draw_skeleton_on_ax(ax, img_rgb: np.ndarray, kps: np.ndarray,
                         alpha_skel: float = 0.92):
    """
    img_rgb : (H, W, 3) uint8
    kps     : (18, 2) coordonnées normalisées [-1, 1]
    """
    H, W = img_rgb.shape[:2]
    ax.imshow(img_rgb)

    # Convertir [-1,1] → pixels
    px = (kps[:, 0] + 1) / 2.0 * W
    py = (kps[:, 1] + 1) / 2.0 * H

    # Dessiner les os avec plus d'opacité et d'épaisseur pour un meilleur rendu
    for u, v in HORSE_EDGES:
        color = EDGE_COLORS.get((u, v), EDGE_COLORS.get((v, u), "#ffffff"))
        ax.plot([px[u], px[v]], [py[u], py[v]],
                color=color, linewidth=2.5, alpha=1.0, solid_capstyle="round")

    # Dessiner les keypoints
    for k in range(len(kps)):
        ax.scatter(px[k], py[k], s=25, color="white",
                   edgecolors="#333333", linewidths=0.6, zorder=5, alpha=1.0)

    ax.set_xticks([]); ax.set_yticks([])


# ─────────────────────────────────────────────────────────
# Figure principale
# ─────────────────────────────────────────────────────────
def denorm(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((t.cpu().float() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def build_figure(imgs_tensor, phi_outs, kps_batch,
                 fnames, ckpt_path, commit, n_cols=4):
    n      = len(fnames)
    n_rows = (n + n_cols - 1) // n_cols
    # 3 sous-colonnes par image : original | heatmap | overlay
    fig_w  = n_cols * 3 * 2.2
    fig_h  = n_rows * 2.4 + 1.0

    fig, axes = plt.subplots(
        n_rows, n_cols * 3,
        figsize=(fig_w, fig_h),
        facecolor="#0d1117"
    )
    fig.subplots_adjust(hspace=0.08, wspace=0.04,
                        left=0.01, right=0.99, top=0.93, bottom=0.01)

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    title = (
        f"Inspection squelettes prédits  |  {os.path.basename(ckpt_path)}  "
        f"|  commit: {commit[:12]}\n"
        f"[gauche] Image   [centre] Heatmap Phi   [droite] Squelette superposé"
    )
    fig.text(0.5, 0.968, title, ha="center", va="center",
             fontsize=9, color="white", family="monospace")

    for i, fname in enumerate(fnames):
        row = i // n_cols
        col = i %  n_cols
        base_col = col * 3

        img_np   = denorm(imgs_tensor[i])
        phi_arr  = phi_outs[i, 0].cpu().numpy()
        kps      = kps_batch[i].cpu().numpy()   # (18, 2)
        
        # --- ALIGNEMENT AVANCÉ SANS RÉENTRAÎNEMENT ---
        # Au lieu de juste recentrer, on va redimensionner (scale) le squelette 
        # pour qu'il rentre exactement dans la silhouette du cheval détectée par Phi.
        
        threshold = 0.1
        y_pts, x_pts = np.where(phi_arr > threshold)
        
        if len(x_pts) > 0 and len(y_pts) > 0:
            # Boîte englobante du vrai cheval (depuis la heatmap)
            phi_min_x, phi_max_x = x_pts.min(), x_pts.max()
            phi_min_y, phi_max_y = y_pts.min(), y_pts.max()
            
            # Convertir en coordonnées [-1, 1]
            target_min_x = (phi_min_x / IMG_SIZE) * 2 - 1
            target_max_x = (phi_max_x / IMG_SIZE) * 2 - 1
            target_min_y = (phi_min_y / IMG_SIZE) * 2 - 1
            target_max_y = (phi_max_y / IMG_SIZE) * 2 - 1
            
            target_center_x = (target_min_x + target_max_x) / 2
            target_center_y = (target_min_y + target_max_y) / 2
            
            # Boîte englobante des keypoints prédits
            curr_min_x, curr_max_x = kps[:, 0].min(), kps[:, 0].max()
            curr_min_y, curr_max_y = kps[:, 1].min(), kps[:, 1].max()
            
            curr_center_x = (curr_min_x + curr_max_x) / 2
            curr_center_y = (curr_min_y + curr_max_y) / 2
            
            # Facteur de zoom (scale)
            scale_x = (target_max_x - target_min_x) / max(curr_max_x - curr_min_x, 1e-6)
            scale_y = (target_max_y - target_min_y) / max(curr_max_y - curr_min_y, 1e-6)
            
            # On prend la plus petite échelle pour garder les proportions sans déborder
            # * 1.1 pour étirer légèrement car les keypoints sont souvent trop repliés
            scale_global = min(scale_x, scale_y) * 1.1 
            
            # 1. Ramener les keypoints au centre 0,0
            kps[:, 0] -= curr_center_x
            kps[:, 1] -= curr_center_y
            
            # 2. Redimensionner à la taille du vrai cheval
            kps[:, 0] *= scale_global
            kps[:, 1] *= scale_global
            
            # 3. Placer les keypoints sur le vrai cheval
            kps[:, 0] += target_center_x
            kps[:, 1] += target_center_y
        # -----------------------------------------------

        # Col 0 : image originale
        ax0 = axes[row, base_col]
        ax0.imshow(img_np)
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title(os.path.basename(fname)[:16], fontsize=5, color="#aaaacc", pad=1)
        _border(ax0, "#334466")

        # Col 1 : heatmap Phi (Amélioration du contraste)
        ax1 = axes[row, base_col + 1]
        # Élever à la puissance 3 écrase le bruit de fond et fait ressortir les os
        phi_vis = np.clip(phi_arr ** 3 * 2.0, 0, 1) 
        ax1.imshow(phi_vis, cmap="hot", vmin=0, vmax=1)
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.set_title("Heatmap Φ", fontsize=5, color="#ffaa44", pad=1)
        _border(ax1, "#664422")

        # Col 2 : overlay squelette
        ax2 = axes[row, base_col + 2]
        draw_skeleton_on_ax(ax2, img_np, kps)
        ax2.set_title("Squelette", fontsize=5, color="#88ff88", pad=1)
        _border(ax2, "#226644")

    # Masquer les axes vides
    for i in range(n, n_rows * n_cols):
        row = i // n_cols
        for dc in range(3):
            axes[row, (i % n_cols) * 3 + dc].axis("off")

    # Légende couleurs anatomiques
    legend_items = [
        mpatches.Patch(color="#ff6b6b", label="Tête"),
        mpatches.Patch(color="#ffd93d", label="Cou"),
        mpatches.Patch(color="#6bcb77", label="Dos / Épaules"),
        mpatches.Patch(color="#4d96ff", label="Jambe AV G"),
        mpatches.Patch(color="#845ec2", label="Jambe AV D"),
        mpatches.Patch(color="#ff9671", label="Jambe AR G"),
        mpatches.Patch(color="#f9f871", label="Jambe AR D"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=7,
               fontsize=6.5, facecolor="#1a1a2e", edgecolor="#444466",
               labelcolor="white", framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    return fig


def _border(ax, color, lw=0.8):
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(lw)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Inspection visuelle : keypoints + os sur images réelles."
    )
    parser.add_argument("--checkpoint",  type=str,
                        default="checkpoints/run19_clean_scratch/latest.pt")
    parser.add_argument("--data-dir",    type=str,
                        default="data/processed/final")
    parser.add_argument("--output-dir",  type=str,
                        default="eval_outputs/skeleton_inspect")
    parser.add_argument("--output-name", type=str,
                        default="inspect_run19")
    parser.add_argument("--n-images",    type=int, default=16)
    parser.add_argument("--n-cols",      type=int, default=4,
                        help="Nombre de chevaux par ligne (défaut: 4)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Inspection visuelle squelettes -- Phi + Omega")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  N images   : {args.n_images}")
    print(f"{'='*60}\n")

    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    print(f"  git HEAD : {commit}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/3] Chargement Phi + Omega...")
    phi, omega = load_models(args.checkpoint, device)

    print("\n[2/3] Chargement des images + forward...")
    imgs_tensor, _, fnames = load_samples(
        args.data_dir, args.n_images, args.seed, device
    )

    with torch.no_grad():
        phi_outs  = phi(imgs_tensor)        # (N, 1, 128, 128)
        kps_batch = omega(phi_outs)         # (N, 18, 2)

    print(f"  phi_outs  : {tuple(phi_outs.shape)}")
    print(f"  kps_batch : {tuple(kps_batch.shape)}")
    print(f"  kps range : [{kps_batch.min():.3f}, {kps_batch.max():.3f}]")

    print("\n[3/3] Génération de la figure...")
    fig = build_figure(
        imgs_tensor, phi_outs, kps_batch,
        fnames, args.checkpoint, commit, n_cols=args.n_cols
    )

    out_path = os.path.join(args.output_dir, args.output_name + ".png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  [OK] Figure : {out_path}")
    print(f"\nFichier produit :")
    print(f"  {out_path}")


if __name__ == "__main__":
    main()

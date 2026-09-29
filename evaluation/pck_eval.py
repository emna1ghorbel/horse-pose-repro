"""
Évaluation PCK (Percentage of Correct Keypoints) -- post-entraînement.

Principe :
  Phi prédit une image squelette (1, 128, 128) pour chaque image d'entrée.
  On binarise la heatmap et on mesure l'overlap avec le masque GT de silhouette.

  Métriques :
    - iou           : intersection-over-union heatmap binarisée / masque GT
    - coverage      : fraction du masque GT couverte par le squelette prédit
    - precision     : fraction du squelette prédit dans le masque GT
    - weighted_frac : fraction d'énergie de heatmap tombant dans le masque

  Note : pas d'annotations keypoints GT paires disponibles → on utilise
  le masque de silhouette comme proxy.

Sorties :
  eval_outputs/pck/<output-name>.png   -- grille (image / masque GT / heatmap / overlay)
  eval_outputs/pck/<output-name>.json  -- métriques numériques

Usage :
    python evaluation/pck_eval.py \\
        --checkpoint checkpoints/run19_clean_scratch/latest.pt \\
        --data-dir data/processed/final \\
        --output-dir eval_outputs/pck \\
        --output-name pck_run19_scratch
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torchvision.transforms as T

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "models"))
sys.path.insert(0, os.path.join(REPO_ROOT, "data"))
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))

from image_to_skeleton import ImageToSkeleton

IMG_SIZE        = 128
N_IMAGES        = 50
BINARIZE_THRESH = 0.3


# ─────────────────────────────────────────────────────────
# Chargement modèle
# ─────────────────────────────────────────────────────────
def load_phi(ckpt_path: str, device) -> ImageToSkeleton:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    phi  = ImageToSkeleton()
    if "phi" in ckpt:
        phi.load_state_dict(ckpt["phi"])
        print(f"  Checkpoint chargé : epoch={ckpt.get('epoch', '?')}")
    else:
        phi.load_state_dict(ckpt)
        print("  Checkpoint chargé (state_dict direct).")
    phi.to(device).eval()
    return phi


# ─────────────────────────────────────────────────────────
# Chargement données
# ─────────────────────────────────────────────────────────
def load_samples(data_dir: str, n: int, seed: int, device):
    """Charge n paires (image, masque) depuis dataset_registry.json."""
    registry_path = os.path.join(data_dir, "dataset_registry.json")
    images_dir    = os.path.join(data_dir, "images")
    masks_dir     = os.path.join(data_dir, "masks")

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
    mask_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])

    imgs, masks, fnames = [], [], []
    for idx in idxs:
        e = entries[idx]
        img       = Image.open(os.path.join(images_dir, e["image"])).convert("RGB")
        mask_path = os.path.join(masks_dir, e["mask"])
        mask      = (Image.open(mask_path).convert("L")
                     if os.path.exists(mask_path)
                     else Image.new("L", img.size, 255))
        imgs.append(img_tf(img))
        masks.append(mask_tf(mask))
        fnames.append(e["image"])

    imgs  = torch.stack(imgs).to(device)
    masks = torch.stack(masks).to(device)
    print(f"  {len(fnames)} échantillons chargés (seed={seed}).")
    return imgs, masks, fnames


# ─────────────────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────────────────
def compute_metrics(phi_heatmap: torch.Tensor,
                    gt_mask: torch.Tensor,
                    thresh: float = BINARIZE_THRESH) -> dict:
    """
    phi_heatmap : (1, H, W)  valeurs [0,1]
    gt_mask     : (1, H, W)  valeurs [0,1]
    """
    pred_bin = (phi_heatmap > thresh).float()
    mask_bin = (gt_mask > 0.5).float()

    inter = (pred_bin * mask_bin).sum().item()
    union = (pred_bin + mask_bin).clamp(0, 1).sum().item()
    iou   = inter / (union + 1e-8)

    mask_px  = mask_bin.sum().item()
    pred_px  = pred_bin.sum().item()
    coverage  = inter / (mask_px + 1e-8)
    precision = inter / (pred_px + 1e-8)

    heat_in   = (phi_heatmap * mask_bin).sum().item()
    heat_tot  = phi_heatmap.sum().item()
    wfrac     = heat_in / (heat_tot + 1e-8)

    return {
        "iou":           iou,
        "coverage":      coverage,
        "precision":     precision,
        "weighted_frac": wfrac,
        "pred_pixels":   int(pred_px),
        "mask_pixels":   int(mask_px),
    }


def aggregate(metrics: list) -> dict:
    keys = metrics[0].keys()
    return {k: float(np.mean([m[k] for m in metrics])) for k in keys}


# ─────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────
def denorm(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = (t.cpu().float() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def border(ax, color, lw=0.8):
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(lw)


def build_figure(imgs, masks, phi_outs, per_img_metrics, agg,
                 fnames, ckpt_path, commit):
    n = len(fnames)
    fig = plt.figure(figsize=(n * 2.6, 4 * 2.6 + 1.6), facecolor="#0d1117")
    gs  = gridspec.GridSpec(4, n, figure=fig,
                             hspace=0.06, wspace=0.04,
                             left=0.01, right=0.99, top=0.92, bottom=0.01)

    title = (
        f"Évaluation PCK  |  {os.path.basename(ckpt_path)}  |  commit: {commit[:12]}\n"
        f"IoU={agg['iou']:.3f}   Coverage={agg['coverage']:.3f}   "
        f"Precision={agg['precision']:.3f}   WeightedFrac={agg['weighted_frac']:.3f}"
    )
    fig.text(0.5, 0.958, title, ha="center", va="center",
             fontsize=8.5, color="white", family="monospace")

    for col in range(n):
        # Ligne 0 : image
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(denorm(imgs[col]))
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title(os.path.basename(fnames[col])[:14],
                      fontsize=4.5, color="#aaaacc", pad=1)
        border(ax0, "#334466")

        # Ligne 1 : masque GT
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(masks[col, 0].cpu().numpy(), cmap="Greens", vmin=0, vmax=1)
        ax1.set_xticks([]); ax1.set_yticks([])
        border(ax1, "#226644")

        # Ligne 2 : heatmap Phi
        ax2 = fig.add_subplot(gs[2, col])
        phi_arr = phi_outs[col, 0].cpu().numpy()
        ax2.imshow(phi_arr, cmap="hot", vmin=0, vmax=1)
        ax2.set_xticks([]); ax2.set_yticks([])
        m = per_img_metrics[col]
        ax2.set_xlabel(f"iou={m['iou']:.2f} cov={m['coverage']:.2f}",
                       fontsize=4.5, color="#ffcc88", labelpad=1)
        border(ax2, "#664422")

        # Ligne 3 : overlay
        ax3 = fig.add_subplot(gs[3, col])
        base_img = denorm(imgs[col]).astype(np.float32) / 255.0
        ov = base_img.copy()
        phi_norm = phi_arr / (phi_arr.max() + 1e-8)
        ov[:, :, 0] = np.clip(ov[:, :, 0] + phi_norm * 0.6, 0, 1)
        ov[:, :, 1] = np.clip(ov[:, :, 1] - phi_norm * 0.2, 0, 1)
        mask_bin = (masks[col, 0].cpu().numpy() > 0.5).astype(np.float32)
        ov[:, :, 1] = np.clip(ov[:, :, 1] + mask_bin * 0.25, 0, 1)
        ax3.imshow(ov)
        ax3.set_xticks([]); ax3.set_yticks([])
        border(ax3, "#553366")

    # Étiquettes de lignes
    for row, (label, color) in enumerate([
        ("Image",     "#aaaaff"),
        ("Masque GT", "#88ff88"),
        ("Phi out",   "#ffaa44"),
        ("Overlay",   "#ff88cc"),
    ]):
        fig.text(0.002, 0.915 - row * 0.22, label,
                 ha="left", va="center", fontsize=5.5,
                 color=color, rotation=90, family="monospace")

    return fig


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Évaluation PCK de Phi (heatmap vs masque silhouette)."
    )
    parser.add_argument("--checkpoint",  type=str,
                        default="checkpoints/run19_clean_scratch/latest.pt")
    parser.add_argument("--data-dir",    type=str,
                        default="data/processed/final")
    parser.add_argument("--output-dir",  type=str,
                        default="eval_outputs/pck")
    parser.add_argument("--output-name", type=str,
                        default="pck_eval")
    parser.add_argument("--n-images",    type=int, default=N_IMAGES,
                        help="Nombre d'images à évaluer (défaut: 50)")
    parser.add_argument("--n-show",      type=int, default=10,
                        help="Colonnes dans la grille visuelle (défaut: 10)")
    parser.add_argument("--thresh",      type=float, default=BINARIZE_THRESH,
                        help="Seuil de binarisation heatmap (défaut: 0.3)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Évaluation PCK -- Phi")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  N images   : {args.n_images}   Seuil bin : {args.thresh}")
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

    print("[1/4] Chargement de Phi...")
    phi = load_phi(args.checkpoint, device)

    print("\n[2/4] Chargement des données...")
    imgs, masks, fnames = load_samples(
        args.data_dir, args.n_images, args.seed, device
    )

    print("\n[3/4] Forward Phi + calcul métriques...")
    all_metrics, phi_outs_list = [], []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            b_imgs  = imgs[i:i + batch_size]
            b_masks = masks[i:i + batch_size]
            phi_out = phi(b_imgs)
            phi_outs_list.append(phi_out.cpu())
            for j in range(phi_out.shape[0]):
                all_metrics.append(
                    compute_metrics(phi_out[j], b_masks[j], args.thresh)
                )

    phi_outputs = torch.cat(phi_outs_list, dim=0)
    agg         = aggregate(all_metrics)

    print(f"\n{'='*60}")
    print(f"  RÉSULTATS  (N={args.n_images}, seuil={args.thresh})")
    print(f"  {'Métrique':<20s}  {'Valeur':>10s}")
    print(f"  {'-'*34}")
    for k, v in agg.items():
        print(f"  {k:<20s}  {v:>10.4f}")
    print(f"{'='*60}")

    iou = agg["iou"]
    if iou > 0.35:
        verdict = "[EXCELLENT] Squelette bien localisé dans la silhouette."
    elif iou > 0.20:
        verdict = "[BON]       Bonne localisation, quelques débordements."
    elif iou > 0.10:
        verdict = "[MOYEN]     Localisation partielle, à améliorer."
    else:
        verdict = "[FAIBLE]    Squelette mal localisé ou trop épars."
    print(f"\n  VERDICT : {verdict}\n")

    print("[4/4] Génération des sorties...")
    n_show = min(args.n_show, len(fnames))
    fig = build_figure(
        imgs[:n_show], masks[:n_show], phi_outputs[:n_show],
        all_metrics[:n_show], agg, fnames[:n_show],
        args.checkpoint, commit
    )
    png_path = os.path.join(args.output_dir, args.output_name + ".png")
    fig.savefig(png_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Figure : {png_path}")

    json_out = {
        "checkpoint": args.checkpoint,
        "commit":     commit,
        "n_images":   args.n_images,
        "thresh":     args.thresh,
        "seed":       args.seed,
        "aggregate":  agg,
        "per_image":  [{"file": f, **m} for f, m in zip(fnames, all_metrics)],
    }
    json_path = os.path.join(args.output_dir, args.output_name + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"  [OK] JSON   : {json_path}")

    print(f"\nFichiers produits :")
    print(f"  {png_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()

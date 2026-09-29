"""
Diagnostic de mode collapse de Phi (F1) -- évaluation post-entraînement.

Métriques calculées :
  - cos_sim   : similarité cosinus moyenne entre toutes les paires de sorties Phi
                (10 images -> 45 paires). cos_sim proche de 1.0 = collapse total.
  - pixel_var : variance de la valeur moyenne par image (inter-images).
                Quasi nulle = collapse (toutes les sorties identiques).
  - intra_std : écart-type moyen intra-image (structure interne de chaque sortie).

Sorties :
  eval_outputs/phi_variance/phi_variance_run5_fixed.png
      Grille 5 colonnes x 4 lignes :
        - Ligne 1-2 : images d'entrée (10 images réelles)
        - Ligne 3-4 : sorties Phi correspondantes (squelette prédit)
      + panneau "variance inter-images" en bas
  eval_outputs/beta_p_check.png
      4 rendus beta(p) depuis le prior (sigma=3.0), pour vérifier la cible.

Usage :
    python evaluation/phi_variance_diag.py \
        --checkpoint checkpoints/run5_fixed/latest.pt \
        --data-dir data/processed/final \
        --prior-poses data/processed/synthetic_prior/synthetic_prior_poses.pt
"""

import argparse
import os
import sys
import math
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
from train import render_skeleton_batch

N_IMAGES = 10


def load_phi_from_checkpoint(ckpt_path: str, device) -> ImageToSkeleton:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    phi = ImageToSkeleton()
    if "phi" in ckpt:
        phi.load_state_dict(ckpt["phi"])
        epoch = ckpt.get("epoch", "?")
        print(f"  Checkpoint charge : epoch={epoch}, cles={list(ckpt.keys())[:6]}...")
    else:
        phi.load_state_dict(ckpt)
        print("  Checkpoint charge (state_dict direct).")
    phi.to(device)
    phi.eval()
    return phi


def load_test_images(data_dir: str, n: int, seed: int, device):
    import json

    registry_path = os.path.join(data_dir, "dataset_registry.json")
    images_dir = os.path.join(data_dir, "images")

    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)
    entries = registry["entries"]

    rng = np.random.RandomState(seed)
    total = len(entries)
    step = max(1, total // n)
    base_indices = list(range(0, min(n * step, total), step))[:n]
    indices = [min(int(i + rng.randint(0, max(1, step // 2))), total - 1)
               for i in base_indices]

    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    images_tensor = []
    images_display = []
    filenames = []

    for idx in indices:
        entry = entries[idx]
        img_path = os.path.join(images_dir, entry["image"])
        img = Image.open(img_path).convert("RGB").resize((128, 128))
        images_display.append(img)
        images_tensor.append(transform(img))
        filenames.append(entry["image"])

    images_tensor = torch.stack(images_tensor).to(device)
    print(f"  {n} images chargees depuis '{data_dir}' (seed={seed}).")
    print(f"  Exemples : {filenames[:3]}...")
    return images_tensor, images_display, filenames


def compute_metrics(phi_outputs: torch.Tensor):
    N = phi_outputs.shape[0]
    flat = phi_outputs.view(N, -1).cpu().float()

    norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = flat / norms

    cos_sims = []
    for i in range(N):
        for j in range(i + 1, N):
            cs = (normed[i] * normed[j]).sum().item()
            cos_sims.append(cs)
    cos_sim = float(np.mean(cos_sims)) if cos_sims else float("nan")

    per_image_means = flat.mean(dim=1)
    pixel_var = float(per_image_means.var().item())

    per_image_stds = flat.std(dim=1)
    intra_std = float(per_image_stds.mean().item())

    return cos_sim, pixel_var, intra_std


def variance_map(phi_outputs: torch.Tensor) -> np.ndarray:
    arr = phi_outputs.squeeze(1).cpu().float().numpy()
    var = arr.var(axis=0)
    vmax = var.max()
    if vmax > 0:
        var = var / vmax
    return var


def denorm_image(tensor_img: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor_img.cpu().float() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def build_grid_figure(images_tensor, phi_outputs, var_map,
                       cos_sim, pixel_var, intra_std,
                       filenames, checkpoint_path, commit_hash):
    n_cols = 5
    n_imgs = len(filenames)
    n_rows_data = math.ceil(n_imgs / n_cols)  # 2

    fig = plt.figure(figsize=(n_cols * 2.8, n_rows_data * 2.8 * 2 + 3.2),
                     facecolor="#1a1a2e")

    gs = gridspec.GridSpec(
        n_rows_data * 2 + 1, n_cols + 1,
        figure=fig,
        hspace=0.05, wspace=0.05,
        left=0.02, right=0.98, top=0.93, bottom=0.02
    )

    fig.text(0.5, 0.965,
             f"Diagnostic Phi -- {os.path.basename(checkpoint_path)}\n"
             f"commit: {commit_hash[:12]}   "
             f"cos_sim={cos_sim:.4f}   pixel_var={pixel_var:.6f}   intra_std={intra_std:.4f}",
             ha="center", va="center", fontsize=9, color="white",
             family="monospace")

    for col in range(n_cols):
        for row_pair in range(n_rows_data):
            img_idx = row_pair * n_cols + col
            if img_idx >= n_imgs:
                continue

            ax_in = fig.add_subplot(gs[row_pair * 2, col])
            inp = denorm_image(images_tensor[img_idx])
            ax_in.imshow(inp)
            ax_in.set_xticks([]); ax_in.set_yticks([])
            fname_short = os.path.basename(filenames[img_idx])[:18]
            ax_in.set_title(fname_short, fontsize=5.5, color="#aaaacc", pad=2)
            for spine in ax_in.spines.values():
                spine.set_edgecolor("#444466")

            ax_out = fig.add_subplot(gs[row_pair * 2 + 1, col])
            phi_arr = phi_outputs[img_idx, 0].cpu().numpy()
            ax_out.imshow(phi_arr, cmap="hot", vmin=0, vmax=1)
            ax_out.set_xticks([]); ax_out.set_yticks([])
            mean_val = phi_arr.mean()
            ax_out.set_xlabel(f"mu={mean_val:.3f}", fontsize=5.5,
                               color="#ffaaaa", labelpad=1)
            for spine in ax_out.spines.values():
                spine.set_edgecolor("#884422")

    ax_var = fig.add_subplot(gs[n_rows_data * 2, :n_cols])
    im = ax_var.imshow(var_map, cmap="plasma", vmin=0, vmax=1)
    ax_var.set_xticks([]); ax_var.set_yticks([])
    ax_var.set_title("Variance inter-images (normalisee) -- blanc = grande variance entre images",
                      fontsize=7, color="#ccccff", pad=3)
    plt.colorbar(im, ax=ax_var, fraction=0.046, pad=0.04)
    for spine in ax_var.spines.values():
        spine.set_edgecolor("#6666aa")

    ax_metrics = fig.add_subplot(gs[n_rows_data * 2, n_cols])
    ax_metrics.axis("off")
    REF = {"cos_sim": 0.943, "pixel_var": 0.00045, "intra_std": 0.1734}
    now = {"cos_sim": cos_sim, "pixel_var": pixel_var, "intra_std": intra_std}

    lines = [
        ("Metrique", "Avant", "Apres", "Delta"),
        ("cos_sim",  f"{REF['cos_sim']:.4f}", f"{now['cos_sim']:.4f}",
         f"{now['cos_sim'] - REF['cos_sim']:+.4f}"),
        ("pixel_var", f"{REF['pixel_var']:.6f}", f"{now['pixel_var']:.6f}",
         f"{now['pixel_var'] - REF['pixel_var']:+.6f}"),
        ("intra_std", f"{REF['intra_std']:.4f}", f"{now['intra_std']:.4f}",
         f"{now['intra_std'] - REF['intra_std']:+.4f}"),
    ]
    table_text = "\n".join(
        f"{l[0]:>10s} | {l[1]:>9s} | {l[2]:>9s} | {l[3]:>9s}"
        for l in lines
    )

    if cos_sim < 0.6:
        verdict = "RESOLU (cos_sim < 0.6)"
        vcolor = "#88ff88"
    elif cos_sim < 0.8:
        verdict = "PARTIEL (cos_sim < 0.8)"
        vcolor = "#ffcc44"
    elif cos_sim < 0.90:
        verdict = "MODERE (cos_sim < 0.90)"
        vcolor = "#ff8844"
    else:
        verdict = "COLLAPSE (cos_sim >= 0.90)"
        vcolor = "#ff4444"

    ax_metrics.text(0.05, 0.95, table_text,
                    transform=ax_metrics.transAxes,
                    fontsize=5.5, color="white", family="monospace",
                    va="top", ha="left")
    ax_metrics.text(0.05, 0.30, verdict,
                    transform=ax_metrics.transAxes,
                    fontsize=7, color=vcolor, family="monospace",
                    va="top", ha="left", fontweight="bold")
    ax_metrics.set_facecolor("#111122")

    return fig


def build_beta_p_figure(prior_poses: torch.Tensor, n: int, device, seed: int = 42):
    torch.manual_seed(seed)
    indices = torch.randint(len(prior_poses), (n,))
    poses = prior_poses[indices].to(device)
    with torch.no_grad():
        renders = render_skeleton_batch(poses, size=128, sigma=3.0)

    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3.2), facecolor="#0d1117")
    fig.suptitle("Rendu beta(p) depuis le prior synthetique (sigma=3.0)\n"
                 "Verifie : squelette articule visible, pas un blob",
                 fontsize=9, color="white")
    for i in range(n):
        arr = renders[i, 0].cpu().numpy()
        axes[i].imshow(arr, cmap="hot", vmin=0, vmax=1)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        axes[i].set_title(f"Prior #{indices[i].item()}", fontsize=7, color="#aaaacc")
        for spine in axes[i].spines.values():
            spine.set_edgecolor("#334455")
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic visuel du mode collapse de Phi."
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/run5_fixed/latest.pt")
    parser.add_argument("--data-dir", type=str,
                        default="data/processed/final")
    parser.add_argument("--prior-poses", type=str,
                        default="data/processed/synthetic_prior/synthetic_prior_poses.pt")
    parser.add_argument("--output-dir", type=str,
                        default="eval_outputs/phi_variance")
    parser.add_argument("--output-name", type=str,
                        default="phi_variance_run5_fixed.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-images", type=int, default=N_IMAGES)
    parser.add_argument("--n-prior", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Diagnostic Phi -- Mode Collapse Analysis")
    print(f"  Device      : {device}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Seed        : {args.seed}")
    print(f"{'='*60}\n")

    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    print(f"  git HEAD    : {commit}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("eval_outputs", exist_ok=True)

    print("[1/4] Chargement de Phi...")
    phi = load_phi_from_checkpoint(args.checkpoint, device)

    print("\n[2/4] Chargement des images de test...")
    images_tensor, images_display, filenames = load_test_images(
        args.data_dir, args.n_images, args.seed, device
    )

    print("\n[3/4] Forward Phi sur les images de test...")
    with torch.no_grad():
        phi_outputs = phi(images_tensor)

    print(f"  phi_outputs : shape={tuple(phi_outputs.shape)}, "
          f"min={phi_outputs.min():.4f}, max={phi_outputs.max():.4f}")

    cos_sim, pixel_var, intra_std = compute_metrics(phi_outputs)
    var_map_arr = variance_map(phi_outputs)

    print(f"\n{'='*60}")
    print(f"  RESULTATS METRIQUES")
    print(f"  {'Metrique':>12s} | {'AVANT (ref)':>14s} | {'APRES (run5)':>12s} | {'Delta':>10s}")
    print(f"  {'-'*56}")
    REF = {"cos_sim": 0.943, "pixel_var": 0.00045, "intra_std": 0.1734}
    print(f"  {'cos_sim':>12s} | {REF['cos_sim']:>14.4f} | {cos_sim:>12.4f} | {cos_sim - REF['cos_sim']:>+10.4f}")
    print(f"  {'pixel_var':>12s} | {REF['pixel_var']:>14.6f} | {pixel_var:>12.6f} | {pixel_var - REF['pixel_var']:>+10.6f}")
    print(f"  {'intra_std':>12s} | {REF['intra_std']:>14.4f} | {intra_std:>12.4f} | {intra_std - REF['intra_std']:>+10.4f}")
    print(f"{'='*60}")

    if cos_sim < 0.6:
        print("  VERDICT : [OK] Collapse resolu (cos_sim < 0.6)")
    elif cos_sim < 0.8:
        print("  VERDICT : [PARTIEL] Amelioration partielle (cos_sim < 0.8)")
    elif cos_sim < 0.90:
        print("  VERDICT : [MODERE] Progres modere, collapse partiel (cos_sim < 0.90)")
    else:
        print("  VERDICT : [COLLAPSE] Collapse quasi identique (cos_sim >= 0.90)")
    print()

    print("[4/4] Generation des figures...")
    fig_main = build_grid_figure(
        images_tensor, phi_outputs, var_map_arr,
        cos_sim, pixel_var, intra_std,
        filenames, args.checkpoint, commit
    )
    out_main = os.path.join(args.output_dir, args.output_name)
    fig_main.savefig(out_main, dpi=140, bbox_inches="tight", facecolor=fig_main.get_facecolor())
    plt.close(fig_main)
    print(f"  [OK] Grille principale : {out_main}")

    prior_poses = torch.load(args.prior_poses, map_location=device)
    print(f"  Prior charge : {prior_poses.shape} poses.")
    fig_beta = build_beta_p_figure(prior_poses, args.n_prior, device, seed=args.seed)
    out_beta = os.path.join("eval_outputs", "beta_p_check.png")
    fig_beta.savefig(out_beta, dpi=140, bbox_inches="tight",
                     facecolor=fig_beta.get_facecolor())
    plt.close(fig_beta)
    print(f"  [OK] Rendu beta(p)    : {out_beta}")

    print(f"\nDiagnostic termine. Commit : {commit}")
    print(f"Fichiers produits :")
    print(f"  {out_main}")
    print(f"  {out_beta}")


if __name__ == "__main__":
    main()

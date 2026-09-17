"""
Boucle d'entraînement complète (Sprint 5, Jour 4) -- assemble F1-F5 et
toutes les pertes (L_D, L_GC, L_Omega) en une seule boucle conjointe,
exactement le mécanisme discuté : tous les modules apprennent ensemble,
à chaque itération, via UNE SEULE perte totale L = L_D + L_GC + L_Omega
(papier équation 7).

Le mode --dummy sert à valider la stabilité numérique de la boucle. Le mode
réel charge des images et masques consolidés ainsi qu'un prior synthétique
non apparié.

Usage :
    python training/train.py --dummy --epochs 3 --batch-size 4
"""
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from image_to_skeleton import ImageToSkeleton
from skeleton_to_pose2d import SkeletonToPose2D
from lifting_2d_3d import Lifting2Dto3D
from renderer import GeometricConsistencyLoop
from discriminator import Discriminator
from adversarial_loss import discriminator_loss, generator_adversarial_loss
from geometric_consistency import geometric_consistency_loss
from real_image_dataset import HorseImageDataset

HORSE_EDGES = [
    (0, 2), (1, 2),              # Tete
    (3, 7), (7, 13), (13, 11),   # Jambe avant gauche
    (4, 8), (8, 14), (14, 12),   # Jambe avant droite
    (5, 9), (9, 15), (15, 17),   # Jambe arriere gauche
    (6, 10), (10, 16), (16, 17), # Jambe arriere droite
    (11, 12),                    # Epaules
    (11, 17), (12, 17),          # Dos
    (2, 11), (2, 12),            # Cou
]

def render_skeleton_batch(poses_2d: torch.Tensor, size: int = 128, sigma: float = 0.5, steps: int = 20) -> torch.Tensor:
    """
    Fonction beta DIFFERENTIABLE : rend un batch de poses 2D (batch, K, 2)
    en images de squelette (batch, 1, size, size).
    Dessine des SEGMENTS DE LIGNE (os) entre les articulations connectees
    (via interpolation lineaire de points le long de chaque arete anatomique),
    en plus des taches gaussiennes sur les points cles eux-memes.
    """
    batch, K, _ = poses_2d.shape
    device = poses_2d.device

    all_pts = [poses_2d]
    if steps > 0:
        alphas = torch.linspace(0, 1, steps=steps, device=device).view(1, steps, 1)
        for u, v in HORSE_EDGES:
            if u < K and v < K:
                pu = poses_2d[:, u:u + 1, :]
                pv = poses_2d[:, v:v + 1, :]
                segment = pu * (1.0 - alphas) + pv * alphas
                all_pts.append(segment)
    
    all_pts = torch.cat(all_pts, dim=1)  # (batch, K_total, 2)

    ys = torch.linspace(-1, 1, size, device=device)
    xs = torch.linspace(-1, 1, size, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (size, size)

    px = all_pts[:, :, 0].view(batch, -1, 1, 1)
    py = all_pts[:, :, 1].view(batch, -1, 1, 1)

    sigma_norm = sigma / (size / 2)  # convertit sigma en pixels vers l'échelle [-1,1]
    dist_sq = (grid_x.view(1, 1, size, size) - px) ** 2 + (grid_y.view(1, 1, size, size) - py) ** 2
    gaussians = torch.exp(-dist_sq / (2 * sigma_norm ** 2))

    canvas, _ = gaussians.max(dim=1, keepdim=True)  # (batch, 1, size, size)
    return canvas


def omega_loss_fn(omega_net, y_pred: torch.Tensor, skeleton_pred: torch.Tensor,
                   prior_poses: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """
    L_Omega (papier équation 6) :
        L_Omega = ||Omega(beta(p)) - p||^2 + lambda * ||beta(y) - s||^2
    p = poses du prior synthétique (non appariées)
    """
    beta_p = render_skeleton_batch(prior_poses)
    omega_beta_p = omega_net(beta_p)
    term1 = ((omega_beta_p - prior_poses) ** 2).mean()

    beta_y = render_skeleton_batch(y_pred)
    term2 = ((beta_y - skeleton_pred) ** 2).mean()

    return term1 + lam * term2


def build_models(n_keypoints: int = 18):
    phi = ImageToSkeleton()
    omega = SkeletonToPose2D(n_keypoints=n_keypoints)
    lambda_net = Lifting2Dto3D(n_keypoints=n_keypoints)
    discriminator = Discriminator()
    geo_loop = GeometricConsistencyLoop(lambda_net)
    return phi, omega, lambda_net, discriminator, geo_loop


def train_step(batch_images, batch_masks, batch_prior_poses, models, optimizers,
               loss_weights=None):
    phi, omega, lambda_net, discriminator, geo_loop = models
    opt_generator, opt_discriminator = optimizers
    loss_weights = loss_weights or {
        "adversarial": 1.0, "geometric": 1.0, "omega": 1.0, "background": 1.0,
    }

    # ---- Forward : mapping principal (F1->F2->F3) ----
    s = phi(batch_images)          # squelette prédit
    y = omega(s)                     # pose 2D
    v = lambda_net(y)                # pose 3D

    # ---- F4 : boucle de cohérence géométrique ----
    loop_results = geo_loop(v)

    # ---- F5 : discriminateur ----
    w = render_skeleton_batch(batch_prior_poses)  # squelette "réel" depuis le prior
    discriminator.train()
    loss_d = discriminator_loss(discriminator(w), discriminator(s.detach()))
    opt_discriminator.zero_grad()
    loss_d.backward()
    opt_discriminator.step()

    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    d_score_fake_for_g = discriminator(s)

    # ---- Pertes (TOUS les forwards terminés AVANT tout optimizer.step) ----
    # Calculer toutes les pertes ici pour éviter que opt_discriminator.step()
    # mute les poids de D en place alors que le graphe de loss_total_generator
    # les référence encore (RuntimeError: version counter mismatch).
    loss_g_adv = generator_adversarial_loss(d_score_fake_for_g)

    gc_losses = geometric_consistency_loss(y, v, loop_results)
    loss_gc = gc_losses["L_GC"]

    loss_omega = omega_loss_fn(omega, y, s, batch_prior_poses)

    # Pénalité de fond L1 (pour forcer le squelette à rester dans la silhouette)
    if batch_masks is not None:
        loss_background = (s * (1.0 - batch_masks)).abs().mean()
    else:
        loss_background = torch.tensor(0.0, device=s.device)

    loss_total_generator = (
        loss_weights["adversarial"] * loss_g_adv
        + loss_weights["geometric"] * loss_gc
        + loss_weights["omega"] * loss_omega
        + loss_weights["background"] * loss_background
    )

    # ---- Backward + step : générateur d'abord ----
    # IMPORTANT : le générateur doit être mis à jour AVANT opt_discriminator.step().
    # loss_total_generator passe par discriminator(s) -- son graphe autograd
    # contient les poids de D. Si opt_discriminator.step() les mute en place
    # (version counter +1) avant que loss_total_generator.backward() s'exécute,
    # PyTorch lève "variable needed for gradient computation has been modified
    # by an inplace operation" (version mismatch).
    # En inversant l'ordre, D est figé pendant le backward du générateur.
    opt_generator.zero_grad()
    loss_total_generator.backward()
    opt_generator.step()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    discriminator.train()

    # ---- Backward + step : discriminateur ensuite ----
    # Les poids de D ne sont plus référencés par un graphe actif → step() sûr.
    return {
        "loss_d": loss_d.item(),
        "loss_g_adv": loss_g_adv.item(),
        "loss_gc": loss_gc.item(),
        "loss_omega": loss_omega.item(),
        "loss_background": loss_background.item(),
        "loss_total": loss_total_generator.item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true", help="Utiliser des données factices (Sprint 5)")
    parser.add_argument("--data-dir", type=str,
                        help="Dossier processed/final contenant images/, masks/ et dataset_registry.json")
    parser.add_argument("--prior-poses", type=str,
                        help="Fichier synthetic_prior_poses.pt produit par data/synthetic_prior.py")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detect-anomaly", action="store_true",
                        help="Active le diagnostic autograd lent (debogage seulement)")
    parser.add_argument("--weight-adversarial", type=float, default=1.0)
    parser.add_argument("--weight-geometric", type=float, default=1.0)
    parser.add_argument("--weight-omega", type=float, default=1.0)
    parser.add_argument("--weight-background", type=float, default=1.0)
    parser.add_argument("--pretrain-omega-epochs", type=int, default=10,
                         help="Nombre d'epochs de pre-entrainement d'Omega seul (defaut: 10)")
    args = parser.parse_args()

    if min(args.weight_adversarial, args.weight_geometric, args.weight_omega,
           args.weight_background) < 0:
        parser.error("Les poids de pertes doivent etre positifs ou nuls.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if not args.dummy and (not args.data_dir or not args.prior_poses):
        parser.error("Le mode réel requiert --data-dir et --prior-poses (ou utilisez --dummy).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    phi, omega, lambda_net, discriminator, geo_loop = build_models()
    phi.to(device); omega.to(device); lambda_net.to(device); discriminator.to(device)

    generator_params = list(phi.parameters()) + list(omega.parameters()) + list(lambda_net.parameters())
    opt_generator = torch.optim.Adam(generator_params, lr=args.lr)
    opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    models = (phi, omega, lambda_net, discriminator, geo_loop)
    optimizers = (opt_generator, opt_discriminator)
    loss_weights = {
        "adversarial": args.weight_adversarial,
        "geometric": args.weight_geometric,
        "omega": args.weight_omega,
        "background": args.weight_background,
    }

    if args.dummy:
        real_loader = None
        prior_poses = None
        batches_per_epoch = args.batches_per_epoch
    else:
        real_dataset = HorseImageDataset(args.data_dir)
        if len(real_dataset) == 0:
            raise ValueError(f"Dataset réel vide : {args.data_dir}")
        real_loader = DataLoader(
            real_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )
        prior_poses = torch.load(args.prior_poses, map_location="cpu")
        if prior_poses.ndim != 3 or prior_poses.shape[1:] != (18, 2):
            raise ValueError(
                f"Prior invalide : attendu (N, 18, 2), reçu {tuple(prior_poses.shape)}"
            )
        if len(prior_poses) == 0:
            raise ValueError("Prior synthétique vide.")
        batches_per_epoch = len(real_loader)
        print(f"Données réelles : {len(real_dataset)} images, {len(prior_poses)} poses du prior.")

    # --- Pré-entraînement d'Omega seul ---
    if args.pretrain_omega_epochs > 0:
        print(f"\n--- Pre-entrainement d'Omega seul ({args.pretrain_omega_epochs} epochs) ---")
        opt_omega_pretrain = torch.optim.Adam(omega.parameters(), lr=args.lr)
        omega.train()
        for p_epoch in range(1, args.pretrain_omega_epochs + 1):
            epoch_loss = 0.0
            for _ in range(batches_per_epoch):
                if prior_poses is None:
                    p_batch = torch.randn(args.batch_size, 18, 2, device=device).mul_(0.3).clamp_(-1, 1)
                else:
                    prior_indices = torch.randint(len(prior_poses), (args.batch_size,))
                    p_batch = prior_poses[prior_indices].to(device, non_blocking=True)
                
                beta_p = render_skeleton_batch(p_batch)
                pred_p = omega(beta_p)
                loss_pretrain = ((pred_p - p_batch) ** 2).mean()

                opt_omega_pretrain.zero_grad()
                loss_pretrain.backward()
                opt_omega_pretrain.step()
                epoch_loss += loss_pretrain.item()

            avg_pretrain_loss = epoch_loss / batches_per_epoch
            print(f"[Pretrain Omega epoch {p_epoch}/{args.pretrain_omega_epochs}] loss={avg_pretrain_loss:.4f}")
        print("--- Fin du pre-entrainement d'Omega ---\n")

        # Réinitialise l'optimiseur du générateur pour ne pas accumuler de momentum obsolète
        opt_generator = torch.optim.Adam(generator_params, lr=args.lr)
        optimizers = (opt_generator, opt_discriminator)

    for epoch in range(1, args.epochs + 1):
        epoch_losses = {"loss_d": 0, "loss_g_adv": 0, "loss_gc": 0, "loss_omega": 0,
                        "loss_background": 0, "loss_total": 0}

        if real_loader is not None:
            batch_iter = iter(real_loader)

        for batch_idx in range(batches_per_epoch):
            if args.dummy:
                batch_images = torch.randn(args.batch_size, 3, 128, 128, device=device)
                batch_masks = torch.ones(args.batch_size, 1, 128, 128, device=device)
                batch_prior_poses = torch.rand(args.batch_size, 18, 2, device=device) * 2 - 1
            else:
                batch_images, batch_masks = next(batch_iter)
                batch_images = batch_images.to(device, non_blocking=True)
                batch_masks = batch_masks.to(device, non_blocking=True)
                prior_indices = torch.randint(len(prior_poses), (batch_images.shape[0],))
                batch_prior_poses = prior_poses[prior_indices].to(device, non_blocking=True)

            losses = train_step(batch_images, batch_masks, batch_prior_poses, models, optimizers,
                                loss_weights=loss_weights)

            for k, v in losses.items():
                epoch_losses[k] += v
                assert not (v != v), f"NaN détecté dans {k} à l'epoch {epoch}, batch {batch_idx} !"  # v != v <=> isnan

        n = batches_per_epoch
        msg = " ".join(f"{k}={v/n:.4f}" for k, v in epoch_losses.items())
        print(f"[Epoch {epoch}/{args.epochs}] {msg}")

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "phi": phi.state_dict(), "omega": omega.state_dict(),
            "lambda_net": lambda_net.state_dict(), "discriminator": discriminator.state_dict(),
            "optimizer_generator": opt_generator.state_dict(),
            "optimizer_discriminator": opt_discriminator.state_dict(),
            "args": vars(args),
            "loss_weights": loss_weights,
            "torch_version": torch.__version__,
        }, os.path.join(args.checkpoint_dir, "latest.pt"))

    print("\nEntraînement terminé sans divergence.")


if __name__ == "__main__":
    main()

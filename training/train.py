"""
Boucle d'entraînement complète (Sprint 5, Jour 4) -- assemble F1-F5 et
toutes les pertes (L_D, L_GC, L_Omega) en une seule boucle conjointe,
exactement le mécanisme discuté : tous les modules apprennent ensemble,
à chaque itération, via UNE SEULE perte totale L = L_D + L_GC + L_Omega
(papier équation 7).

Pour ce test (Sprint 5), on utilise des données FACTICES (--dummy) pour
valider la stabilité numérique de la boucle assemblée, avant de brancher
les vraies données (Sprint 6).

Usage :
    python training/train.py --dummy --epochs 3 --batch-size 4
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))

from image_to_skeleton import ImageToSkeleton
from skeleton_to_pose2d import SkeletonToPose2D
from lifting_2d_3d import Lifting2Dto3D
from renderer import GeometricConsistencyLoop
from discriminator import Discriminator
from adversarial_loss import discriminator_loss, generator_adversarial_loss
from geometric_consistency import geometric_consistency_loss


def render_skeleton_batch(poses_2d: torch.Tensor, size: int = 128, sigma: float = 3.0) -> torch.Tensor:
    """
    Fonction beta simplifiée et DIFFERENTIABLE : rend un batch de poses 2D
    (batch, K, 2) en images de squelette (batch, 1, size, size), via des
    "taches" gaussiennes autour de chaque point (pas d'assignation en place,
    contrairement à la version précédente qui cassait le graphe de calcul
    de PyTorch -- cf. erreur "modified by an inplace operation").

    Chaque point clé produit une gaussienne 2D sur le canevas ; toutes les
    gaussiennes sont ensuite combinées par un max (pas une somme, pour éviter
    la saturation quand des points sont proches).
    """
    batch, K, _ = poses_2d.shape
    device = poses_2d.device

    ys = torch.linspace(-1, 1, size, device=device)
    xs = torch.linspace(-1, 1, size, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (size, size)

    # Broadcast : (batch, K, 1, 1) vs (size, size) -> (batch, K, size, size)
    px = poses_2d[:, :, 0].view(batch, K, 1, 1)
    py = poses_2d[:, :, 1].view(batch, K, 1, 1)

    sigma_norm = sigma / (size / 2)  # convertit sigma en pixels vers l'échelle [-1,1]
    dist_sq = (grid_x.view(1, 1, size, size) - px) ** 2 + (grid_y.view(1, 1, size, size) - py) ** 2
    gaussians = torch.exp(-dist_sq / (2 * sigma_norm ** 2))  # (batch, K, size, size)

    canvas, _ = gaussians.max(dim=1, keepdim=True)  # (batch, 1, size, size), pas d'assignation en place
    return canvas


def omega_loss_fn(omega_net, phi_net, y_pred: torch.Tensor, skeleton_pred: torch.Tensor,
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


def build_models(n_keypoints: int = 20):
    phi = ImageToSkeleton()
    omega = SkeletonToPose2D(n_keypoints=n_keypoints)
    lambda_net = Lifting2Dto3D(n_keypoints=n_keypoints)
    discriminator = Discriminator()
    geo_loop = GeometricConsistencyLoop(lambda_net)
    return phi, omega, lambda_net, discriminator, geo_loop


def train_step(batch_images, batch_prior_poses, models, optimizers):
    phi, omega, lambda_net, discriminator, geo_loop = models
    opt_generator, opt_discriminator = optimizers

    # ---- Forward : mapping principal (F1->F2->F3) ----
    s = phi(batch_images)          # squelette prédit
    y = omega(s)                     # pose 2D
    v = lambda_net(y)                # pose 3D

    # ---- F4 : boucle de cohérence géométrique ----
    loop_results = geo_loop(v)

    # ---- F5 : discriminateur ----
    w = render_skeleton_batch(batch_prior_poses)  # squelette "réel" depuis le prior
    d_score_real = discriminator(w)
    d_score_fake = discriminator(s.detach())  # detach : ne pas propager dans Phi ici
    d_score_fake_for_g = discriminator(s)     # sans detach, pour que le gradient atteigne Phi

    # ---- Pertes (TOUS les forwards terminés AVANT tout optimizer.step) ----
    # Calculer toutes les pertes ici pour éviter que opt_discriminator.step()
    # mute les poids de D en place alors que le graphe de loss_total_generator
    # les référence encore (RuntimeError: version counter mismatch).
    loss_d = discriminator_loss(d_score_real, d_score_fake)

    loss_g_adv = generator_adversarial_loss(d_score_fake_for_g)

    gc_losses = geometric_consistency_loss(y, v, loop_results)
    loss_gc = gc_losses["L_GC"]

    loss_omega = omega_loss_fn(omega, phi, y, s, batch_prior_poses)

    loss_total_generator = loss_g_adv + loss_gc + loss_omega

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

    # ---- Backward + step : discriminateur ensuite ----
    # Les poids de D ne sont plus référencés par un graphe actif → step() sûr.
    opt_discriminator.zero_grad()
    loss_d.backward()
    opt_discriminator.step()

    return {
        "loss_d": loss_d.item(),
        "loss_g_adv": loss_g_adv.item(),
        "loss_gc": loss_gc.item(),
        "loss_omega": loss_omega.item(),
        "loss_total": loss_total_generator.item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true", help="Utiliser des données factices (Sprint 5)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    if not args.dummy:
        raise NotImplementedError(
            "Mode données réelles pas encore implémenté -- Sprint 6. "
            "Utilise --dummy pour ce test de stabilité (Sprint 5)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    phi, omega, lambda_net, discriminator, geo_loop = build_models()
    phi.to(device); omega.to(device); lambda_net.to(device); discriminator.to(device)

    generator_params = list(phi.parameters()) + list(omega.parameters()) + list(lambda_net.parameters())
    opt_generator = torch.optim.Adam(generator_params, lr=args.lr)
    opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    models = (phi, omega, lambda_net, discriminator, geo_loop)
    optimizers = (opt_generator, opt_discriminator)

    for epoch in range(1, args.epochs + 1):
        epoch_losses = {"loss_d": 0, "loss_g_adv": 0, "loss_gc": 0, "loss_omega": 0, "loss_total": 0}

        for batch_idx in range(args.batches_per_epoch):
            batch_images = torch.randn(args.batch_size, 3, 128, 128, device=device)
            batch_prior_poses = torch.rand(args.batch_size, 20, 2, device=device) * 2 - 1  # [-1,1]

            losses = train_step(batch_images, batch_prior_poses, models, optimizers)

            for k, v in losses.items():
                epoch_losses[k] += v
                assert not (v != v), f"NaN détecté dans {k} à l'epoch {epoch}, batch {batch_idx} !"  # v != v <=> isnan

        n = args.batches_per_epoch
        msg = " ".join(f"{k}={v/n:.4f}" for k, v in epoch_losses.items())
        print(f"[Epoch {epoch}/{args.epochs}] {msg}")

    print("\nEntraînement (données factices) terminé sans divergence -- Sprint 5 validé.")


if __name__ == "__main__":
    main()
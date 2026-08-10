"""
Pertes de cohérence géométrique (papier §3.2.2, équations 2-5).

Utilise les 5 variables produites par la boucle F4 (models/renderer.py) :
    y, y_hat, y_prime (poses 2D)
    v, v_hat, v_hat_prime, v_prime (poses 3D)

L_2D  = ||y' - y||^2                                          (eq. 2)
L_3D  = ||(v'_j - v'_k) - (v_j - v_k)||^2                       (eq. 3)
        -- comparaison de la DEFORMATION entre 2 échantillons j,k
        du batch, pas une comparaison directe v vs v' (cf. justification
        du papier : évite de pénaliser des ambiguïtés de profondeur légitimes)
L_r3D = ||v_hat' - v_hat||^2                                    (eq. 4)
L_GC  = L_2D + L_3D + L_r3D                                     (eq. 5)
"""
import torch


def l_2d(y: torch.Tensor, y_prime: torch.Tensor) -> torch.Tensor:
    """Equation 2 : cohérence de la pose 2D avant/après le cycle."""
    return ((y_prime - y) ** 2).mean()


def l_3d(v: torch.Tensor, v_prime: torch.Tensor) -> torch.Tensor:
    """
    Equation 3 : compare la déformation entre deux échantillons j,k du batch,
    plutôt que v et v_prime directement. On utilise un décalage circulaire
    du batch (roll) pour obtenir des paires (j,k) sans double boucle.
    """
    batch_size = v.shape[0]
    if batch_size < 2:
        # Pas de deuxième échantillon disponible (batch de taille 1) -- terme nul
        return torch.tensor(0.0, device=v.device)

    idx_k = torch.roll(torch.arange(batch_size, device=v.device), shifts=1)

    diff_v = v - v[idx_k]              # (v_j - v_k)
    diff_v_prime = v_prime - v_prime[idx_k]  # (v'_j - v'_k)

    return ((diff_v_prime - diff_v) ** 2).mean()


def l_r3d(v_hat: torch.Tensor, v_hat_prime: torch.Tensor) -> torch.Tensor:
    """Equation 4 : cohérence de la pose tournée elle-même."""
    return ((v_hat_prime - v_hat) ** 2).mean()


def geometric_consistency_loss(y: torch.Tensor, v: torch.Tensor, loop_results: dict) -> dict:
    """
    Calcule L_GC = L_2D + L_3D + L_r3D (équation 5), à partir de la pose
    d'origine (y, v) et du dict retourné par GeometricConsistencyLoop (F4).

    Retourne un dict avec chaque terme séparé (utile pour le monitoring
    pendant l'entraînement, cf. plan §Sprint6) + le total.
    """
    loss_2d = l_2d(y, loop_results["y_prime"])
    loss_3d = l_3d(v, loop_results["v_prime"])
    loss_r3d = l_r3d(loop_results["v_hat"], loop_results["v_hat_prime"])

    total = loss_2d + loss_3d + loss_r3d

    return {
        "L_2D": loss_2d,
        "L_3D": loss_3d,
        "L_r3D": loss_r3d,
        "L_GC": total,
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))

    from image_to_skeleton import ImageToSkeleton
    from skeleton_to_pose2d import SkeletonToPose2D
    from lifting_2d_3d import Lifting2Dto3D
    from renderer import GeometricConsistencyLoop

    batch_size = 4
    phi = ImageToSkeleton()
    omega = SkeletonToPose2D(n_keypoints=20)
    lambda_net = Lifting2Dto3D(n_keypoints=20)
    geo_loop = GeometricConsistencyLoop(lambda_net)

    dummy_image = torch.randn(batch_size, 3, 128, 128)
    skeleton = phi(dummy_image)
    y = omega(skeleton)
    v = lambda_net(y)
    loop_results = geo_loop(v)

    losses = geometric_consistency_loss(y, v, loop_results)

    for name, val in losses.items():
        print(f"{name:8s} : {val.item():.6f}")
        assert not torch.isnan(val), f"{name} est NaN !"

    # Vérifie que le gradient de L_GC atteint Phi (via toute la chaîne F1->F2->F3->F4)
    losses["L_GC"].backward()
    phi_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in phi.parameters())
    print(f"\nGradient de L_GC atteint Phi (F1) : {phi_has_grad}")
    assert phi_has_grad, "Le gradient de L_GC n'atteint pas Phi !"

    print("\nOK : les pertes de cohérence géométrique (L_2D, L_3D, L_r3D, L_GC) fonctionnent correctement.")

"""
Module F4 -- Boucle de rotation/projection (auto-supervision géométrique).
Pas un réseau : une opération géométrique déterministe qui réutilise Lambda
(F3). Permet d'apprendre une représentation 3D plausible sans aucune vérité
terrain 3D (papier §3.2.2).

Séquence exacte (5 étapes) :
    1. v -> rotation aléatoire (azimut + élévation) -> v_hat
    2. v_hat -> projection 2D -> y_hat
    3. y_hat -> Lambda (même réseau que F3) -> v_hat'
    4. v_hat' -> rotation inverse -> v'
    5. v' -> projection 2D -> y'

Ces 5 sorties (v_hat, y_hat, v_hat', v', y') servent ensuite au calcul de
L_GC (losses/geometric_consistency.py, prochaine étape).
"""
import torch
import math


def random_rotation_matrix(batch_size: int, device=None,
                            azimuth_range=(0, 360), elevation_range=(-30, 30)) -> torch.Tensor:
    """
    Génère une matrice de rotation 3D aléatoire par élément du batch, en
    échantillonnant azimut et élévation depuis une distribution uniforme
    (bornes choisies comme hyperparamètre, non précisées dans le papier --
    cf. question ouverte notée dans docs/literature_notes.md).

    Retourne : (batch, 3, 3)
    """
    azimuth = torch.empty(batch_size, device=device).uniform_(*azimuth_range) * math.pi / 180
    elevation = torch.empty(batch_size, device=device).uniform_(*elevation_range) * math.pi / 180

    cos_a, sin_a = torch.cos(azimuth), torch.sin(azimuth)
    cos_e, sin_e = torch.cos(elevation), torch.sin(elevation)

    zeros = torch.zeros(batch_size, device=device)
    ones = torch.ones(batch_size, device=device)

    # Rotation autour de l'axe vertical (azimut)
    R_azimuth = torch.stack([
        torch.stack([cos_a, zeros, sin_a], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sin_a, zeros, cos_a], dim=-1),
    ], dim=1)  # (batch, 3, 3)

    # Rotation autour de l'axe horizontal (élévation)
    R_elevation = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cos_e, -sin_e], dim=-1),
        torch.stack([zeros, sin_e, cos_e], dim=-1),
    ], dim=1)  # (batch, 3, 3)

    return torch.bmm(R_elevation, R_azimuth)  # (batch, 3, 3)


def apply_rotation(v: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Applique une matrice de rotation (batch,3,3) à une pose 3D (batch,K,3)."""
    return torch.bmm(v, R.transpose(1, 2))  # (batch, K, 3)


def project_to_2d(v: torch.Tensor) -> torch.Tensor:
    """
    Projection orthographique simple : on garde juste (x, y), on jette z.
    Choix cohérent avec la simplification du papier (pas de caméra perspective
    complexe, cf. §3.1 -- même esprit que la profondeur constante de Lambda).
    """
    return v[:, :, :2]  # (batch, K, 2)


class GeometricConsistencyLoop:
    """
    Orchestration de la boucle F4. Prend Lambda (F3, déjà instancié) et
    l'applique deux fois, comme décrit dans le papier.
    """

    def __init__(self, lambda_net):
        self.lambda_net = lambda_net

    def __call__(self, v: torch.Tensor) -> dict:
        """
        v : pose 3D d'origine (batch, K, 3), sortie de F3 sur les vraies données.
        Retourne un dict avec toutes les variables intermédiaires nécessaires
        au calcul de L_GC.
        """
        batch = v.shape[0]
        R = random_rotation_matrix(batch, device=v.device)

        v_hat = apply_rotation(v, R)              # étape 1 : rotation
        y_hat = project_to_2d(v_hat)               # étape 2 : projection 2D
        v_hat_prime = self.lambda_net(y_hat)        # étape 3 : re-lifting (même Lambda)
        v_prime = apply_rotation(v_hat_prime, R.transpose(1, 2))  # étape 4 : rotation inverse
        y_prime = project_to_2d(v_prime)            # étape 5 : projection 2D

        return {
            "v_hat": v_hat,
            "y_hat": y_hat,
            "v_hat_prime": v_hat_prime,
            "v_prime": v_prime,
            "y_prime": y_prime,
        }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from image_to_skeleton import ImageToSkeleton
    from skeleton_to_pose2d import SkeletonToPose2D
    from lifting_2d_3d import Lifting2Dto3D

    phi = ImageToSkeleton()
    omega = SkeletonToPose2D(n_keypoints=18)
    lambda_net = Lifting2Dto3D(n_keypoints=18)
    loop = GeometricConsistencyLoop(lambda_net)

    dummy_image = torch.randn(4, 3, 128, 128)
    skeleton = phi(dummy_image)
    y = omega(skeleton)
    v = lambda_net(y)

    results = loop(v)

    print("Test chaînage complet F1->F2->F3->F4 :")
    print(f"  image     : {dummy_image.shape}")
    print(f"  squelette : {skeleton.shape}")
    print(f"  y (pose2D): {y.shape}")
    print(f"  v (pose3D): {v.shape}")
    for key, val in results.items():
        print(f"  {key:15s}: {val.shape}")
        assert not torch.isnan(val).any(), f"NaN détecté dans {key} !"

    print("\nOK : le module F4 (boucle rotation/projection) fonctionne correctement.")

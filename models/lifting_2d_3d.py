"""
Module F3 -- Lambda : réseau fully-connected (pas convolutif, contrairement
à Phi et Omega) qui relève ("lift") la pose 2D en pose 3D.
v = Lambda(y)

Simplification du papier (§3.1) : pas d'apprentissage d'angle d'élévation de
caméra (contrairement à [27] et [31]). Pour chaque point 2D (xi, yi), le
réseau estime seulement un décalage de profondeur delta_i, ajouté à une
profondeur constante d (choisie ici comme hyperparamètre documenté).

Lambda est réutilisé deux fois par forward pendant la boucle de cohérence
géométrique (F4) : une fois sur la pose 2D d'origine, une fois sur la pose 2D
projetée après rotation -- c'est le MÊME réseau, mêmes poids, appelé deux fois.

Entrée  : pose 2D (batch, K, 2)
Sortie  : pose 3D (batch, K, 3)
"""
import torch
import torch.nn as nn

# Profondeur constante d (hyperparamètre du papier, valeur non précisée --
# choix d'implémentation à documenter et ajuster empiriquement, cf. plan §Sprint4/Jour3)
DEFAULT_CONSTANT_DEPTH = 0.0


class Lifting2Dto3D(nn.Module):
    """Lambda : pose 2D (K, 2) -> pose 3D (K, 3), via un MLP par point clé."""

    def __init__(self, n_keypoints: int = 20, hidden_dim: int = 256,
                 constant_depth: float = DEFAULT_CONSTANT_DEPTH):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.constant_depth = constant_depth

        in_dim = n_keypoints * 2  # toutes les coordonnées 2D concaténées

        # MLP global : voit la pose entière (pas juste un point isolé), ce qui
        # permet de capturer les relations entre points (ex: la profondeur
        # d'un sabot dépend de la posture générale du cheval)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, n_keypoints),  # un delta de profondeur par point clé
            nn.Tanh(),  # borne delta dans [-1, 1], évite des profondeurs aberrantes
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y : (batch, K, 2)
        batch = y.shape[0]
        y_flat = y.reshape(batch, -1)          # (batch, K*2)
        delta_z = self.mlp(y_flat)              # (batch, K)

        z = self.constant_depth + delta_z       # (batch, K), profondeur par point

        v = torch.cat([y, z.unsqueeze(-1)], dim=-1)  # (batch, K, 3) -- (x, y, z)
        return v


if __name__ == "__main__":
    model = Lifting2Dto3D(n_keypoints=20)
    dummy_pose2d = torch.randn(4, 20, 2)
    output = model(dummy_pose2d)

    print(f"Entrée  : {dummy_pose2d.shape}")
    print(f"Sortie  : {output.shape}")
    assert output.shape == (4, 20, 3), "Forme de sortie incorrecte !"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres : {n_params:,}")

    # Test de chaînage complet F1 -> F2 -> F3
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from image_to_skeleton import ImageToSkeleton
    from skeleton_to_pose2d import SkeletonToPose2D

    phi = ImageToSkeleton()
    omega = SkeletonToPose2D(n_keypoints=20)

    dummy_image = torch.randn(4, 3, 128, 128)
    skeleton = phi(dummy_image)
    pose_2d = omega(skeleton)
    pose_3d = model(pose_2d)

    print(f"\nTest chaînage F1->F2->F3 :")
    print(f"  image   : {dummy_image.shape}")
    print(f"  squelette : {skeleton.shape}")
    print(f"  pose 2D : {pose_2d.shape}")
    print(f"  pose 3D : {pose_3d.shape}")

    # Vérifie que Lambda est bien réutilisable (même réseau, appelé une 2e fois,
    # comme dans la boucle de cohérence géométrique F4 à venir)
    pose_2d_bis = torch.randn(4, 20, 2)
    pose_3d_bis = model(pose_2d_bis)  # même instance "model", 2e appel
    print(f"  Réutilisation de Lambda (2e appel, même poids) : {pose_3d_bis.shape}")

    print("\nOK : le module F3 (Lambda) fonctionne correctement, seul et chaîné avec F1->F2.")

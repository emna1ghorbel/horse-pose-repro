"""
Module F2 -- Omega : CNN qui extrait les coordonnées des points clés (pose 2D)
à partir de l'image squelette produite par Phi (F1).
y = Omega(s) = Omega(Phi(x))

Approche : Omega produit des heatmaps (une carte de probabilité de présence
par point clé), puis un soft-argmax convertit chaque heatmap en coordonnées
(x, y) continues et différentiables -- nécessaire pour que le gradient
puisse remonter jusqu'à Phi pendant l'entraînement (cf. discussion sur la
rétropropagation à travers toute la chaîne).

Entrée  : image squelette (batch, 1, 128, 128)
Sortie  : coordonnées 2D (batch, K, 2), normalisées dans [-1, 1]
          K = nombre de points clés (20, cohérent avec le papier)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.block(x)


def soft_argmax_2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Convertit des heatmaps (batch, K, H, W) en coordonnées (batch, K, 2)
    continues et différentiables, via une moyenne pondérée par une
    distribution softmax sur chaque heatmap (au lieu d'un argmax classique,
    non différentiable).
    """
    batch, K, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(batch, K, -1)
    weights = F.softmax(heatmaps_flat, dim=-1)  # distribution de probabilité par point clé

    ys = torch.linspace(-1, 1, H, device=heatmaps.device)
    xs = torch.linspace(-1, 1, W, device=heatmaps.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    x_coord = (weights * grid_x).sum(dim=-1)
    y_coord = (weights * grid_y).sum(dim=-1)

    return torch.stack([x_coord, y_coord], dim=-1)  # (batch, K, 2)


class SkeletonToPose2D(nn.Module):
    """Omega : image squelette (1,128,128) -> pose 2D (K, 2)."""

    def __init__(self, n_keypoints: int = 20, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.n_keypoints = n_keypoints

        self.features = nn.Sequential(
            ConvBlock(1, c),
            nn.MaxPool2d(2),          # 64x64
            ConvBlock(c, c * 2),
            nn.MaxPool2d(2),          # 32x32
            ConvBlock(c * 2, c * 4),
            nn.MaxPool2d(2),          # 16x16
            ConvBlock(c * 4, c * 4),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2),  # 32x32
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(c * 2, c, 2, stride=2),      # 64x64
            nn.ReLU(inplace=False),
        )

        self.heatmap_head = nn.Conv2d(c, n_keypoints, kernel_size=1)  # (batch, K, 64, 64)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        feat = self.features(s)
        feat = self.upsample(feat)
        heatmaps = self.heatmap_head(feat)  # (batch, K, 64, 64)
        coords = soft_argmax_2d(heatmaps)   # (batch, K, 2)
        return coords


if __name__ == "__main__":
    model = SkeletonToPose2D(n_keypoints=20)
    dummy_skeleton = torch.randn(4, 1, 128, 128)
    output = model(dummy_skeleton)

    print(f"Entrée  : {dummy_skeleton.shape}")
    print(f"Sortie  : {output.shape}")
    assert output.shape == (4, 20, 2), "Forme de sortie incorrecte !"
    assert output.min() >= -1.01 and output.max() <= 1.01, "Coordonnées hors [-1,1] !"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres : {n_params:,}")

    # Test de chaînage F1 -> F2
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from image_to_skeleton import ImageToSkeleton

    phi = ImageToSkeleton()
    dummy_image = torch.randn(4, 3, 128, 128)
    skeleton = phi(dummy_image)      # sortie de F1
    pose_2d = model(skeleton)         # devient l'entrée de F2
    print(f"\nTest chaînage F1->F2 : image {dummy_image.shape} -> squelette {skeleton.shape} -> pose2D {pose_2d.shape}")

    print("OK : le module F2 (Omega) fonctionne correctement, seul et chaîné avec F1.")

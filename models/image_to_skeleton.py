"""
Module F1 -- Phi : CNN qui apprend le mapping image -> image squelette.
s = Phi(x)

Architecture : U-Net léger (encodeur-décodeur avec skip connections),
adapté à une tâche de traduction image-à-image sur des images 128x128.
L'architecture exacte n'est pas spécifiée dans le papier (cf. docs/literature_notes.md,
question ouverte notée pour l'encadrant) -- ce choix est documenté ici comme
décision d'implémentation.

Entrée  : image RGB (batch, 3, 128, 128)
Sortie  : image squelette (batch, 1, 128, 128), valeurs dans [0, 1] (sigmoid)
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Bloc conv -> batchnorm -> ReLU, répété deux fois."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.block(x)


class ImageToSkeleton(nn.Module):
    """Phi : U-Net léger, image (3,128,128) -> squelette (1,128,128)."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels

        # Encodeur
        self.enc1 = ConvBlock(3, c)          # 128x128
        self.pool1 = nn.MaxPool2d(2)         # 64x64
        self.enc2 = ConvBlock(c, c * 2)      # 64x64
        self.pool2 = nn.MaxPool2d(2)         # 32x32
        self.enc3 = ConvBlock(c * 2, c * 4)  # 32x32
        self.pool3 = nn.MaxPool2d(2)         # 16x16

        # Goulot d'étranglement
        self.bottleneck = ConvBlock(c * 4, c * 8)  # 16x16

        # Décodeur (avec skip connections)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)   # -> 32x32
        self.dec3 = ConvBlock(c * 8, c * 4)  # concat avec enc3 (c*4 + c*4 = c*8)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)   # -> 64x64
        self.dec2 = ConvBlock(c * 4, c * 2)  # concat avec enc2

        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)       # -> 128x128
        self.dec1 = ConvBlock(c * 2, c)      # concat avec enc1

        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)
        self.out_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, 3, 128, 128)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)
        return self.out_activation(out)  # (batch, 1, 128, 128), valeurs [0,1]


if __name__ == "__main__":
    # Auto-test rapide
    model = ImageToSkeleton()
    dummy_input = torch.randn(4, 3, 128, 128)
    output = model(dummy_input)
    print(f"Entrée  : {dummy_input.shape}")
    print(f"Sortie  : {output.shape}")
    assert output.shape == (4, 1, 128, 128), "Forme de sortie incorrecte !"
    assert output.min() >= 0 and output.max() <= 1, "Valeurs hors [0,1] !"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres : {n_params:,}")
    print("OK : le module F1 (Phi) fonctionne correctement.")

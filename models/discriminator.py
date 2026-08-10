"""
Module F5 -- Discriminateur D : juge si une image squelette ressemble à un
squelette "authentique" (rendu depuis le prior synthétique) ou à un
squelette prédit par Phi (F1) sur une vraie image.

Papier §3.2.1, équation 1 :
    L_D = E_w[log(D(w))] + E_s[log(1 - D(s))]
    où w = squelette réaliste (rendu du prior synthétique via beta)
          s = squelette prédit par Phi

Entrée  : image squelette (batch, 1, 128, 128) -- soit w, soit s
Sortie  : score de plausibilité (batch, 1), dans [0, 1] (1 = jugé réaliste)
"""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """D : image squelette (1,128,128) -> score de plausibilité scalaire."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels

        self.features = nn.Sequential(
            nn.Conv2d(1, c, 4, stride=2, padding=1),        # 128 -> 64
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(c, c * 2, 4, stride=2, padding=1),      # 64 -> 32
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(c * 4, c * 8, 4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (batch, c*8, 1, 1)
            nn.Flatten(),               # (batch, c*8)
            nn.Linear(c * 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, skeleton_image: torch.Tensor) -> torch.Tensor:
        feat = self.features(skeleton_image)
        score = self.classifier(feat)
        return score  # (batch, 1)


if __name__ == "__main__":
    model = Discriminator()

    dummy_skeleton = torch.randn(4, 1, 128, 128)
    output = model(dummy_skeleton)

    print(f"Entrée  : {dummy_skeleton.shape}")
    print(f"Sortie  : {output.shape}")
    assert output.shape == (4, 1), "Forme de sortie incorrecte !"
    assert output.min() >= 0 and output.max() <= 1, "Scores hors [0,1] !"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres : {n_params:,}")

    # Test avec la sortie réelle de F1 (Phi), pas juste du bruit
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from image_to_skeleton import ImageToSkeleton

    phi = ImageToSkeleton()
    dummy_image = torch.randn(4, 3, 128, 128)
    skeleton_from_phi = phi(dummy_image)  # s = Phi(x)
    score_on_fake = model(skeleton_from_phi)

    print(f"\nTest sur sortie de Phi (s) : score = {score_on_fake.squeeze().tolist()}")
    print("(scores proches de 0.5 attendus : D non entraîné, ne sait pas encore juger)")

    print("\nOK : le module F5 (Discriminateur) fonctionne correctement.")
"""
Perte adversariale L_D (papier §3.2.1, équation 1) :

    L_D = E_w[log(D(w))] + E_s[log(1 - D(s))]

où :
    w = squelette "réel" (rendu depuis le prior synthétique via beta)
    s = squelette prédit par Phi (F1) sur une vraie image

Cette perte sert à entraîner LE DISCRIMINATEUR (le maximiser -- il veut bien
distinguer w de s). Le générateur (Phi/F1) veut au contraire MINIMISER le
deuxième terme seul (tromper D) -- d'où les deux optimizers séparés dans la
boucle d'entraînement (Sprint 5, Jour 4).
"""
import torch


def discriminator_loss(d_score_real: torch.Tensor, d_score_fake: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """
    Perte du DISCRIMINATEUR : à maximiser par D (donc on retourne l'opposé,
    pour minimiser avec un optimizer standard).

    d_score_real : D(w), scores sur les squelettes réels du prior (batch, 1)
    d_score_fake : D(s), scores sur les squelettes prédits par Phi (batch, 1)
    """
    loss_real = torch.log(d_score_real + eps).mean()
    loss_fake = torch.log(1 - d_score_fake + eps).mean()
    loss_d = -(loss_real + loss_fake)  # signe négatif : on MINIMISE l'opposé du terme à maximiser
    return loss_d


def generator_adversarial_loss(d_score_fake: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Perte adversariale côté GÉNÉRATEUR (Phi/F1) : veut que D(s) soit proche
    de 1 (tromper le discriminateur). C'est cette perte qui, via son
    gradient, corrige F1 -- le coeur du mécanisme discuté précédemment.

    Formulation "non-saturating" (standard en pratique GAN, plus stable que
    minimiser directement log(1 - D(s)) qui sature tôt) :
        L_G = -E_s[log(D(s))]
    """
    loss_g = -torch.log(d_score_fake + eps).mean()
    return loss_g


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))

    from models.image_to_skeleton import ImageToSkeleton
    from models.discriminator import Discriminator

    phi = ImageToSkeleton()
    D = Discriminator()

    batch_size = 4
    dummy_real_skeleton = torch.rand(batch_size, 1, 128, 128)  # simule w (rendu du prior)
    dummy_image = torch.randn(batch_size, 3, 128, 128)

    s = phi(dummy_image)  # squelette prédit par F1

    d_score_real = D(dummy_real_skeleton)
    d_score_fake = D(s)

    loss_d = discriminator_loss(d_score_real, d_score_fake)
    loss_g = generator_adversarial_loss(d_score_fake)

    print(f"D(w) [réel]  : {d_score_real.squeeze().tolist()}")
    print(f"D(s) [faux]  : {d_score_fake.squeeze().tolist()}")
    print(f"L_D (discriminateur) : {loss_d.item():.4f}")
    print(f"L_G (générateur/F1)  : {loss_g.item():.4f}")

    assert not torch.isnan(loss_d), "L_D est NaN !"
    assert not torch.isnan(loss_g), "L_G est NaN !"

    # Vérifie que le gradient de L_G remonte bien jusqu'à Phi (pas juste jusqu'à D)
    loss_g.backward()
    phi_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in phi.parameters())
    print(f"\nGradient de L_G atteint Phi (F1) : {phi_has_grad}")
    assert phi_has_grad, "Le gradient de L_G n'atteint pas Phi !"

    print("\nOK : les pertes adversariales (L_D, L_G) fonctionnent correctement.")

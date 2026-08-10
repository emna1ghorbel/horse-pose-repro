"""
Utilitaire de redimensionnement image + masque avec padding (letterbox).

Remplace les simples .resize() qui déforment le cheval quand l'image
n'est pas carrée — problème particulièrement fréquent sur les frames
YouTube (formats 16:9, 9:16 Shorts, etc.) et sur certains shots TigDog.

Fonctionnement :
  1. Calcule le ratio qui fait tenir l'image dans target_size × target_size
     sans dépasser (scale = min(target_size/W, target_size/H)).
  2. Redimensionne image et masque avec ce ratio (BILINEAR / NEAREST).
  3. Centre le résultat dans un canvas carré rempli de 0 (noir pour
     l'image RGB, 0 pour le masque).

L'image de sortie est toujours un objet PIL.Image.
"""

from PIL import Image
import numpy as np


def resize_pair_with_padding(
    image: Image.Image,
    mask: Image.Image,
    target_size: int,
) -> tuple[Image.Image, Image.Image]:
    """
    Redimensionne image + masque de manière isotrope avec padding centré.

    Parameters
    ----------
    image       : PIL.Image.Image  — image RGB (ou L) source.
    mask        : PIL.Image.Image  — masque binaire (mode L, valeurs 0/255).
    target_size : int              — taille du carré de sortie (côté unique).

    Returns
    -------
    (image_resized, mask_resized) : deux PIL.Image.Image de taille
                                    (target_size × target_size).
    """
    orig_w, orig_h = image.size

    # --- 1. Ratio isotrope (letterbox) ---
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    # --- 2. Redimensionnement ---
    img_scaled = image.resize((new_w, new_h), Image.BILINEAR)
    mask_scaled = mask.resize((new_w, new_h), Image.NEAREST)

    # --- 3. Canvas carré + centrage ---
    img_mode = image.mode  # "RGB" ou "L"
    pad_value_img = (0, 0, 0) if img_mode == "RGB" else 0

    canvas_img = Image.new(img_mode, (target_size, target_size), pad_value_img)
    canvas_mask = Image.new("L", (target_size, target_size), 0)

    x_off = (target_size - new_w) // 2
    y_off = (target_size - new_h) // 2

    canvas_img.paste(img_scaled, (x_off, y_off))
    canvas_mask.paste(mask_scaled, (x_off, y_off))

    return canvas_img, canvas_mask

"""
Teste le module F1 (Phi) sur une vraie image du dataset prétraité, pas juste
sur du bruit aléatoire. Le modèle n'étant pas encore entraîné (poids
aléatoires), la sortie sera visuellement du bruit -- ce test valide la
PLOMBERIE (formats, dimensions, absence de crash), pas la qualité.

Usage :
    python test_f1_real_image.py --data-dir ../data/processed/final --index 0
"""
import argparse
import json
import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(__file__))
from image_to_skeleton import ImageToSkeleton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                         help="Dossier processed/final (contenant images/ et dataset_registry.json)")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    registry_path = os.path.join(args.data_dir, "dataset_registry.json")
    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)

    entry = registry["entries"][args.index]
    img_path = os.path.join(args.data_dir, "images", entry["image"])
    print(f"Image utilisée : {entry['image']}")

    img_pil = Image.open(img_path).convert("RGB")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img_pil).unsqueeze(0)  # (1, 3, 128, 128)

    print(f"Tenseur d'entrée : {img_tensor.shape}")

    model = ImageToSkeleton()
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)  # (1, 1, 128, 128)

    print(f"Tenseur de sortie : {output.shape}")
    print(f"Min/Max sortie : {output.min().item():.4f} / {output.max().item():.4f}")
    print(f"NaN présents : {torch.isnan(output).any().item()}")

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_pil)
    axes[0].set_title("Image réelle (entrée)")
    axes[0].axis("off")

    skeleton_out = output.squeeze().numpy()
    axes[1].imshow(skeleton_out, cmap="gray")
    axes[1].set_title("Sortie de Phi (modèle NON entraîné -> bruit attendu)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("f1_test_real_image.png", dpi=120)
    print("Sauvegardé : f1_test_real_image.png")
    plt.show()


if __name__ == "__main__":
    main()

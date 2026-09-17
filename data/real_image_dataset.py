"""
Dataset PyTorch pour les images d'entraînement réelles (processed/final/,
consolidation TigDog + YouTube, Sprint 2). Remplace le mode --dummy de
training/train.py.

Usage (test) :
    python real_image_dataset.py --data-dir ./processed/final
"""
import argparse
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class HorseImageDataset(Dataset):
    """
    Charge les images réelles (déjà en 128x128, cf. Sprint 2) à partir du
    registre consolidé (dataset_registry.json). Les masques de segmentation
    sont retournés avec les images : ils servent à pénaliser les pixels de
    squelette prédits hors de la silhouette, sans fournir de pose annotée.
    """

    def __init__(self, data_dir: str, image_size: int = 128):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "masks")

        registry_path = os.path.join(data_dir, "dataset_registry.json")
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
        self.entries = registry["entries"]

        # Normalisation standard ImageNet (choix courant pour un encodeur CNN
        # entraîné from scratch ; à ajuster si besoin après premiers essais)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),  # sécurité, déjà 128x128 normalement
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.images_dir, entry["image"])
        img = Image.open(img_path).convert("RGB")
        image = self.transform(img)

        mask_name = entry.get("mask")
        if mask_name is None:
            raise KeyError(f"Entrée {idx} sans masque dans dataset_registry.json")
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        # Toute valeur non nulle correspond à la silhouette du cheval.
        mask = (self.mask_transform(mask) > 0).float()
        return image, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    dataset = HorseImageDataset(args.data_dir)
    print(f"Dataset chargé : {len(dataset)} images.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    images, masks = next(iter(loader))
    print(f"Images shape : {images.shape}")  # [batch_size, 3, 128, 128]
    print(f"Masques shape : {masks.shape}")  # [batch_size, 1, 128, 128]
    print(f"NaN dans les images : {torch.isnan(images).any().item()}")
    print(f"Min/Max images : {images.min().item():.3f} / {images.max().item():.3f}")


if __name__ == "__main__":
    main()

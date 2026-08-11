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
    registre consolidé (dataset_registry.json). Ne charge PAS les masques
    ici : le masque n'a servi qu'au filtrage (Sprint 2), il n'entre pas dans
    l'entraînement du modèle lui-même (méthode auto-supervisée, cf. §RM1 du
    plan -- aucune annotation n'est utilisée).
    """

    def __init__(self, data_dir: str, image_size: int = 128):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")

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

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.images_dir, entry["image"])
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    dataset = HorseImageDataset(args.data_dir)
    print(f"Dataset chargé : {len(dataset)} images.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(loader))
    print(f"Batch shape : {batch.shape}")  # attendu : [batch_size, 3, 128, 128]
    print(f"NaN dans le batch : {torch.isnan(batch).any().item()}")
    print(f"Min/Max valeurs : {batch.min().item():.3f} / {batch.max().item():.3f}")


if __name__ == "__main__":
    main()

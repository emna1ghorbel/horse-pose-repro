"""
Consolidation finale du dataset prétraité (TigDog + YouTube) avec
enregistrement de checksums pour traçabilité.

Usage :
    python consolidate_dataset.py --sources ./processed/tigdog ./processed/youtube --output-dir ./processed/final
"""
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True,
                         help="Dossiers sources (doivent contenir images/ et masks/)")
    parser.add_argument("--output-dir", type=str, default="./processed/final")
    args = parser.parse_args()

    images_out = os.path.join(args.output_dir, "images")
    masks_out = os.path.join(args.output_dir, "masks")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)

    registry = {
        "created_at": datetime.now().isoformat(),
        "sources": args.sources,
        "entries": [],
    }

    total_pairs = 0
    total_missing_pairs = 0

    for source in args.sources:
        src_images = os.path.join(source, "images")
        src_masks = os.path.join(source, "masks")

        if not os.path.isdir(src_images):
            print(f"ATTENTION : {src_images} introuvable, source ignorée.")
            continue

        image_files = sorted(f for f in os.listdir(src_images) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        print(f"Source {source} : {len(image_files)} images trouvées.")

        for img_name in image_files:
            base_name = os.path.splitext(img_name)[0]
            mask_name = base_name + "_mask.png"

            src_img_path = os.path.join(src_images, img_name)
            src_mask_path = os.path.join(src_masks, mask_name)

            if not os.path.exists(src_mask_path):
                total_missing_pairs += 1
                continue

            dst_img_path = os.path.join(images_out, img_name)
            dst_mask_path = os.path.join(masks_out, mask_name)

            shutil.copy2(src_img_path, dst_img_path)
            shutil.copy2(src_mask_path, dst_mask_path)

            registry["entries"].append({
                "image": img_name,
                "mask": mask_name,
                "source": source,
                "image_sha256": sha256_of_file(dst_img_path),
                "mask_sha256": sha256_of_file(dst_mask_path),
            })
            total_pairs += 1

    registry["total_pairs"] = total_pairs
    registry["total_missing_pairs"] = total_missing_pairs

    registry_path = os.path.join(args.output_dir, "dataset_registry.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"\n=== Consolidation terminée ===")
    print(f"Paires copiées      : {total_pairs}")
    print(f"Paires incomplètes  : {total_missing_pairs} (masque manquant, ignorées)")
    print(f"Registre (checksums): {registry_path}")


if __name__ == "__main__":
    main()
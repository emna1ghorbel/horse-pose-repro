"""
Génération des masques de segmentation pour les frames YouTube filtrées,
via YOLOv8-seg (Ultralytics) — plus simple à installer que Detectron2
sous Windows, cohérent avec le YOLOv8m déjà utilisé pour le filtrage.

TigDog fournit déjà ses masques (horseSeg/) — ce script ne concerne QUE
les frames YouTube, qui n'ont pas de masque natif.

Prérequis :
    pip install ultralytics opencv-python

Usage :
    python generate_youtube_masks.py --input-dir ./raw/youtube_frames_filtered --output-dir ./processed/youtube
"""
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from resize_utils import resize_pair_with_padding

# Classe "horse" dans COCO (dataset sur lequel YOLOv8-seg est pré-entraîné)
COCO_HORSE_CLASS_ID = 17


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True,
                         help="Dossier contenant les frames YouTube filtrées (.jpg)")
    parser.add_argument("--output-dir", type=str, default="./processed/youtube")
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--model", type=str, default="yolov8m-seg.pt",
                         help="Poids YOLOv8-seg (téléchargés automatiquement au premier lancement)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None,
                         help="Limiter le nombre d'images traitées (test rapide)")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)

    model = YOLO(args.model)

    filenames = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if args.limit:
        filenames = filenames[: args.limit]

    kept, rejected = 0, 0
    log = []

    for i, fname in enumerate(filenames, 1):
        img_path = os.path.join(args.input_dir, fname)
        image_bgr = cv2.imread(img_path)

        if image_bgr is None:
            rejected += 1
            log.append(f"{fname}: image illisible, ignorée")
            continue

        results = model.predict(image_bgr, conf=args.confidence_threshold, verbose=False)
        result = results[0]

        if result.masks is None or len(result.masks.data) == 0:
            rejected += 1
            log.append(f"{fname}: aucune instance détectée par YOLOv8-seg, rejetée")
            continue

        classes = result.boxes.cls.cpu().numpy().astype(int)
        horse_indices = np.where(classes == COCO_HORSE_CLASS_ID)[0]

        if len(horse_indices) == 0:
            rejected += 1
            log.append(f"{fname}: aucun cheval détecté (autres classes présentes), rejetée")
            continue

        masks_data = result.masks.data.cpu().numpy()  # (N, H, W)
        horse_masks = masks_data[horse_indices]

        # Garder la plus grande instance (cheval le plus visible/proche)
        areas = horse_masks.sum(axis=(1, 2))
        best_mask = horse_masks[np.argmax(areas)]

        # Redimensionner image + masque en target_size x target_size (isotrope avec padding centré)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_mask = Image.fromarray((best_mask.astype(np.uint8) * 255), mode="L")

        pil_image_resized, pil_mask_resized = resize_pair_with_padding(pil_image, pil_mask, args.target_size)

        image_resized = np.array(pil_image_resized)  # (H, W, 3) RGB
        mask_resized = np.array(pil_mask_resized)    # (H, W)  0/255

        out_name = os.path.splitext(fname)[0]
        cv2.imwrite(
            os.path.join(args.output_dir, "images", out_name + ".jpg"),
            cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(os.path.join(args.output_dir, "masks", out_name + "_mask.png"), mask_resized)

        kept += 1
        if i % 50 == 0:
            print(f"[{i}/{len(filenames)}] {kept} gardées, {rejected} rejetées")

    print(f"\n=== Terminé ===")
    print(f"Total gardées  : {kept}")
    print(f"Total rejetées : {rejected}")
    if kept + rejected > 0:
        print(f"Taux de rejet  : {100 * rejected / (kept + rejected):.1f}%")

    report_path = os.path.join(args.output_dir, "masking_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Modele : YOLOv8-seg ({args.model}), seuil de confiance={args.confidence_threshold}\n")
        f.write(f"Frames gardees : {kept}\n")
        f.write(f"Frames rejetees : {rejected}\n")
        f.write("\nDetails :\n")
        for line in log:
            f.write(f" - {line}\n")
    print(f"Rapport sauvegarde : {report_path}")


if __name__ == "__main__":
    main()
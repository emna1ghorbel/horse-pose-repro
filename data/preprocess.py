"""
Prétraitement du sous-ensemble cheval de TigDog (Sprint 2).

Pour chaque frame de chaque shot :
  1. Charge l'image (horse/) + le masque (horseSeg/)
  2. Filtre : rejette si le masque est vide ou trop petit (cheval non visible/trop partiel)
  3. Redimensionne image + masque en 128x128 (isotrope avec padding centré)
  4. Sauvegarde dans data/processed/tigdog/images/ et data/processed/tigdog/masks/

Usage (depuis le dossier horse-pose-repro/data) :
    python preprocess.py --raw-dir ./raw/behaviorDiscovery2.0 --output-dir ./processed/tigdog
"""
import argparse
import os
import numpy as np
import scipy.io
from PIL import Image
from resize_utils import resize_pair_with_padding


def load_ranges(raw_dir: str):
    """Retourne un tableau (99, 3) : [shot_id, first_frame, last_frame]."""
    path = os.path.join(raw_dir, "ranges", "horse", "ranges.mat")
    data = scipy.io.loadmat(path)
    return data["ranges"]


def load_segments(raw_dir: str, shot_id: int):
    """Retourne le cell array 'segments' pour un shot donné (un masque par frame)."""
    path = os.path.join(raw_dir, "horseSeg", f"seg{shot_id}.mat")
    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    return data["segments"]
    


def mask_quality_ok(mask: np.ndarray, min_ratio: float = 0.02, max_ratio: float = 0.85) -> bool:
    """
    Filtre de qualité (cf. §2.5 du plan, cas limite 'masque de mauvaise qualité').
    Rejette si le masque est quasi vide (cheval absent/minuscule) ou quasi plein
    (probable erreur de segmentation couvrant toute l'image).
    """
    fg_ratio = mask.mean()
    return min_ratio <= fg_ratio <= max_ratio


def process_shot(raw_dir: str, output_dir: str, shot_id: int, first_frame: int, last_frame: int,
                  target_size: int = 128, log=None):
    try:
        segments = load_segments(raw_dir, shot_id)
    except FileNotFoundError:
        if log is not None:
            log.append(f"Shot {shot_id}: fichier de segmentation introuvable, shot ignoré")
        return 0, 0

    n_frames_shot = last_frame - first_frame + 1
    kept, rejected = 0, 0

    for local_idx in range(n_frames_shot):
        frame_number = first_frame + local_idx
        image_path = os.path.join(raw_dir, "horse", f"{frame_number:08d}.jpg")

        if not os.path.exists(image_path):
            rejected += 1
            continue
        if local_idx >= len(segments):
            rejected += 1
            continue

        mask = np.asarray(segments[local_idx]).astype(np.uint8)
        if mask.max() > 1:  # normaliser si le masque n'est pas déjà en {0,1}
            mask = (mask > 0).astype(np.uint8)

        if not mask_quality_ok(mask):
            rejected += 1
            continue

        image = Image.open(image_path).convert("RGB")
        mask_img = Image.fromarray(mask * 255)

        image_resized, mask_resized = resize_pair_with_padding(image, mask_img, target_size)

        out_name = f"tigdog_{shot_id}_{frame_number:08d}"
        image_resized.save(os.path.join(output_dir, "images", out_name + ".jpg"), quality=95)
        mask_resized.save(os.path.join(output_dir, "masks", out_name + "_mask.png"))

        kept += 1

    return kept, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="./raw/behaviorDiscovery2.0")
    parser.add_argument("--output-dir", type=str, default="./processed/tigdog")
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--limit-shots", type=int, default=None,
                         help="Limiter le nombre de shots traités (pour un test rapide)")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)

    ranges = load_ranges(args.raw_dir)
    if args.limit_shots:
        ranges = ranges[: args.limit_shots]

    total_kept, total_rejected = 0, 0
    log = []

    for i, (shot_id, first_frame, last_frame) in enumerate(ranges, 1):
        print(f"[{i}/{len(ranges)}] Shot {shot_id} (frames {first_frame}-{last_frame})...", end=" ")
        kept, rejected = process_shot(
            args.raw_dir, args.output_dir, int(shot_id), int(first_frame), int(last_frame),
            target_size=args.target_size, log=log,
        )
        print(f"{kept} gardées, {rejected} rejetées")
        total_kept += kept
        total_rejected += rejected

    print(f"\n=== Terminé ===")
    print(f"Total gardées   : {total_kept}")
    print(f"Total rejetées  : {total_rejected}")
    if total_kept + total_rejected > 0:
        print(f"Taux de rejet   : {100 * total_rejected / (total_kept + total_rejected):.1f}%")

    if log:
        print("\nAvertissements :")
        for line in log:
            print(" -", line)

    # Rapport de qualité (cf. Sprint 2, Jour 5 du plan)
    report_path = os.path.join(args.output_dir, "preprocessing_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Frames gardées : {total_kept}\n")
        f.write(f"Frames rejetées : {total_rejected}\n")
        if total_kept + total_rejected > 0:
            f.write(f"Taux de rejet : {100 * total_rejected / (total_kept + total_rejected):.1f}%\n")
        f.write("\nAvertissements :\n")
        for line in log:
            f.write(f" - {line}\n")
    print(f"\nRapport sauvegardé : {report_path}")


if __name__ == "__main__":
    main()
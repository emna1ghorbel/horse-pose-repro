"""
Extraction de frames depuis vidéos YouTube.

Modes disponibles :

1) Recherche automatique -> fichier de CANDIDATS A REVOIR (pas d'extraction) :
   python extract_youtube_frames.py --search-youtube --candidates-out candidates.txt

   Ouvre ensuite candidates.txt : chaque ligne ressemble à
       https://www.youtube.com/watch?v=XXXX   # 245s | Channel Name | Horse jumping compilation 2023
   Supprime les lignes hors-sujet (interviews, compilations, reviews de
   matériel équestre, vlogs, vidéos trop courtes/longues, doublons de
   chaîne...), puis passe ce fichier nettoyé en --urls-file.

   C'est l'étape de revue manuelle attendue par le papier (§4.1 : "manually
   defined group of YouTube videos") et par le plan (Sprint 2, Jour 2.1).
   Elle reste rapide : tu lis des titres/durées, tu ne regardes pas les
   vidéos une par une.

2) Extraction à partir d'un fichier d'URLs déjà revu :
   python extract_youtube_frames.py --urls-file candidates.txt

Pipeline :
YouTube
   |
   | yt-dlp ytsearch (métadonnées seules, rien n'est téléchargé)
   |
candidates.txt   <-- REVUE MANUELLE ICI (obligatoire avant extraction)
   |
   | yt-dlp (téléchargement vidéo)
   |
   | ffmpeg (extraction frames)
   |
Frames JPG + provenance.csv
"""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime, timezone


# =====================================================
# Vérification dépendances
# =====================================================

def check_dependencies():
    missing = []
    for tool in ["yt-dlp", "ffmpeg"]:
        try:
            subprocess.run([tool, "--version"], capture_output=True)
        except FileNotFoundError:
            missing.append(tool)
    if missing:
        print("Outils manquants :", ", ".join(missing))
        print("Installer : pip install yt-dlp + ffmpeg")
        sys.exit(1)


# =====================================================
# Lecture fichier URL (accepte aussi le format candidates.txt avec commentaire)
# =====================================================

def read_urls(path):
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # tolère "URL # commentaire" -> on ne garde que la partie URL
            url = line.split("#", 1)[0].strip()
            if url:
                urls.append(url)
    return urls


# =====================================================
# Lecture des video_id déjà traités (idempotence)
# =====================================================

def read_processed_ids(csv_path):
    processed = set()
    if not os.path.exists(csv_path):
        return processed
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                processed.add(row[0])
    return processed


# =====================================================
# Recherche automatique YouTube -> fichier de CANDIDATS (pas d'extraction)
# =====================================================

def search_youtube(query, number=15, shorts_only=False):
    # Cibler les Shorts : pas de filtre natif "shorts" dans ytsearch, mais
    # ajouter "#shorts" à la requête biaise fortement les résultats vers des
    # clips verticaux courts -- combiné au filtre de durée dur plus bas,
    # c'est suffisant en pratique.
    q = f"{query} #shorts" if shorts_only else query
    print(f"Recherche YouTube ({'Shorts' if shorts_only else 'vidéos'}) : {query}")
    result = subprocess.run(
        [
            "yt-dlp", "--flat-playlist", "--print",
            "%(id)s|||%(title)s|||%(channel)s|||%(duration)s",
            f"ytsearch{number}:{q}",
        ],
        capture_output=True, text=True,
    )
    entries = []
    for line in result.stdout.splitlines():
        parts = line.strip().split("|||", 3)
        if len(parts) == 4:
            entries.append({
                "id": parts[0], "title": parts[1],
                "channel": parts[2], "duration": parts[3] or "NA",
            })
    return entries


# Mots-clés à exclure des titres -- pré-filtre grossier, ne remplace PAS la
# revue manuelle mais réduit le nombre de lignes à lire.
TITLE_EXCLUDE_KEYWORDS = [
    "interview", "reaction", "review", "unboxing", "compilation",
    "vs", "top 10", "top10", "funny", "fail", "fails", "meme",
]


def looks_off_topic(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in TITLE_EXCLUDE_KEYWORDS)


def generate_candidates(keywords, per_keyword, min_duration, max_duration,
                         shorts_only=False):
    seen_ids = set()
    candidates = []
    skipped_not_short = 0
    for kw in keywords:
        for entry in search_youtube(kw, per_keyword, shorts_only=shorts_only):
            if entry["id"] in seen_ids:
                continue
            seen_ids.add(entry["id"])

            try:
                dur = float(entry["duration"])
            except (TypeError, ValueError):
                dur = None

            # En mode Shorts, la durée est un vrai filtre (exclusion), pas un
            # simple repère -- une vidéo de 10 min qui remonte sur "#shorts"
            # n'est de toute façon pas un Short, autant l'écarter tout de suite.
            if shorts_only:
                if dur is None or dur > max_duration:
                    skipped_not_short += 1
                    continue

            flagged_off_topic = looks_off_topic(entry["title"])
            flagged_duration = (
                dur is not None and (dur < min_duration or dur > max_duration)
            )
            candidates.append({**entry, "duration_s": dur,
                                "flag_offtopic": flagged_off_topic,
                                "flag_duration": flagged_duration})

    if shorts_only and skipped_not_short:
        print(f"{skipped_not_short} résultat(s) écarté(s) car pas au format Short "
              f"(> {int(max_duration)}s)")
    return candidates


def write_candidates_file(candidates, path):
    n_flagged = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Candidats trouvés automatiquement -- A REVOIR AVANT EXTRACTION.\n")
        f.write("# Supprime les lignes hors-sujet, puis relance avec --urls-file "
                f"{os.path.basename(path)}\n")
        f.write("# Lignes marquées [A VERIFIER] : pré-filtre suspecte un hors-sujet "
                "(titre) ou une durée atypique -- pas une décision automatique, juste un repère.\n\n")
        for c in candidates:
            url = f"https://www.youtube.com/watch?v={c['id']}"
            flag = ""
            if c["flag_offtopic"] or c["flag_duration"]:
                flag = " [A VERIFIER]"
                n_flagged += 1
            dur_str = f"{int(c['duration_s'])}s" if c["duration_s"] is not None else "durée inconnue"
            f.write(f"{url}  # {flag}{dur_str} | {c['channel']} | {c['title']}\n")
    print(f"\n{len(candidates)} candidats écrits dans {path}")
    print(f"{n_flagged} ligne(s) marquée(s) [A VERIFIER] par le pré-filtre (titre/durée suspects)")
    print("-> Ouvre ce fichier, supprime les lignes hors-sujet, "
          "puis relance avec --urls-file " + os.path.basename(path))


# =====================================================
# Metadata vidéo (pour le mode extraction, à partir d'une URL déjà revue)
# =====================================================

def get_metadata(url):
    result = subprocess.run(
        ["yt-dlp", "--skip-download", "--print",
         "%(id)s|||%(title)s|||%(channel)s|||%(duration)s", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("yt-dlp erreur metadata:",
              result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "inconnue")
        return None
    data = result.stdout.strip().split("|||", 3)
    if len(data) != 4:
        return None
    return {"id": data[0], "title": data[1], "channel": data[2], "duration": data[3] or "NA"}


# =====================================================
# Téléchargement + extraction
# =====================================================

def download_video(url, video_id, folder):
    output = os.path.join(folder, video_id + ".%(ext)s")
    subprocess.run(
        ["yt-dlp", "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
         "--merge-output-format", "mp4", "-o", output, url],
        check=True,
    )
    return os.path.join(folder, video_id + ".mp4")


def extract_frames(video, output, video_id, fps):
    os.makedirs(output, exist_ok=True)
    pattern = os.path.join(output, video_id + "_%06d.jpg")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-vf", f"fps={fps}", "-qscale:v", "2", pattern],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("ffmpeg erreur:",
              result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "inconnue")
    return len(os.listdir(output))


# =====================================================
# Programme principal
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls-file")
    parser.add_argument("--search-youtube", action="store_true",
                         help="Génère un fichier de candidats à revoir -- NE TÉLÉCHARGE RIEN.")
    parser.add_argument("--candidates-out", default="candidates.txt")
    parser.add_argument("--keywords", nargs="+",
                         default=["horse galloping close up", "horse trotting",
                                  "horse cantering", "horse walking slow motion",
                                  "horse rearing", "horse bucking", "horse grazing",
                                  "horse standing still", "horse running in field",
                                  "horse jumping obstacle", "mustang running wild",
                                  "horse in slow motion", "solo horse no rider",
                                  "horse turning around", "horse stretching"])
    parser.add_argument("--per-keyword", type=int, default=15)
    parser.add_argument("--output-dir", default="./raw/youtube_frames")
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--shorts-only", action="store_true",
                         help="Cible les YouTube Shorts (vertical, courts) au lieu des vidéos "
                              "classiques. Change les défauts de durée (voir --max-duration).")
    parser.add_argument("--max-duration", type=float, default=None,
                         help="Durée max en secondes pour la recherche/l'extraction. "
                              "Défaut : 180s en mode --shorts-only, 1200s sinon.")
    parser.add_argument("--min-duration", type=float, default=5,
                         help="Durée min en secondes -- écarte les clips quasi vides.")
    args = parser.parse_args()

    if args.max_duration is None:
        args.max_duration = 180 if args.shorts_only else 1200

    check_dependencies()

    # --- Mode recherche : produit un fichier à revoir, s'arrête là ---
    if args.search_youtube:
        candidates = generate_candidates(
            args.keywords, args.per_keyword, args.min_duration, args.max_duration,
            shorts_only=args.shorts_only,
        )
        write_candidates_file(candidates, args.candidates_out)
        return

    # --- Mode extraction : nécessite un fichier d'URLs déjà revu ---
    if not args.urls_file:
        print("Donner --urls-file (après revue) ou --search-youtube (pour générer des candidats)")
        return

    urls = read_urls(args.urls_file)
    if args.limit_videos:
        urls = urls[: args.limit_videos]

    frames_root = os.path.join(args.output_dir, "frames")
    video_tmp = os.path.join(args.output_dir, "_videos")
    os.makedirs(video_tmp, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "provenance.csv")
    already_done = read_processed_ids(csv_path)
    if already_done:
        print(f"{len(already_done)} vidéo(s) déjà traitée(s), seront ignorées")

    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if os.stat(csv_path).st_size == 0:
        writer.writerow(["video_id", "url", "title", "channel", "duration", "frames", "date"])

    total = 0
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}]")
        meta = get_metadata(url)
        if meta is None:
            print("Impossible metadata")
            continue

        vid = meta["id"]
        if vid in already_done:
            print(f"{vid} déjà traité, skip")
            continue

        if args.max_duration:
            try:
                duration_s = float(meta["duration"])
            except (TypeError, ValueError):
                duration_s = None
            if duration_s is not None and duration_s > args.max_duration:
                print(f"{vid} trop long ({int(duration_s)}s > {int(args.max_duration)}s), skip")
                continue

        video = None
        try:
            video = download_video(url, vid, video_tmp)
            output = os.path.join(frames_root, vid)
            n = extract_frames(video, output, vid, args.fps)
            total += n

            writer.writerow([vid, url, meta["title"], meta["channel"], meta["duration"],
                              n, datetime.now(timezone.utc).isoformat()])
            csv_file.flush()
            already_done.add(vid)
            print(f"{n} frames extraites")

        except Exception as e:
            print("Erreur:", e)

        finally:
            if video and os.path.exists(video):
                os.remove(video)

    csv_file.close()
    print("\n================")
    print("Terminé")
    print("Total frames:", total)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
import os
import shutil
from ultralytics import YOLO


INPUT_DIR = "./raw/youtube_frames/frames"
OUTPUT_DIR = "./raw/youtube_frames/filtered_frames"


model = YOLO("yolov8m.pt")

HORSE_CLASS_ID = 17


kept = 0
deleted = 0


for video_folder in os.listdir(INPUT_DIR):

    video_path = os.path.join(INPUT_DIR, video_folder)

    if not os.path.isdir(video_path):
        continue


    output_video_folder = os.path.join(
        OUTPUT_DIR,
        video_folder
    )

    os.makedirs(
        output_video_folder,
        exist_ok=True
    )


    for image in os.listdir(video_path):

        if not image.endswith(".jpg"):
            continue


        img_path = os.path.join(
            video_path,
            image
        )


        results = model(
            img_path,
            verbose=False
        )


        horse_found = False


        for r in results:

            for cls in r.boxes.cls:

                if int(cls) == HORSE_CLASS_ID:

                    horse_found = True
                    break



        if horse_found:

            shutil.copy(
                img_path,
                os.path.join(
                    output_video_folder,
                    image
                )
            )

            kept += 1


        else:

            deleted += 1



print("================")
print("Images gardées :", kept)
print("Images supprimées :", deleted)
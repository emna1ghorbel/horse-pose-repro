"""
Script de vérification : charge une image de horse/ + son masque de horseSeg/,
les affiche côte à côte pour valider que tout correspond.
"""
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# --- Paramètres à ajuster ---
RAW_DIR = r"E:\horse-pose-repro\data\raw\behaviorDiscovery2.0"
SHOT_ID = 100          # une des vidéos qu'on a vues dans ranges.mat
FRAME_INDEX = 0        # 0 = première frame de ce shot

# --- 1. Charger ranges.mat pour trouver les frames du shot ---
ranges_data = scipy.io.loadmat(f"{RAW_DIR}\\ranges\\horse\\ranges.mat")
ranges = ranges_data["ranges"]  # (99, 3) : [shot_id, first_frame, last_frame]

row = ranges[ranges[:, 0] == SHOT_ID][0]
shot_id, first_frame, last_frame = row
print(f"Shot {shot_id} : frames {first_frame} à {last_frame} ({last_frame - first_frame + 1} frames)")

# --- 2. Charger l'image correspondante ---
frame_number = first_frame + FRAME_INDEX
image_path = f"{RAW_DIR}\\horse\\{frame_number:08d}.jpg"
image = Image.open(image_path)
print(f"Image chargée : {image_path}, taille {image.size}")

# --- 3. Charger le masque de segmentation ---
# --- 3. Charger le masque de segmentation ---
seg_data = scipy.io.loadmat(
    f"{RAW_DIR}\\horseSeg\\seg{SHOT_ID}.mat",
    squeeze_me=True, struct_as_record=False
)
segments = seg_data["segments"]
print("Nombre de frames dans segments :", len(segments))

mask = segments[FRAME_INDEX]
print("Forme du masque :", mask.shape, "type :", mask.dtype)

# --- 4. Affichage côte à côte ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)
axes[0].set_title(f"Image (frame {frame_number})")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Masque de segmentation")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("check_output.png")
print("Résultat sauvegardé dans check_output.png")
plt.show()
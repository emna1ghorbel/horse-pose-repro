
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

data_dir = Path("data/raw/horse_synthetic/horse_combineds5r5_texture")
base_name = "1000_SK_horse_skeleton_horse_idleB_anim_0.65_450.00_160.00_150.00.png"

img = Image.open(data_dir / f"{base_name}_img.png")
kpts = np.load(data_dir / f"{base_name}_kpts.npy")

# Les 18 indices des vraies articulations anatomiques
joint_idxs = np.array([1718,1684,1271,1634,1650,1643,1659,925,392,564,993,726,1585,1556,427,1548,967,877])
joints = kpts[joint_idxs]

print(f"Nombre de points denses : {kpts.shape[0]}")
print(f"Nombre de joints anatomiques extraits : {joints.shape[0]}")

plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.scatter(joints[:, 0], joints[:, 1], c='lime', s=80, edgecolors='black', linewidths=1.5)
for i, (x, y, v) in enumerate(joints):
    plt.annotate(str(i), (x, y), color='yellow', fontsize=9, fontweight='bold')
plt.title(f"{joints.shape[0]} joints anatomiques extraits")
plt.savefig("kpts_visualization_18joints.png", dpi=150)
print("Image sauvegardee : kpts_visualization_18joints.png")

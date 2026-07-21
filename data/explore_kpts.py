import numpy as np
from pathlib import Path

# Prends le premier fichier .npy trouvé
kpts_dir = Path("data/raw/horse_synthetic/horse_combineds5r5_texture")
npy_files = list(kpts_dir.glob("*_kpts.npy"))

print(f"Nombre total de fichiers _kpts.npy trouvés : {len(npy_files)}")

# Charge et inspecte le premier
sample = np.load(npy_files[0])
print(f"\nFichier exemple : {npy_files[0].name}")
print(f"Shape : {sample.shape}")
print(f"Dtype : {sample.dtype}")
print(f"Contenu :\n{sample}")
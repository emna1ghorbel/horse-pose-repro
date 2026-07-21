import torch

print("Version PyTorch :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU détecté :", torch.cuda.get_device_name(0))
    vram_go = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    print("VRAM totale (Go) :", vram_go)
else:
    print("Aucun GPU détecté — entraînement sur CPU")
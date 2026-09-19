import torch

def diversity_loss(poses_2d: torch.Tensor) -> torch.Tensor:
    """
    Pénalise le mode collapse en s'assurant que les poses 2D générées varient 
    d'une image à l'autre.
    
    Au lieu de forcer les pixels bruts du squelette à être orthogonaux (ce qui 
    empêche de dessiner un cheval au centre de l'image), on maximise l'écart-type 
    des coordonnées des points clés prédits sur le batch.
    
    poses_2d : (batch_size, K, 2) dans [-1, 1]
    Retourne l'opposé de l'écart-type moyen (à minimiser).
    """
    batch_size = poses_2d.shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=poses_2d.device)
    
    # Écart-type des coordonnées sur la dimension du batch : shape (K, 2)
    std_per_keypoint = poses_2d.std(dim=0)
    
    # On veut maximiser cette diversité, donc on renvoie son opposé
    return -std_per_keypoint.mean()

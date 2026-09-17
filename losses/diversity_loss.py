import torch

def diversity_loss(skeletons: torch.Tensor) -> torch.Tensor:
    """
    Pénalise la similarité (mode collapse) entre les squelettes d'un même batch.
    Calcule la similarité cosinus moyenne entre toutes les paires distinctes du batch.
    À minimiser par le générateur pour forcer des sorties variées.
    """
    batch_size = skeletons.shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=skeletons.device)
    
    flat = skeletons.view(batch_size, -1)
    flat_norm = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
    sim_matrix = torch.mm(flat_norm, flat_norm.t())
    
    mask = torch.eye(batch_size, device=skeletons.device).bool()
    sim_off_diag = sim_matrix[~mask]
    
    return sim_off_diag.mean()

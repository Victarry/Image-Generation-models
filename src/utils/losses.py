import torch
import torch.nn.functional as F

def adversarial_loss(pred, target_is_real=True, loss_mode="vanilla"):
    if loss_mode == "vanilla":
        if target_is_real:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)
        return F.binary_cross_entropy_with_logits(pred, target)
    elif loss_mode == "lsgan":
        if target_is_real:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)
        return F.mse_loss(pred, target)
    else:
        raise NotImplementedError

def normal_kld(mu, log_sigma):
    kl_divergence = -0.5 * torch.sum(1 + 2 * log_sigma - mu ** 2 - torch.exp(2 * log_sigma), dim=-1).mean(dim=0)
    return kl_divergence

def symmetry_contra_loss(feat1, feat2, temperature=0.07):
    logits = torch.einsum("ik,jk->ij", feat1, feat2) / temperature # (d, d)
    d = logits.shape[0]

    labels = torch.arange(d).to(feat1.device) # (d)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)
    contra_loss = (loss_i + loss_j) / 2
    return contra_loss
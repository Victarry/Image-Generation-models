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
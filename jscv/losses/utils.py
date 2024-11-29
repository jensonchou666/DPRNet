import torch.nn.functional as F

def loss_map(pred, target, ignore_index=None, epsilon=0.05,
             pred_logits=False, dim=1):
    if pred_logits:
        pred = F.log_softmax(pred, dim=dim)
    if target.dim() == pred.dim() - 1:
        target = target.unsqueeze(dim)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -pred.gather(dim=dim, index=target)
        smooth_loss = -pred.sum(dim=dim, keepdim=True)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -pred.gather(dim=dim, index=target)
        smooth_loss = -pred.sum(dim=dim, keepdim=True)

    nll_loss = nll_loss.squeeze(dim)
    smooth_loss = smooth_loss.squeeze(dim)
    eps_i = epsilon / pred.size(dim)
    _loss_map = nll_loss * (1.0 - epsilon) + smooth_loss * eps_i
    return _loss_map




def sum_losses(losses):
    if isinstance(losses, dict):
        s = 0
        for loss in losses.values():
            s += loss
        return s
    else:
        return losses

def losses_weighted(losses, weight):
    if isinstance(losses, dict):
        d2 = {}
        for key, loss in losses.items():
            d2[key] = loss * weight
    else:
        return losses * weight

def losses_add_suffix(losses, suffix):
    d2 = {}
    if isinstance(losses, dict):
        for key, loss in losses.items():
            d2[f"{key}{suffix}"] = loss
    else:
        d2[f"loss{suffix}"] = losses
    return d2

def losses_add_suffix_detached(losses, suffix):
    d2 = {}
    if isinstance(losses, dict):
        for key, loss in losses.items():
            d2[f"{key}{suffix}"] = loss.detach()
    else:
        d2[f"loss{suffix}"] = losses.detach()
    return d2

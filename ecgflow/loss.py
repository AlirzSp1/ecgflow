"""Loss related functions.
"""
import torch


def smooth_binary_target(x: torch.Tensor, target: torch.Tensor,
                         smoothing: float) -> torch.Tensor:
    # From timm/loss/binary_cross_entropy.py
    # Here we assume `target` has same shape as `x` (batch_size, 2)
    batch_size = target.shape[0]
    num_classes = 2
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    target = target.long()[:,1]
    target = torch.full(
        (batch_size, num_classes),
        off_value,
        device=x.device, dtype=x.dtype).scatter_(1, target, on_value)
    return target


class MSLQLoss(torch.nn.Module):
    """Mean squared log(Q) loss.

    This provides a relative error in terms of Q=predicted/actual.
    The target should be strictly positive.
    (cf. Tofallis 2015)
    """
    def __init__(self, denom_min=1e-3):
        super().__init__()
        self.dmin = torch.tensor([denom_min])

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        assert torch.all(target > 0)
        dmin = self.dmin.to(x.device)
        Q = torch.maximum((x / torch.maximum(target, dmin)), dmin)
        return torch.mean((torch.log(Q))**2).to(x.device)
                 

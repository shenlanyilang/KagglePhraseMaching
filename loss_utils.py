from torch.nn import MSELoss
from torch import nn
from torch import Tensor
import torch
from torch.nn import functional as F


class ShrinkageLoss(nn.Module):
    def __init__(self, a=10, c=0.1) -> None:
        super(ShrinkageLoss, self).__init__()
        self.a = a
        self.c = c
    
    def forward(self, input:Tensor, target:Tensor, reduction='mean'):
        """_summary_
        def shrinkage_loss(l, a=0.1, c=0.1):
            return l**2 / (1 + np.exp(a*(c-l)))

        Args:
            input (Tensor): _description_
            target (Tensor): _description_
        """
        input_flatten = input.flatten()
        target_flatten = target.flatten()
        mse = F.mse_loss(input_flatten, target_flatten, reduction='none')
        abs_diff = torch.abs(target_flatten - input_flatten)
        loss = mse / (1. + torch.exp(self.a * (self.c - abs_diff)))
        return torch.mean(loss)

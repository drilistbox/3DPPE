import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import LOSSES


@LOSSES.register_module()
class ScaleInvariantLoss(nn.Module):
    def __init__(self, lamda=0.85, alpha=10.0, loss_weight=1.0, ):
        super(ScaleInvariantLoss, self).__init__()
        self.lamda = lamda
        self.alpha = alpha
        self.loss_weight = loss_weight

    def forward(self, pred, target, avg_factor=None):
        """
        Args:
            pred: (N, )
            target: (N, )
            avg_factor: N

        Returns:

        """
        g = torch.log(pred) - torch.log(target)
        Dg = torch.var(g) + (1 - self.lamda) * torch.pow(torch.mean(g), 2)
        loss = self.loss_weight * self.alpha * torch.sqrt(Dg)
        return loss

import torch
from torch import nn
from . import AIITLoss



class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        bce = self.bceloss(pred_, target_.float())
        return bce


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class AIITBceDiceLoss(AIITLoss):
    def __init__(self, dice_weight=1.0, BCE_weight=1.0):
        super().__init__(name="BceDiceLoss", num_classes=1, weight=None)
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = BCE_weight
        self.wd = dice_weight

    def forward(self, pred, target):
        pred = nn.functional.sigmoid(pred)
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss

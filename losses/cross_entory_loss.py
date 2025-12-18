
from torch import nn, Tensor
from losses import AIITLoss
from typing import Optional


class AIITCrossEntropyLoss(AIITLoss):

    def __init__(self, num_classes, loss_weight: float=1.0, cls_weight: Optional[Tensor] = None) -> None:
        super().__init__(name="entropy loss", num_classes=num_classes, weight=loss_weight)
        self.cls_weight = cls_weight
        self.ce = nn.CrossEntropyLoss(weight=cls_weight) if self.num_classes > 1 else nn.BCEWithLogitsLoss()

    def forward(self, x_pred, y_truth):
        return self.ce(x_pred, y_truth.long()) * self.weight


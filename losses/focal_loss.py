from monai.losses.focal_loss import FocalLoss
from typing import Optional
from losses import AIITLoss
from torch import Tensor

class AIITFocalLoss(AIITLoss):
    def __init__(
        self,
        num_classes,
        weight: float = 1.0,
        gamma: float = 2.0,
        alpha: float = None,
        class_weights: Optional[Tensor] = None,
    ) -> None:
        super().__init__("focal loss", num_classes=num_classes, weight=weight)
        if num_classes == 1:
            use_softmax, to_onehot_y = False, False
        else:
            use_softmax, to_onehot_y = True, True
        self.fl = FocalLoss(
            include_background=True,
            to_onehot_y=to_onehot_y,
            gamma=gamma,
            alpha=alpha,
            weight=class_weights,
            use_softmax=use_softmax
        )


    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input tensor shape:
            input: [b, c, h, w]; target: [b, 1, h, w]
        output tensor shape: [0]
        """
        return self.fl(input, target[:].long()) * self.weight
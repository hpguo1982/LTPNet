import torch
from typing import Optional
from torch import Tensor
from . import AIITLoss



def one_hot(input: Tensor, num_classes: int) -> Tensor:
    """
    input tensor shape: [b, h, w]
    output tensor shape: [b, num_classes, h, w]
    """
    tensors = []
    for i in range(num_classes):
        # noinspection PyUnresolvedReferences
        tensors.append((input == i).unsqueeze(1))
    input = torch.cat(tensors, dim=1)
    return input.float()

def binary_dice_loss(input: Tensor, target: Tensor) -> Tensor:
    """
    input tensor shape:
        input: [b, h, w]; target: [b, h, w]
    output tensor shape: [0]
    """
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(input * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(input * input)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return 1 - loss

def multiclass_dice_loss(input: Tensor, target: Tensor, weight: Optional[Tensor] = None, softmax: bool = True) -> Tensor:
    """
    input tensor shape:
        input: [b, c, h, w]; target: [b, h, w]; weights: [c]
    output tensor shape: [0]
    """
    num_classes = input.shape[1]
    if softmax:
        input = torch.softmax(input, dim=1)
    target = one_hot(target, num_classes)
    if weight is None:
        weight = [1.] * num_classes
    assert input.size() == target.size(), \
        "predict {} & target {} shape do not match".format(input.size(), target.size())

    loss = 0.0
    for i in range(0, num_classes):
        dice = binary_dice_loss(input[:, i], target[:, i])
        loss += dice * weight[i]
    # noinspection PyTypeChecker
    return loss / num_classes

class AIITDiceLoss(AIITLoss):

    def __init__(self, num_classes, loss_weight: float=1.0, cls_weight: Optional[Tensor] = None) -> None:
        """
        input tensor shape:
            weights: [c]
        """
        super().__init__(name="dice loss",
                         num_classes=num_classes,
                         weight=loss_weight)
        self.dice = lambda x, y: multiclass_dice_loss(x, y, softmax=True, weight=cls_weight)\
            if num_classes > 1 else\
            lambda x, y: binary_dice_loss(torch.sigmoid(x), y)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input tensor shape:
            input: [b, c, h, w]; target: [b, h, w]
        output tensor shape: [0]
        """
        return self.dice(input, target) * self.weight


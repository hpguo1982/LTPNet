import torch
import numpy as np
from evals import AIITMetric

def dc(result: torch.Tensor, reference: torch.Tensor) -> float:
    result = torch.atleast_1d(result.type(torch.bool))
    reference = torch.atleast_1d(reference.type(torch.bool))

    intersection = torch.count_nonzero(result & reference)

    size_i1 = torch.count_nonzero(result)
    size_i2 = torch.count_nonzero(reference)

    try:
        dc = (2. * intersection / float(size_i1 + size_i2)).item()
    except ZeroDivisionError:
        dc = 0.0

    return dc

def calc_dice_gpu(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    input tensor shape:
        pred: [[d,] h, w]; gt: [[d,] h, w]
    """
    if pred.sum() > 0 and gt.sum() > 0:
        return dc(pred, gt)
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1.
    return 0.


class AIITDice(AIITMetric):

    def __init__(self, num_classes, name: str = None, ignore_index: int = 0):
        super().__init__(num_classes,
                         name=name if name is not None else "dice",
                         ignore_index=ignore_index)
        self.dices = np.zeros(self.num_classes)
        self.batch = 0

    def forward(self, pred, label, threshold=0.5):

        dices = np.zeros(self.num_classes)

        if self.num_classes == 1:
            # ----- 二值分割 -----
            probs = torch.sigmoid(pred)  # [N,1,H,W]
            p = torch.asarray(probs.squeeze(1) > threshold,  dtype=torch.int) # [N,H,W]
            t = torch.asarray(label,  dtype=torch.int) # [N,H,W]
            dices[0] = calc_dice_gpu(p, t)
        else:
            # ----- 多类分割 -----
            preds = torch.argmax(pred, dim=1)  # [N,H,W]

            for c in range(self.num_classes):
                if self.ignore_index is not None and c == self.ignore_index:
                    continue

                p = torch.asarray(preds == c, dtype=torch.int)
                t = torch.asarray(label == c, dtype=torch.int)

                dices[c] = calc_dice_gpu(p, t)

        self.dices += dices
        self.batch += 1

        if self.num_classes > 1:
            r_ = np.r_[0: self.ignore_index, self.ignore_index + 1:self.num_classes]
            dices = dices[r_]
        return dices.mean()

    def zero_metric(self):
        self.dices = np.zeros(self.num_classes)
        self.batch = 0

    @property
    def metric_value(self):
        if self.num_classes == 1:
             dices = np.copy(self.dices)
        else:
            r_ = np.r_[0: self.ignore_index, self.ignore_index + 1:self.num_classes]
            dices =  self.dices[r_]
        dices /= self.batch

        return {self._name:(dices.mean(), dices)}


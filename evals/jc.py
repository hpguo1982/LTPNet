import torch
import numpy as np
from medpy import metric
from . import AIITMetric


def Jc(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    input ndarray shape:
        pred: [depth, height, width]; gt: [depth, height, width]
    output float: jaccard
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        jaccard = metric.binary.jc(pred, gt)
        return jaccard
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1
    else:
        return 0

class AIITJc(AIITMetric):
    def __init__(self, num_classes, name: str = None, ignore_index: int = 0):
        super().__init__(num_classes, name=name if name is not None else "jaccard",
                         ignore_index=ignore_index)
        self._jc = np.zeros(num_classes)
        self._batch = 0


    def forward(self, pred, label, threshold=0.5):

        _jc = np.zeros(self.num_classes)
        if self.num_classes == 1:
            # ----- 二值分割 -----
            p = torch.sigmoid(pred)  # [N,1,H,W]
            p = p.squeeze(1) > threshold # [N,H,W]
            t = label.long()
            _jc[0] = Jc(p.detach().cpu().numpy(), t.detach().cpu().numpy())
        else:
            # ----- 多类分割 -----
            preds = torch.argmax(pred, dim=1)  # [N,H,W]
            for c in range(self.num_classes):
                if self.ignore_index is not None and c == self.ignore_index:
                    continue

                p = preds == c# [N,H,W]
                t = label == c # [N,H,W]

                _jc[c] = Jc(p.detach().cpu().numpy().astype(np.uint8), t.detach().cpu().numpy().astype(np.uint8))


        self._jc += _jc
        self._batch += 1

        if self.num_classes > 1:
            _jc = _jc[np.r_[0: self.ignore_index, self.ignore_index + 1:self.num_classes]]

        return {self._name:(_jc.mean())}

    def zero_metric(self):
        self._jc = np.zeros(self.num_classes)
        self._batch = 0

    @property
    def metric_value(self):
        if self.num_classes == 1:
            _jc = self._jc
        else:
            _jc = self._jc[np.r_[0: self.ignore_index, self.ignore_index + 1:self.num_classes]]

        _jc = _jc/self._batch

        return {self._name:(_jc.mean(), _jc)}

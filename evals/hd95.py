import torch
import numpy as np
from medpy import metric
from evals import AIITMetric


def HD95(pred: np.array, gt: np.array):
    """
    input ndarray shape:
        pred: [depth, height, width]; gt: [depth, height, width]
    output float: (dice, hd95, jaccard, asd)
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 0

class AIITHD95(AIITMetric):

    def __init__(self, num_classes, name: str = None, ignore_index: int = 0):
        super().__init__(num_classes, name=name if name is not None else "HD95", ignore_index=ignore_index)
        self._hd95 = np.zeros(num_classes)
        self._batch = 0

    def forward(self, pred, label, threshold=0.5):



        _hd95 = np.zeros(self.num_classes)
        if self.num_classes == 1:
            # ----- 二值分割 -----
            probs = torch.sigmoid(pred)  # [N,1,H,W]
            preds = probs.squeeze(1) > threshold  # [N,H,W]
            tgts = label.long()  # [N,H,W]
            _hd95[0] = HD95(preds.detach().cpu().numpy(), tgts.detach().cpu().numpy())
        else:
            # ----- 多类分割 -----
            preds = torch.argmax(pred, dim=1)  # [N,H,W]
            for c in range(self.num_classes):
                if self.ignore_index is not None and c == self.ignore_index:
                    continue

                p = (preds == c) # [N,H,W]
                t = (label == c) # [N,H,W]

                _hd95[c] = HD95(
                    p.detach().cpu().numpy().astype(np.uint8),
                    t.detach().cpu().numpy().astype(np.uint8))

        self._hd95 += _hd95
        self._batch += 1

        if self.num_classes > 1:
            _hd95 = _hd95[np.r_[0: self.ignore_index, self.ignore_index + 1:self.num_classes]]
        return {self._name: (_hd95.mean(), _hd95)}

    def zero_metric(self):
        self._hd95 = np.zeros(self.num_classes)
        self._batch = 0

    @property
    def metric_value(self):
        _hd95 = self._hd95 if self.num_classes == 1 \
            else self._hd95[np.r_[0: self.ignore_index, self.ignore_index + 1:self.num_classes]]
        _hd95 /= self._batch
        return {self._name: (_hd95.mean(), _hd95)}


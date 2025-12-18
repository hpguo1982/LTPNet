import torch
import numpy as np
from medpy import metric
from . import AIITMetric


def asd(pred, gt):
    """
    Average surface distance metric.

    Computes the average surface distance (ASD) between the binary objects in two images.

    size: [N, H, W] - N-pair images with size of [H, W]
    """
    if pred.sum() > 0 and gt.sum() > 0:
        return np.mean([metric.binary.asd(pred, gt), metric.binary.asd(gt, pred)])
    else:
        return 0


class AIITAsd(AIITMetric):

    def __init__(self, num_classes, name: str = None, ignore_index: int = 0):
        super().__init__(num_classes, name=name if name is not None else "asd",
                         ignore_index=ignore_index)
        self._asd = np.zeros(num_classes)
        self._batch = 0

    def forward(self, pred, label, threshold=0.5):

        _asd = np.zeros(self.num_classes)
        if self.num_classes == 1:
            # ----- 二值分割 -----
            probs = torch.sigmoid(pred)  # [N,1,H,W]
            preds = (probs.squeeze(1) > threshold).detach().cpu().numpy() # [N,H,W]
            tgts = label.long().detach().cpu().numpy()  # [N,H,W]

            _asd[0] += asd(preds, tgts)
        else:
            # ----- 多类分割 -----
            preds = torch.argmax(pred, dim=1)  # [N,H,W]
            for c in range(self.num_classes):
                if self.ignore_index is not None and c == self.ignore_index:
                    continue

                preds_c = (preds == c).detach().cpu().numpy()# [N,H,W]
                tgts_c = (label == c).detach().cpu().numpy() # [N,H,W]
                _asd[c] += asd(preds_c, tgts_c)

        self._asd += _asd
        self._batch += 1

        if self.num_classes > 1:
            _asd = _asd[np.r_[0: self.ignore_index, self.ignore_index + 1:self.num_classes]]

        return {"asd": (_asd.mean())}

    def zero_metric(self):
        self._asd = np.zeros(self.num_classes)
        self._batch = 0

    @property
    def metric_value(self):
        _asd = self._asd if self.num_classes == 1 else\
            self._asd[np.r_[0: self.ignore_index, self.ignore_index + 1:self.num_classes]]
        _asd = self._asd/self._batch
        return {self._name: (_asd.mean(), _asd)}




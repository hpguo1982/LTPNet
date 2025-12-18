import torch
import numpy as np
from evals import AIITMetric

def confusion_matrix(pred, target, num_classes=2):
    """
    pred: torch.Tensor (N, ...) 预测 (可以是 logits/prob，需先阈值化或 argmax)
    target: torch.Tensor (N, ...) 标签
    num_classes: 类别数
    返回: confusion matrix
    """
    pred = pred.view(-1).to(torch.long)
    target = target.view(-1).to(torch.long)
    k = (target >= 0) & (target < num_classes)  # 只考虑合法标签
    inds = target[k] * num_classes + pred[k]   # 组合索引

    cm = torch.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)

    # cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=pred.device)
    # # 累计混淆矩阵
    # for t, p in zip(target, pred):
    #     cm[t.long(), p.long()] += 1
    return cm


class AIITGeneralMetrics(AIITMetric):
    """
    including overall_accuracy, accuracy_per_class,
        precision_per_class, recall_per_class,
        iou_per_class, dice_per_class,
        mean_iou, mean_dice
    """
    def __init__(self, num_classes, name: str = None, ignore_index: int = 0):
        super().__init__(num_classes, name=name if name is not None else "general metrics", ignore_index=ignore_index)

        # 初始化混淆矩阵
        self._cm = np.zeros((num_classes, num_classes), dtype=np.int64)


    def forward(self, pred, label, threshold=0.5):

        if self.num_classes == 1:
            # ----- 二值分割 -----
            probs = torch.sigmoid(pred)  # [N,1,H,W]
            preds = probs.squeeze(1)  # [N,H,W]
            y_pre = np.where(preds >= threshold, 1, 0)
            target = np.where(label >= 0.5, 1, 0)
            num_classes = 2
        else:
            # ----- 多类分割 -----
            y_pre = torch.argmax(pred, dim=1)  # [N,H,W]
            target = label.long()
            num_classes = self.num_classes
        _cm = confusion_matrix(y_pre, target, num_classes).detach().cpu().numpy()
        self._cm += _cm

        metrics = self._cal_metrics(_cm)
        return metrics


    def _cal_metrics(self, cm):


        metrics = {
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "IoU": [],
            "Dice": [],
        }

        num_classes = 2 if self.num_classes == 1 \
            else self.num_classes

        for c in range(num_classes):
            TP = float(cm[c, c])
            FP = float(cm[:, c].sum()) - TP
            FN = float(cm[c, :].sum()) - TP
            TN = float(cm.sum()) - (TP + FP + FN)

            acc = (TP + TN) / cm.sum() if cm.sum() > 0 else 0.0
            precision = TP / (TP + FP) if TP + FP > 0 else 0.0
            recall = TP / (TP + FN)  if TP + FN > 0 else 0.0
            iou = TP / (TP + FP + FN)  if TP + FP + FN > 0 else 0.0
            dice = (2 * TP) / (2 * TP + FP + FN)  if 2 * TP + FP + FN > 0 else 0.0

            if c != self.ignore_index:
                metrics["Accuracy"].append(acc)
                metrics["Precision"].append(precision)
                metrics["Recall"].append(recall)
                metrics["IoU"].append(iou)
                metrics["Dice"].append(dice)

        for key, value in metrics.items():
            metrics[key] = (np.mean(value), value)

        return metrics


    def zero_metric(self):
        if self.num_classes <= 1:
            self._hist = np.zeros((2, 2), dtype=np.int64)
        else:
            self._hist = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @property
    def metric_value(self):
        return self._cal_metrics(self._cm)









from torch import nn
from registers import AIITModel


class AIITLoss(nn.Module, AIITModel):

    def __init__(self, name, num_classes, weight:float=1.0):
        super().__init__()
        self._name = name
        self._num_classes = num_classes
        self._weight = weight

    @property
    def loss_name(self):
        return self._name

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def weight(self):
        return self._weight

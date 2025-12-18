from torch import nn
from registers import AIITModel


class AIITMetric(nn.Module, AIITModel):

    def __init__(self, num_classes, name = "metric", ignore_index: int=0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._name = name

    def forword(self, pred, label):
        pass

    def zero_metric(self):
        pass

    @property
    def metric_name(self)->str:
        return self._name

    @property
    def metric_value(self)->dict[str, tuple[float,list[float]]]:
        return None

    @property
    def description(self):
        metric_value = self.metric_value
        string = []
        for key, value in metric_value.items():
            tmp_key, tmp_value = key, value[0]
            sum_ = f"{tmp_key:>10}:\t{tmp_value:.4f}\t"
            # sub_keys = []
            # sub_values = []
            # for i in range(1, self.num_classes):
            #     sub_keys.append("class" + str(i))
            #     sub_values.append(f"{value[1][i - 1]:.4f}")
            sub_str = []
            for i in range(1, self.num_classes):
                sub_str.append(f"c{i}:{value[1][i - 1]:.4f}")
            sub_str = " ".join(sub_str)

            string.append(sum_ + sub_str)

        return "\n".join(string)





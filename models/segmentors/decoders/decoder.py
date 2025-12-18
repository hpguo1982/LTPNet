from registers import AIITModel
from torch.nn import Module

class AIITDecoder(Module, AIITModel):

    def __init__(self):
        super().__init__()

    def forword(self, x):
        pass
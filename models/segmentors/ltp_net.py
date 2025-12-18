import torch

from models.segmentors.encoder_decoder import AIITEncoderDecoder
from losses import AIITDiceLoss



class AIITLTPNet(AIITEncoderDecoder):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)







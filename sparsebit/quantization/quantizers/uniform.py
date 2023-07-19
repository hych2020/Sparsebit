import torch
from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from sparsebit.quantization.common import Granularity, QuantTarget
from .quant_tensor import STE


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "uniform"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)

    def _forward(self, x_f, scale, zero_point):
        if self.qdesc.target == QuantTarget.WEIGHT and self.qdesc.groupsize != -1:
            oc = x_f.shape[0]
            ic = x_f.shape[1]
            if ic % self.qdesc.groupsize != 0:
                groupsize = ic
            else:
                groupsize = self.qdesc.groupsize
            groups = ic // groupsize
            groups = ic
            groupsize = 1
            assert len(x_f.shape) in [2, 4]
            if len(x_f.shape) == 2:
                x_f = x_f.reshape(oc*groups, groupsize)
            else:
                x_f = x_f.reshape(oc*groups, groupsize, x_f.shape[2], x_f.shape[3])
        x_dq = STE.apply(x_f, scale, zero_point, self.qdesc, self.backend)
        if self.qdesc.target == QuantTarget.WEIGHT and self.qdesc.groupsize != -1:
            if len(x_f.shape) == 2:
                x_dq = x_dq.reshape(oc, ic)
            else:
                x_dq = x_dq.reshape(oc, ic, x_f.shape[2], x_f.shape[3])
        return x_dq

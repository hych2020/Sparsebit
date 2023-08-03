import torch
from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from .quant_tensor import STE


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "pwlq"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)
        if self.qdesc.bit < 8:
            self.scale = None
            self.zero_point = None
            self.register_buffer(
                "pos_scale", torch.tensor([1.0], dtype=torch.float).to(self.device)
            )
            self.register_buffer(
                "pos_zero_point", torch.tensor([0.0], dtype=torch.float).to(self.device)
            )
            self.register_buffer(
                "neg_scale", torch.tensor([1.0], dtype=torch.float).to(self.device)
            )
            self.register_buffer(
                "neg_zero_point", torch.tensor([0.0], dtype=torch.float).to(self.device)
            )
            assert self.cfg.OBSERVER.TYPE in ["MSE"], "pwlq quantizer only support mse observer when bit<8!"

    def calc_qparams(self):
        if self.fake_fused:
            if self.qdesc.bit < 8:
                return self.pos_scale, self.pos_zero_point, self.neg_scale, self.neg_zero_point
            return self.scale, self.zero_point
        
        if self.qdesc.bit < 8:
            pos_scale, pos_zero_point = self.observer.calc_qparams(filter_mode="pos")
            self.pos_scale = self._broadcast_qparams(pos_scale)
            self.pos_zero_point = self._broadcast_qparams(pos_zero_point)
            neg_scale, neg_zero_point = self.observer.calc_qparams(filter_mode="neg")
            self.neg_scale = self._broadcast_qparams(neg_scale)
            self.neg_zero_point = self._broadcast_qparams(neg_zero_point)
            self.observer.data_cache.reset()
            return self.pos_scale, self.pos_zero_point, self.neg_scale, self.neg_zero_point
        else:
            scale, zero_point = self.observer.calc_qparams()
            self.scale = self._broadcast_qparams(scale)
            self.zero_point = self._broadcast_qparams(zero_point)
            return self.scale, self.zero_point

    def _forward(self, x_f, scale, zero_point):
        x_dq = STE.apply(x_f, scale, zero_point, self.qdesc, self.backend)
        return x_dq

    def forward(self, x):
        if self.is_enable:
            if self.qdesc.bit < 8:
                x_dq_pos = self._forward(x, self.pos_scale, self.pos_zero_point)
                x_dq_neg = self._forward(x, self.neg_scale, self.neg_zero_point)
                x_dq = x_dq_pos + x_dq_neg
            else:
                x_dq = self._forward(x, self.scale, self.zero_point)
        else:
            x_dq = x
        return x_dq
    
    def __repr__(self):
        info = "{}, {}, observer={},".format(self.TYPE, self.qdesc, self.observer.TYPE)
        if self.qdesc.scheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            if self.qdesc.bit<8:
                info += " pos_scale={:.4f}, pos_zp={:.4f}, neg_scale={:.4f}, neg_zp={:.4f}".format(
                    self.pos_scale.item(), self.pos_zero_point.item(), self.neg_scale.item(), self.neg_zero_point.item()
                )
            else:
                info += " scale={:.4f}, zp={:.4f}".format(
                    self.scale.item(), self.zero_point.item()
                )
        elif self.qdesc.scheme in [
            torch.per_channel_affine,
            torch.per_channel_symmetric,
            "per-group-symmetric",
            "per-group-affine",
        ]:
            if self.qdesc.bit<8:
                info += " pos_scale=[{:.4f}, {:.4f}], pos_zp=[{}, {}], neg_scale=[{:.4f}, {:.4f}], neg_zp=[{}, {}]".format(
                    self.pos_scale.min(),
                    self.pos_scale.max(),
                    self.pos_zero_point.min(),
                    self.pos_zero_point.max(),
                    self.neg_scale.min(),
                    self.neg_scale.max(),
                    self.neg_zero_point.min(),
                    self.neg_zero_point.max(),
                )
            else:
                info += " scale=[{:.4f}, {:.4f}], zp=[{}, {}]".format(
                    self.scale.min(),
                    self.scale.max(),
                    self.zero_point.min(),
                    self.zero_point.max(),
                )
        else:
            raise NotImplementedError
        return info

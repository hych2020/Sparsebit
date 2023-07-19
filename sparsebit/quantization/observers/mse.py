import torch
import math
from .utils import mse_loss
from sparsebit.quantization.observers import Observer as BaseObserver
from sparsebit.quantization.observers import register_observer
from sparsebit.quantization.quantizers.quant_tensor import STE
from sparsebit.quantization.common import Backend, Granularity

def quantize(data, scales, zeros, qmax):
    qdata = torch.clamp(torch.round(data / scales) + zeros, 0, qmax)
    return qdata

def decompose_group_scales(scales, groups, pack_bit=4):
    if groups < 4:
        return scales
    # scales is [oc*groups]
    print("Run groups scales decompose")
    scales = scales.reshape(-1, groups)
    assert torch.all(scales > 0), "all quant-scales must greater than zero"
    scales_max, scales_min = (
        scales.max(axis=1, keepdim=True)[0],
        scales.min(axis=1, keepdim=True)[0],
    )
    step_qmax = 2 ** pack_bit - 1
    step = scales - scales_min  # oc, gorups
    step_scales = (scales_max - scales_min) / step_qmax  # oc, 1
    step_scales = torch.maximum(step_scales, torch.tensor(1e-6))
    q_step = quantize(step, step_scales, 0, step_qmax)
    #from IPython import embed;embed();exit(1)
    dq_scales = step_scales * q_step + scales_min
    return dq_scales.reshape(-1)

@register_observer
class Observer(BaseObserver):
    TYPE = "mse"

    def __init__(self, config, qdesc):
        super(Observer, self).__init__(config, qdesc)
        self.alpha = config.OBSERVER.PERCENTILE.ALPHA

    def calc_minmax(self, data_c_first):
        if self.is_perchannel:
            max_val = data_c_first.max(axis=1).values
            min_val = data_c_first.min(axis=1).values
        else:
            min_val, max_val = data_c_first.min(), data_c_first.max()
        self.min_val = min_val.to(self.device)
        self.max_val = max_val.to(self.device)
        return self.min_val, self.max_val

    def calc_qparams(self):
        data_c_first = self.data_cache.get_data_for_calibration(Granularity.CHANNELWISE)
        self.data_cache.reset()
        if self.data_cache._groups is not None:
            assert self.is_perchannel
            assert self.groupsize > 0
            oc, ic = self.data_cache._oc_ic
            data_c_first = data_c_first.reshape(oc, ic, -1)
            kernel_abs_max = data_c_first.max(axis=-1)[0]   #(oc, ic)
            perm = torch.argsort(kernel_abs_max, descending=True)
            data_ls = []
            for i in range(oc):
                data_ls.append(data_c_first[i][perm[i]].unsqueeze(0))
            data_c_first = torch.cat(data_ls, dim=0)
            invperm = torch.argsort(perm)
            data_c_first = data_c_first.reshape(oc*self.data_cache._groups, -1)
            #from IPython import embed;embed();exit(1)
        # if self.groupsize != -1:
        #     assert self.is_perchannel
        #     channels = data_c_first.shape[0]
        #     if channels % self.groupsize != 0:
        #         print("warning: channels is not an integer multiple of groupsize, set groupsize as number of channels")
        #         groupsize = channels
        #     else:
        #         groupsize = self.groupsize
        #     groups = channels // groupsize
        #     data_c_first = data_c_first.reshape(groups, -1)
        min_val, max_val = self.calc_minmax(data_c_first)
        x_f = data_c_first.to(self.device)
        # old_ch_axis = self.qdesc.ch_axis
        # if old_ch_axis !=0:
        #     print("warning: self.qdesc.ch_axis!=0, This will lead to wrong calibration results, temporarily set it to 0")
        #     self.qdesc._ch_axis = 0
        if self.is_perchannel:
            best_scale = torch.tensor(
                [1.0 for _ in range(data_c_first.shape[0])], device=self.device
            )
            best_zero_point = torch.tensor(
                [0.0 for _ in range(data_c_first.shape[0])], device=self.device
            )
            loss_min = torch.tensor(
                [1e10 for _ in range(data_c_first.shape[0])], device=self.device
            )
        else:
            best_scale, best_zero_point = None, None
            loss_min = 1e10
        for i in range(80):
            cur_min_val = min_val * (1.0 - (i * 0.01))
            cur_max_val = max_val * (1.0 - (i * 0.01))
            scale, zero_point = self.calc_qparams_with_minmax(cur_min_val, cur_max_val)
            x_dq = STE.apply(x_f, scale, zero_point, self.qdesc, Backend.VIRTUAL)
            if self.is_perchannel:
                loss = mse_loss(x_f, x_dq, is_perchannel=True)
                best_scale[loss < loss_min] = scale[loss < loss_min]
                best_zero_point[loss < loss_min] = zero_point[loss < loss_min]
                loss_min[loss < loss_min] = loss[loss < loss_min]
            else:
                loss = mse_loss(x_f, x_dq, is_perchannel=False)
                if loss < loss_min:
                    loss_min = loss
                    best_scale = scale
                    best_zero_point = zero_point
        assert len(self.data_cache) == 0, "free data cache after calc_qparams"
        # if self.groupsize != -1:
        #     best_scale = best_scale.reshape(-1, 1).repeat(1, groupsize).reshape(-1)
        #     best_zero_point = best_zero_point.reshape(-1, 1).repeat(1, groupsize).reshape(-1)
        # self.qdesc._ch_axis = old_ch_axis
        if self.data_cache._groups is not None:
            groupsize = self.data_cache._groupsize
            best_scale = decompose_group_scales(best_scale, self.data_cache._groups)
            best_scale = best_scale.reshape(-1, 1).repeat(1, groupsize).reshape(-1)
            best_zero_point = best_zero_point.reshape(-1, 1).repeat(1, groupsize).reshape(-1)
            index_per_kernel = []
            for i in range(len(invperm)):
                index_per_kernel.extend([x+i*ic for x in invperm[i]])
            best_scale = best_scale[index_per_kernel]
            best_zero_point = best_zero_point[index_per_kernel]
            #from IPython import embed;embed();#exit(1)
        return best_scale, best_zero_point

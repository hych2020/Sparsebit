import torch

def quantize(data, scales, zeros, qmax):
    qdata = torch.clamp(torch.round(data / scales) + zeros, 0, qmax)
    return qdata


def decompose_group_scales(scales, pack_bit=4):
    # scales is [oc, groups]
    print("Run groups scales decompose")
    assert torch.all(scales > 0), "all quant-scales must greater than zero"
    scales_max, scales_min = (
        scales.max(axis=1, keepdim=True)[0],
        scales.min(axis=1, keepdim=True)[0],
    )
    step_qmax = 2 ** pack_bit - 1
    step = scales - scales_min  # oc, gorups
    step_scales = (scales_max - scales_min) / step_qmax  # oc, 1
    q_step = quantize(step, step_scales, 0, step_qmax)
    from IPython import embed;embed();exit(1)

scales = torch.rand(3, 4)
decompose_group_scales(scales)


from IPython import embed;embed();exit(1)
kernel_abs_max = data.reshape(oc, ic, -1).abs().max(axis=-1)[0]
perm = torch.argsort(kernel_abs_max, descending=True)
data_ls = []
for i in range(oc):
    data_ls.append(data[i][perm[i]].unsqueeze(0))
data = torch.cat(data_ls, dim=0)
self.invperm = torch.argsort(perm)


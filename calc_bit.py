import torch
import torch.nn as nn

import torchvision.models as models

model = models.resnet18(pretrained=True)
model.cuda()

def get_input_shape(module, x):
    x, = x
    module.input_shape = x.shape


hooks = []
def register_fake_quant_input_hook(model):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_pre_hook(get_input_shape))

register_fake_quant_input_hook(model)
inp = torch.rand((1,3,224,224)).cuda()
with torch.no_grad():
    _ = model(inp)

for x in hooks:
    x.remove()


for gs in [1,4,8,16,32,65536]:
    w_nums=0
    f_nums=0
    w_bits_total=0
    f_bits_total=0
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            #print(n)
            b, c, h, w = m.input_shape
            oc, ic, k, k = m.weight.shape
            assert c == ic

            ksize= oc * ic * k * k
            fsize= c * h * w
            w_nums += ksize
            f_nums += fsize
            """
            if ic % gs==0:
                ks_nums = oc * (ic // gs)
                fs_nums = ic // gs
            else:
                ks_nums = oc
                fs_nums = 1
            
            if k==1 and gs==1:
                ks_nums /= 2

            w_bits_total += ksize * 4 + ks_nums * (32 + 4)
            f_bits_total += fsize * 8 + fs_nums * (32 + 8)
            """
            if ic % gs==0:
                groups = ic // gs
                if groups < 4:
                    ks_nums = oc * groups * (32 + 4)
                else:
                    ks_nums = oc * 2 * 32 + oc * groups * 4 * 2
                #ks_nums = oc * groups * (32 + 4)
                fs_nums = groups * (32 + 8)
            else:
                ks_nums = oc * (32 + 4)
                fs_nums = 1 * (32 + 8)
            w_bits_total += ksize * 4 + ks_nums
            f_bits_total += fsize * 8 + fs_nums

    print("gs: ", gs)
    print("equal w bit: ", w_bits_total/w_nums)
    print("equal f bit: ", f_bits_total/f_nums)


import torch
import torch.nn as nn


class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups):
        super(CBL, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=1, padding=pad,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1),
        )

        ## hook相关代码
        self.mid_fea = {}
        for index_i, (name, module) in enumerate(self.named_modules()):
            if index_i == 1:        # conv在模型中的序号是1
                module.register_forward_hook(hook=self.layer_hook)
                # 必须在前向推理之前声明hook
                break

    def layer_hook(self, module, fea_in, fea_out):
        self.mid_fea[fea_in[0].device].append(fea_out)

    def forward(self, x):
        self.mid_fea[x.device] = []
        out = self.conv(x)
        return out, self.mid_fea[x.device][0]		# 返回模型输出以及中间层特征


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CBL(8, 16, 3, 1).to(device)
    model = nn.DataParallel(model)          # 使用多张gpu

    x = torch.ones(2, 8, 10, 10)
    out, mid_fea = model(x)

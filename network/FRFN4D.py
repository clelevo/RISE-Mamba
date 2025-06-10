import torch
import torch.nn as nn
from einops import rearrange, repeat
import math

class FRFN(nn.Module):
    def __init__(self, dim=128, hidden_dim=1024, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2), act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        # 处理视频数据，形状为 BFCHW
        batch_size, frames, channels, height, width = x.shape

        # 将视频数据展平成 (batch_size * frames) C H W 的形式
        x = x.reshape(batch_size * frames, channels, height, width)

        # 原始代码中的处理逻辑
        hh = x.shape[2]
        ww = x.shape[3]

        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        x = rearrange(x, 'b c h w -> b (h w) c', h=hh, w=ww)

        x = self.linear1(x)
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, 'b (h w) (c) -> b c h w', h=hh, w=ww)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, 'b c h w -> b (h w) c', h=hh, w=ww)
        x = x_1 * x_2

        x = self.linear2(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=ww)

        # 恢复原始形状，即 BFCHW
        x = x.reshape(batch_size, frames, channels, height, width)

        return x

if __name__ == '__main__':
    block = FRFN()
    input = torch.rand(2, 5, 128, 64, 64)
    output = block(input)
    print(f"Input size: {input.size()}")
    print(f"Output size: {output.size()}")
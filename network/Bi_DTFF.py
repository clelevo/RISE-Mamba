import torch
import time

from torch import nn as nn


class Bi_DTFF(nn.Module):
    def __init__(self, num_feat=64):
        super().__init__()
        self.num_feat = num_feat
        self.conv1 = nn.Conv2d(num_feat * 3, num_feat * 3, 3, 1, 1, bias=True)  # 修改为3倍特征
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.kernel_conv_pixel = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.resblock_bcakward2d = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, feature):
        # predefine
        b, t, c, h, w = feature.size()  # b t 64 256 256
        forward_list = []
        feat_prop = feature.new_zeros(b, c, h, w)
        feat_prev = feature.new_zeros(b, c, h, w)  # 初始化前一帧特征
        for i in range(0, t):
            x_feat = feature[:, i, :, :, :]
            if i > 0:
                feat_prev = feature[:, i - 1, :, :, :]  # 获取前一帧特征
            else:
                feat_prev = feature.new_zeros(b, c, h, w)  # 如果没有前一帧，用零填充
            # fusion propagation
            feat_fusion = torch.cat([x_feat, feat_prop, feat_prev], dim=1)  # b 192 256 256
            feat_fusion = self.lrelu(self.conv1(feat_fusion))  # b 192 256 256
            feat_prop1, feat_prop2, feat_prop3 = torch.split(feat_fusion, self.num_feat, dim=1)
            feat_prop1 = feat_prop1 * torch.sigmoid(self.conv2(feat_prop1))
            feat_prop2 = feat_prop2 * torch.sigmoid(self.conv3(feat_prop2))
            feat_prop3 = feat_prop3 * torch.sigmoid(self.conv3(feat_prop3))
            feat_prop = feat_prop1 + feat_prop2 + feat_prop3
            # dynamic conv
            feat_prop = self.kernel_conv_pixel(feat_prop)
            # resblock2D
            feat_prop = self.resblock_bcakward2d(feat_prop)
            forward_list.append(feat_prop)

        conv3d_feature = torch.stack(forward_list, dim=1)  # b 64 t 256 256
        return conv3d_feature

if __name__ == "__main__":
    model = manual_conv3d_propagation_forward().to("cuda")
    input = torch.randn(1, 5, 64, 256, 256).to("cuda")
    output = model(input)
    print(output.shape)
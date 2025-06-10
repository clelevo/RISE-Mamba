import torch
import torch.nn as nn
# DSTNet
# Deep Discriminative Spatial and Temporal Network for Efficient Video Deblurring

class DTFF(nn.Module):
    def __init__(self, num_feat=64):
        super().__init__()
        self.num_feat = num_feat
        self.conv1 = nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True,groups=num_feat)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True,groups=num_feat)
        self.kernel_conv_pixel = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.resblock_bcakward2d = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)

    def forward(self, feature):
        # predefine
        b, t, c, h, w = feature.size()                           # b t 64 256 256
        backward_list = []
        feat_prop = feature.new_zeros(b, c, h, w)
        # propagation
        for i in range(t - 1, -1, -1):
            x_feat = feature[:, i, :, :, :]
            # fusion propagation
            feat_fusion = torch.cat([x_feat, feat_prop], dim=1)       # b 128 256 256
            feat_fusion = self.lrelu(self.conv1(feat_fusion))   # b 128 256 256
            feat_prop1, feat_prop2 = torch.split(feat_fusion, self.num_feat, dim=1)
            feat_prop1 = feat_prop1 * torch.sigmoid(self.conv2(feat_prop1))
            feat_prop2 = feat_prop2 * torch.sigmoid(self.conv3(feat_prop2))
            feat_prop = feat_prop1 + feat_prop2
            # dynamic conv
            feat_prop = self.kernel_conv_pixel(feat_prop)
            # resblock2D
            feat_prop = self.resblock_bcakward2d(feat_prop)
            backward_list.append(feat_prop)

        backward_list = backward_list[::-1]
        conv3d_feature = torch.stack(backward_list, dim=1)      # b 64 t 256 256
        return conv3d_feature

if __name__ == '__main__':
    from thop import profile
    model = DTFF().to('cuda')
    input = torch.randn(1, 10, 64, 128, 128).to('cuda')
    output = model(input)
    print(output.shape)

    flops, params = profile(model, inputs=(input,))
    print('params:', params / 1e6, 'M')
    print('flops:', flops / 1e9, 'G')
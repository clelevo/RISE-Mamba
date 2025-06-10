import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# from mmcv.runner import load_checkpoint
from mmengine.runner import load_checkpoint
# from mmedit.models.registry import

# from mmedit.utils import get_root_logger
import logging

from convnext import ConvNeXt
from head import ProjectionHead
from mambablock import MambaLayerglobal, MambaLayerlocal
from IDC import InceptionDWConv3d
from DTFF import DTFF



# @BACKBONES.register_module()
class RISE_Mamba(nn.Module):
    """RainMamba network structure.

    Paper:
        RainMamba: Enhanced Locality Learning with State Space Models for Video Deraining

    Args:
        num_features (int, optional): Channel number of the intermediate
            features. Default: 128.

    """

    def __init__(self,
                 num_features=128,
                 feat_pretrained='/media/byc/MyDisk/Pycharm_Projects/RISE_Mamba/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'):

        super().__init__()
        self.num_features = num_features

        self.TRAIN_MODE = True
        self.feat_extract = ConvNeXt(
            in_channels=1,
            arch='tiny',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.0,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            init_cfg=dict(type='Pretrained', checkpoint=feat_pretrained, prefix='backbone.'))

        self.head = ProjectionHead(in_channels=[96, 192, 384, 768],
                                   out_channels=num_features,
                                   num_outs=4
                                   )

        self.backbone = nn.ModuleDict()

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        # downsample
        self.conv1 = nn.Conv3d(num_features, num_features*2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))

        # upsample
        self.upconv2 = nn.ConvTranspose3d(num_features*2, num_features, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                                          output_padding=(0, 1, 1))
        self.conv_before_upsample1 = nn.Conv3d(num_features, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.upsample1 = nn.PixelShuffle(2)
        self.conv_before_upsample2 = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.conv_before_upsample1_M1 = nn.Conv3d(num_features, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.upsample1 = nn.PixelShuffle(2)
        self.conv_before_upsample2_M1 = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.conv_before_upsample1_M2 = nn.Conv3d(num_features, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.upsample1 = nn.PixelShuffle(2)
        self.conv_before_upsample2_M2 = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.upsample2 = nn.PixelShuffle(2)
        self.conv_last = nn.Conv3d(16, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv_last_M1 = nn.Conv3d(16, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv_last_M2 = nn.Conv3d(16, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # 64 * 64 * 128
        self.IDC1 = InceptionDWConv3d(num_features)
        self.GlobalMambaBlock1 = MambaLayerglobal(dim=num_features)
        self.IDC2 = InceptionDWConv3d(num_features)
        self.LocalMambaBlock1 = MambaLayerlocal(dim=num_features)
        self.IDC3 = InceptionDWConv3d(num_features)
        self.GlobalMambaBlock2 = MambaLayerglobal(dim=num_features)
        self.IDC4 = InceptionDWConv3d(num_features)
        self.LocalMambaBlock2 = MambaLayerlocal(dim=num_features)
        self.FRFN1 = DTFF(128)

        # 32 * 32 * 256
        self.IDC5 = InceptionDWConv3d(num_features * 2)
        self.GlobalMambaBlockLowRes1 = MambaLayerglobal(dim=num_features * 2)
        self.IDC6 = InceptionDWConv3d(num_features * 2)
        self.LocalMambaBlockLowRes1 = MambaLayerlocal(dim=num_features * 2)
        self.IDC7 = InceptionDWConv3d(num_features * 2)
        self.GlobalMambaBlockLowRes2 = MambaLayerglobal(dim=num_features * 2)
        self.IDC8 = InceptionDWConv3d(num_features * 2)
        self.LocalMambaBlockLowRes2 = MambaLayerlocal(dim=num_features * 2)
        self.IDC9 = InceptionDWConv3d(num_features * 2)
        self.GlobalMambaBlockLowRes3 = MambaLayerglobal(dim=num_features * 2)
        self.IDC10 = InceptionDWConv3d(num_features * 2)
        self.LocalMambaBlockLowRes3 = MambaLayerlocal(dim=num_features * 2)
        # self.FRFN2 = FRFN(dim=256, hidden_dim=2048)

        # 64 * 64 * 128
        self.IDC11 = InceptionDWConv3d(num_features)
        self.GlobalMambaBlock3 = MambaLayerglobal(dim=num_features)
        self.IDC12 = InceptionDWConv3d(num_features)
        self.LocalMambaBlock3 = MambaLayerlocal(dim=num_features)
        self.IDC13 = InceptionDWConv3d(num_features)
        self.GlobalMambaBlock4 = MambaLayerglobal(dim=num_features)
        self.IDC14 = InceptionDWConv3d(num_features)
        self.LocalMambaBlock4 = MambaLayerlocal(dim=num_features)
        # self.FRFN3 = FRFN(dim=128, hidden_dim=1024)

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def forward(self, lqs, hilbert_curve_large_scale='/media/byc/MyDisk/Pycharm_Projects/RISE_Mamba/hilbert_curve/5_64.pt',
                hilbert_curve_small_scale='/media/byc/MyDisk/Pycharm_Projects/RISE_Mamba/hilbert_curve/5_32.pt'):
        """Forward function for RainMamba.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lqs.size()

        self.check_if_mirror_extended(lqs)

        feats_ = self.feat_extract(lqs.view(-1, c, h, w))
        down1, down2, down3, down4 = self.head(feats_)
        # print('in line 427', down1.shape, down2.shape, down3.shape, down4.shape, lqs.shape)
        down2_up = F.interpolate(down2, size=down1.size()[2:], mode='bilinear', align_corners=True)
        down3_up = F.interpolate(down3, size=down1.size()[2:], mode='bilinear', align_corners=True)
        down4_up = F.interpolate(down4, size=down1.size()[2:], mode='bilinear', align_corners=True)
        f = (down1 + down2_up + down3_up + down4_up) / 4
        f = self.refine(f) + f
        f_ori = f

        x_new = f_ori.view(n, t, self.num_features, int(h / 4), int(w / 4))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hilbert_curve_large_scale = torch.load(hilbert_curve_large_scale).to(device)
        hilbert_curve_small_scale = torch.load(hilbert_curve_small_scale).to(device)

        # 64 * 64 * 128
        M1 = self.IDC1(x_new)
        M1 = self.GlobalMambaBlock1(M1)
        M1 = self.IDC2(M1)
        M1 = self.LocalMambaBlock1(M1, hilbert_curve_large_scale)
        M1 = self.IDC3(M1)
        M1 = self.GlobalMambaBlock2(M1)
        M1 = self.IDC4(M1)
        M1 = self.LocalMambaBlock2(M1, hilbert_curve_large_scale)
        M_mid = self.FRFN1(M1)

        if self.TRAIN_MODE:
            x_re_1 = rearrange(M1, 'n d c h w ->  n c d h w')
            M1_Mid_Feature = self.up_1(x_re_1) # 2, 16, 5, 256, 256
            M1_Mid_Feature = self.conv_last_M1(M1_Mid_Feature).transpose(1, 2)

        # 32 * 32 * 256
        x_down = rearrange(M1, 'n d c h w ->  n c d h w')
        x_down = F.relu(self.conv1(x_down))
        x_down = rearrange(x_down, 'n c d h w ->  n d c h w')

        x_down = self.IDC5(x_down)
        M2 = self.GlobalMambaBlockLowRes1(x_down)
        M2 = self.IDC6(M2)
        M2 = self.LocalMambaBlockLowRes1(M2, hilbert_curve_small_scale)
        M2 = self.IDC7(M2)
        M2 = self.GlobalMambaBlockLowRes2(M2)
        M2 = self.IDC8(M2)
        M2 = self.LocalMambaBlockLowRes2(M2, hilbert_curve_small_scale)
        M2 = self.IDC9(M2)
        M2 = self.GlobalMambaBlockLowRes3(M2)
        M2 = self.IDC10(M2)
        M2 = self.LocalMambaBlockLowRes3(M2, hilbert_curve_small_scale)
        # M2 = self.FRFN2(M2)

        # x_re_2 = rearrange(M2, 'n d c h w ->  n c d h w')
        # M2_Mid_Feature = self.up_2(x_re_2) # 2, 16, 5, 256, 256
        # M2_Mid_Feature = self.conv_last_M2(M2_Mid_Feature).transpose(1, 2)

        x_up = rearrange(M2, 'n d c h w ->  n c d h w')
        x_up = F.relu(self.upconv2(x_up))
        x_up = rearrange(x_up, 'n c d h w ->  n d c h w')

        if self.TRAIN_MODE:
            x_re_2 = rearrange(x_up, 'n d c h w ->  n c d h w')
            M2_Mid_Feature = self.up_2(x_re_2) # 2, 16, 5, 256, 256
            M2_Mid_Feature = self.conv_last_M2(M2_Mid_Feature).transpose(1, 2)

        # 64 * 64 * 128
        M3 = self.IDC11(x_up+M_mid)
        M3 = self.GlobalMambaBlock3(M3)
        M3 = self.IDC12(M3)
        M3 = self.LocalMambaBlock3(M3, hilbert_curve_large_scale)
        M3 = self.IDC13(M3)
        M3 = self.GlobalMambaBlock4(M3)
        M3 = self.IDC14(M3)
        M3 = self.LocalMambaBlock4(M3, hilbert_curve_large_scale) # 2, 5, 128, 64, 64
        # M3 = self.FRFN3(M3)

        x_re = rearrange(M3, 'n d c h w ->  n c d h w') # 2, 128, 5, 64, 64

        x_re = self.up(x_re)  # 2, 16, 5, 256, 256

        final = self.conv_last(x_re).transpose(1, 2) # 2, 5, 1, 256, 256

        if self.TRAIN_MODE:
            return final, M1, x_up, M3, M1_Mid_Feature, M2_Mid_Feature
        else:
            return final


    def up(self, x):
        x = self.conv_before_upsample1(x)
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample1(x)
        x = rearrange(x, 'n d c h w -> n c d h w')
        x = self.conv_before_upsample2(x)
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample2(x)
        x = rearrange(x, 'n d c h w -> n c d h w')
        return x

    def up_1(self, x):
        x = self.conv_before_upsample1_M1(x)
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample1(x)
        x = rearrange(x, 'n d c h w -> n c d h w')
        x = self.conv_before_upsample2_M1(x)
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample2(x)
        x = rearrange(x, 'n d c h w -> n c d h w')
        return x

    def up_2(self, x):
        x = self.conv_before_upsample1_M2(x)
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample1(x)
        x = rearrange(x, 'n d c h w -> n c d h w')
        x = self.conv_before_upsample2_M2(x)
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample2(x)
        x = rearrange(x, 'n d c h w -> n c d h w')
        return x

    def init_weights(self, pretrained=None, strict=False):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """

        if isinstance(pretrained, str):
            # logger = get_root_logger()
            logger = logging.getLogger()
            logger.info(f"Init weights: {pretrained}")
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif self.feat_extract.init_cfg is not None:
            self.feat_extract.init_weights()

        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
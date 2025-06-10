import torch
from Hilbert3d import Hilbert3d

def hilbert_curve_large_scale( ):
    nf = 15
    H = 64
    W = 64

    hilbert_curve = list(
        Hilbert3d(width=H, height=W, depth=nf))
    hilbert_curve = torch.tensor(hilbert_curve).long()
    hilbert_curve = hilbert_curve[:, 0] * W * nf + hilbert_curve[:, 1] * nf + hilbert_curve[:, 2]

    return {
        'hilbert_curve_large_scale': hilbert_curve
    }


def hilbert_curve_small_scale( ):
    nf = 15
    H = 32
    W = 32

    hilbert_curve = list(
        Hilbert3d(width=H, height=W, depth=nf))
    hilbert_curve = torch.tensor(hilbert_curve).long()
    hilbert_curve = hilbert_curve[:, 0] * W * nf + hilbert_curve[:, 1] * nf + hilbert_curve[:, 2]

    return {
        'hilbert_curve_small_scale': hilbert_curve
    }


def save_hilbert_curve_large_scale( hilbert_curve_large_scale, filename='/media/byc/MyDisk/Pycharm_Projects/RISE_Mamba/hilbert_curve/15_64_64.pt'):
    torch.save(hilbert_curve_large_scale, filename)


def save_hilbert_curve_small_scale( hilbert_curve_small_scale, filename='/media/byc/MyDisk/Pycharm_Projects/RISE_Mamba/hilbert_curve/15_32_32.pt'):
    torch.save(hilbert_curve_small_scale, filename)

result_large_scale = hilbert_curve_large_scale()
hilbert_curve_large_scale = result_large_scale['hilbert_curve_large_scale'].to('cuda')
result_small_scale = hilbert_curve_small_scale()
hilbert_curve_small_scale = result_small_scale['hilbert_curve_small_scale'].to('cuda')

save_hilbert_curve_large_scale(hilbert_curve_large_scale)
save_hilbert_curve_small_scale(hilbert_curve_small_scale)

def hilbert_curve_large_scale_channel( ):
    nf = 256
    H = 32
    W = 40

    hilbert_curve = list(
        Hilbert3d(width=H, height=W, depth=nf))
    hilbert_curve = torch.tensor(hilbert_curve).long()
    hilbert_curve = hilbert_curve[:, 0] * W * nf + hilbert_curve[:, 1] * nf + hilbert_curve[:, 2]

    return {
        'hilbert_curve_large_scale_channel': hilbert_curve
    }

def hilbert_curve_small_scale_channel( ):
    nf = 128
    H = 64
    W = 80

    hilbert_curve = list(
        Hilbert3d(width=H, height=W, depth=nf))
    hilbert_curve = torch.tensor(hilbert_curve).long()
    hilbert_curve = hilbert_curve[:, 0] * W * nf + hilbert_curve[:, 1] * nf + hilbert_curve[:, 2]

    return {
        'hilbert_curve_small_scale_channel': hilbert_curve
    }

def save_hilbert_curve_large_scale_channel( hilbert_curve_large_scale_channel, filename='/media/byc/MyDisk/Pycharm_Projects/RISE_Mamba/hilbert_curve/32_40_256.pt'):
    torch.save(hilbert_curve_large_scale_channel, filename)


def save_hilbert_curve_small_scale_channel( hilbert_curve_small_scale_channel, filename='/media/byc/MyDisk/Pycharm_Projects/RISE_Mamba/hilbert_curve/64_80_128.pt'):
    torch.save(hilbert_curve_small_scale_channel, filename)

result_large_scale_channel = hilbert_curve_large_scale_channel()
hilbert_curve_large_scale_channel = result_large_scale_channel['hilbert_curve_large_scale_channel'].to('cuda')
result_small_scale_channel = hilbert_curve_small_scale_channel()
hilbert_curve_small_scale_channel = result_small_scale_channel['hilbert_curve_small_scale_channel'].to('cuda')

# save_hilbert_curve_large_scale_channel(hilbert_curve_large_scale_channel)
# save_hilbert_curve_small_scale_channel(hilbert_curve_small_scale_channel)
import torch
import torch.fft



def frequency_mask(height, width, cutoff=0.3, gamma=1.5, device="cuda"):
    """
    自适应频域掩膜生成器
    对应论文公式(1)-(3)
    """
    # 生成径向基坐标（公式1）
    y_coords = torch.linspace(-1, 1, height, device=device)
    x_coords = torch.linspace(-1, 1, width, device=device)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
    R = torch.sqrt(X ** 2 + Y ** 2)

    # 计算动态截止阈值（公式2）
    D_c = cutoff * (2 ** 0.5)  # 最大归一化半径

    # 生成平滑过渡掩膜（公式3）
    mask = torch.exp(-gamma * (R - D_c) ** 2)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask.unsqueeze(0)  # 增加通道维度


def STL_loss(denoised, target):


    params = {
        'cutoff': 0.3,
        'gamma': 1.5,
        'alpha': 0.8,
        'beta': 0.3,
        'gamma_t': 0.2
    }

    N, T, C, H, W = denoised.shape
    device = denoised.device


    alpha = torch.nn.Parameter(torch.tensor(params['alpha']))
    beta = torch.nn.Parameter(torch.tensor(params['beta']))


    mask = frequency_mask(H, W, params['cutoff'], params['gamma'], device)


    F_den = torch.fft.fftn(denoised, dim=(-2, -1))
    F_gt = torch.fft.fftn(target, dim=(-2, -1))
    amp_loss = torch.mean(mask * (torch.abs(F_den) - torch.abs(F_gt)) ** 2)


    phase_den = torch.angle(F_den)
    phase_gt = torch.angle(F_gt)
    grad_den = torch.stack(torch.gradient(phase_den, dim=(-2, -1)), -1)
    grad_gt = torch.stack(torch.gradient(phase_gt, dim=(-2, -1)), -1)
    phase_loss = torch.norm(grad_den - grad_gt, p=2, dim=-1).mean()


    temporal_diff = denoised[:, 1:] - denoised[:, :-1]
    F_temp = torch.fft.fftn(temporal_diff, dim=(-2, -1))
    temporal_loss = torch.mean(torch.abs(F_temp) ** 2)


    total_loss = alpha * amp_loss + beta * phase_loss + params['gamma'] * temporal_loss

    return total_loss

params = {
    'cutoff': 0.3,
    'gamma': 1.5,
    'alpha': 0.8,
    'beta': 0.3,
    'gamma_t': 0.2
}


denoised = torch.randn(4, 5, 1, 256, 256, device='cuda')
target = torch.randn(4, 5, 1, 256, 256, device='cuda')
total = STL_loss(denoised, target)
print("Loss:", total)
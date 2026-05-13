import torch
from torch import nn


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5):
    """
    PyTorch implementation of multi-kernel RBF Gaussian kernel
    Args:
        source: source domain data (Tensor [n_s, m])
        target: target domain data (Tensor [n_t, m])
        kernel_mul: kernel scale multiplier (float)
        kernel_num: number of kernels (int)
    Returns:
        kernels: combined kernel matrix (Tensor [n, n])
    """
    n_s = source.size(0)
    n_t = target.size(0)
    n = n_s + n_t
    total = torch.cat([source, target], dim=0)

    # Compute L2 distance matrix (n x n)
    total_0 = total.unsqueeze(0).expand(n, n, -1)
    total_1 = total.unsqueeze(1).expand(n, n, -1)
    L2_distance = ((total_0 - total_1) ** 2).sum(dim=2)

    # Compute base bandwidth (length scale)
    bandwidth = torch.sum(L2_distance.data) / (n ** 2 - n + 1e-8)
    bandwidth /= kernel_mul ** (kernel_num // 2)

    # Create multiple bandwidths
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # Compute multi-kernel matrix
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val)  # Sum over all kernels


def MK_MMD(source, target, kernel_mul=2.0, kernel_num=5):
    """
    PyTorch implementation of multi-kernel MMD
    Args:
        source: source domain data (Tensor [n_s, m])
        target: target domain data (Tensor [n_t, m])
        kernel_mul: kernel scale multiplier (float)
        kernel_num: number of kernels (int)
    Returns:
        mmd_loss: MMD distance (scalar Tensor)
    """
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num)
    n_s = source.size(0)
    n_t = target.size(0)

    # Extract sub-matrices
    XX = kernels[:n_s, :n_s]
    YY = kernels[n_s:, n_s:]
    XY = kernels[:n_s, n_s:]
    YX = kernels[n_s:, :n_s]

    # Compute MMD components
    XX_sum = torch.sum(XX) / (n_s ** 2)
    YY_sum = torch.sum(YY) / (n_t ** 2)
    XY_sum = torch.sum(XY) / (n_s * n_t)
    YX_sum = torch.sum(YX) / (n_s * n_t)

    # Combine and return absolute value
    return torch.abs(XX_sum + YY_sum - XY_sum - YX_sum)

class EnhancedMK_MMD(nn.Module):
    def __init__(self, kernels_mul=2.0, kernels_num=5, fix_sigma=None):
        super().__init__()
        self.kernels_mul = kernels_mul
        self.kernels_num = kernels_num
        self.fix_sigma = fix_sigma

    def forward(self, source, target):
        batch_size = source.size(0)
        total_mmd = 0

        # 多尺度MMD
        for scale in [1.0, 2.0, 4.0]:
            source_scaled = source * scale
            target_scaled = target * scale
            total_mmd += MK_MMD(source_scaled, target_scaled)

        # 特征对齐损失 - 二阶矩匹配
        if source.size(0) > 1 and target.size(0) > 1:
            source_cov = torch.mm(source.t(), source) / (source.size(0) - 1)
            target_cov = torch.mm(target.t(), target) / (target.size(0) - 1)
            coral_loss = torch.norm(source_cov - target_cov, p='fro') ** 2
            total_mmd += 0.1 * coral_loss / (4 * source.size(1) ** 2)

        return total_mmd / 3.0  # 平均多尺度MMD
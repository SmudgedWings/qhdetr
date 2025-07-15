import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm
import os
from thop import profile

def plot_distribution(output, title='output', save_path=None,xlim=(-5, 5), ylim=(0, 1.3)):
    # output shape: [B, N, C]
    data = output.detach().cpu().numpy().flatten()
    
    # 直方图
    plt.figure(figsize=(6, 4))
    count, bins, _ = plt.hist(data, bins=200, density=True, alpha=0.6, color='steelblue', label='output values')

    # 高斯 PDF 拟合
    mu, std = data.mean(), data.std()
    pdf = norm.pdf(bins, mu, std)
    plt.plot(bins, pdf, 'r', label='PDF curve')

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(xlim)
    plt.ylim(ylim)

    ax = plt.gca()
    ax.set_facecolor('#f0f0f0')
    for spine in ax.spines.values():
        spine.set_visible(False)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    plt.close()

def get_model_size(model):
    """model size(MB)"""
    param_num = sum(p.numel() for p in model.parameters())
    param_size_bytes = param_num * 4  # 默认FP32，每个参数4字节
    size_mb = param_size_bytes / (1024 ** 2)
    return size_mb

def get_flops_and_params(model, input_size):
    """FLOPs and parameters, input_size: (N,C,H,W)"""
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops_g = flops / 1e9  # 转成Giga-Operations
    params_m = params / 1e6  # 转成百万参数
    return flops_g, params_m

import torch
import torch.nn as nn
import functools

def calculate_model_flops(model: nn.Module, quant_bits: int = 32) -> float:

    total_flops = 0.0

    for name, module in model.named_modules():
        # Conv2d and Linear
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            input_shape = tuple(module.weight.shape[1:])
            output_shape =module.weight.shape[0]
            macs = output_shape * functools.reduce(lambda x, y: x * y, input_shape)
            if quant_bits < 32:
                macs /= (32 / quant_bits)  # Adjust for quantization
            flops = 2 * macs  # Multiply + Add
            total_flops += flops
            # print(f"[Conv/Linear] {name}: {flops / 1e6:.3f} MFLOPs")

        # Multi-head attention
        elif isinstance(module, nn.MultiheadAttention):
            input_shape = module.in_proj_weight.shape[:2]
            d_head = module.in_proj_weight.shape[0] // module.num_heads
            seq_len= input_shape[0]
            macs = seq_len * module.num_heads * (d_head ** 2)
            if quant_bits < 32:
                macs /= (32 / quant_bits)
            flops = 3 * macs
            total_flops += flops
            # print(f"[Attention] {name}: {flops / 1e6:.3f} MFLOPs")
            
    print(f"\n[Total] Model FLOPs: {total_flops / 1e9:.3f} GFLOPs")
    print(f"\n[Total] Model FLOPs: {total_flops / 1e6:.3f} MFLOPs")
    return total_flops / 1e9

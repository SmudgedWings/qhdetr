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

def plot_references_on_image(references, image, image_size, level, save_path):
    import matplotlib.pyplot as plt
    """
    在输入图像上绘制指定层级的 reference 点并保存到本地。
    
    参数:
    - references (torch.Tensor): 归一化的 reference 坐标, 形状为 [batch_size, num_queries, 2 或 4]。
    - image (np.ndarray): 输入图像数组，形状为 (H, W, 3)。
    - image_size (tuple): 图像的尺寸 (width, height)。
    - level (int): 当前参考点的 decoder 层数。
    - save_path (str): 保存图像的本地路径。
    """
    img_width, img_height = image_size

    if references.dim() == 2:  # 若 references 为二维，添加一个维度
        references = references.unsqueeze(0)

    ref_points = references[..., :2].detach().cpu().numpy()
    x_points = (ref_points[:, :, 0] * img_width).flatten()
    y_points = (ref_points[:, :, 1] * img_height).flatten()

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.scatter(x_points, y_points, color="red", s=10, alpha=0.6, label=f'Decoder Layer {level}')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(f"Reference Points for Decoder Layer {level} on Input Image")
    plt.legend()
    plt.axis('off')
    plt.savefig(f"{save_path}_layer_{level}.png")
    plt.close() 

def save_decoder_layer_output_as_3d_image(output_tensor, file_path="decoder_layer_output_3d_visualization.png"):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # 移除batch维度，得到 (1800, 256) 的矩阵
    output = output_tensor.squeeze(0).detach().cpu().numpy()
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(output.shape[0])  # 1800个tokens位置
    y = np.arange(output.shape[1])  # 256个特征维度
    X, Y = np.meshgrid(x, y)
    Z = output.T  # 转置为 (256, 1800) 以匹配X和Y的维度
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title("3D Visualization of Decoder Layer Output")
    ax.set_xlabel("Token Position (1800)")
    ax.set_ylabel("Feature Dimension (256)")
    ax.set_zlabel("Feature Value")

    plt.savefig(file_path, format="png")
    plt.close()
    print(f"图像已保存到: {file_path}")

def plot_histogram_with_stats(tensor, save_path):
 
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    output_flat = tensor.cpu().numpy().flatten()
    
    # 创建绘图
    plt.figure(figsize=(6, 4))
    sns.histplot(output_flat, bins=500, kde=True, color="skyblue", stat="count")

    plt.xlim((-1, 1))
    
    # 计算均值和标准差
    mean = np.mean(output_flat)
    std_dev = np.std(output_flat)
    
    # 添加均值和标准差到图中
    plt.title("quantizated output")
    plt.xlabel("values")
    plt.ylabel("count")
    plt.text(0.05, max(plt.gca().get_ylim()) * 0.8, f"mean: {mean:.5f}\nstd: {std_dev:.5f}", fontsize=10)
    
    # 去除网格线
    plt.grid(False)
    
    # 保存图像到指定路径
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭图像以释放内存
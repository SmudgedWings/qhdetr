import torch
import torch.nn as nn
import torch.nn.functional as F
from .lsq_plus import *
from ._quan_base_plus import *

class SeparableConv2dLSQ(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True, nbits_w=4):
        super(SeparableConv2dLSQ, self).__init__()
        self.depthwise = Conv2dLSQ(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, groups=in_channels, nbits_w=nbits_w)
        self.pointwise = Conv2dLSQ(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=bias, nbits_w=nbits_w)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class AMP(nn.Module):   # c,h,w -> 1,h,w
    def __init__(self, in_channels=256, q=1, groups=32, tau=1.2):
        super(AMP, self).__init__()
        self.conv1 = SeparableConv2dLSQ(in_channels, in_channels, kernel_size=5, padding=2, bias=False, nbits_w=4)
        self.conv2 = SeparableConv2dLSQ(in_channels, in_channels, kernel_size=3, padding=1, bias=False, nbits_w=4)
        self.conv3 = SeparableConv2dLSQ(in_channels, q, kernel_size=3, padding=1, nbits_w=4)
        self.gn1 = nn.GroupNorm(groups, in_channels)
        self.gn2 = nn.GroupNorm(groups, in_channels)
        self.prelu1 = nn.PReLU(in_channels)
        self.prelu2 = nn.PReLU(in_channels)
        self.tau = tau 

    def forward(self, x):
        x = self.prelu1(self.gn1(self.conv1(x)))
        x = self.prelu2(self.gn2(self.conv2(x)))
        x = self.conv3(x)
    
        b, c, h, w = x.size()
        x = x.flatten(start_dim=2) 
        x = F.softmax(x * self.tau, dim=2)  
        x = x.view(b, c, h, w) 
        return x
    
class WP(nn.Module):
    def __init__(self):
        super(WP, self).__init__()
        
        self.A_act = ActLSQ(nbits_a=4, in_features=49)
        self.F_act = ActLSQ(nbits_a=4, in_features=49) 

    def forward(self, F, A):
        b, c, h, w = F.shape
        q = A.size(1)
        F = F.view(b, c, h * w).permute(0, 2, 1)  
        A = A.view(b, q, h * w)              
        A = self.A_act(A)
        F = self.F_act(F)
        return torch.bmm(A, F)
    
class CR(nn.Module):
    def __init__(self, in_channels=256):
        super(CR, self).__init__()
        self.fc1 = LinearLSQ(in_channels, in_channels, nbits_w=4, bias=True)
        self.fc2 = LinearLSQ(in_channels, in_channels, nbits_w=4, bias=True)
        self.prelu = nn.PReLU(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = x.transpose(1, 2)
        x = self.prelu(x)
        x = x.transpose(1, 2)
        x = self.sigmoid(self.fc2(x))
        return x
    
class SAPM(nn.Module): # [2*300, 256, 7, 7] ->  [2, 300, 256]
    def __init__(self, in_channels=256, q=1, groups=32, tau=1.2):
        super(SAPM, self).__init__()
        self.amp = AMP(in_channels, q, groups, tau)
        self.wp = WP()
        self.cr = CR(in_channels)
        self.F_C_act = ActLSQ(nbits_a=4,in_features=in_channels)
        self.F_P_act = ActLSQ(nbits_a=4,in_features=in_channels)

    def forward(self, F):
        A = self.amp(F)
        F_P = self.wp(F, A)
        F_C = self.cr(F_P)

        F_C = self.F_C_act(F_C)
        F_P = self.F_P_act(F_P)

        # F_P: b x q x c 
        F_O = F_C * F_P
        return F_O
    
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            LinearLSQ(n, k, nbits_w=4, bias=True) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def convert_to_rois(outputs_coord, feature_shape):
    batch_size, q, _ = outputs_coord.shape
    height, width = feature_shape

    # 将中心坐标和宽高转换为左上角和右下角坐标
    center_x = outputs_coord[..., 0] * width
    center_y = outputs_coord[..., 1] * height
    half_w = outputs_coord[..., 2] * width / 2
    half_h = outputs_coord[..., 3] * height / 2

    x1 = center_x - half_w
    y1 = center_y - half_h
    x2 = center_x + half_w
    y2 = center_y + half_h

    # 确保坐标在特征图范围内
    x1 = torch.clamp(x1, 0, width)
    y1 = torch.clamp(y1, 0, height)
    x2 = torch.clamp(x2, 0, width)
    y2 = torch.clamp(y2, 0, height)

    # 拼接坐标到形状为 [batch_size, 5]
    rois = torch.stack([x1, y1, x2, y2], dim=-1)
    batch_idx = torch.arange(batch_size, device=outputs_coord.device).view(-1, 1, 1).expand(batch_size, q, 1)
    rois = torch.cat((batch_idx, rois), dim=-1).view(batch_size * q, 5)

    return rois
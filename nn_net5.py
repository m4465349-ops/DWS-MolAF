import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
class tanhLU(nn.Module):
    def __init__(self, alpha=1.0 ,beta=0.0 ,gama=1.0):
        super(tanhLU, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=True)#requires_grad=True表示在反向传播时计算该参数的梯度
        self.beta = torch.nn.Parameter(torch.tensor(beta), requires_grad=True)
        self.gama = torch.nn.Parameter(torch.tensor(gama), requires_grad=True)

      #输入x是该模块的输入张量
    def forward(self, x):
        return self.alpha*torch.tanh(self.gama*x)+ self.beta*x

class Logish(nn.Module):
    def __init__(self):
        super(Logish, self).__init__()

    def forward(self, x):
        return x*torch.log(1+torch.sigmoid(x))



class SwishReLU(nn.Module):
    def __init__(self):
        super(SwishReLU, self).__init__()
    def forward(self, x):
        swish = x * torch.sigmoid(x)  # Swish for x <= 0
        relu = x             # ReLU for x > 0
        return torch.where(x > 0, relu, swish)


class SReLU(nn.Module):
    def __init__(self, alpha=1.0, beta=-0.5, lambda_=1.0):
        super(SReLU, self).__init__()
        self.alpha = alpha  # α parameter
        self.beta = beta    # β parameter
        self.lambda_ = lambda_  # λ parameter

    def forward(self, x):
        # Compute the Swish-like part for x < 0
        swish_like = self.alpha * (torch.sigmoid(x) + self.beta)
        # Compute the linear part for x >= 0
        relu_like = self.lambda_ * x
        # Use torch.where to apply the conditions
        return torch.where(x >= 0, relu_like, swish_like)


class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x):
        return x*(torch.tanh(F.softplus(x)))


class Cosine(nn.Module):
    def forward(self, x):
        return torch.cos(x)
class Sinusoid(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class HF11(nn.Module):
    def __init__(self):
        super(HF11, self).__init__()

    def forward(self, x):
        # 正负区间均使用 arctan(x)
        return torch.atan(x)  # 等价于 torch.arctan(x)


class HFP(nn.Module):
    def __init__(self, alpha=0.2, beta=0.2):
        super(HFP, self).__init__()
        self.alpha = alpha  # x >= 0时的系数
        self.beta = beta  # x < 0时的系数

    def forward(self, x):
        H = self.beta * torch.tanh(x)  # H for x < 0
        F = self.alpha * torch.arctan(x)  # F for x >= 0
        return torch.where(x >= 0, F, H)
class M101(nn.Module):
    def __init__(self, in_size, dropout=0.37, alpha=1.2, beta=0.8):  # 添加num_heads参数
        super().__init__()
        # self.af1 = nn.ReLU()
        # self.af1 = nn.PReLU()
        # self.af1 = nn.LeakyReLU()
        # self.af1 = nn.RReLU(lower=0.125, upper=0.333)
        # self.af1 = nn.ELU()  #表示当前启用ELU函数
        # self.af1 = nn.GELU()
        # self.af1 = nn.SiLU()#swish
        # self.af1 = nn.Tanh()
        # self.af1 = Mish()
        # self.af1 = torch.nn.LogSigmoid()
        # self.af1 = nn.Softplus()
        # self.af1 = Cosine()
        # self.af1 = nn.Hardshrink()
        # self.af1 = nn.Sigmoid()
        # self.af1 = HF11()  #纯arctan
        self.af1 = HFP(alpha=alpha, beta=beta)

        # 初始投影层（保持不变）
        self.initial_proj = nn.Sequential(
            nn.Linear(in_size, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout * 0.7)
        )

        # 卷积模块（修改：四层改为一层）
        self.conv_block = nn.Sequential(
            # 只有一层卷积
            nn.Conv1d(1, 300, kernel_size=3, padding=1),
            nn.BatchNorm1d(300),
            nn.AdaptiveAvgPool1d(1)
        )

        # 用线性层替换自注意力机制
        self.linear_transform = nn.Linear(300, 300)

        # 全连接部分（保持不变）
        self.fc = nn.Sequential(
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_proj(x).unsqueeze(1)  # [batch, 1, 100]
        x = self.conv_block(x)  # [batch, 300, 1]
        x = x.squeeze(2)  # [batch, 300]

        # 用线性变换替换自注意力
        x = self.linear_transform(x)  # [batch, 300]

        x = self.fc(x)
        return self.output(x)


class M1012(nn.Module):
    def __init__(self, in_size, dropout=0.3,  alpha=1.0, beta=1.0):  # 添加num_heads参数
        super().__init__()
        # self.af1 = nn.ReLU()
        # self.af1 = nn.PReLU()
        # self.af1 = nn.LeakyReLU()
        # self.af1 = nn.RReLU(lower=0.125, upper=0.333)
        # self.af1 = nn.ELU()  #表示当前启用ELU函数
        # self.af1 = nn.GELU()
        # self.af1 = nn.SiLU()#swish
        # self.af1 = nn.Tanh()
        # self.af1 = Mish()
        # self.af1 = torch.nn.LogSigmoid()
        # self.af1 = nn.Softplus()
        # self.af1 = Cosine()
        # self.af1 = nn.Hardshrink()
        # self.af1 = nn.Sigmoid()
        # self.af1 = HF11()  #纯arctan
        self.af1 = HFP(alpha=alpha, beta=beta)

        # 初始投影层（保持不变）
        self.initial_proj = nn.Sequential(
            nn.Linear(in_size, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout * 0.7)
        )

        # 卷积模块（保持不变）
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 300, kernel_size=3, padding=1),
            nn.BatchNorm1d(300),
            nn.AdaptiveAvgPool1d(1)
        )

        # 最小修改：用线性变换替换自注意力（维度300→300，保持dropout一致）
        self.linear_proj = nn.Sequential(
            nn.Linear(300, 300),  # 等价原自注意力退化后的线性映射
            nn.Dropout(dropout)  # 保留原自注意力的dropout概率
        )

        # 全连接部分（保持不变）
        self.fc = nn.Sequential(
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_proj(x).unsqueeze(1)  # [batch, 1, 100]
        x = self.conv_block(x)  # [batch, 300, 1]

        # 调整维度：最小修改（保留原permute逻辑，也可简化为x.squeeze(2)，效果一致）
        x = x.permute(0, 2, 1)  # [batch, 1, 300]
        x = x.squeeze(1)  # [batch, 300]（直接删除长度为1的维度）

        # 最小修改：用线性变换替换自注意力调用
        x = self.linear_proj(x)  # [batch, 300]

        x = self.fc(x)
        return self.output(x)


class M103(nn.Module):
    def __init__(self, in_size, dropout=0.3,  alpha=1.5, beta=2.0):  # 添加num_heads参数
        super().__init__()
        # self.af1 = nn.ReLU()
        # self.af1 = nn.PReLU()
        # self.af1 = nn.LeakyReLU()
        # self.af1 = nn.RReLU(lower=0.125, upper=0.333)
        # self.af1 = nn.ELU()  #表示当前启用ELU函数
        # self.af1 = nn.GELU()
        # self.af1 = nn.SiLU()#swish
        # self.af1 = nn.Tanh()
        # self.af1 = Mish()
        # self.af1 = torch.nn.LogSigmoid()
        # self.af1 = nn.Softplus()
        # self.af1 = Cosine()
        # self.af1 = nn.Hardshrink()
        # self.af1 = nn.Sigmoid()
        # self.af1 = HF11()  #纯arctan
        self.af1 = HFP(alpha=alpha, beta=beta)

        # 初始投影层（保持不变）
        self.initial_proj = nn.Sequential(
            nn.Linear(in_size, 100),
            nn.BatchNorm1d(100),
            # self.af1,
            nn.Dropout(dropout * 0.7)
        )

        # 卷积模块（保持不变，仍输出[batch, 300, 1]）
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 150, kernel_size=3, padding=1),
            nn.BatchNorm1d(150),
            self.af1,
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(150, 300, kernel_size=3, padding=1),
            nn.BatchNorm1d(300),
            self.af1,
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(300, 300, kernel_size=3, padding=1),
            nn.BatchNorm1d(300),
            nn.AdaptiveAvgPool1d(1)
        )

        # 核心修改：用线性变换替换自注意力（输入300维→输出300维，与原逻辑一致）
        self.linear_proj = nn.Sequential(
            nn.Linear(300, 300),  # 等价于自注意力退化后的线性映射
            nn.BatchNorm1d(300),  # 可选：添加BatchNorm提升稳定性（原自注意力无，可按需删除）
            nn.Dropout(dropout)  # 保持与原自注意力一致的dropout概率
        )

        # 全连接部分（保持不变）
        self.fc = nn.Sequential(
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_proj(x).unsqueeze(1)  # [batch, 1, 100]
        x = self.conv_block(x)  # [batch, 300, 1]

        # 维度调整（与原逻辑一致，仅需squeeze无需permute，更简洁）
        x = x.squeeze(2)  # [batch, 300]（直接删除长度为1的维度，等价于原permute+squeeze）

        # 应用线性变换（替换原自注意力调用）
        x = self.linear_proj(x)  # [batch, 300]

        x = self.fc(x)
        return self.output(x)




class M102(nn.Module):
    def __init__(self, in_size, dropout=0.37135066533871786, num_heads=4, alpha=1.5, beta=1.5):
        super().__init__()
        self.af1 = HFP(alpha=alpha, beta=beta)

        # 初始投影层（保持不变）
        self.initial_proj = nn.Sequential(
            nn.Linear(in_size, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout * 0.7)
        )

        # 卷积模块（修改：移除最后的AdaptiveAvgPool1d）
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 150, kernel_size=3, padding=1),
            nn.BatchNorm1d(150),
            self.af1,
            nn.MaxPool1d(2),  # 序列长度从100变为50
            nn.Dropout(dropout),

            # 最后一层卷积（移除AdaptiveAvgPool1d）
            nn.Conv1d(150, 300, kernel_size=3, padding=1),
            nn.BatchNorm1d(300),
            # 移除 nn.AdaptiveAvgPool1d(1) 以保持序列长度
        )

        # 自注意力机制（保持不变）
        self.attention = nn.MultiheadAttention(
            embed_dim=300,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        # 全连接部分（保持不变）
        self.fc = nn.Sequential(
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_proj(x).unsqueeze(1)  # [batch, 1, 100]
        x = self.conv_block(x)  # [batch, 300, 50] - 现在有50个时间步

        # 调整维度以适应注意力机制
        x = x.permute(0, 2, 1)  # [batch, 50, 300]

        # 应用自注意力（现在有50个时间步，注意力可以正常工作）
        attn_output, _ = self.attention(x, x, x)

        # 取所有时间步的平均值
        x = attn_output.mean(dim=1)  # [batch, 300]

        x = self.fc(x)
        return self.output(x)


class M102_SE(nn.Module):
    def __init__(self, in_size, dropout=0.37135066533871786, num_heads=4, alpha=1.5, beta=1.5):
        super().__init__()
        self.af1 = HFP(alpha=alpha, beta=beta)

        # 初始投影层（保持不变）
        self.initial_proj = nn.Sequential(
            nn.Linear(in_size, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout * 0.7)
        )

        # 卷积模块（保持不变）
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 150, kernel_size=3, padding=1),
            nn.BatchNorm1d(150),
            self.af1,
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(150, 300, kernel_size=3, padding=1),
            nn.BatchNorm1d(300),
        )

        # SE注意力机制 (reduction=16)
        self.se_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(300, 300 // 16),
            nn.ReLU(),
            nn.Linear(300 // 16, 300),
            nn.Sigmoid()
        )

        # 全连接部分（保持不变）
        self.fc = nn.Sequential(
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_proj(x).unsqueeze(1)  # [batch, 1, 100]
        x = self.conv_block(x)  # [batch, 300, 50]

        # SE注意力
        se_weights = self.se_attention(x).unsqueeze(2)  # [batch, 300, 1]
        x = x * se_weights  # 通道注意力

        # 全局平均池化
        x = x.mean(dim=2)  # [batch, 300]

        x = self.fc(x)
        return self.output(x)


class M102_ECA(nn.Module):
    def __init__(self, in_size, dropout=0.37135066533871786, num_heads=4, alpha=1.5, beta=1.5):
        super().__init__()
        self.af1 = HFP(alpha=alpha, beta=beta)

        # 初始投影层（保持不变）
        self.initial_proj = nn.Sequential(
            nn.Linear(in_size, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout * 0.7)
        )

        # 卷积模块（保持不变）
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 150, kernel_size=3, padding=1),
            nn.BatchNorm1d(150),
            self.af1,
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(150, 300, kernel_size=3, padding=1),
            nn.BatchNorm1d(300),
        )

        # ECA注意力机制 (kernel_size=3)
        self.eca_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(300, 300, kernel_size=3, padding=1, groups=300),
            nn.Sigmoid()
        )

        # 全连接部分（保持不变）
        self.fc = nn.Sequential(
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_proj(x).unsqueeze(1)  # [batch, 1, 100]
        x = self.conv_block(x)  # [batch, 300, 50]

        # ECA注意力
        eca_weights = self.eca_attention(x)  # [batch, 300, 1]
        x = x * eca_weights  # 通道注意力

        # 全局平均池化
        x = x.mean(dim=2)  # [batch, 300]

        x = self.fc(x)
        return self.output(x)


class SimAM(nn.Module):
    def __init__(self, lambda_val=1e-4):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        # SimAM注意力机制
        n, c, h, w = x.size()
        # 由于我们是一维数据，将w视为序列长度
        n, c, l = x.size()

        # 计算能量函数
        d = (x - x.mean(dim=2, keepdim=True)).pow(2)
        v = l
        e_inv = d / (4 * (d.sum(dim=2, keepdim=True) / v + self.lambda_val)) + 0.5

        return x * e_inv


class M102_SimAM(nn.Module):
    def __init__(self, in_size, dropout=0.37135066533871786, num_heads=4, alpha=1.5, beta=1.5):
        super().__init__()
        self.af1 = HFP(alpha=alpha, beta=beta)

        # 初始投影层（保持不变）
        self.initial_proj = nn.Sequential(
            nn.Linear(in_size, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout * 0.7)
        )

        # 卷积模块（保持不变）
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 150, kernel_size=3, padding=1),
            nn.BatchNorm1d(150),
            self.af1,
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(150, 300, kernel_size=3, padding=1),
            nn.BatchNorm1d(300),
        )

        # SimAM注意力机制 (lambda=1e-4)
        self.simam_attention = SimAM(lambda_val=1e-4)

        # 全连接部分（保持不变）
        self.fc = nn.Sequential(
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_proj(x).unsqueeze(1)  # [batch, 1, 100]
        x = self.conv_block(x)  # [batch, 300, 50]

        # SimAM注意力
        x = self.simam_attention(x)  # [batch, 300, 50]

        # 全局平均池化
        x = x.mean(dim=2)  # [batch, 300]

        x = self.fc(x)
        return self.output(x)


class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=3):
        super().__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa

        return x


class M102_CBAM(nn.Module):
    def __init__(self, in_size, dropout=0.37135066533871786, num_heads=4, alpha=1.5, beta=1.5):
        super().__init__()
        self.af1 = HFP(alpha=alpha, beta=beta)

        self.initial_proj = nn.Sequential(
            nn.Linear(in_size, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout * 0.7)
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 150, kernel_size=3, padding=1),
            nn.BatchNorm1d(150),
            self.af1,
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(150, 300, kernel_size=3, padding=1),
            nn.BatchNorm1d(300),
        )

        # CBAM注意力 (reduction=16, kernel_size=3)
        self.cbam_attention = CBAM1D(channels=300, reduction=16, kernel_size=3)

        self.fc = nn.Sequential(
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_proj(x).unsqueeze(1)  # [batch, 1, 100]
        x = self.conv_block(x)  # [batch, 300, 50]

        # CBAM注意力
        x = self.cbam_attention(x)  # [batch, 300, 50]

        x = x.mean(dim=2)  # [batch, 300]
        x = self.fc(x)
        return self.output(x)
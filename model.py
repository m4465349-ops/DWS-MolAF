import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class HFP(nn.Module):
    def __init__(self, alpha=0.2, beta=0.2):
        super(HFP, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        H = self.beta * torch.tanh(x)
        F = self.alpha * torch.arctan(x)
        return torch.where(x >= 0, F, H)

class M_Carcino(nn.Module):
    def __init__(self, in_size, dropout=0.37, alpha=1.2, beta=0.8):
        super().__init__()
        self.af1 = HFP(alpha=alpha, beta=beta)

        self.initial_proj = nn.Sequential(
            nn.Linear(in_size, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout * 0.7)
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 300, kernel_size=3, padding=1),
            nn.BatchNorm1d(300),
            nn.AdaptiveAvgPool1d(1)
        )

        self.linear_transform = nn.Linear(300, 300)

        self.fc = nn.Sequential(
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_proj(x).unsqueeze(1)
        x = self.conv_block(x)
        x = x.squeeze(2)

        x = self.linear_transform(x)

        x = self.fc(x)
        return self.output(x)


class M_Cardio(nn.Module):
    def __init__(self, in_size, dropout=0.3, alpha=1.0, beta=1.0):
        super().__init__()
        self.af1 = HFP(alpha=alpha, beta=beta)

        self.initial_proj = nn.Sequential(
            nn.Linear(in_size, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout * 0.7)
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 300, kernel_size=3, padding=1),
            nn.BatchNorm1d(300),
            nn.AdaptiveAvgPool1d(1)
        )

        self.linear_proj = nn.Sequential(
            nn.Linear(300, 300),
            nn.Dropout(dropout)
        )

        self.fc = nn.Sequential(
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_proj(x).unsqueeze(1)
        x = self.conv_block(x)

        x = x.permute(0, 2, 1)
        x = x.squeeze(1)

        x = self.linear_proj(x)

        x = self.fc(x)
        return self.output(x)


class M_Hepa(nn.Module):
    def __init__(self, in_size, dropout=0.3, alpha=1.5, beta=2.0):
        super().__init__()
        self.af1 = HFP(alpha=alpha, beta=beta)

        self.initial_proj = nn.Sequential(
            nn.Linear(in_size, 100),
            nn.BatchNorm1d(100),
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
            self.af1,
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(300, 300, kernel_size=3, padding=1),
            nn.BatchNorm1d(300),
            nn.AdaptiveAvgPool1d(1)
        )

        self.linear_proj = nn.Sequential(
            nn.Linear(300, 300),
            nn.BatchNorm1d(300),
            nn.Dropout(dropout)
        )

        self.fc = nn.Sequential(
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_proj(x).unsqueeze(1)
        x = self.conv_block(x)

        x = x.squeeze(2)

        x = self.linear_proj(x)

        x = self.fc(x)
        return self.output(x)


class M_Muta(nn.Module):
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

        self.attention = nn.MultiheadAttention(
            embed_dim=300,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            self.af1,
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_proj(x).unsqueeze(1)
        x = self.conv_block(x)

        x = x.permute(0, 2, 1)

        attn_output, _ = self.attention(x, x, x)

        x = attn_output.mean(dim=1)

        x = self.fc(x)
        return self.output(x)


class M_Acute(nn.Module):
    def __init__(self, in_size, dropout=0.3, norm_type='batch', alpha=1.5, beta=1.5):
        super().__init__()
        self.fc1 = nn.Linear(in_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

        if norm_type == 'batch':
            self.norms = nn.ModuleList([
                nn.BatchNorm1d(256),
                nn.BatchNorm1d(128),
                nn.BatchNorm1d(64),
                nn.BatchNorm1d(32)
            ])
        elif norm_type == 'layer':
            self.norms = nn.ModuleList([
                nn.LayerNorm(256),
                nn.LayerNorm(128),
                nn.LayerNorm(64),
                nn.LayerNorm(32)
            ])
        else:
            self.norms = nn.ModuleList([nn.Identity()] * 4)

        self.dropout = nn.Dropout(dropout)
        self.af1 = HFP()

        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        nn.init.xavier_normal_(self.fc5.weight)
        nn.init.constant_(self.fc5.bias, 0.0)

    def extract_features(self, x):
        with torch.no_grad():
            x = self.dropout(self.af1(self.norms[0](self.fc1(x))))
            x = self.dropout(self.af1(self.norms[1](self.fc2(x))))
            x = self.dropout(self.af1(self.norms[2](self.fc3(x))))
            x = self.dropout(self.af1(self.norms[3](self.fc4(x))))

            features = x

            return features

    def forward(self, x):
        x = self.dropout(self.af1(self.norms[0](self.fc1(x))))
        x = self.dropout(self.af1(self.norms[1](self.fc2(x))))
        x = self.dropout(self.af1(self.norms[2](self.fc3(x))))
        x = self.dropout(self.af1(self.norms[3](self.fc4(x))))
        return self.fc5(x)
import torch
import torch.nn as nn


# 定义一个简单的自编码器模型
class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # 定义解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 编码器
        z = self.encoder(x)

        # 解码器
        x_hat = self.decoder(z)

        return x_hat

    def loss_function(self, x_hat, x):
        # 重构损失
        reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')

        return reconstruction_loss


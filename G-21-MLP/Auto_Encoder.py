import torch
import torchvision
import torch.nn as nn


# 定义自编码器结构
class Auto_Encoder(nn.Module):

    def __init__(self, obs_size):
        super(Auto_Encoder, self).__init__()
        self.obs_size = obs_size

        # 定义编码器结构
        self.Encoder = nn.Sequential(
            nn.Linear(self.obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )

        # 定义解码器结构
        self.Decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, self.obs_size),
            nn.Sigmoid()
        )

    def forward(self, input):
        code = input.view(input.size(0), -1)
        code = self.Encoder(code)

        output = self.Decoder(code)
        output = output.view(input.size(0), -1)

        return output

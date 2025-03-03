import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int, output_size: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Tanh()  # 保证输出范围 [-1, 1]
        )

    def forward(self, x):
        return self.model(x).view(x.size(0), 3, 16, 16)
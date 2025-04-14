import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels: int = 3, n_classes: int = 3, img_size: int = 16) -> None:
        super().__init__()
        # 将条件 embedding 到一个向量，再 reshape 为 [batch, 1, img_size, img_size]
        self.label_embedding = nn.Linear(n_classes, img_size * img_size)
        self.model = nn.Sequential(
            nn.Conv2d(img_channels + 1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(64 * (img_size // 2) * (img_size // 2), 1),
            nn.Sigmoid()
        )
    
    def forward(self, img: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch_size = img.size(0)
        img_size = img.size(2)
        cond = self.label_embedding(condition).view(batch_size, 1, img_size, img_size)
        x = torch.cat((img, cond), dim=1)
        return self.model(x)

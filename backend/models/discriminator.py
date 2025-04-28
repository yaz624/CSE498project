import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_classes: int, img_size: int = 16):
        super().__init__()
        self.label_embed = nn.Linear(n_classes, img_size * img_size)
        self.img_size = img_size

        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),  
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * (img_size // 4) * (img_size // 4), 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        b, c, h, w = img.shape
        cond = self.label_embed(labels).view(b, 1, h, w)
        x = torch.cat((img, cond), dim=1)
        return self.model(x)

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim: int, n_classes: int, img_size: int = 16):
        super().__init__()
        self.img_size = img_size
        self.label_emb = nn.Linear(n_classes, 10)  # 条件嵌入为10维
        input_dim = latent_dim + 10

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128 * (img_size // 4) * (img_size // 4)),
            nn.BatchNorm1d(128 * (img_size // 4) * (img_size // 4)),
            nn.ReLU(True),
            nn.Unflatten(1, (128, img_size // 4, img_size // 4)),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # -> (64, img_size/2, img_size/2)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # -> (3, img_size, img_size)
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_emb(labels)
        x = torch.cat((noise, label_embed), dim=1)
        return self.model(x)

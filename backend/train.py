# backend/train.py

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import DataLoader

# 加路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from configs.config import latent_dim, train_batch_size, num_epochs, learning_rate_G, learning_rate_D
from dataset.dataset import PixelDataset
from models.generator import Generator
from models.discriminator import Discriminator
from backend.utils.device import get_device

device = get_device()
n_classes = 4
output_size = 3 * 16 * 16
print("Using device:", device)

# 预处理
transform = transforms.Compose([
    transforms.Resize(16),
    transforms.CenterCrop(16),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3)
])

# 路径
csv_file = os.path.join(project_root, "resources", "pixel_dataset", "labels.csv")
images_dir = os.path.join(project_root, "resources", "pixel_dataset", "images")

# 数据
dataset = PixelDataset(csv_file=csv_file, root_dir=images_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

# 模型
generator = Generator(latent_dim, n_classes=n_classes, img_size=16).to(device)
discriminator = Discriminator(n_classes=n_classes, img_size=16).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_G, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_D, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Checkpoint
CHECKPOINT_DIR = os.path.join(project_root, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(epoch):
    torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch}.pth"))

# 训练
for epoch in range(num_epochs):
    for batch_idx, (real_imgs, conditions) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        conditions = conditions.to(device)
        batch_size = real_imgs.size(0)

        real_labels = torch.ones((batch_size, 1), device=device) * 0.9
        fake_labels = torch.zeros((batch_size, 1), device=device)

        # === 训练 D ===
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(noise, conditions)

        preds_real = discriminator(real_imgs, conditions)
        preds_fake = discriminator(fake_imgs.detach(), conditions)
        d_loss = (criterion(preds_real, real_labels) + criterion(preds_fake, fake_labels)) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # === 训练 G ===
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(noise, conditions)
        preds = discriminator(fake_imgs, conditions)
        g_loss = criterion(preds, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if batch_idx % 50 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {batch_idx}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    if epoch % 10 == 0 and epoch > 0:
        save_checkpoint(epoch)
        with torch.no_grad():
            sample_noise = torch.randn(16, latent_dim, device=device)
            sample_labels = torch.randint(0, n_classes, (16,), device=device)
            onehot = torch.zeros(16, n_classes, device=device)
            onehot.scatter_(1, sample_labels.unsqueeze(1), 1)
            fake_imgs = generator(sample_noise, onehot)
            utils.save_image(fake_imgs.data, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.png"), normalize=True, nrow=4)

save_checkpoint("final")
print("Training completed.")

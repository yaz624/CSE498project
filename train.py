import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import DataLoader

import sys
import os

# 获取当前脚本（train.py）的所在目录
project_root = os.path.abspath(os.path.dirname(__file__))
# 或者再加一层保险：如果你想从 project_root 的父目录开始
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 将该目录加入 sys.path，让 Python 能在此目录下搜索模块
sys.path.insert(0, project_root)

from dataset.dataset import PixelDataset
# 现在应该能找到 datasets/pixel_dataset.py 了


from backend.models.generator import Generator
from backend.models.discriminator import Discriminator
from backend.utils.device import get_device
from configs.config import latent_dim, train_batch_size, num_epochs, learning_rate_G, learning_rate_D

# 设定 checkpoint 目录为项目根目录下的 checkpoints/
CHECKPOINT_DIR = os.path.join(project_root, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = get_device()
print("Using device:", device)

criterion = nn.BCELoss()
# 输出图像大小： 3x16x16
output_size = 3 * 16 * 16

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(16),
    transforms.CenterCrop(16),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# 加载数据集：读取 CSV 文件中的图像路径和条件标签
csv_file = os.path.join(project_root, "resources", "pixel_dataset", "labels.csv")
root_dir = os.path.join(project_root, "resources", "pixel_dataset", "images")
train_dataset = PixelDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
print("Dataset size:", len(train_dataset))

dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

# 角色类别数 = 3
n_classes = 3

generator = Generator(latent_dim, n_classes=n_classes, output_size=output_size).to(device)
discriminator = Discriminator(n_classes=n_classes, img_size=16).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_G, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_D, betas=(0.5, 0.999))

def save_checkpoint(epoch, generator, discriminator):
    torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch}.pth"))
    print(f"Saved checkpoint at epoch {epoch}")

for epoch in range(num_epochs):
    for batch_idx, (real_imgs, conditions) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        conditions = conditions.to(device)  # conditions: [batch, 3]
        batch_size = real_imgs.size(0)

        # 真实标签平滑, 设置为 0.9；假标签为 0.0
        real_targets = torch.ones((batch_size, 1), device=device) * 0.9
        fake_targets = torch.zeros((batch_size, 1), device=device)

        # 生成随机噪声
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(noise, conditions)

        # 判别器对真实图片和假图片的预测
        real_preds = discriminator(real_imgs, conditions)
        fake_preds = discriminator(fake_imgs.detach(), conditions)
        
        d_loss_real = criterion(real_preds, real_targets)
        d_loss_fake = criterion(fake_preds, fake_targets)
        d_loss = (d_loss_real + d_loss_fake) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器，让生成器产生的图片能够骗过判别器
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(noise, conditions)
        fake_preds = discriminator(fake_imgs, conditions)
        g_loss = criterion(fake_preds, real_targets)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if batch_idx % 50 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {batch_idx}/{len(dataloader)}] " +
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
    
    if epoch % 20 == 0 and epoch != 0:
        save_checkpoint(epoch, generator, discriminator)
        with torch.no_grad():
            sample_noise = torch.randn(16, latent_dim, device=device)
            # 随机选择 16 条条件；这里为了简单我们直接随机生成条件 one-hot
            sample_labels = torch.randint(0, n_classes, (16,), device=device)
            sample_conditions = torch.zeros(16, n_classes, device=device)
            sample_conditions.scatter_(1, sample_labels.unsqueeze(1), 1)
            sample_conditions = sample_conditions.float()
            gen_imgs = generator(sample_noise, sample_conditions)
            utils.save_image(gen_imgs, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.png"), normalize=True, nrow=4)

torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, "generator.pth"))
print("Training Done!")

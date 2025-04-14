import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import DataLoader

# 将项目根目录加入 sys.path，保证可以导入configs、datasets、models等模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from configs.config import latent_dim, train_batch_size, num_epochs, learning_rate_G, learning_rate_D
from dataset.dataset import PixelDataset
from models.generator import Generator
from models.discriminator import Discriminator
from backend.utils.device import get_device

device = get_device()
print("Using device:", device)

# 输出图像尺寸：3×16×16
output_size = 3 * 16 * 16
# 角色类别数为3（例如：monster, human, prop）
n_classes = 3

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(16),
    transforms.CenterCrop(16),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# CSV 文件路径和图片所在目录
csv_file = os.path.join(project_root, "resources", "pixel_dataset", "labels.csv")
images_dir = os.path.join(project_root, "resources", "pixel_dataset", "images")
print("CSV file:", csv_file)
print("Images directory:", images_dir)

# 实例化数据集，PixelDataset 负责读取 CSV 并正确解析标签、标准化文件路径（用os.path.basename）
train_dataset = PixelDataset(csv_file=csv_file, root_dir=images_dir, transform=transform)
print("Dataset size:", len(train_dataset))

dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

# 实例化条件生成器和条件判别器
generator = Generator(latent_dim, n_classes=n_classes, output_size=output_size).to(device)
discriminator = Discriminator(n_classes=n_classes, img_size=16).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_G, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_D, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# checkpoint 保存目录（设为项目根目录下的 checkpoints/）
CHECKPOINT_DIR = os.path.join(project_root, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(epoch, generator, discriminator):
    gen_path = os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch}.pth")
    disc_path = os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch}.pth")
    torch.save(generator.state_dict(), gen_path)
    torch.save(discriminator.state_dict(), disc_path)
    print(f"Saved checkpoint at epoch {epoch}")

# 开始训练
for epoch in range(num_epochs):
    for batch_idx, (real_imgs, conditions) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        # conditions: 假设已转换为 one-hot 形状 [batch, 3]
        conditions = conditions.to(device)
        batch_size = real_imgs.size(0)

        # 真实标签 (label smoothing, 0.9) 与假标签 (0.0)
        real_targets = torch.ones((batch_size, 1), device=device) * 0.9
        fake_targets = torch.zeros((batch_size, 1), device=device)

        # ----- 训练判别器 -----
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(noise, conditions)
        # 判别器对真实与假图预测
        preds_real = discriminator(real_imgs, conditions)
        preds_fake = discriminator(fake_imgs.detach(), conditions)
        loss_real = criterion(preds_real, real_targets)
        loss_fake = criterion(preds_fake, fake_targets)
        d_loss = (loss_real + loss_fake) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ----- 训练生成器 -----
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(noise, conditions)
        preds_fake_for_g = discriminator(fake_imgs, conditions)
        g_loss = criterion(preds_fake_for_g, real_targets)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if batch_idx % 50 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {batch_idx}/{len(dataloader)}] " +
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
    
    # 每 20 个 epoch 保存一次 checkpoint，同时保存样本网格图
    if epoch != 0 and epoch % 10 == 0:
        save_checkpoint(epoch, generator, discriminator)
        with torch.no_grad():
            sample_noise = torch.randn(16, latent_dim, device=device)
            # 从数据集中随机生成条件，或者直接生成随机 one-hot
            sample_labels = torch.randint(0, n_classes, (16,), device=device)
            sample_conditions = torch.zeros(16, n_classes, device=device)
            sample_conditions.scatter_(1, sample_labels.unsqueeze(1), 1)
            sample_conditions = sample_conditions.float()
            gen_imgs = generator(sample_noise, sample_conditions)
            grid_filename = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.png")
            utils.save_image(gen_imgs.data, grid_filename, normalize=True, nrow=4)
            print(f"Saved sample grid image at epoch {epoch}: {grid_filename}")

# 最后保存一个 final checkpoint
save_checkpoint("final", generator, discriminator)
print("Training completed!")

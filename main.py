import torch
import os
import matplotlib.pyplot as plt
from typing import Any
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from torch import nn, optim, Tensor

from configs.config import train_batch_size,  num_epochs, latent_dim, learning_rate_G, learning_rate_D
from datasets.pixel_dataset import PixelDataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils.device import get_device


CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Save model
def save_checkpoint(epoch, generator, discriminator):
    torch.save(generator.state_dict(), f"{CHECKPOINT_DIR}/generator_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"{CHECKPOINT_DIR}/discriminator_epoch_{epoch}.pth")
    print(f"Saved checkpoint at epoch {epoch}")


# Device
device = get_device()
print(f"Using device: {device}")

# Loss function
criterion = nn.BCELoss()

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Three channels normalization
])

# Dataset and DataLoader
train_dataset = PixelDataset(transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    drop_last=True
)

# Models
discriminator = Discriminator().to(device)
generator = Generator(latent_dim, 16 * 16 * 3).to(device)

# Optimizers
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate_D, betas=(0.5, 0.999))
generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate_G, betas=(0.5, 0.999))

# start_epoch = 150
# generator.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/generator_epoch_150.pth"))
# discriminator.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/discriminator_epoch_150.pth"))
# print(f"Loaded checkpoint from epoch 150")

# Training loop
for epoch in range( num_epochs):
    loss_discriminator: Any = None
    loss_generator: Any = None
    generated_samples: Tensor = torch.Tensor()
    real_samples: Tensor = torch.Tensor()

    for batch_idx, real_samples in enumerate(train_loader):
        real_samples = real_samples.to(device)

        # Labels
        real_labels = torch.ones((train_batch_size, 1), device=device) * 0.9
        fake_labels = torch.zeros((train_batch_size, 1), device=device)

        # Generate fake samples
        latent_space = torch.randn((train_batch_size, latent_dim), device=device)
        fake_samples = generator(latent_space)

        # Combine real and fake samples
        all_samples = torch.cat((real_samples, fake_samples))
        all_labels = torch.cat((real_labels, fake_labels))

        # Train Discriminator
        if batch_idx % 2 == 0:
            discriminator.zero_grad()
            all_samples = torch.cat((real_samples, fake_samples))
            all_labels = torch.cat((real_labels, fake_labels))
            predictions = discriminator(all_samples)
            loss_discriminator = criterion(predictions, all_labels)
            loss_discriminator.backward()
            discriminator_optimizer.step()

        # Train Generator
        latent_space = torch.randn((train_batch_size, latent_dim), device=device)
        generator.zero_grad()
        generated_samples = generator(latent_space)
        predictions = discriminator(generated_samples)
        loss_generator = criterion(predictions, real_labels)
        loss_generator.backward()
        generator_optimizer.step()

    print(f"Epoch: {epoch} | Loss D.: {loss_discriminator:.4f} | Loss G.: {loss_generator:.4f}")

    # 每 50 epoch 保存一次 checkpoint
    if epoch % 50 == 0 and epoch != 0:
        save_checkpoint(epoch, generator, discriminator)

    # Visualization every 5 epochs
    if epoch % 5 == 0:
        with torch.no_grad():
            cpu_fake_sample = generated_samples[0].cpu()
            cpu_real_sample = real_samples[0].cpu()
            plt.imshow(to_pil_image(cpu_fake_sample))
            plt.title(f"Fake Sample - Epoch {epoch}")
            plt.show()
            plt.imshow(to_pil_image(cpu_real_sample))
            plt.title(f"Real Sample - Epoch {epoch}")
            plt.show()

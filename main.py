import torch
import matplotlib.pyplot as plt
from typing import Any
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from torch import nn, optim, Tensor

from my_configs.config import train_batch_size,  num_epochs, latent_dim
from datasets.pixel_dataset import PixelDataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils.device import get_device


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
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# Training loop
for epoch in range(num_epochs):
    loss_discriminator: Any = None
    loss_generator: Any = None
    generated_samples: Tensor = torch.Tensor()
    real_samples: Tensor = torch.Tensor()

    for real_samples in train_loader:
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
        discriminator.zero_grad()
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

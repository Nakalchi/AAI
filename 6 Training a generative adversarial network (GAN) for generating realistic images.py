python -m pip install tensorflow numpy pandas matplotlib


import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# Step 1: Define Generator and Discriminator

class Generator(nn.Module):

    def __init__(self, noise_dim, img_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, img_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Step 2: Define Constants and Hyperparameters

device = "cuda" if torch.cuda.is_available() else "cpu"

img_size = 28
img_dim = img_size * img_size

noise_dim = 100
batch_size = 64
epochs = 50
lr = 0.0002

print("Using device:", device)


# Step 3: Prepare the MNIST Dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root="data",
    train=True,
    transform=transform,
    download=True
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


# Step 4: Initialize Models, Loss, and Optimizers

generator = Generator(
    noise_dim,
    img_dim
).to(device)

discriminator = Discriminator(
    img_dim
).to(device)

criterion = nn.BCELoss()

optimizer_g = optim.Adam(
    generator.parameters(),
    lr=lr
)

optimizer_d = optim.Adam(
    discriminator.parameters(),
    lr=lr
)


# Step 5: Training Loop

for epoch in range(epochs):

    for real_images, _ in dataloader:

        real_images = real_images.view(
            -1,
            img_dim
        ).to(device)

        current_batch_size = real_images.size(0)

        # Labels for real and fake images
        real_labels = torch.ones(
            current_batch_size,
            1
        ).to(device)

        fake_labels = torch.zeros(
            current_batch_size,
            1
        ).to(device)

        # Train Discriminator

        noise = torch.randn(
            current_batch_size,
            noise_dim
        ).to(device)

        fake_images = generator(noise)

        real_preds = discriminator(real_images)

        fake_preds = discriminator(
            fake_images.detach()
        )

        loss_d_real = criterion(
            real_preds,
            real_labels
        )

        loss_d_fake = criterion(
            fake_preds,
            fake_labels
        )

        loss_d = (
            loss_d_real + loss_d_fake
        ) / 2

        optimizer_d.zero_grad()

        loss_d.backward()

        optimizer_d.step()

        # Train Generator

        noise = torch.randn(
            current_batch_size,
            noise_dim
        ).to(device)

        fake_images = generator(noise)

        fake_preds = discriminator(fake_images)

        loss_g = criterion(
            fake_preds,
            real_labels
        )

        optimizer_g.zero_grad()

        loss_g.backward()

        optimizer_g.step()

    # Print training progress
    print(
        f"Epoch [{epoch + 1}/{epochs}] | "
        f"Loss D: {loss_d.item():.4f} | "
        f"Loss G: {loss_g.item():.4f}"
    )

    # Save generated samples every 10 epochs
    if (epoch + 1) % 10 == 0:

        noise = torch.randn(
            16,
            noise_dim
        ).to(device)

        with torch.no_grad():
            generated_images = generator(
                noise
            ).view(
                -1,
                1,
                img_size,
                img_size
            ).cpu()

        plt.figure(figsize=(8, 8))

        for i in range(16):

            plt.subplot(4, 4, i + 1)

            plt.imshow(
                generated_images[i].squeeze(),
                cmap="gray"
            )

            plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            f"generated_images_epoch_{epoch + 1}.png"
        )

        plt.close()

        print(
            f"Generated images saved for epoch {epoch + 1}"
        )


# Step 6: Save the Generator Model

torch.save(
    generator.state_dict(),
    "gan_generator.pth"
)

print("\nGenerator model saved as gan_generator.pth")

print("GAN training completed successfully!")

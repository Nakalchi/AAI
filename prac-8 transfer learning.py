python -m pip install torch torchvision matplotlib numpy--in cmd

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 10
batch_size = 32
epochs = 10
learning_rate = 0.001

print("Using device:", device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)



model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


for param in model.parameters():
    param.requires_grad = False


model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)



criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=learning_rate
)



def train(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total

    return running_loss / len(loader), accuracy



def test(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total

    return running_loss / len(loader), accuracy


train_losses = []
test_losses = []

train_accuracies = []
test_accuracies = []

for epoch in range(epochs):

    train_loss, train_acc = train(
        model,
        train_loader,
        criterion,
        optimizer
    )

    test_loss, test_acc = test(
        model,
        test_loader,
        criterion
    )

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"\nEpoch {epoch + 1}/{epochs}")

    print(
        f"Train Loss: {train_loss:.4f}, "
        f"Train Accuracy: {train_acc:.2f}%"
    )

    print(
        f"Test Loss: {test_loss:.4f}, "
        f"Test Accuracy: {test_acc:.2f}%"
    )

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")

plt.legend()
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")


plt.subplot(1, 2, 2)

plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")

plt.legend()
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.tight_layout()
plt.show()



torch.save(
    model.state_dict(),
    "resnet18_cifar10.pth"
)

print("\nModel saved as 'resnet18_cifar10.pth'.")


# PyTorch Neural Network practice using MNISTNet

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir="runs/mnist_idlerun")
num_epochs = 10  
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalizing using MNIST mean and std
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
        
        # Weight initialization example
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instantiate Model and Test on One Batch
model = MNISTNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

l1_lambda = 1e-5
l1_norm = sum(p.abs().sum() for p in model.parameters())
# original loss (0.5) is slightly increased due to the l1 regularisation norm
loss = 0.5 + l1_lambda * l1_norm

# Get one batch from the DataLoader
for images, labels in train_loader:
    output = model(images)  # Forward pass
    print(output.shape)     # Should be: torch.Size([64, 10])
    break

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    # loops through mini-batches in training data loader
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Clears gradients from previous step and completes pass to get predictions
        optimizer.zero_grad()
        outputs = model(inputs)

        # calculates how wrong the predictions are, backpropagates and updates weights using optimizer
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        # Select class with highest probability        
        _, predicted = outputs.max(1)

        # Compare predictions with targets
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Log batch loss
        step = epoch * len(train_loader) + batch_idx
        writer.add_scalar("Loss/Batch", loss.item(), step)

    writer.add_scalar("Loss/Epoch", epoch_loss / len(train_loader), epoch)
    writer.add_scalar("Accuracy/Epoch", 100. * correct / total, epoch)

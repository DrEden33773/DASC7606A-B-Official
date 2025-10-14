import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A simple CNN architecture for image classification
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional layers: progressively increase number of filters (3 -> 32 -> 64 -> 128)
        # 3x3 kernels with padding=1 maintain spatial dimensions before pooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling reduces spatial dimensions by half
        # Fully connected layers: flatten feature maps and classify
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 128 channels * 4x4 spatial resolution
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def create_model(num_classes, device):
    """Create and initialize the model"""
    model = SimpleCNN(num_classes=num_classes)
    model = model.to(device)
    return model

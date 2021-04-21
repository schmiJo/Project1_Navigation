import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, square_image_res=84, action_size=4, seed=0, fc1_units=120, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            square_image_res (int): The dimensions of the input image (assuming it is a square)
            action_size (int): Dimension of the action space
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=1)  # same size
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # halves size
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)  # same size
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # halves size
        self.fc1 = nn.Linear(16 * 20 * 20, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        print('Initialized updated')

    def forward(self, x):
        """Build a network that maps state -> action values."""
        # Convolutional Layers first
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # flatten x into an (1,0) tensor
        x = x.view(1, -1)

        # Fully connected Layers next
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

"""Linear Kernel Network module."""
import torch
from torch import nn
from torch.nn import functional as F


class LinearKernelNetwork(nn.Module):
    """Linear Kernel Network used in fourier network."""

    def __init__(self, size_in: int, num_of_modes: int) -> None:
        """Inits Linear Kernel Network."""
        super().__init__()
        self.size_in: int = size_in
        self.modes: int = num_of_modes
        self.fc0 = nn.Linear(3, self.size_in)
        self.fc1 = nn.Linear(1568, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(self.modes, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        # shape of x: [batch_size, #gridpoints in x, #gridpoints in y,3] since (a(x,y),x,y).
        xft = self.fc0(x)
        xft = xft.permute(0, 2, 1)

        xft = F.relu(self.fc1(xft))
        xft = F.relu(self.fc2(xft))
        xft = xft.permute(0, 2, 1)
        xft = F.relu(self.fc3(xft))
        return xft

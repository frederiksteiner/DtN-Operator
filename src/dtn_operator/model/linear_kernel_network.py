"""Linear Kernel Network module."""
from torch import nn
import torch.nn.functional as F


class LinearKernelNetwork(nn.Module):
    """Linear Kernel Network used in fourier network."""
    def __init__(self, size_in, numOfmodes, outsize, outker, outmodes):
        """Inits Linear Kernel Network."""
        super().__init__()
        self.outker, self.outmodes = outker, outmodes
        self.size_in, self.modes = size_in, numOfmodes
        self.outsize = outsize
        self.halfsize = int(self.outsize / 2)
        self.fc0 = nn.Linear(3, self.size_in)
        self.fc1 = nn.Linear(1568, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(self.modes, 32)

    def forward(self, x):
        """Forward method."""
        # shape of x: [batch_size, #gridpoints in x, #gridpoints in y,3] since (a(x,y),x,y).
        xft = self.fc0(x)
        xft = xft.permute(0, 2, 1)

        xft = F.relu(self.fc1(xft))
        xft = F.relu(self.fc2(xft))
        xft = xft.permute(0, 2, 1)
        xft = F.relu(self.fc3(xft))
        return xft

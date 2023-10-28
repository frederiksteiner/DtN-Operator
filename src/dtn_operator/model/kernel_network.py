"""Kernel Network module."""
import torch
from torch import fft
from torch import nn
from torch.nn import functional as F


class KernelNetwork(nn.Module):
    """Kernel Network used in the Fourier Neural Operator."""

    def __init__(
        self,
        size_in: int,
        num_of_modes: int,
        outsize: int,
        outker: int,
        outmodes: int,
    ):
        """Inits Kernel Network."""
        super().__init__()
        self.outker, self.outmodes = outker, outmodes
        self.size_in, self.modes = size_in, num_of_modes
        self.outsize = outsize
        self.halfsize = int(self.outsize / 2)
        self.fc0 = nn.Linear(3, self.size_in)
        self.fc1 = nn.Linear(self.size_in, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc4 = nn.Linear(self.modes, self.outsize)
        self.fc1c = nn.Linear(self.size_in, 128)
        self.fc3c = nn.Linear(128, 128)
        self.fc2c = nn.Linear(128, 1)
        self.fc4c = nn.Linear(self.modes, self.outsize)

    def forward(self, x: torch.Tensor):
        """Forward method."""
        # shape of x: [batch_size, #gridpoints in x, #gridpoints in y,3] since (a(x,y),x,y).
        xft = self.fc0(x)
        xft = xft.permute(0, 2, 1)

        # pylint: disable=not-callable
        xft = fft.rfftn(xft, dim=2, norm="ortho")
        xft = xft[:, :, : self.modes]
        xft = xft.permute(0, 2, 1)
        xftreal = F.relu(self.fc1(xft.real))
        xftreal = F.relu(self.fc3(xftreal))
        xftreal = F.relu(self.fc2(xftreal))
        xftreal = F.relu(self.fc4(xftreal.squeeze()))
        xftreal = xftreal.view(-1, self.outker, self.outker, self.outmodes)

        xftcomp = F.relu(self.fc1(xft.imag))
        xftcomp = F.relu(self.fc3(xftcomp))
        xftcomp = F.relu(self.fc2(xftcomp))
        xftcomp = F.relu(self.fc4(xftcomp.squeeze()))
        xftcomp = xftcomp.view(-1, self.outker, self.outker, self.outmodes)

        return xftreal + 1j * xftcomp

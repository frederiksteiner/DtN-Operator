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
    ) -> None:
        """Inits Kernel Network."""
        super().__init__()
        self.outker: int = outker
        self.outmodes: int = outmodes
        self.modes: int = num_of_modes
        self.outsize: int = outsize
        self.fc0: nn.Linear = nn.Linear(3, size_in)
        self.real_fc_layers: list[nn.Linear] = [
            nn.Linear(size_in, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 1),
            nn.Linear(self.modes, self.outsize),
        ]
        self.img_fc_layers: list[nn.Linear] = [
            nn.Linear(size_in, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 1),
            nn.Linear(self.modes, self.outsize),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        # shape of x: [batch_size, #gridpoints in x, #gridpoints in y,3] since (a(x,y),x,y).
        xft = self.fc0(x)
        xft = xft.permute(0, 2, 1)

        # pylint: disable=not-callable
        xft = fft.rfftn(xft, dim=2, norm="ortho")
        xft = xft[:, :, : self.modes]
        xft = xft.permute(0, 2, 1)
        xftreal = self._apply_fc_layers(self.real_fc_layers, xft.real)
        xftreal = xftreal.view(-1, self.outker, self.outker, self.outmodes)

        xftcomp = self._apply_fc_layers(self.img_fc_layers, xft.imag)
        xftcomp = xftcomp.view(-1, self.outker, self.outker, self.outmodes)

        return xftreal + 1j * xftcomp

    @staticmethod
    def _apply_fc_layers(
        fc_layers: list[nn.Linear], tensor: torch.Tensor
    ) -> torch.Tensor:
        for i, layer in enumerate(fc_layers):
            if i == len(fc_layers) - 1:
                tensor = tensor.squeeze()
            tensor = F.relu(layer(tensor))
        return tensor

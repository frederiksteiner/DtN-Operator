"""Fourier Network module."""
import torch
from torch import nn
from torch.nn import functional as F

from dtn_operator.model.network_parts.fourier_kernel import FourierKernel
from dtn_operator.model.network_parts.kernel_network import KernelNetwork

# pylint: disable=not-callable


def batch_conv_1d(x: torch.Tensor, batch_kernel: torch.Tensor) -> torch.Tensor:
    """Batch convolution."""
    kernel_shape = batch_kernel.shape
    shape = x.shape
    x = torch.reshape(x, [x.shape[0] * x.shape[1], x.shape[2]])
    x = torch.unsqueeze(x, 0)
    # x = x.view([x.shape[0]*x.shape[1], x.shape[2]])
    weights = batch_kernel.view([kernel_shape[0] * kernel_shape[1], kernel_shape[2], 1])
    # pylint: disable=not-callable
    outputs_grouped = F.conv1d(x, weights, groups=kernel_shape[0])
    outputs_grouped = outputs_grouped.view(shape)
    return outputs_grouped


class FourierNetwork(nn.Module):
    """Fourier Network class."""

    def __init__(
        self,
        size_in: int,
        num_of_modes: int,
        kernel_in: int,
        kernel_modes: int,
        num_of_layers: int,
    ) -> None:
        """Inits Fourier network."""
        super().__init__()
        self.size_in = size_in
        self.outsize: int = size_in * size_in * num_of_modes
        self.num_of_layers: int = num_of_layers
        self.fc0: nn.Linear = nn.Linear(3, size_in)
        self.fourreal: nn.ModuleList = nn.ModuleList(
            [
                FourierKernel(size_in, size_in, num_of_modes)
                for _ in range(self.num_of_layers)
            ]
        )
        self.green: nn.ModuleList = nn.ModuleList(
            [
                KernelNetwork(
                    kernel_in, kernel_modes, self.outsize, size_in, num_of_modes
                )
                for _ in range(self.num_of_layers)
            ]
        )
        self.wreal = nn.ModuleList(
            [nn.Conv1d(size_in, size_in, 1) for _ in range(self.num_of_layers)]
        )
        self.scale: float = 1 / (size_in * size_in)

        self.fc1: nn.Linear = nn.Linear(self.size_in, 128)
        self.fc2: nn.Linear = nn.Linear(128, 1)

        self.weightx: nn.Parameter = nn.Parameter(torch.rand(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        # shape of x: [batch_size, #gridpoints in x, #gridpoints in y,3] since (a(x,y),x,y).
        batchsize = x.shape[0]
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i in range(self.num_of_layers - 1):
            weights = self.green[i](y)
            x1 = self.fourreal[i](x, weights)
            shape = x1.shape
            x2 = self.wreal[i](x.view(batchsize, self.size_in, -1))
            x = x1 + x2.view(shape)
            x = F.gelu(x)

        weights = self.green[self.num_of_layers - 1](y)
        x1 = self.fourreal[self.num_of_layers - 1](x, weights)
        x2 = self.wreal[self.num_of_layers - 1](x.view(batchsize, self.size_in, -1))
        # pyre-ignore[61]
        x = x1 + x2.view(shape)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x * self.weightx

"""Fourier Network module."""
import torch
from torch import nn
from torch.nn import functional as F

from dtn_operator.model.fourier_kernel import FourierKernel
from dtn_operator.model.kernel_network import KernelNetwork
from dtn_operator.model.linear_kernel_network import LinearKernelNetwork


def batch_conv_1d(x: torch.Tensor, batch_kernel: torch.Tensor):
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
    ):
        """Inits Fourier network."""
        super().__init__()
        self.size_in, self.modes = size_in, num_of_modes
        self.outsize = size_in * size_in * num_of_modes
        self.kernel_modes = kernel_modes
        self.kernel_in = kernel_in
        self.num_of_layers = num_of_layers
        self.fc0 = nn.Linear(3, self.size_in)
        self.fourreal = nn.ModuleList()
        self.green = nn.ModuleList()
        self.lin_layer_real = nn.ModuleList()
        self.scale = 1 / (size_in * size_in)
        for _ in range(num_of_layers):
            self.fourreal.append(FourierKernel(self.size_in, self.size_in, self.modes))
            self.green.append(
                KernelNetwork(
                    self.kernel_in,
                    self.kernel_modes,
                    self.outsize,
                    self.size_in,
                    self.modes,
                )
            )

            self.lin_layer_real.append(
                LinearKernelNetwork(
                    self.kernel_in,
                    self.kernel_modes,
                    self.size_in,
                    self.size_in,
                    self.modes,
                )
            )

        self.fc1 = nn.Linear(self.size_in, 128)
        self.fc2 = nn.Linear(128, 1)

        self.weightx = nn.Parameter(torch.rand(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Forward method."""
        # shape of x: [batch_size, #gridpoints in x, #gridpoints in y,3] since (a(x,y),x,y).
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i in range(self.num_of_layers - 1):
            weights = self.green[i](y)
            x1 = self.fourreal[i](x, weights)
            shape = x1.shape
            x2 = batch_conv_1d(x, torch.unsqueeze(self.lin_layer_real[i](y), -1))
            x = x1 + x2.view(shape)
            # pylint: disable=not-callable
            x = F.gelu(x)

        weights = self.green[self.num_of_layers - 1](y)
        x1 = self.fourreal[self.num_of_layers - 1](x, weights)
        x2 = batch_conv_1d(
            x, torch.unsqueeze(self.lin_layer_real[self.num_of_layers - 1](y), -1)
        )
        x = x1 + x2.view(shape)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        # pylint: disable=not-callable
        x = F.gelu(x)
        x = self.fc2(x)
        return x * self.weightx

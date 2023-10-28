"""Fourier Kernel file."""
import torch
from torch import nn


class FourierKernel(nn.Module):
    """Fourier Kernel used in fourier network."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        """Inits Fourier Kernel."""
        super(FourierKernel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )

    # Complex multiplication

    @staticmethod
    def _compl_mul1d(input_tensor: torch.Tensor, weights: torch.Tensor):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bjk, bljk -> blk", input_tensor, weights)

    def forward(self, x: torch.Tensor, weights: torch.Tensor):
        """Forward method."""
        batchsize = x.shape[0]
        weights = weights.to(torch.cfloat)
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)  # pylint: disable=not-callable

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self._compl_mul1d(
            x_ft[:, :, : self.modes1], weights
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))  # pylint: disable=not-callable
        return x

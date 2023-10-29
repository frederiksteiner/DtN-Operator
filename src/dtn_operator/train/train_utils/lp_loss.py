"""Lp Loss."""
import torch


class LpLoss:
    """Lp loss which can be used for training."""

    def __init__(
        self, d: int = 2, p: int = 2, size_average: bool = True, reduction: bool = True
    ) -> None:
        """Inits lp loss."""
        super().__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d: int = d
        self.p: int = p
        self.reduction: bool = reduction
        self.size_average: bool = size_average

    def abs(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Abs method."""
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            return torch.sum(all_norms)

        return all_norms

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Relative lp loss."""
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates lp loss."""
        return self.rel(x, y)

"""Lightning module used for training."""
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dtn_operator.common.config import Config
from dtn_operator.common.config import resolve_config
from dtn_operator.model.fourier_network import FourierNetwork
from dtn_operator.train.train_utils.lp_loss import LpLoss

# pylint: disable=arguments-differ, unused-argument
# pyre-ignore-all-errors[14]


class LightningNet(pl.LightningModule):
    """Class to train fourier neural operator net."""

    def __init__(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        """Inits lightning net."""
        super().__init__()
        self.config: Config = resolve_config()
        self.myloss: LpLoss = LpLoss(size_average=False)
        self.train_loader: DataLoader = train_loader
        self.test_loader: DataLoader = test_loader
        self.net: FourierNetwork = get_model_architecture(
            architecture=self.config.training.architecture
        )(
            self.config.network.size_in,
            self.config.network.num_of_modes,
            self.config.network.kernel_in,
            self.config.network.kernel_modes,
            self.config.network.num_of_layers,
        )
        self.training_step_outputs: list[torch.Tensor] = []
        self.test_step_outputs: list[torch.Tensor] = []
        self.val_step_outputs: list[torch.Tensor] = []
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        out = self.net(x, y)
        return out

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.StepLR]]:
        """Configures optimizers for training."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.training.learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.training.step_size,
            gamma=self.config.training.gamma,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, a, y = batch
        out = self(x, a)
        loss = self.myloss(
            out.view(self.config.training.batchsize, -1),
            y.view(self.config.training.batchsize, -1),
        )
        self.log("train_loss", loss)
        self.training_step_outputs.append(loss)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test step."""
        x, a, y = batch
        out = self(x, a)
        loss = self.myloss(
            out.view(self.config.training.batchsize, -1),
            y.view(self.config.training.batchsize, -1),
        ).detach()
        self.log("test_loss", loss)
        self.test_step_outputs.append(loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, a, y = batch
        out = self(x, a)
        loss = self.myloss(
            out.view(self.config.training.batchsize, -1),
            y.view(self.config.training.batchsize, -1),
        ).detach()
        self.log("val_loss", loss)
        self.val_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Epoch end for val epoch."""
        test_loss = torch.mean(torch.stack(self.val_step_outputs), dim=0)
        self.log("mse_test_loss", test_loss)
        self.val_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        """Epoch end for test epoch."""
        test_loss = torch.mean(torch.stack(self.test_step_outputs), dim=0)
        self.log("mse_test_loss", test_loss)
        self.test_step_outputs.clear()

    def on_train_epoch_end(self) -> None:
        """Epoch end for train epoch."""
        train_loss = torch.mean(torch.stack(self.training_step_outputs), dim=0)
        self.log("train-loss-mean", train_loss)
        self.training_step_outputs.clear()

    def train_dataloader(self) -> DataLoader:
        """Training data loader."""
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Validation data loader."""
        return self.test_loader

    def test_dataloader(self) -> DataLoader:
        """Test data loader."""
        return self.test_loader


def get_model_architecture(
    architecture: str = "NewKernelFixed",
) -> type[FourierNetwork]:
    """Returns specific model architecture."""
    if architecture == "NewKernelFixed":
        return FourierNetwork
    raise NotImplementedError

"""Main module."""
import pytorch_lightning as pl
import torch

from dtn_operator.common.config import Config
from dtn_operator.common.config import resolve_config
from dtn_operator.data.data_reader import DataReader
from dtn_operator.train.lightning_net import LightningNet

if __name__ == "__main__":
    NUM_OF_MAT_FILES = 14
    data_reader = DataReader("../data-folder/dtn-data/", ".mat", NUM_OF_MAT_FILES)
    # pyre-ignore[5]
    x_train, y_train, a_train, x_test, y_test, a_test = data_reader.load_data()
    swapind = [2, 0, 1]
    x_train, a_train, x_test, a_test = (
        x_train[:, :, swapind],
        a_train[:, :, swapind],
        x_test[:, :, swapind],
        a_test[:, :, swapind],
    )
    config: Config = resolve_config()
    train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, a_train, y_train),
        batch_size=config.training.batchsize,
        shuffle=True,
        num_workers=23,
    )
    test_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, a_test, y_test),
        batch_size=config.training.batchsize,
        shuffle=False,
        num_workers=23,
    )
    model = LightningNet(train_loader=train_loader, test_loader=test_loader)

    trainer = pl.Trainer(max_epochs=600, enable_progress_bar=True)
    trainer.fit(model)

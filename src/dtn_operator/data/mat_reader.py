"""Mat reader module to read mat files."""
from typing import Optional

import h5py
import numpy as np
import scipy
import torch


class MatReader:
    """Mat reader class."""

    def __init__(
        self,
        file_path: str,
        to_cuda: bool = False,
    ) -> None:
        """Inits mat reader class."""
        self.to_cuda: bool = to_cuda
        self.file_path: str = file_path

        self._data: Optional[dict[str, torch.Tensor]] = None
        self.old_mat: bool = False
        self._load_file()

    @property
    def data(self) -> dict[str, torch.Tensor]:
        """Returns data."""
        if not self._data:
            raise RuntimeError("Data has not been set yet.")
        return self._data

    def _load_file(self) -> None:
        try:
            self._data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except ValueError:
            # pyre-ignore[8]
            self._data = h5py.File(self.file_path)

    def load_file(self, file_path: str) -> None:
        """Loads file."""
        self.file_path = file_path
        self._load_file()

    def read_field(self, field: str) -> torch.Tensor:
        """Reads field from data."""
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        x = torch.from_numpy(x)

        if self.to_cuda:
            x = x.cuda()

        return x

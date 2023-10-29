"""Data reader module."""
from typing import Optional

import torch
from sklearn.model_selection import train_test_split

from dtn_operator.data.mat_reader import MatReader


class DataReader:
    """Data reader class to read mat files."""

    a_data_name: str = "a_data"
    dn_data_name: str = "dn_data"
    b_data_name: str = "u_data"

    def __init__(
        self, file_prefix: str, file_suffix: str, num_of_mat_files: int
    ) -> None:
        """Inits data reader."""
        self.file_prefix: str = file_prefix
        self.file_suffix: str = file_suffix
        self.num_of_mat_files: int = num_of_mat_files
        self.reader: MatReader = MatReader()

    def _get_data_path(self, data_type: str, file_id: Optional[int] = None) -> str:
        if not file_id:
            return f"{self.file_prefix}{data_type}{self.num_of_mat_files}{self.file_suffix}"
        return f"{self.file_prefix}{data_type}{str(file_id).zfill(2)}{self.file_suffix}"

    def _read_data(self, data_type: str, file_id: int) -> torch.Tensor:
        data_path = self._get_data_path(data_type, file_id)
        self.reader.load_file(data_path)
        data = self.reader.read_field(data_type)
        if data_type == self.dn_data_name and data.ndim == 3:
            return data.squeeze()
        return data

    def _read_and_append(
        self, data_type: str, file_id: int, data: torch.Tensor
    ) -> torch.Tensor:
        if data.shape[0] == 0:
            return self._read_data(data_type, file_id)
        return torch.cat((data, self._read_data(data_type, file_id)), 0)

    @staticmethod
    def _get_train_test_idxs(
        num_samples: int, test_size: float = 0.1
    ) -> tuple[list[int], list[int]]:
        return train_test_split([i for i in range(num_samples)], test_size=test_size)

    def load_data(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Loads data."""
        a_data = torch.zeros((0))
        y_data = torch.zeros((0))
        x_data = torch.zeros((0))
        for i in range(self.num_of_mat_files):
            a_data = self._read_and_append(self.a_data_name, i, a_data)
            y_data = self._read_and_append(self.dn_data_name, i, y_data)
            x_data = self._read_and_append(self.b_data_name, i, x_data)
        train_idxs, test_idxs = self._get_train_test_idxs(x_data.shape[0])
        return (
            x_data[train_idxs, :, :],
            y_data[train_idxs, :],
            a_data[train_idxs, :, :],
            x_data[test_idxs, :, :],
            y_data[test_idxs, :],
            a_data[test_idxs, :, :],
        )

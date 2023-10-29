"""Data reader module."""
from typing import Optional

import torch

from dtn_operator.data.mat_reader import MatReader


class DataReader:
    """Data reader class to read mat files."""

    a_data_name: str = "a_data"
    dn_data_name: str = "dn_data"
    b_data_name: str = "uboundary_data"

    def __init__(
        self, file_prefix: str, file_suffix: str, num_of_mat_files: int
    ) -> None:
        """Inits data reader."""
        self.file_prefix: str = file_prefix
        self.file_suffix: str = file_suffix
        self.num_of_mat_files: int = num_of_mat_files
        self.reader: MatReader = MatReader(
            f"{self.file_prefix}uboundary_data{num_of_mat_files}{file_suffix}"
        )

    def _get_data_path(self, data_type: str, file_id: Optional[int] = None) -> str:
        if not file_id:
            return f"{self.file_prefix}{data_type}{self.num_of_mat_files}{self.file_suffix}"
        return f"{self.file_prefix}{data_type}{file_id}{self.file_suffix}"

    def _read_data(self, data_type: str, file_id: int) -> torch.Tensor:
        data_path = self._get_data_path(data_type, file_id)
        self.reader.load_file(data_path)
        data = self.reader.read_field(data_type)
        if data_type == self.dn_data_name and data.ndim == 3:
            return data.squeeze()
        return data

    def _read_and_append(
        self, data_type: str, file_id: int, data: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if not data:
            return self._read_data(data_type, file_id)
        return torch.cat((data, self._read_data(data_type, file_id)), 0)

    def load_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,]:
        """Loads data."""
        a_data = torch.empty()
        y_data = torch.empty()
        x_data = torch.empty()
        for i in range(self.num_of_mat_files):
            a_data = self._read_and_append(self.a_data_name, i, a_data)
            y_data = self._read_and_append(self.dn_data_name, i, y_data)
            x_data = self._read_and_append(self.b_data_name, i, x_data)

        return a_data, x_data, y_data

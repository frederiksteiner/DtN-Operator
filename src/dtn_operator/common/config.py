"""Common config class."""


class NetworkConfig:
    """Network configs."""

    size_in: int = 32
    num_of_layers: int = 4
    num_of_modes: int = 32
    kernel_in: int = 100
    kernel_modes: int = 100


class TrainingConfig:
    """Training configs."""

    batchsize: int = 20
    learning_rate: float = 0.001
    gamma: float = 0.5
    step_size: int = 75
    architecture: str = "NewKernelFixed"


class Config:
    """Configs."""

    network: NetworkConfig = NetworkConfig()
    training: TrainingConfig = TrainingConfig()


def resolve_config() -> Config:
    """Returns configs."""
    config = Config()
    return config

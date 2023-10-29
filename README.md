## Setup

### Install Python

If not already installed, install Python. The recommended way is to use [pyenv](https://github.com/pyenv/pyenv), which allows multiple parallel Python installations which can be automatically selected per project you're working on.

```shell
# Install Python if necessary
pyenv install 3.10.13
pyenv local 3.10.13
```

### Install Poetry

If not already installed, get Poetry according to <https://python-poetry.org/docs/#installation>.
If your are new to Poetry, you may find <https://python-poetry.org/docs/basic-usage/> interesting.

### Setup Environment

```shell
# Create venv and install all dependencies
poetry env use 3.10
poetry install

# add pre-commit hooks
poetry run pre-commit install
```

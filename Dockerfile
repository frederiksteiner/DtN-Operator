FROM python:3.10

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_CACHE_DIR="/tmp/.cache/pypoetry" \
  POETRY_HOME="/etc/poetry" \
  PATH="/etc/poetry/bin:$PATH"

# System deps:
RUN apt-get update && apt-get install -y \
  # needed due to numpy error numpy.distutils.system_info.NotFoundError: No BLAS/LAPACK libraries found
  libblas-dev liblapack-dev \
  # needed for scipy
  gfortran \
  && \
  curl -sSL https://install.python-poetry.org | python3 -

# Copy only requirements to cache them in docker layer
WORKDIR /usr/src/app
COPY pyproject.toml poetry.lock ./

# Project initialization:
RUN poetry config --local virtualenvs.create false \
  && poetry install --only main --no-interaction --no-ansi --no-root \
  && chmod g+r poetry.toml

# Creating directories, and files for a project:
COPY src/ src/
COPY README.md README.md

# Install src directory as python package:
RUN poetry install --only main --no-interaction --no-ansi

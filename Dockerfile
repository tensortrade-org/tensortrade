FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -yq --assume-yes --no-install-recommends \
        git \
        libgl1-mesa-glx \
        rsync \
        wget \
        zip \
        build-essential && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    wget -O config.sub 'https://git.savannah.gnu.org/cgit/config.git/plain/config.sub' && \
    wget -O config.guess 'https://git.savannah.gnu.org/cgit/config.git/plain/config.guess' && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* ta-lib-0.4.0-src.tar.gz ta-lib/

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY tensortrade/ ./tensortrade/
COPY examples/ ./examples/
COPY docs/ ./docs/
COPY README.md ./

# Install dependencies using uv
RUN uv sync --frozen --all-extras

# Install the package in development mode
RUN uv pip install -e ".[examples]"

# Set the entrypoint to use uv's virtual environment
ENTRYPOINT ["uv", "run"]
CMD ["python"]

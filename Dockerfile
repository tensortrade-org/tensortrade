FROM python:3.12-slim

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -yq --assume-yes --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        build-essential \
        ca-certificates \
        rsync \
        wget \
        zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install TA-Lib (with ARM64 support)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    wget -O config.sub 'https://git.savannah.gnu.org/cgit/config.git/plain/config.sub' && \
    wget -O config.guess 'https://git.savannah.gnu.org/cgit/config.git/plain/config.guess' && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# Copy project files
COPY . ./

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r examples/requirements.txt && \
    pip install --no-cache-dir -e ".[docs,tests]"

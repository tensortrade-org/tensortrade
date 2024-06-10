FROM tensorflow/tensorflow:2.7.0-gpu

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

COPY . ./

SHELL ["/bin/bash", "-c"]

RUN rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    apt-get install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -yq --assume-yes --no-install-recommends git \
                                                             libgl1-mesa-glx \
                                                             python3-pip \
                                                             python3-dev \
                                                             python3-setuptools \
                                                             rsync \
                                                             zip && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r "./examples/requirements.txt" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* ta-lib-0.4.0-src.tar.gz

# Faster compilation for tests
RUN pip3 install --no-cache-dir -e ".[docs,tests]"


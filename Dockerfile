FROM tensorflow/tensorflow:2.8.0-gpu

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

COPY . ./

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -yq --assume-yes --no-install-recommends cmake \
                                                             libgl1-mesa-glx \
                                                             libopenmpi-dev \
                                                             python3-pip \
                                                             python3-dev \
                                                             python3-setuptools \
                                                             rsync \
                                                             wget \
                                                             zip \
                                                             zlib1g-devgit && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -e ".[ray,docs,tests]" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* ta-lib-0.4.0-src.tar.gz


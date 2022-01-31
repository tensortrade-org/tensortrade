FROM tensorflow/tensorflow:2.4.2-gpu

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
    apt-get install -yq --assume-yes --no-install-recommends git \
                                                             libgl1-mesa-glx \
                                                             python3-pip \
                                                             python3-dev \
                                                             python3-setuptools \
                                                             rsync \
                                                             wget \
                                                             zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -e ".[docs,tests]"
RUN pip3 install --no-cache-dir -r "./examples/requirements.txt"


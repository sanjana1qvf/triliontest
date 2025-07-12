#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgcc-s1 \
    libc6-dev \
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-dev \
    python3-venv \
    python3-setuptools \
    python3-wheel \
    curl \
    wget

# Install yt-dlp
curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp
chmod a+rx /usr/local/bin/yt-dlp

# Install Node.js dependencies
npm install

# Install Python dependencies
pip3 install --upgrade pip
pip3 install -r requirements-render.txt

echo "Build completed successfully!" 
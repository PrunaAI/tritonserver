FROM nvcr.io/nvidia/tritonserver:23.12-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && \
    apt-get install -y wget curl git vim sudo cmake build-essential \
    libssl-dev libffi-dev python3-dev python3-venv python3-pip libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install packaging psutil pexpect ipywidgets jupyterlab ipykernel \
    librosa soundfile

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Pruna
RUN pip3 install pruna[gpu]==0.1.2 --extra-index-url https://prunaai.pythonanywhere.com/
# If required for your model, install the full version of Pruna with
RUN pip3 install pruna[full]==0.1.2 --extra-index-url https://prunaai.pythonanywhere.com

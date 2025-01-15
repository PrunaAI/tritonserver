# Triton Server with Pruna Integration

This repository demonstrates how to optimize and deploy machine learning models using Pruna with NVIDIA's Triton Inference Server.

## Overview

This project showcases the integration of Pruna's model optimization capabilities with Triton Server for efficient model serving. It includes an example of deploying a Stable Diffusion model that has been optimized using Pruna's step caching compiler.

## Features

- Integration of Pruna optimization with Triton Server
- Example implementation of Stable Diffusion model serving
- Step caching optimization for improved inference performance
- Easy-to-follow setup and deployment instructions

## Getting Started

1. Clone this repository
2. Install tritonclient[grpc] `pip install tritonclient[grpc]`
3. Build Docker image `docker build -t tritonserver_pruna .`
4. Run Docker image `docker run --rm --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
   -v "path/to/your/model_repository:/models" \
   tritonserver_pruna tritonserver --model-repository=/models`
   Don't forget to replace `path/to/your/model_repository` with the actual path to your model repository.
5. Put your token in `model_repository/stable_diffusion/1/model.py`
6. Call the model using the following example script `python3 client.py`

## Model Configuration

The repository includes a sample model configuration that demonstrates, this is all in `model_repository/stable_diffusion/1/model.py`:
- Loading a Stable Diffusion pipeline
- Applying Pruna's step caching optimization
- Configuring the model for Triton Server deployment
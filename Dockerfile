FROM python:3.10-slim-buster

# Update package lists
RUN apt-get update

# Install libstdc++ (if needed)
RUN apt-get install -y libstdc++6

## Force install of bitsandbyes non-CUDA version
RUN pip install --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl'

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

## Install g++ build-essentials
RUN apt-get update && apt-get install -y g++ build-essential


ENV TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache"
ENV BNB_CUDA_OFFLOAD=0

# Create and set permissions for cache directories
RUN mkdir -p /.cache && chmod -R 777 /.cache

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Command to run the app
CMD ["python", "app.py"]
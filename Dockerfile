FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git ffmpeg wget python3-pip python3-dev python3-opencv

# Work directory inside container
WORKDIR /workspace

# Copy repo into container
COPY . /workspace

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Set python to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Default entrypoint (can be overridden)
ENTRYPOINT ["python3", "upscale_job.py"]

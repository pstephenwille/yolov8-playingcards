FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
LABEL authors="stephen.wille"

WORKDIR /home
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    wget \
    python3-pip \
    python3.10-venv

RUN pip install ultralytics
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install jupyterlab
RUN pip install random2 \
                opendatasets \
                opencv-contrib-python-headless \
                pandas \
                seaborn \
                matplotlib \
                torch \
                ipykernel \
                virtualenv

EXPOSE 8888

ENTRYPOINT ["/bin/bash", "./docker-entry-point.bash"]

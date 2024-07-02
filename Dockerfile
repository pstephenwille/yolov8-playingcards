FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
LABEL authors="stephen.wille"

WORKDIR /home
COPY ./requiremnets.txt .
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    wget \
    curl \
    python3-pip \
    python3.10-venv \
    libusb-1.0-0

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
                virtualenv \
                tf_keras \
                sng4onnx \
                onnx_graphsurgeon \
                onnx \
                onnx2tf \
                onnxslim \
                tflite_support \
                onnxruntime \
                tensorflow \
                tflite_support \
                tensorflow-io-gcs-filesystem \
                numpy

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

EXPOSE 8888

ENTRYPOINT ["/bin/bash", "./docker-entry-point.bash"]

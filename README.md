# Playing Card Detection with YOLOv8

### Guide:
* [Having Fun with YOLOv8: How Good Your Model in Detecting Playing Card?](https://medium.com/@sdwiulfah/having-fun-with-yolov8-how-good-your-model-in-detecting-playing-card-a468a02e4775).


### Steps

1. install nvidia-container-toolkit https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
2. build the dockerfile: `docker build -t cuda-12-1:latest .`
2. ``` 
    docker run \
    -p 8888:8888 \
    -v $PWD:/home \
    --rm --runtime=nvidia \
    --gpus all \
    --ipc=host \
    cuda-12-1:latest
    ```
1. verify container is running on GPU:
    ``` 
    python3
    import torch
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
     ```
1. check `localhost:8888`
2. run the notebook


### todo
1. increase dataset to obtain >90% confidence in all cards


# Notes
>ultralytics missing deps:
'tf_keras', 'sng4onnx>=1.0.1', 'onnx_graphsurgeon>=0.3.26', 'onnx>=1.12.0', 'onnx2tf>1.17.5,<=1.22.3', 'onnxslim>=0.1.31', 'tflite_support', 'onnxruntime'

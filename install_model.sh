#!/bin/bash

pip install openvino_dev

if [ -d "model" ]
then
    echo "Model directory already exists"
else
    mkdir model
fi

omz_downloader --name face-detection-adas-0001 -o model
omz_downloader --name facial-landmarks-35-adas-0002 -o model
omz_downloader --name head-pose-estimation-adas-0001 -o model
omz_downloader --name gaze-estimation-adas-0002 -o model

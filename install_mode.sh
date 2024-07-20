#!/bin/bash

pip install -r requirements.txt

if [ -d "model" ]
then
    echo "Model directory already exists"
else
    mkdir model
fi

omz_downloader --name face-detection-adas-0001 -o model

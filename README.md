# age-gender-openvino

## Requirements

- Python 3.12.3
- pip 24.1.2

## Installation

```bash
pip install -r requirements.txt
mkdir model
omz_downloader --name face-detection-adas-0001 -o model
omz_downloader --name facial-landmarks-35-adas-0002 -o model
omz_downloader --name head-pose-estimation-adas-0001 -o model
omz_downloader --name gaze-estimation-adas-0002 -o model
```
or (Linux or Mac only)
```bash
./install_model.sh
```

## Usage

```bash
python main.py \
  --device CPU \
  --model FP16 \
  --image image.png
```

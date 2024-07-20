# age-gender-openvino

## Requirements

- Python 3.12.3
- pip 24.1.2

## Installation

```bash
pip install -r requirements.txt
mkdir model
omz_downloader --name face-detection-adas-0001 -o model
```
or
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

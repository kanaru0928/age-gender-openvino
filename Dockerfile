FROM python:3.12-slim AS base

RUN apt-get update && apt-get -y install libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -U pip && pip install -r requirements.txt

FROM base AS final

COPY . .

ENTRYPOINT ["python", "main.py"]

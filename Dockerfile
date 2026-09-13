# Reproduction image: regenerates every table and figure from the committed caches, no API calls.
#   docker build -t longtail-txcat .
#   docker run --rm -v "$PWD/results:/app/results" longtail-txcat
# Data files under data/raw and data/processed are not in the image; mount them or run
# `python data/download.py mcc_codes dc oklahoma` inside the container first.
FROM python:3.12-slim

ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=0

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir --no-deps -e .

CMD ["python", "reproduce.py", "--config", "configs/dc.yaml", "--seeds", "42", "43", "44"]

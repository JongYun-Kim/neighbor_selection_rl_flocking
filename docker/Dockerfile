# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# system
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates curl wget git build-essential \
    sudo nano x11-apps \
    iproute2 iputils-ping net-tools dnsutils \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# python
RUN add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-distutils python3.9-venv \
 && rm -rf /var/lib/apt/lists/*

# fix python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
 && /usr/bin/python3.9 /tmp/get-pip.py \
 && rm -f /tmp/get-pip.py \
 # enable python and python3 in cli
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 # get py3.9 at the entrypoint
 && ln -sf /usr/local/bin/pip3.9 /usr/local/bin/pip \
 && ln -sf /usr/local/bin/pip3.9 /usr/local/bin/pip3

# double check!
RUN python -V && python3 -V && pip -V

# get pip update-to-dated
RUN python -m pip install --upgrade pip setuptools wheel

# turn on your torch; let there be light, there's light! Thanks Gradient!!
RUN python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu113 \
    torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113

# get the apples
RUN cat > /tmp/requirements.txt <<'REQ'
absl-py==2.1.0
aiosignal==1.3.1
annotated-types==0.7.0
attrs==23.2.0
certifi==2024.2.2
charset-normalizer==3.3.2
click==8.0.4
cloudpickle==3.0.0
colorama==0.4.6
contourpy==1.2.1
cycler==0.12.1
distlib==0.3.8
dm-tree==0.1.8
filelock==3.14.0
fonttools==4.52.4
frozenlist==1.4.1
grpcio==1.64.0
gym==0.23.1
gym-notices==0.0.8
idna==3.7
imageio==2.34.1
importlib_metadata==7.1.0
importlib_resources==6.4.0
jsonschema==4.22.0
jsonschema-specifications==2023.12.1
kiwisolver==1.4.5
lazy_loader==0.4
lz4==4.3.3
Markdown==3.6
MarkupSafe==2.1.5
matplotlib==3.9.0
msgpack==1.0.8
networkx==3.2.1
numpy==1.23.4
packaging==24.0
pandas==2.2.2
pillow==10.3.0
pipdeptree==2.22.0
platformdirs==4.2.2
protobuf==4.25.3
pydantic==1.10.13
pydantic_core==2.18.4
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
PyYAML==6.0.1
ray==2.1.0
referencing==0.35.1
requests==2.32.3
rpds-py==0.18.1
scikit-image==0.22.0
scipy==1.13.1
six==1.16.0
tabulate==0.9.0
tensorboard==2.17.0
tensorboard-data-server==0.7.2
tensorboardX==2.6.2.2
tifffile==2024.5.22
tqdm==4.66.4
typing_extensions==4.12.0
tzdata==2024.1
urllib3==2.2.1
virtualenv==20.26.2
Werkzeug==3.0.3
zipp==3.19.0
REQ
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# connect to the host
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} flocking || true \
 && useradd -m -s /bin/bash -u ${UID} -g ${GID} flocking \
 && echo 'flocking:neighbor' | chpasswd

RUN mkdir -p /workspace && chown -R ${UID}:${GID} /workspace
WORKDIR /workspace
USER flocking

# set cuda params
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# check at the cmd
CMD ["bash", "-lc", "python -V && python3 -V && pip -V && python -c 'import torch, torchvision, torchaudio, ray, gym, numpy as np; print(\"torch\", torch.__version__); print(\"torchvision\", torchvision.__version__); print(\"torchaudio\", torchaudio.__version__); print(\"ray\", ray.__version__); print(\"gym\", gym.__version__); print(\"numpy\", np.__version__)' || true; exec bash"]

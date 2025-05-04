#! /bin/bash

sudo apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    curl \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    build-essential \
    libssl-dev \
    libffi-dev \
    git \
    golang \
    wget \
    curl \
    && apt-get clean

git clone https://github.com/what-the-fawk/blockchain-benchmarks && cd blockchain-benchmarks
git checkout checkpoint1 & cd ..

VIRTUAL_ENV=/opt/venv
python3.10 -m venv $VIRTUAL_ENV
PATH="$VIRTUAL_ENV/bin:$PATH"
pip3 install --no-cache-dir -r blockchain-benchmarks/ml/requirements.txt

sudo curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs

npm install --only=prod @hyperledger/caliper-cli && \
npx caliper bind --caliper-bind-sut fabric:2.4

sudo apt-get update && apt-get install -y ca-certificates curl && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME:-$VERSION_CODENAME}) stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update -y && \
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo systemctl start docker
sudo systemctl enable docker
# 使用するベースイメージ
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 環境変数を設定
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# 必要なパッケージのインストール
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    gcc \
    curl \
    wget \
    sudo \
    pciutils \
    python3-all-dev \
    python-is-python3 \
    python3-pip \
    ffmpeg \
    portaudio19-dev \
    libsndfile1 \
    locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

RUN apt-get install -y git

# 必要なライブラリのインストール
RUN pip install -U pip && \
    pip install setuptools-rust && \
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install git+https://github.com/m-bain/whisperx.git && \
    git clone https://github.com/m-bain/whisperX.git /whisperx && \
    pip3 install -e /whisperx 

WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# VSCode特有の設定

ENV USERNAME=vscode
RUN echo "root:root" | chpasswd \
    && adduser --disabled-password --gecos "" "${USERNAME}" \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
ARG WKDIR=/home/${USERNAME}/workdir
WORKDIR ${WKDIR}
RUN sudo chown ${USERNAME}:${USERNAME} ${WKDIR}
RUN echo 'export QT_X11_NO_MITSHM=1' >> ${HOME}/.bashrc
# RUN echo 'sudo chmod 776 /dev/video*' >> ${HOME}/.bashrc


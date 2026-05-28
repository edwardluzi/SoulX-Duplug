FROM nvidia/cuda:12.4.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    git wget curl ca-certificates \
    libsndfile1 ffmpeg sox \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# 升级 pip
RUN python -m pip install --upgrade pip setuptools wheel

# 安装 PyTorch (CUDA 12.4)
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 分步安装核心依赖（避免同时解析超时）
RUN pip install \
    transformers==4.52.1 \
    accelerate==1.9.0 \
    peft==0.16.0 \
    omegaconf==2.3.0 \
    pytorch-lightning==2.5.2

RUN pip install \
    fastapi==0.115.3 \
    uvicorn==0.32.0 \
    websockets==12.0 \
    python-multipart==0.0.12

RUN pip install \
    soundfile==0.12.1 \
    librosa==0.10.2 \
    soxr==0.5.0.post1

RUN pip install \
    modelscope==1.28.2 \
    funasr==1.2.6

# PAI-EAS 在 GPU 环境下可以安装这些 CUDA 包
RUN pip install \
    deepspeed==0.18.0 \
    xformers==0.0.29.post2 \
    triton==3.2.0 || true

# 文本处理依赖
RUN pip install \
    pypinyin==0.55.0 \
    zhon==2.1.1 \
    cn2an==0.5.23 \
    addict==2.4.0 \
    datasets==4.8.5 \
    multiprocess==0.70.19

# WeTextProcessing（含 pynini）在 Linux + CUDA 镜像上通常可以编译成功
RUN pip install WeTextProcessing==1.0.3 || echo "WeTextProcessing install skipped"

# 创建工作目录
WORKDIR /app

# 复制项目代码（不包含模型权重）
COPY config/ ./config/
COPY model/ ./model/
COPY service/ ./service/
COPY utils/ ./utils/
COPY server.py ./
COPY run.sh ./

# 修改 config 适配云端 GPU
RUN sed -i 's/device: cpu/device: cuda/' config/config.yaml && \
    sed -i 's/precision: fp32/precision: bf16/' config/config.yaml && \
    sed -i 's/model_name: paraformer/model_name: sensevoice/' config/config.yaml

# 暴露 WebSocket 端口
EXPOSE 8000

# PAI-EAS 通过该命令启动服务
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

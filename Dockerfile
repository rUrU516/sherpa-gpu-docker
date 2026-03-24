# sherpa-onnx GPU Docker Image
# 基于 NVIDIA CUDA 镜像，包含完整的 GPU 支持

FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装 Python 3.10 和其他依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# 设置 CUDA 环境变量
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 从源码编译 sherpa-onnx，启用 GPU 和 Python 绑定
RUN git clone --depth 1 https://github.com/k2-fsa/sherpa-onnx.git /tmp/sherpa-onnx && \
    cd /tmp/sherpa-onnx && \
    mkdir build && cd build && \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DSHERPA_ONNX_ENABLE_GPU=ON \
        -DBUILD_PYTHON=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} \
        .. && \
    make -j$(nproc) install && \
    rm -rf /tmp/sherpa-onnx

# 设置 Python 路径
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages:${PYTHONPATH}

# 安装其他依赖
RUN pip3 install --no-cache-dir \
    numpy \
    oss2

# 创建目录
RUN mkdir -p /app/models/funasr-nano-int8 \
             /app/models/vad \
             /app/wavs \
             /app/output

# 复制脚本
COPY scripts/ /app/scripts/

WORKDIR /app
CMD ["/bin/bash"]
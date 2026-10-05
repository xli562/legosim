# =============================================================================
# Chiplet Simulator 镜像 (Ubuntu 18.04 + CUDA 11.3 + ZeroMQ)
#
# 相比 apt-cmake 版本的关键改动：
#   - Python 保持 18.04 自带的 3.6（不动）
#   - pip 升级到 21.3.1（Python 3.6 支持的最后版本，能拉 cmake 3.27 wheel）
#   - cmake 改用 pip install 的 3.27.9（对齐宿主机）
# =============================================================================
FROM nvidia/cuda:11.3.1-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# 1) apt 源 -> 清华
RUN rm -f /etc/apt/sources.list && \
    echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse'         >  /etc/apt/sources.list && \
    echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse' >> /etc/apt/sources.list && \
    cat /etc/apt/sources.list

# 2) 装包（不装 apt cmake，其他照旧）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    make git pkg-config ca-certificates wget curl gnupg \
    vim nano less unzip tree \
    locales tzdata \
    software-properties-common \
    scons m4 \
    bison flex doxygen graphviz \
    libsqlite3-dev zlib1g-dev libbz2-dev libboost-all-dev \
    python3-dev libprotobuf-dev protobuf-compiler libgoogle-perftools-dev \
    python2.7 python3 python3-pip python3-setuptools \
    libzmq3-dev libzmq5 \
    xutils-dev \
    libgl1-mesa-dev libglu1-mesa-dev \
    netcat iputils-ping iproute2 net-tools tmux htop \
 && rm -rf /var/lib/apt/lists/*

# 3) locale / timezone
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8 && \
    ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 4) python 软链（gem5 老脚本依赖 python 指向 python2.7）
RUN ln -sf /usr/bin/python2.7 /usr/bin/python

# 5) 升级 pip + 装 CMake 3.27.9（对齐宿主机版本）
#    Python 3.6 支持的最后 pip 版本是 21.3.1，正好能拉 cmake 3.27 预编译 wheel
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

RUN python3 -m pip install --no-cache-dir 'pip==21.3.1' && \
    python3 -m pip install --no-cache-dir 'cmake==3.27.9' && \
    which cmake && \
    cmake --version && \
    pip3 --version && \
    python3 --version

# 6) cppzmq（头文件，interchiplet 需要）
RUN set -eux; \
    cd /tmp; \
    (git clone --depth=1 --branch v4.10.0 https://gitee.com/mirrors_zeromq/cppzmq.git || \
     git clone --depth=1 --branch v4.10.0 https://gitclone.com/github.com/zeromq/cppzmq.git || \
     git clone --depth=1 --branch v4.10.0 https://github.com/zeromq/cppzmq.git); \
    install -m 0644 /tmp/cppzmq/zmq.hpp       /usr/local/include/zmq.hpp; \
    install -m 0644 /tmp/cppzmq/zmq_addon.hpp /usr/local/include/zmq_addon.hpp; \
    rm -rf /tmp/cppzmq; \
    ls -l /usr/local/include/zmq.hpp /usr/local/include/zmq_addon.hpp

# 7) 环境变量
ENV CUDA_INSTALL_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH}

# 8) 工作目录
WORKDIR /workspace
RUN mkdir -p /workspace/sim
CMD ["/bin/bash"]

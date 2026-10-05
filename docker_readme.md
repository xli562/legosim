# Chiplet Simulator Docker 镜像使用说明

基于 Ubuntu 18.04 + CUDA 11.3 + ZeroMQ 的 LegoSim 仿真器容器化方案。
镜像本身**不打包仿真器代码**，代码通过 `-v` 在运行时挂载进容器。

## 目录

1. [镜像清单](#1-镜像清单)
2. [构建镜像](#2-构建镜像)
3. [验证镜像](#3-验证镜像)
4. [导出 / 导入镜像](#4-导出--导入镜像)
5. [首次进入容器：编译所有仿真器](#5-首次进入容器编译所有仿真器)
6. [单容器模拟多机分布式运行](#6-单容器模拟多机分布式运行)
7. [多容器真分布式运行](#7-多容器真分布式运行)
8. [常见问题](#8-常见问题)

---

## 1. 镜像清单

| 组件 | 版本 | 来源 |
|------|------|------|
| Ubuntu | 18.04 | `nvidia/cuda:11.3.1-devel-ubuntu18.04` |
| CUDA Toolkit | 11.3.1 | 基础镜像自带 |
| GCC / G++ | 7.5.0 | apt |
| CMake | 3.27.9 | pip install（对齐宿主机） |
| Python 3 | 3.6.9 | apt |
| pip3 | 21.3.1 | pip 自升级 |
| Python 2 | 2.7.17 | apt（gem5 兼容） |
| libzmq | 4.2.5 | apt |
| cppzmq | v4.10.0 | 从 gitee / gitclone 镜像拉头文件 |
| Boost | 1.65 | apt（libboost-all-dev） |
| GPGPUSim 构建依赖 | bison / flex / doxygen / makedepend / libGL | apt |

---

## 2. 构建镜像

**前提**：在一台装有 Docker 的机器上，当前目录存放本 Dockerfile。

```bash
cd /path/to/dockerfile_directory

docker build --network=host -t chiplet-sim:u1804 -f Dockerfile .
```

关键参数：
- `--network=host`：让 build 阶段的容器共享宿主机网络，apt 清华源、pip 清华源都能直连。**不需要**配 `--build-arg HTTP_PROXY`。
- `-t chiplet-sim:u1804`：镜像 tag，可自定义。

**耗时**：首次构建 15–25 分钟（基础镜像 3 GB + apt 约 600 MB + pip cmake wheel 26 MB）。
**最终镜像大小**：约 5 GB。

构建成功标志：最后一行输出 `Successfully tagged chiplet-sim:u1804`。

```bash
# 查看镜像
docker images chiplet-sim
```

---

## 3. 验证镜像

```bash
docker run --rm chiplet-sim:u1804 bash -lc '
  echo "== OS ==";        cat /etc/lsb-release | grep DESCRIPTION
  echo "== gcc ==";       gcc --version | head -1
  echo "== nvcc ==";      nvcc --version | tail -1
  echo "== cmake ==";     cmake --version | head -1
  echo "== python3 ==";   python3 --version
  echo "== pip3 ==";      pip3 --version
  echo "== makedepend =="; which makedepend
  echo "== libGL ==";     ls /usr/lib/x86_64-linux-gnu/libGL.so
  echo "== libzmq ==";    dpkg -l libzmq3-dev 2>/dev/null | awk "/^ii/{print \$3}"
  echo "== zmq.hpp ==";   ls /usr/local/include/zmq.hpp
'
```

期望输出：
```
== OS ==
DISTRIB_DESCRIPTION="Ubuntu 18.04.6 LTS"
== gcc ==
gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
== nvcc ==
Build cuda_11.3.r11.3/compiler.29920130_0
== cmake ==
cmake version 3.27.9
== python3 ==
Python 3.6.9
== pip3 ==
pip 21.3.1 from /usr/local/lib/python3.6/dist-packages/pip (python 3.6)
== makedepend ==
/usr/bin/makedepend
== libGL ==
/usr/lib/x86_64-linux-gnu/libGL.so
== libzmq ==
4.2.5-1ubuntu0.2
== zmq.hpp ==
/usr/local/include/zmq.hpp
```

---

## 4. 导出 / 导入镜像

### 导出为 tar（方便拷到无网机器）

```bash
docker save -o chiplet-sim-u1804.tar chiplet-sim:u1804
du -h chiplet-sim-u1804.tar
# 约 5.0 GB
```

### 拷到目标机器 + 导入

```bash
# 本机 -> 目标机
scp chiplet-sim-u1804.tar user@target:~/

# 目标机上
docker load -i chiplet-sim-u1804.tar
docker images | grep chiplet-sim
```

---

## 5. 首次进入容器：编译所有仿真器

> 仿真器代码假定放在宿主机 `~/Chiplet_Heterogeneous_newVersion`
> （请按实际路径替换）。这个目录**必须已经 `git submodule update` 拉好了所有子模块**，即已经完成了git submodule init和git submodule update。
> （snipersim / gpgpu-sim / popnet_chiplet / gem5 / interchiplet/thirdparty/*）。

### 5.1 启动容器

```bash
docker run -it --rm \
    --name chiplet \
    --network host \
    -e http_proxy=http://127.0.0.1:7890 \
    -e https_proxy=http://127.0.0.1:7890 \
    -v ~/Chiplet_Heterogeneous_newVersion:/workspace/sim \
    -w /workspace/sim \
    chiplet-sim:u1804 bash
```

关键参数：
- `--network host`：共享宿主网络，让容器内的 ZeroMQ 端口可直接被宿主机/其他机器访问
- `-e http_proxy / https_proxy`：**如果你本机有代理**，sniper 的 `make` 会从 GitHub 拉 mbuild/xed、从 snipersim.org 下 pin tool，需要走代理。把 `127.0.0.1:7890` 改成你自己的代理端口。**如果不需要代理**，去掉这两行。
- `-v`：仿真器代码挂载到容器内 `/workspace/sim`
- `-w`：容器启动时的工作目录

### 5.2 容器内：一次性配置

```bash
# 让 git 不报 dubious ownership（宿主机普通用户的仓库在容器 root 下会报错）
git config --global --add safe.directory '*'

# 让 wget 走代理（sniper Makefile 里用的是 wget，不读 $https_proxy），如果不需要代理，则不需要这几行
cat > /root/.wgetrc <<'EOF'
use_proxy = on
http_proxy = http://127.0.0.1:7890
https_proxy = http://127.0.0.1:7890
check_certificate = off
EOF

# 让 git 走代理，如果不需要代理，则不需要这几行
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# source 仿真器环境
source setup_env.sh
# 期望输出末尾: setup_environment succeeded
```

> 如果不需要代理，上面 wget / git 代理配置跳过。

### 5.3 容器内：打 patch（首次）

```bash
./apply_patch.sh
# 首次应当无报错。若报 "already exists" 说明 patch 已应用过，忽略即可。
```

### 5.4 容器内：依次编译 4 个仿真器

```bash
# 1) sniper
#    首次会从 snipersim.org 下 pinplay tar (~44MB)、从 GitHub 拉 mbuild/xed
cd $SIMULATOR_ROOT/snipersim
make -j4

ls run-sniper      # 验证产物

# 2) gpgpu-sim
cd $SIMULATOR_ROOT/gpgpu-sim
source setup_environment    # 必须 source，会 export GPGPUSIM_CONFIG
make -j4

ls lib/$GPGPUSIM_CONFIG/libcudart.so   # 验证产物

# 3) popnet
cd $SIMULATOR_ROOT/popnet_chiplet
mkdir build
cd build
cmake ..
make -j4

ls popnet

# 4) interchiplet
cd $SIMULATOR_ROOT/interchiplet
mkdir build
cd build
cmake ..
make

ls ../bin/          # 期望: interchiplet  interchiplet_distributed
ls ../lib/          # 期望: libinterchiplet_c.a
```

### 5.5 容器内：编 matmul benchmark

```bash
cd $SIMULATOR_ROOT/benchmark/matmul_test
make

ls bin/    # 期望: matmul_c  matmul_cu
```

### 5.6 容器内：跑原版 matmul（验证仿真器链路）

```bash
cd $SIMULATOR_ROOT/benchmark/matmul_test

# 清理上次运行产物（如果有）
rm -rf proc_r* bench.txt delayInfo.txt buffer* *.log

$SIMULATOR_ROOT/interchiplet/bin/interchiplet ./matmul.yml
```

成功标志：屏幕滚动后自动退出，当前目录产生 `proc_r1_p1_t0`..`t3` 子目录，有 `bench.txt` / `delayInfo.txt`。

---

## 6. 单容器模拟多机分布式运行

> 在同一容器内启动 coordinator + worker 两个进程，通过 `127.0.0.1` 互通，验证 ZeroMQ 代码逻辑。

### 6.1 容器内运行

在容器内（已经 source 好 setup_env.sh 和 gpgpu-sim/setup_environment）：

```bash
cd $SIMULATOR_ROOT/benchmark/matmul_test

# 清理残留
rm -rf machine_* proc_r* bench.txt delayInfo.txt buffer*

# 启动分布式仿真
bash $SIMULATOR_ROOT/run_distributed.sh ./matmul_distributed.yml
```

脚本自动做：
1. 创建 `machine_0/` 和 `machine_1/` 工作目录
2. 在 `machine_0/` 启动 coordinator（绑 127.0.0.1:15000）
3. 2 秒后在 `machine_1/` 启动 worker
4. 等所有进程结束

### 6.2 产物

```bash
cd $SIMULATOR_ROOT/benchmark/matmul_test

# 各机器日志
cat machine_0/coordinator.log | tail -30
cat machine_1/worker_1.log | tail -30

# 同步耗时统计（项目核心数据）
cat machine_0/sync_time_machine_0.csv
cat machine_1/sync_time_machine_1.csv

# 通信基准
cat machine_0/bench.txt
cat machine_0/delayInfo.txt
```

---

## 7. 多容器真分布式运行

> 在**同一宿主机**起 2 个容器模拟两台机器，通过 docker bridge 网络用容器名互相寻址。
> 这是真实跨机器部署前的验证环节。

### 7.1 宿主机准备（一次性）

```bash
# 创建 docker bridge 网络
docker network create chiplet-net 2>/dev/null || true

# 生成多容器版 YAML（把 127.0.0.1 改成容器名）
cd ~/Chiplet_Heterogeneous_newVersion/benchmark/matmul

cp matmul_distributed.yml matmul_multi_container.yml

# YAML 里有两处 127.0.0.1：machine 0 改成 coord，machine 1 改成 worker1
awk 'BEGIN{cnt=0} /host:/ && /127\.0\.0\.1/ {
    cnt++;
    if (cnt==1) sub(/127\.0\.0\.1/, "coord");
    else        sub(/127\.0\.0\.1/, "worker1");
} {print}' matmul_distributed.yml > matmul_multi_container.yml

# 验证
grep "host:" matmul_multi_container.yml
# 期望输出两行，一行 coord，一行 worker1
```

### 7.2 终端 A：启动 coordinator 容器

```bash
docker run -it --rm \
    --name coord --hostname coord \
    --network chiplet-net \
    -e http_proxy=http://127.0.0.1:7890 \
    -e https_proxy=http://127.0.0.1:7890 \
    -v ~/Chiplet_Heterogeneous_newVersion:/workspace/sim \
    -w /workspace/sim/benchmark/matmul \
    chiplet-sim:u1804 bash
```

容器内：
```bash
source /workspace/sim/setup_env.sh
source /workspace/sim/gpgpu-sim/setup_environment

# 清理上次残留
rm -rf proc_r* bench.txt delayInfo.txt buffer* *.log sync_time*.csv

# 启动 coordinator
$SIMULATOR_ROOT/interchiplet/bin/interchiplet_distributed \
    --role coordinator --machine-id 0 \
    ./matmul_multi_container.yml 2>&1 | tee coordinator.log
```

Coordinator 启动后会 hang 打印：
```
[info] Coordinator ROUTER bound to tcp://*:15000
[info] Coordinator waiting for 1 workers to connect...
```

**这是正常的**，它在等 worker。保持此终端不要关闭。

### 7.3 终端 B：启动 worker 容器

**另开一个终端**：

```bash
docker run -it --rm \
    --name worker1 --hostname worker1 \
    --network chiplet-net \
    -e http_proxy=http://127.0.0.1:7890 \
    -e https_proxy=http://127.0.0.1:7890 \
    -v ~/Chiplet_Heterogeneous_newVersion:/workspace/sim \
    -w /workspace/sim/benchmark/matmul \
    chiplet-sim:u1804 bash
```

容器内：
```bash
source /workspace/sim/setup_env.sh
source /workspace/sim/gpgpu-sim/setup_environment

# 先验证网络连通
ping -c 2 coord
nc -zv coord 15000   # 应当显示 succeeded

# 清理残留
rm -rf proc_r* buffer* *.log sync_time*.csv

# 启动 worker
$SIMULATOR_ROOT/interchiplet/bin/interchiplet_distributed \
    --role worker --machine-id 1 \
    ./matmul_multi_container.yml 2>&1 | tee worker_1.log
```

### 7.4 查看结果

仿真产物在挂载目录中，**宿主机直接可见**：

```bash
# 宿主机
cd ~/Chiplet_Heterogeneous_newVersion/benchmark/matmul

# 同步耗时（核心数据）
cat sync_time_machine_0.csv
cat sync_time_machine_1.csv

# 各机器日志
tail -30 coordinator.log
tail -30 worker_1.log

# 通信记录
cat bench.txt
cat delayInfo.txt
```

> 注：coordinator 容器和 worker 容器都 `-w /workspace/sim/benchmark/matmul`，两边写的文件落在同一个宿主机目录里。如果想把两个容器的产物彻底分开，可以给 worker 改用 `/workspace/sim/benchmark/matmul/worker_workdir/`、自己 mkdir，并修改启动命令。

### 7.5 跨真实物理机部署

把上面流程扩展到两台物理机只需要：

1. 两台机器都 `docker load` 同一个 tar
2. 不用 `chiplet-net`，改用 `--network host`（直接用物理机的网络）
3. YAML 里的 `host` 字段写**对方物理机的 IP/hostname**（而不是容器名）
4. 确保防火墙开放 15000 和 16000-17000 端口

---

## 8. 常见问题

### Q1：make 时 GitHub 连不上
sniper 首次 make 需要从 GitHub 拉 mbuild/xed、从 snipersim.org 下 pin tool。需要代理：

1. 宿主机有代理（如 `127.0.0.1:7890`）
2. 启容器时带 `--network host -e http_proxy=http://127.0.0.1:7890 -e https_proxy=http://127.0.0.1:7890`
3. 容器内配 wget / git 代理（见 [5.2](#52-容器内一次性配置)）

### Q2：coordinator 启动后一直 hang
正常 —— 它在等 worker 发 `READY` 消息。在另一个终端启动 worker 就会继续。

### Q3：worker 报 "Connection refused" 或找不到 coord
检查：
```bash
# worker 容器里
ping -c 2 coord            # DNS 必须通
nc -zv coord 15000         # TCP 能连上
```
如果 ping 不通，通常是两个容器没加同一个 `--network chiplet-net`。

### Q4：容器里 `run-sniper` 提示找不到 libcudart
没 source 过 gpgpu-sim/setup_environment。每次进新容器都需要：
```bash
source /workspace/sim/setup_env.sh
source /workspace/sim/gpgpu-sim/setup_environment
```

### Q5：YAML 里 host 改成容器名后，原版 run_distributed.sh 还能跑吗
不能 —— 本地 `127.0.0.1` 和容器名 `coord` / `worker1` 是互斥的。建议保留**两个**YAML 文件：
- `matmul_distributed.yml` （127.0.0.1，单容器用）
- `matmul_multi_container.yml`（容器名，多容器用）

### Q6：镜像里没有仿真器代码，每次起容器都要重编吗
**不用**。仿真器代码和编译产物都在宿主机挂载目录里，容器退出不会丢。下次起新容器挂载同一目录，所有 `.so` / `.a` / 可执行文件都在。

### Q7：跑完想清理产物
```bash
cd $SIMULATOR_ROOT/benchmark/matmul_test
make clean
rm -rf machine* proc_r* buffer* *.log sync_time*.csv
```

### Q8：想加第二个 worker（3 台机器）
1. 修改 YAML 在 `machines:` 和 `phase1:` 中加 machine 2 的配置
2. 另开第三个容器 `--name worker2 --hostname worker2`，启动参数里改 `--machine-id 2`
# GCN Benchmark README

## 1. 总体目标
`benchmark/gcn` 用于把一次 GCN 推理拆成 `CPU`、`PIM`、`IO` 三个芯粒，并生成可直接用于 `gem5 + interchiplet + PopNet` 的输入与运行计划。

当前目录采用的实现口径是：
- 统一输入是 `ONNX` 图。
- `CPU` 与 `IO` 在 `gem5 SE` 中运行真实 `RISC-V` workload。
- `PIM` 运行本地功能模型，并注入离线估算出的计算时间与片内网络时间。
- `PIM` 片内网络延迟离线建模，`phase2` 只做片间网络仿真。

## 2. 仿真方案
当前我们明确采用“片内离线，片间在线”的两段式口径：

### 2.1 PIM 片内
`bench_pim_intra.txt` 只用于 `PIM` 芯粒内部的离线 `PopNet` 仿真，流程是：
1. `chiplet_task_estimator.py` 生成 `bench_pim_intra.txt`。
2. `run_popnet_intra_pim.sh` 使用 `topology/pim_vertical_4_4.gv` 跑片内 `PopNet`。
3. 输出 `delayInfo_pim_intra.txt`。
4. `build_pim_intra_delay_profile.py` 将其整理成 `reports/pim_intra_delay_profile.txt`。
5. `pim.cpp` 运行时读取 `reports/pim_intra_delay_profile.txt`，把片内通信延迟注入到每个 epoch 的执行过程中。

也就是说，`PIM` 片内网络已经在 `phase1` 前通过离线方式折算成时间，不再进入 `phase2` 重复仿真。

### 2.2 片间网络
`bench.txt` 只表示芯粒级通信事务，由 `interchiplet` 在线运行时记录，参与 `phase2`：
1. `CPU / PIM / IO` 在 `phase1` 联机运行。
2. `interchiplet` 记录芯粒间事务，生成在线 `bench.txt`。
3. `phase2` 由 `gcn.yml` 直接调用 `popnet`。
4. `phase2` 使用 `topology/mesh_2_2.gv` 跑片间 `PopNet`，输出 `delayInfo.txt`。

### 2.3 为什么必须这样拆
如果把 `bench_pim_intra.txt` 再并入 `phase2`，就会出现重复建模：
- 一次是在 `pim.cpp` 中通过 `pim_intra_delay_profile.txt` 注入；
- 一次是在 `phase2` 中再次被 `PopNet` 回放。

这会导致 `PIM` 片内延迟被重复计算。当前口径 A 的核心，就是避免这一点。

## 3. 目录中的主要模块

### 3.1 模型输入
- `build_gcn_onnx.py`

作用：
- 生成默认的 `GCN ONNX` 输入图 `gcn_3chiplet.onnx`。
- 这是整个离线分析与运行计划生成的统一入口。

### 3.2 离线估算入口
- `chiplet_task_estimator.py`
- `chiplet_estimator/main.py`

作用：
- 读取 `ONNX` 图；
- 做算子划分；
- 统计算量与访存；
- 生成 `CPU / PIM / IO` 的运行计划；
- 生成片间 `bench.txt` 与片内 `bench_pim_intra.txt`。

### 3.3 CPU 芯粒
- `cpu.cpp`

作用：
- 读取 `data/cora.graph` 和 `data/cora.svmlight`；
- 组织图结构、节点特征和标签；
- 通过 `interchiplet` 将输入数据流发给 `PIM`；
- 接收 `PIM` 返回的 `logits`；
- 执行 `softmax / logsoftmax / none` 后处理；
- 将结果转发给 `IO`。

### 3.4 PIM 芯粒
- `pim.cpp`
- `run_pim_backend.sh`

作用：
- 接收 `CPU` 发来的数据流；
- 按 `runtime_plan_pim.txt` 执行 GCN 主干的功能建模；
- 读取 `reports/pim_intra_delay_profile.txt`，把片内网络延迟注入运行时；
- 将每轮结果回传给 `CPU`。

### 3.5 IO 芯粒
- `io.cpp`

作用：
- 接收 `CPU` 转发的结果；
- 按 `runtime_plan_io.txt` 模拟写回、输出或存储阶段的耗时。

### 3.6 PIM 片内网络与分块
- `chiplet_estimator/pim_sram_tiling.py`
- `run_popnet_intra_pim.sh`
- `build_pim_intra_delay_profile.py`

作用：
- 对每个映射到 `PIM` 的节点执行 `128KB SRAM` 约束下的 `tile` 搜索；
- 估计激活块、权重块、累加器块、输出块的驻留情况；
- 为 `16` 个 `NPU` 的 `4x4` 纵向拓扑生成片内事务；
- 运行离线 `PopNet` 得到片内延迟分布。

## 4. 关键输出文件

### 4.1 离线估算结果
- `partition_info.py`
- `reports/execution_plan.json`
- `reports/runtime_plan_pim.txt`
- `reports/runtime_plan_io.txt`
- `reports/pim_mapping.json`
- `reports/pim_pe_workload.txt`

### 4.2 网络相关文件
- `bench.txt`
  - 芯粒级在线事务，供 `phase2` 使用。
- `bench_pim_intra.txt`
  - PIM 片内离线事务，只供 `run_popnet_intra_pim.sh` 使用。
- `delayInfo_pim_intra.txt`
  - PIM 片内 `PopNet` 原始输出。
- `reports/pim_intra_delay_profile.txt`
  - `pim.cpp` 运行时读取的片内延迟画像。
- `delayInfo.txt`
  - `phase2` 片间 `PopNet` 输出。

## 5. 当前执行流程

### 5.1 离线阶段
```bash
cd benchmark/gcn
make ir-plan
```

这一步会：
- 编译 `cpu / pim / io`；
- 生成 `gcn_3chiplet.onnx`；
- 读取 `ONNX`；
- 生成 `runtime_plan_pim.txt`、`runtime_plan_io.txt`；
- 生成 `bench_pim_intra.txt`；
- 如果启用了片间 bench，也会生成 `bench.txt`。

### 5.2 PIM 片内延迟画像
```bash
cd benchmark/gcn
make pim-intra-delay
```

这一步会：
- 跑 `run_popnet_intra_pim.sh`；
- 生成 `delayInfo_pim_intra.txt`；
- 调用 `build_pim_intra_delay_profile.py`；
- 输出 `reports/pim_intra_delay_profile.txt`。

### 5.3 联机三芯粒运行
```bash
cd benchmark/gcn
make run
```

这一步会：
- 启动 `CPU` gem5 进程；
- 启动 `PIM` 本地后端；
- 启动 `IO` gem5 进程；
- 在 `phase1` 中跑完整的三芯粒协同；
- 由 `interchiplet` 记录在线片间事务；
- 在 `phase2` 中仅回放片间 `bench.txt`。

## 6. 运行依赖

### 6.1 Python
至少需要：
- `python3`
- `numpy`
- `onnx`

可参考：
```bash
python3 -m pip install numpy onnx
```

### 6.2 编译工具
至少需要：
- `g++`
- `riscv64-unknown-linux-gnu-g++`
- `make`
- `bash`

### 6.3 仿真器
至少需要：
- `gem5/build/RISCV/gem5.opt`
- `interchiplet/bin/interchiplet`
- `popnet_chiplet/build/popnet`

### 6.4 数据与配置
至少需要：
- `data/cora.graph`
- `data/cora.svmlight`
- `chiplet_config.json`

## 7. 推荐测试顺序
标准使用时，建议直接执行：

```bash
cd benchmark/gcn
make run
```

这和 `artifact/matmul` 的使用方式一致：由 `make run` 调用 `interchiplet`，再由 yml 自动执行 `phase1` 和 `phase2`。

如果你在做分阶段排错，再按下面顺序拆开检查：

1. `make clean-output`
2. `make`
3. `make gen-onnx`
4. `make ir-plan`
5. `make pim-intra-delay`
6. `make run`

当前目录不再保留单独的 `phase2` 调试脚本。片间 `PopNet` 由 `gcn.yml` 在正式运行流程中直接调用，因此推荐只维护 `make run` 这一条主线。

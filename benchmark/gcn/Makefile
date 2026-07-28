SIMULATOR_ROOT ?= $(abspath $(CURDIR)/../..)
BENCHMARK_ROOT := $(SIMULATOR_ROOT)/benchmark/gcn

RISCV_CC := riscv64-unknown-linux-gnu-g++
HOST_CC := g++
CFLAGS := -Wall -Werror -g -I$(SIMULATOR_ROOT)/interchiplet/includes -I$(SIMULATOR_ROOT)/interchiplet/includes -pthread
RISCV_WARN_RELAX_FLAGS := -Wno-error
RISCV_BUILD_CFLAGS := $(CFLAGS) $(RISCV_WARN_RELAX_FLAGS)
HOST_BUILD_CFLAGS := $(CFLAGS)
LDFLAGS := -pthread
RISCV_LDFLAGS := $(LDFLAGS) -static
PYTHON ?= python3
ONNX_MODEL ?= gcn_3chiplet.onnx
INTERCHIPLET_BIN ?= $(SIMULATOR_ROOT)/interchiplet/bin/interchiplet
INTERCHIPLET_CFG ?= gcn.yml
INTERCHIPLET_CFG_RUBY ?= gcn_ruby.yml
MEMORY_PROFILE ?= baseline

CPU_SRCS := cpu.cpp
CPU_OBJS := obj/cpu.o
CPU_TARGET := bin/cpu

PIM_SRCS := pim.cpp
PIM_OBJS := obj/pim.o
PIM_TARGET := bin/pim

IO_SRCS := io.cpp
IO_OBJS := obj/io.o
IO_TARGET := bin/io

.PHONY: all build cpu_target pim_target io_target gen-onnx partition estimate ir-plan bench bench-calib pim-intra-delay run run-ruby run-debug run_popnet_intra clean clean-output help

all: bin_dir obj_dir cpu_target pim_target io_target
build: all

cpu_target: $(CPU_OBJS)
	@echo "Building CPU chiplet with RISC-V cross-compiler (statically linked)..."
	$(RISCV_CC) $(RISCV_BUILD_CFLAGS) $(CPU_OBJS) $(RISCV_LDFLAGS) -o $(CPU_TARGET)
	@echo "CPU chiplet built successfully (RISC-V format, statically linked)"

pim_target: $(PIM_OBJS)
	$(HOST_CC) $(HOST_BUILD_CFLAGS) $(PIM_OBJS) $(LDFLAGS) -o $(PIM_TARGET)
	@echo "PIM chiplet built successfully"

io_target: $(IO_OBJS)
	@echo "Building IO chiplet with RISC-V cross-compiler (statically linked)..."
	$(RISCV_CC) $(RISCV_BUILD_CFLAGS) $(IO_OBJS) $(RISCV_LDFLAGS) -o $(IO_TARGET)
	@echo "IO chiplet built successfully (RISC-V format, statically linked)"

obj/cpu.o: cpu.cpp gcn_runtime.h
	@echo "Compiling cpu.cpp with RISC-V cross-compiler..."
	$(RISCV_CC) $(RISCV_BUILD_CFLAGS) -c $< -o $@

obj/pim.o: pim.cpp gcn_runtime.h
	@echo "Compiling pim.cpp..."
	$(HOST_CC) $(HOST_BUILD_CFLAGS) -c $< -o $@

obj/io.o: io.cpp gcn_runtime.h
	@echo "Compiling io.cpp with RISC-V cross-compiler..."
	$(RISCV_CC) $(RISCV_BUILD_CFLAGS) -c $< -o $@

gen-onnx:
	$(PYTHON) build_gcn_onnx.py --out $(ONNX_MODEL)

partition: all gen-onnx
	$(PYTHON) chiplet_task_estimator.py --onnx $(ONNX_MODEL) --partition-only

estimate: all gen-onnx
	$(PYTHON) chiplet_task_estimator.py --onnx $(ONNX_MODEL) --report reports/execution_plan.json --memory-profile $(MEMORY_PROFILE)

ir-plan: all gen-onnx
	$(PYTHON) chiplet_task_estimator.py --onnx $(ONNX_MODEL) --no-inter-bench --bench-intra bench_pim_intra.txt --pim-mapping-report reports/pim_mapping.json --pim-pe-workload reports/pim_pe_workload.txt --report reports/execution_plan.json --memory-profile $(MEMORY_PROFILE)

bench: all gen-onnx
	$(PYTHON) chiplet_task_estimator.py --onnx $(ONNX_MODEL) --bench bench.txt --bench-intra bench_pim_intra.txt --pim-mapping-report reports/pim_mapping.json --pim-pe-workload reports/pim_pe_workload.txt --report reports/execution_plan.json --memory-profile $(MEMORY_PROFILE)

bench-calib: all gen-onnx
	$(PYTHON) chiplet_task_estimator.py --onnx $(ONNX_MODEL) --no-inter-bench --bench-intra bench_pim_intra.txt --pim-mapping-report reports/pim_mapping.json --pim-pe-workload reports/pim_pe_workload.txt --report reports/execution_plan.json --memory-profile $(MEMORY_PROFILE) --enable-gem5-calibration --gem5-stats-root . --interchiplet-cmd "../../interchiplet/bin/interchiplet ./gcn.yml"

run_popnet_intra: ir-plan
	bash ./run_popnet_intra_pim.sh

pim-intra-delay: ir-plan
	bash ./run_popnet_intra_pim.sh
	$(PYTHON) build_pim_intra_delay_profile.py --delay delayInfo_pim_intra.txt --out reports/pim_intra_delay_profile.txt --npu-count 16

run: all pim-intra-delay
	@if [ ! -x "$(INTERCHIPLET_BIN)" ]; then \
		echo "[gcn] ERROR: interchiplet binary not found: $(INTERCHIPLET_BIN)"; \
		echo "[gcn] Build it first: (cd $(SIMULATOR_ROOT)/interchiplet && mkdir -p build && cd build && cmake .. && make -j)"; \
		exit 1; \
	fi
	@if ! grep -q "chiplet_se.py" $(INTERCHIPLET_CFG); then \
		echo "[gcn] ERROR: $(INTERCHIPLET_CFG) does not look like chiplet_se.py-based gem5 SE configuration."; \
		echo "[gcn] Please keep CPU/IO chiplets in gem5 SE + Ruby/Garnet mode."; \
		exit 1; \
	fi
	@echo "[gcn] Launching interchiplet with $(INTERCHIPLET_CFG) (CPU/IO via gem5 SE, PIM native by default)..."
	$(INTERCHIPLET_BIN) $(INTERCHIPLET_CFG)

run-ruby: INTERCHIPLET_CFG=$(INTERCHIPLET_CFG_RUBY)
run-ruby: run

run-debug: all pim-intra-delay
	@if [ ! -x "$(INTERCHIPLET_BIN)" ]; then \
		echo "[gcn] ERROR: interchiplet binary not found: $(INTERCHIPLET_BIN)"; \
		exit 1; \
	fi
	@echo "[gcn] Launching interchiplet in debug mode with $(INTERCHIPLET_CFG)..."
	$(INTERCHIPLET_BIN) $(INTERCHIPLET_CFG) --debug

bin_dir:
	mkdir -p bin

obj_dir:
	mkdir -p obj

clean-output:
	rm -rf bench.txt bench_pim_intra.txt bench_pim_intra_local.txt delayInfo.txt delayInfo_pim_intra.txt buffer* message_record.txt
	rm -rf proc_r*_t* *.log
	rm -rf output_epoch_*.txt
	rm -f partition_info.py
	rm -rf reports

clean: clean-output
	rm -rf obj bin

help:
	@echo "GCN Benchmark Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all             - Build all chiplets (default)"
	@echo "  build           - Alias of all"
	@echo "  cpu_target      - Build CPU chiplet only"
	@echo "  pim_target      - Build native PIM chiplet (pim.cpp)"
	@echo "  io_target       - Build IO chiplet only"
	@echo "  gen-onnx        - Generate default 3-chiplet GCN ONNX model ($(ONNX_MODEL))"
	@echo "  partition       - Generate partition_info.py from ONNX graph"
	@echo "  estimate        - Generate execution_plan.json with timing estimation"
	@echo "  ir-plan         - Generate PIM intra bench + PIM/IO runtime plans"
	@echo "  bench           - Generate bench.txt + bench_pim_intra.txt for PopNet"
	@echo "  bench-calib     - Generate IR plan with gem5/interchiplet calibration"
	@echo "  pim-intra-delay - Run intra PopNet + build reports/pim_intra_delay_profile.txt"
	@echo "  run             - Build chiplets + pim-intra-delay + run interchiplet (CPU/IO in gem5 SE)"
	@echo "  run-ruby        - Same as run but use gcn_ruby.yml"
	@echo "  run-debug       - Same as run but enable interchiplet --debug logs"
	@echo "  run_popnet_intra  - Run standalone PIM intra-chiplet PopNet simulation"
	@echo "  clean           - Remove build artifacts and runtime outputs"
	@echo "  clean-output    - Remove runtime outputs only, keep compiled binaries"
	@echo "  help            - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  SIMULATOR_ROOT  - Path to simulator root directory"
	@echo "  BENCHMARK_ROOT  - Path to benchmark directory (derived from SIMULATOR_ROOT)"
	@echo "  RISCV_CC        - RISC-V cross-compiler (default: riscv64-unknown-linux-gnu-g++)"
	@echo "  HOST_CC         - Host compiler (default: g++)"
	@echo "  ONNX_MODEL      - ONNX model path used by partition/estimate/bench targets"
	@echo "  MEMORY_PROFILE  - Named memory-optimization profile used by the estimator"
	@echo ""
	@echo "Requirements:"
	@echo "  - RISC-V cross-compiler: riscv64-unknown-linux-gnu-g++"
	@echo "  - gem5 RISC-V simulator: gem5/build/RISCV/gem5.opt"
	@echo ""
	@echo "Example usage:"
	@echo "  make              # Build all chiplets"
	@echo "  make gen-onnx     # Build default GCN ONNX model"
	@echo "  make ir-plan      # Generate PIM intra bench + runtime plans"
	@echo "  make bench        # Build chiplets + generate inter/intra bench files from ONNX graph"
	@echo "  make bench-calib  # ir-plan + interchiplet/gem5 stats calibration"
	@echo "  make pim-intra-delay # ir-plan + intra PopNet + delay profile generation"
	@echo "  make run          # Build chiplets + pim-intra-delay + run full interchiplet flow"
	@echo "  make run-ruby     # Build chiplets + pim-intra-delay + run Ruby/Garnet flow"
	@echo "  make run_popnet_intra # Run PIM intra-chiplet PopNet with vertical-only topology"
	@echo "  make clean        # Clean build artifacts and runtime outputs"

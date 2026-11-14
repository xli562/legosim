# Project environment
# SIMULATOR_ROOT, defined by setup_env.sh
BENCHMARK_ROOT=$(SIMULATOR_ROOT)/benchmark/matmul

# Compiler environment of C/C++
CC=g++
CFLAGS=-Wall -Werror -g -I$(SIMULATOR_ROOT)/interchiplet/includes
INTERCHIPLET_C_LIB=$(SIMULATOR_ROOT)/interchiplet/lib/libinterchiplet_c.a
INTERCHIPLET=../../interchiplet/bin/interchiplet
# C/C++ Source file
C_SRCS=matmul.cpp
C_OBJS=obj/matmul.o
C_TARGET=bin/matmul_c
CHECKED_C_TARGET=bin/matmul_checked_c
NPU_TARGET=bin/matmul_npu

# Compiler environment of CUDA
NVCC=nvcc
CUFLAGS=--compiler-options -Wall -I$(SIMULATOR_ROOT)/interchiplet/includes

# CUDA Source file
CUDA_SRCS=matmul.cu
CUDA_OBJS=cuobj/matmul.o
CUDA_TARGET=bin/matmul_cu

all: bin_dir obj_dir cuobj_dir C_target CUDA_target

# Build the checked CPU and the GPU worker from their current sources.
checked: bin_dir obj_dir cuobj_dir $(CHECKED_C_TARGET) $(CUDA_TARGET)

npu_checked: bin_dir obj_dir $(NPU_TARGET)

$(NPU_TARGET): obj/matmul_npu.o
	$(CC) $< $(INTERCHIPLET_C_LIB) -o $@

obj/matmul_npu.o: matmul_npu.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(CHECKED_C_TARGET): obj/matmul_checked.o
	$(CC) $< $(INTERCHIPLET_C_LIB) -o $@

obj/matmul_checked.o: matmul.cpp
	$(CC) $(CFLAGS) -DVALIDATE_MATMUL -c $< -o $@

# C language target
C_target: $(C_OBJS)
	$(CC) $(C_OBJS) $(INTERCHIPLET_C_LIB) -o $(C_TARGET)

# CUDA language target
CUDA_target: $(CUDA_TARGET)

$(CUDA_TARGET): $(CUDA_OBJS)
	$(NVCC) -L$(SIMULATOR_ROOT)/gpgpu-sim/lib/$(GPGPUSIM_CONFIG) --cudart shared $(CUDA_OBJS) -o $@

# Rule for C object
obj/%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Rule for Cuda object
cuobj/%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

# Directory for binary files.
bin_dir:
	mkdir -p bin

# Directory for object files for C.
obj_dir:
	mkdir -p obj

# Directory for object files for CUDA.
cuobj_dir:
	mkdir -p cuobj

run:
	$(INTERCHIPLET) ./matmul.yml -w 6 -f 2

# Clean generated files.
clean:
	rm -rf bench.txt delayInfo.txt buffer* message_record.txt
	rm -rf proc_r*_t* *.log
	rm -rf obj cuobj bin

BENCHMARK_ROOT = $(SIMULATOR_ROOT)/benchmark/gemm_2dmesh

# Compiler environment
CC = g++
CFLAGS = -Wall -Werror -g -std=c++11 \
         -I$(SIMULATOR_ROOT)/interchiplet/includes \
         -I$(SIMULATOR_ROOT)/interchiplet/thirdparty/spdlog/include
INTERCHIPLET_C_LIB=$(SIMULATOR_ROOT)/interchiplet/lib/libinterchiplet_c.a
INTERCHIPLET=../../interchiplet/bin/interchiplet

# Targets
ASIC_TARGET = bin/asic
CPU_TARGET  = bin/cpu

all: bin_dir obj_dir $(ASIC_TARGET) $(CPU_TARGET)

# Native ASIC model: header-only sync protocol, no interchiplet library.
$(ASIC_TARGET): obj/asic.o
	$(CC) $< -o $@ -pthread

# CPU program run under Sniper: needs the syscall API library.
$(CPU_TARGET): obj/cpu.o
	$(CC) $< $(INTERCHIPLET_C_LIB) -o $@

# Rule for C object
obj/%.o: %.cpp comms_protocol.h
	$(CC) $(CFLAGS) -c $< -o $@

# Directory for binary files.
bin_dir:
	mkdir -p bin

# Directory for object files for C.
obj_dir:
	mkdir -p obj

# Clean generated files.
clean:
	rm -rf bench.txt delayInfo.txt buffer* message_record.txt
	rm -rf proc_r*_t* *.log
	rm -rf obj bin

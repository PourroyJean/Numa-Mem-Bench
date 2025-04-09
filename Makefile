# =============================================================================
# Makefile for numa_mem_bench
# Author: Jean Pourroy
# Organization: HPE
# License: MIT (See LICENSE file for details)
# =============================================================================
# This Makefile compiles the numa_mem_bench C program, which measures memory
# access latency across NUMA domains using MPI, hwloc, and OpenMP.
#
# Variables:
#   CC:         The C compiler (usually the MPI wrapper, e.g., 'cc' on Cray)
#   HWLOC_DIR:  Path to the hwloc installation (defaults to ~/TOOLS/hwloc)
#               Can be overridden, e.g., make HWLOC_DIR=/path/to/hwloc
#
# Targets:
#   all:        Builds the numa_mem_bench executable (default target)
#   clean:      Removes object files and the executable
#
# Usage:
#   make          # Compile numa_mem_bench
#   make clean    # Remove build artifacts
#   make HWLOC_DIR=/custom/path/to/hwloc # Compile with custom hwloc path
# =============================================================================

# Compiler: Use the MPI C compiler wrapper (e.g., 'cc' on Cray)
CC = cc

# [LUMI] Check if module craype-x86-trento is loaded and warn if not
ifeq ($(shell module list 2>&1 | grep -q craype-x86-trento; echo $$?), 1)
    $(warning [LUMI] Warning: craype-x86-trento module is not loaded)
endif

# Path to hwloc installation (change if needed or set environment variable)
HWLOC_DIR ?= $(HOME)/TOOLS/hwloc

# Source directory
SRC_DIR = src

# Build directory (for object files)
BUILD_DIR = build

# Compiler Flags:
# -Wall -Wextra: Enable most warnings
# -O3: Optimization level 3
# -I...: Include paths for hwloc
# -fopenmp: Enable OpenMP support (for omp.h include)
CFLAGS = -Wall -Wextra -O3 -I$(HWLOC_DIR)/include -I$(SRC_DIR) -fopenmp -D_GNU_SOURCE

# Linker Flags:
# -lnuma: Link against the NUMA library
# -L... -lhwloc: Link against the hwloc library
# -fopenmp: Link OpenMP runtime library
LDFLAGS = -lnuma -L$(HWLOC_DIR)/lib -lhwloc -fopenmp

# Target executable name (placed in the root directory)
TARGET = numa_mem_bench

# Source files (with paths)
SRCS = $(SRC_DIR)/numa_mem_bench.c $(SRC_DIR)/common/bench_common.c $(SRC_DIR)/benchmark/memory_latency_bench.c

# Object files (placed in build directory)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))

# Phony targets (targets that don't represent files)
.PHONY: all clean

# Default target: build the executable
all: $(TARGET)

# Create build directory if it doesn't exist
$(BUILD_DIR): 
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/common
	mkdir -p $(BUILD_DIR)/benchmark

# Rule to link the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Build complete! Find the executable at $(TARGET)"

# Rule to compile source files into object files in the build directory
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to clean up build artifacts
clean:
	rm -rf $(BUILD_DIR) $(TARGET) 
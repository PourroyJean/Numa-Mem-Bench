// =============================================================================
// memory_latency_bench.h
// Memory latency benchmark functionality
// Author: Jean Pourroy
// Organization: HPE
// License: MIT (See LICENSE file for details)
// =============================================================================

#ifndef MEMORY_LATENCY_BENCH_H
#define MEMORY_LATENCY_BENCH_H

#include "../common/bench_common.h"

// Number of iterations for the latency benchmark
#define LATENCY_ITERATIONS 100000  // Chosen for good accuracy/speed balance
#define WARMUP_ITERATIONS 1000     // Enough to warm caches effectively

// Fisher-Yates shuffle algorithm for randomizing memory access patterns
void shuffle(size_t *array, size_t n);

// Function to measure memory latency
double measure_memory_latency(void *memory, size_t size);

// Function to run the memory latency benchmark
int run_memory_latency_benchmark(int rank, int size, int serial_mode, int num_sizes, size_t *sizes, 
                               int cpu_id, int core_numa, char *csv_filename);

#endif // MEMORY_LATENCY_BENCH_H 
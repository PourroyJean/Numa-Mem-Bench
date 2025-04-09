// =============================================================================
// bench_common.h
// Common functionality for NUMA benchmarks
// Author: Jean Pourroy
// Organization: HPE
// License: MIT (See LICENSE file for details)
// =============================================================================

#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <mpi.h>
#include <omp.h>
#include <numa.h>
#include <hwloc.h>
#include <sys/mman.h>
#include <errno.h>
#include <linux/mempolicy.h>  // For MPOL_F_NODE and MPOL_F_ADDR
#include <numaif.h>          // For get_mempolicy
#include <time.h>            // For time() function for random seed

// Default allocation size in MB
#define DEFAULT_ALLOC_SIZE_MB 512

// Maximum number of different sizes to test
#define MAX_SIZES 18

// Standard sizes for range expansion
#define NUM_STANDARD_SIZES 18
extern const size_t STANDARD_SIZES[NUM_STANDARD_SIZES];

// Default CSV delimiter
#define DEFAULT_CSV_DELIMITER ","

// Mapping file name output
#define MAPPING_FILE "mapping_check.log"

// Structure for MPI rank mapping information
struct mapping_info {
    int rank;
    int cpu_id;
    int cpu_numa;
    int memory_numa;
};

// Benchmark parameters structure
typedef struct {
    size_t sizes[MAX_SIZES];
    int sizes_count;
    int serial_mode;
    char *csv_file;
    char csv_delimiter[8];
} bench_params_t;

// Common functions
void get_cpu_info(hwloc_topology_t topology, int *cpu_id, int *core_numa);
int get_numa_node_of_address(void *addr);
int write_mapping_info(const char *filename, struct mapping_info *info, int world_size);
int collect_mapping_info(int rank, int size, int cpu_id, int core_numa, void *addr);
int parse_args(int argc, char **argv, bench_params_t *params);
void print_help(const char *program_name);
void print_results_table_header(int num_sizes, size_t *sizes);
void print_results_table_row(int rank, int world_size, int cpu_id, int core_numa, void *addr, int num_sizes, double *latencies);
void print_results_table_footer(int num_sizes);

#endif // BENCH_COMMON_H 
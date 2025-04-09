// =============================================================================
// Program: numa_mem_bench.c
// Author: Jean Pourroy
// Organization: HPE
// License: MIT (See LICENSE file for details)
// =============================================================================
// This program measures memory access latency across NUMA domains using MPI.
// It provides detailed analysis of memory access patterns and NUMA binding effects.
//
// Features:
// - Measures memory latency using pointer chasing technique
// - Supports both parallel and serial execution modes
// - Configurable memory sizes and allocation patterns
// - Detailed CPU and NUMA topology reporting
// - CSV output for data analysis
// - Mapping information for process placement
//
// Usage:
//   The program is typically launched using the wrapper script wrapper_numa.sh:
//   srun --nodes=1 --ntasks=N ./wrapper_numa.sh [options] -- ./numa_mem_bench [options]
//
// Wrapper Options:
//   --numa=VALUE     NUMA binding configuration
//                    - Single value: Bind all ranks to specified NUMA node
//                    - Comma-separated list: Bind each rank to corresponding node
//                    - 'auto': Automatically distribute ranks across NUMA nodes
//   --quiet          Disable verbose output
//   --serial         Run in serial mode
//   --dry-run        Print commands without execution
//
// Program Options:
//   --size=SIZE       Memory size in MB (single value, range, or comma-separated list)
//   --serial          Run in serial mode (one rank at a time)
//   --csv=FILE        Output results to CSV file
//   --mapping=FILE    Save process mapping information
//
// Example:
//   srun --nodes=1 --ntasks=56 ./wrapper_numa.sh --numa=3 -- ./numa_mem_bench --size=2048 --csv=results.csv
//   srun --nodes=1 --ntasks=4 ./wrapper_numa.sh --numa=0,1,2,3 --quiet -- --serial -- ./numa_mem_bench --size=128-1024
// =============================================================================

#include "common/bench_common.h"
#include "benchmark/memory_latency_bench.h"

int main(int argc, char *argv[]) {
    int rank, size;
    hwloc_topology_t topology;
    int cpu_id, core_numa;
    bench_params_t params;

    // Initialize MPI
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "MPI initialization failed\n");
        return 1;
    }

    // Get MPI rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Print debug information for rank 0 only
    if (rank == 0) {
        printf("\n===================== Benchmark Information =====================\n");
        printf("MPI Configuration:\n");
        printf("  Number of ranks: %d\n", size);
        printf("  Command line arguments:\n");
        for (int i = 0; i < argc; i++) {
            printf("    argv[%d] = %s\n", i, argv[i]);
        }
        printf("\nSystem Information:\n");
        printf("  Page size: %ld bytes\n", sysconf(_SC_PAGESIZE));
        printf("  Number of NUMA nodes: %d\n", numa_max_node() + 1);
        printf("  NUMA available: %s\n", numa_available() == -1 ? "No" : "Yes");
        printf("  Number of CPUs: %ld\n", sysconf(_SC_NPROCESSORS_ONLN));
        printf("  Current CPU: %d\n", sched_getcpu());
        printf("  Current NUMA node: %d\n", numa_node_of_cpu(sched_getcpu()));
        printf("\nNote: NUMA memory binding should be controlled externally using numactl --membind=<node>\n");
        printf("Mapping information written to %s\n", MAPPING_FILE);
        printf("=================================================================\n\n");
        fflush(stdout);
    }

    // Parse command line arguments
    int result = parse_args(argc, argv, &params);
    if (result != 0) {
        // If result is 1, help was displayed, exit cleanly
        if (result == 1) {
            MPI_Finalize();
            return 0;
        }
        // Otherwise, there was an error
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Initialize hwloc topology with specific flags
    hwloc_topology_init(&topology);
    hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM);
    hwloc_topology_load(topology);

    // Get CPU and NUMA information
    get_cpu_info(topology, &cpu_id, &core_numa);

    // Run the memory latency benchmark
    result = run_memory_latency_benchmark(rank, size, params.serial_mode, params.sizes_count, 
                                       params.sizes, cpu_id, core_numa, params.csv_file);

    // Cleanup
    hwloc_topology_destroy(topology);
    if (params.csv_file) {
        free(params.csv_file);
    }
    MPI_Finalize();
    return result;
}

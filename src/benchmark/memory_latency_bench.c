// =============================================================================
// memory_latency_bench.c
// Memory latency benchmark functionality
// Author: Jean Pourroy
// Organization: HPE
// License: MIT (See LICENSE file for details)
// =============================================================================

#include "memory_latency_bench.h"

// Implementation of Fisher-Yates shuffle algorithm for randomizing memory access patterns
void shuffle(size_t *array, size_t n) {
    if (n <= 1) return;
    
    // Use different seed for each MPI rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(time(NULL) + rank);
    
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

/*
 * Measure memory latency using pointer chasing technique
 * 
 * This function:
 * 1. Creates a linked list of pointers in the allocated memory
 * 2. Randomizes the pointer chain to prevent CPU prefetching
 * 3. Uses volatile pointers to prevent compiler optimizations
 * 4. Measures the time taken to traverse this random path
 * 
 * Why pointer chasing?
 * - Forces actual memory accesses
 * - Prevents CPU prefetching by using random access pattern
 * - Each access must wait for the previous one to complete
 * 
 * Why volatile?
 * - Prevents compiler from optimizing away the pointer chase loop
 * - Ensures each memory access is actually performed
 * - Gives more accurate latency measurements
 * 
 * Parameters:
 *   memory: Pointer to the allocated memory region
 *   size: Size of the memory region in bytes
 * 
 * Returns:
 *   Average latency per memory access in nanoseconds
 */
double measure_memory_latency(void *memory, size_t size) {
    // Set up the pointer-chasing linked list
    size_t num_pointers = size / sizeof(void*);
    void **pointers = (void**)memory;
    size_t *indices = NULL;
    
    // Create an array of indices and shuffle them
    indices = malloc(num_pointers * sizeof(size_t));
    if (!indices) {
        fprintf(stderr, "Failed to allocate indices array for latency test\n");
        return -1.0;
    }
    
    for (size_t i = 0; i < num_pointers; i++) {
        indices[i] = i;
    }
    
    // Shuffle the indices to create a random path through memory
    shuffle(indices, num_pointers);
    
    // Create the pointer chain using the shuffled indices
    for (size_t i = 0; i < num_pointers - 1; i++) {
        pointers[indices[i]] = &pointers[indices[i + 1]];
    }
    // Complete the cycle
    pointers[indices[num_pointers - 1]] = &pointers[indices[0]];
    
    // Warm up the cache and ensure memory is paged in
    volatile void **p = (volatile void **)&pointers[indices[0]];
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        p = (volatile void **)*p;
    }
    
    // Measure memory access time
    double start_time = MPI_Wtime();
    
    // Chase pointers through memory
    p = (volatile void **)&pointers[indices[0]];
    for (int i = 0; i < LATENCY_ITERATIONS; i++) {
        p = (volatile void **)*p;
    }
    
    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;
    
    // Calculate average latency in nanoseconds
    double latency_ns = (total_time * 1e9) / LATENCY_ITERATIONS;
    
    // Add a dummy use of p to prevent compiler optimization
    if (p == NULL) {
        fprintf(stderr, "Should never happen but prevents optimization\n");
    }
    
    // Clean up indices array
    free(indices);
    return latency_ns;
}

// Run the memory latency benchmark with provided configuration
int run_memory_latency_benchmark(int rank, int size, int serial_mode, int num_sizes, size_t *sizes, 
                               int cpu_id, int core_numa, char *csv_filename) {
    void *allocated_memory = NULL;
    double *latencies = NULL;
    
    // Allocate memory for storing latency results
    latencies = (double *)malloc(num_sizes * sizeof(double));
    if (!latencies) {
        fprintf(stderr, "Rank %d: Failed to allocate latency results array\n", rank);
        return 1;
    }

    // Allocate memory for the first size to get NUMA information
    allocated_memory = malloc(sizes[0] * 1024 * 1024);
    if (allocated_memory == NULL) {
        fprintf(stderr, "Rank %d: Memory allocation failed for size %zu MB\n", rank, sizes[0]);
        free(latencies);
        return 1;
    }

    // Collect and write mapping information
    if (collect_mapping_info(rank, size, cpu_id, core_numa, allocated_memory) != 0) {
        free(latencies);
        free(allocated_memory);
        return 1;
    }

    // Loop through each memory size
    for (int i = 0; i < num_sizes; i++) {
        size_t current_size_mb = sizes[i];
        
        // Free previous allocation if any
        if (allocated_memory != NULL) {
            free(allocated_memory);
            allocated_memory = NULL;
        }
        
        // Allocate memory using standard malloc
        // Note: NUMA binding should be controlled externally using numactl --membind=<node>
        allocated_memory = malloc(current_size_mb * 1024 * 1024);

        if (allocated_memory == NULL) {
            fprintf(stderr, "Rank %d: Memory allocation failed for size %zu MB\n", rank, current_size_mb);
            free(latencies);
            return 1;
        }

        // Ensure all ranks proceed to measurement at the same time
        MPI_Barrier(MPI_COMM_WORLD);

        // Measure memory latency
        if (serial_mode) {
            // In serial mode, only one rank runs the benchmark at a time
            double *all_latencies = malloc(size * sizeof(double));
            if (!all_latencies) {
                fprintf(stderr, "Failed to allocate latency array\n");
                free(latencies);
                free(allocated_memory);
                return 1;
            }

            for (int current_rank = 0; current_rank < size; current_rank++) {
                if (rank == current_rank) {
                    // Current rank runs the benchmark
                    all_latencies[current_rank] = measure_memory_latency(allocated_memory, current_size_mb * 1024 * 1024);
                    // Broadcast the result to all ranks
                    MPI_Bcast(&all_latencies[current_rank], 1, MPI_DOUBLE, current_rank, MPI_COMM_WORLD);
                } else {
                    // Other ranks wait for the current rank to finish
                    MPI_Bcast(&all_latencies[current_rank], 1, MPI_DOUBLE, current_rank, MPI_COMM_WORLD);
                }
                // Synchronize all ranks before moving to the next one
                MPI_Barrier(MPI_COMM_WORLD);
            }

            // Each rank uses its own latency result
            latencies[i] = all_latencies[rank];
            free(all_latencies);
        } else {
            // In parallel mode, all ranks run the benchmark simultaneously
            latencies[i] = measure_memory_latency(allocated_memory, current_size_mb * 1024 * 1024);
        }

        // Ensure all ranks have completed this size before moving to the next
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Print results in a table format
    if (rank == 0) {
        print_results_table_header(num_sizes, sizes);
    }
    
    // Ensure header is printed before rows
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Print each rank's row
    print_results_table_row(rank, size, cpu_id, core_numa, allocated_memory, num_sizes, latencies);
    
    // Ensure all rows are printed before footer
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Small delay to ensure output is flushed
    if (rank == 0) {
        usleep(100000); // 100ms delay
        print_results_table_footer(num_sizes);
    }

    // Need to ensure all ranks have processed the table output before proceeding
    MPI_Barrier(MPI_COMM_WORLD);

    // All ranks should wait a moment to ensure table output is complete
    MPI_Barrier(MPI_COMM_WORLD);

    // Only the last rank displays NUMA statistics
    // Use strict sequential ordering like we did with table rows
    for (int r = 0; r < size; r++) {
        if (rank == r && rank == size - 1) {
            // Initialize the memory to ensure it's allocated
            size_t total_size = sizes[num_sizes-1] * 1024 * 1024;
            char *mem = (char *)allocated_memory;
            for (size_t i = 0; i < total_size; i += 4096) {  // Touch each page
                mem[i] = 0;
            }
            
            // Print NUMA statistics header
            printf("\n[%d] ===================== NUMA Statistics =====================\n", rank);
            printf("[%d] Last Process: Rank %d\n", rank, rank);
            printf("[%d] Process ID: %d\n", rank, getpid());
            printf("[%d] Allocated Memory Size: %zu MB\n", rank, sizes[num_sizes-1]);
            printf("[%d] Running numastat...\n", rank);
            fflush(stdout);
            
            char cmd[256];
            snprintf(cmd, sizeof(cmd), "numastat -p %d | sed 's/^/[%d] /'", getpid(), rank);
            system(cmd);
            
            printf("[%d]==========================================================\n\n", rank);
            fflush(stdout);
        }
        
        // Synchronize after the rank prints (or would have printed)
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // CSV output handling if enabled
    if (csv_filename != NULL) {
        // Gather all results to rank 0
        double *all_latencies = NULL;
        if (rank == 0) {
            all_latencies = malloc(num_sizes * size * sizeof(double));
            if (!all_latencies) {
                fprintf(stderr, "Error: Failed to allocate memory for gathering results\n");
                free(latencies);
                free(allocated_memory);
                return 1;
            }
        }

        // Gather results from all ranks to rank 0
        MPI_Gather(latencies, num_sizes, MPI_DOUBLE, all_latencies, num_sizes, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Write CSV file on rank 0
        if (rank == 0) {
            FILE *csv_file = fopen(csv_filename, "w");
            if (!csv_file) {
                fprintf(stderr, "Error: Failed to open CSV file %s\n", csv_filename);
                free(all_latencies);
                free(latencies);
                free(allocated_memory);
                return 1;
            }

            // Write header
            fprintf(csv_file, "size (MB)");
            for (int r = 0; r < size; r++) {
                fprintf(csv_file, ",%d", r);
            }
            fprintf(csv_file, "\n");

            // Write data rows
            for (int i = 0; i < num_sizes; i++) {
                fprintf(csv_file, "%zu", sizes[i]);
                for (int r = 0; r < size; r++) {
                    fprintf(csv_file, ",%.2f", all_latencies[r * num_sizes + i]);
                }
                fprintf(csv_file, "\n");
            }

            fclose(csv_file);
            free(all_latencies);
        }

        // Note: csv_filename is now freed in main, not here
    }

    // Clean up
    free(latencies);
    if (allocated_memory != NULL) {
        free(allocated_memory);
    }
    
    return 0;
} 
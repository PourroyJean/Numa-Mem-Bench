// =============================================================================
// bench_common.c
// Common functionality for NUMA benchmarks
// Author: Jean Pourroy
// Organization: HPE
// License: MIT (See LICENSE file for details)
// =============================================================================

#include "bench_common.h"

// Standard sizes for range expansion
const size_t STANDARD_SIZES[NUM_STANDARD_SIZES] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
};

void get_cpu_info(hwloc_topology_t topology, int *cpu_id, int *core_numa) {
    // Get current CPU directly using sched_getcpu
    #ifdef _GNU_SOURCE
    *cpu_id = sched_getcpu();
    #else
    *cpu_id = -1;
    #endif
    
    if (*cpu_id >= 0) {
        // Get NUMA node for the current CPU
        *core_numa = numa_node_of_cpu(*cpu_id);
        if (*core_numa == -1) {
            fprintf(stderr, "Warning: Could not determine NUMA node for CPU %d, defaulting to 0\n", *cpu_id);
            *core_numa = 0;
        }
    } else {
        // Fallback to hwloc if sched_getcpu fails
        hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
        if (hwloc_get_cpubind(topology, cpuset, 0) < 0) {
            hwloc_get_last_cpu_location(topology, cpuset, 0);
        }
        
        hwloc_obj_t obj = hwloc_get_first_largest_obj_inside_cpuset(topology, cpuset);
        if (obj) {
            *cpu_id = obj->logical_index;
            *core_numa = numa_node_of_cpu(*cpu_id);
        } else {
            *cpu_id = -1;
            *core_numa = -1;
        }
        
        hwloc_bitmap_free(cpuset);
    }
}

int get_numa_node_of_address(void *addr) {
    unsigned long node;
    
    if (get_mempolicy((int *)&node, NULL, 0, addr, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
        return (int)node;
    }
    
    return -1;
}

// Helper function to add a size to the size array if it's not already there
static int add_size_if_unique(size_t *sizes, int *num_sizes, size_t size_to_add) {
    // Check if we've hit the maximum number of sizes
    if (*num_sizes >= MAX_SIZES) {
        return -1; // Too many sizes
    }
    
    // Verify the size is positive
    if (size_to_add <= 0) {
        return -2; // Invalid size
    }
    
    // Check if size already exists in the array
    for (int i = 0; i < *num_sizes; i++) {
        if (sizes[i] == size_to_add) {
            return 0; // Size already exists, no need to add
        }
    }
    
    // Add the size to the array
    sizes[*num_sizes] = size_to_add;
    (*num_sizes)++;
    return 0;
}

int parse_args(int argc, char **argv, bench_params_t *params) {
    memset(params, 0, sizeof(bench_params_t));
    params->sizes_count = 0;
    params->serial_mode = 0;
    params->csv_file = NULL;

    // Default values
    strcpy(params->csv_delimiter, DEFAULT_CSV_DELIMITER);
    
    // Check for help flag first
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help(argv[0]);
            return 1; // Signal that we should exit after help
        }
    }

    // Parse other arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--serial") == 0) {
            params->serial_mode = 1;
            continue;
        }
        
        // Check for --csv option
        if (strncmp(argv[i], "--csv=", 6) == 0) {
            params->csv_file = strdup(argv[i] + 6);
            if (!params->csv_file) {
                fprintf(stderr, "Error: Failed to allocate memory for CSV filename\n");
                return -1;
            }
            continue;
        }
        
        // Check for --size option
        if (strncmp(argv[i], "--size=", 7) == 0) {
            char *size_str = argv[i] + 7;
            
            // Check for range format (e.g., 128-1024)
            char *dash = strchr(size_str, '-');
            if (dash) {
                *dash = '\0'; // Split the string at the dash
                char *start_str = size_str;
                char *end_str = dash + 1;
                
                // Parse start and end of range
                char *endptr;
                long start_size = strtol(start_str, &endptr, 10);
                if (*endptr != '\0' || start_size <= 0) {
                    fprintf(stderr, "Error: Invalid range start value '%s'\n", start_str);
                    return -1;
                }
                
                long end_size = strtol(end_str, &endptr, 10);
                if (*endptr != '\0' || end_size <= 0) {
                    fprintf(stderr, "Error: Invalid range end value '%s'\n", end_str);
                    return -1;
                }
                
                if (start_size > end_size) {
                    fprintf(stderr, "Error: Range start (%ld) is greater than range end (%ld)\n", 
                            start_size, end_size);
                    return -1;
                }
                
                // Find indices in standard size array
                int start_idx = -1, end_idx = -1;
                for (int j = 0; j < NUM_STANDARD_SIZES; j++) {
                    if (STANDARD_SIZES[j] >= (size_t)start_size && start_idx == -1) {
                        start_idx = j;
                    }
                    if (STANDARD_SIZES[j] <= (size_t)end_size) {
                        end_idx = j;
                    }
                }
                
                if (start_idx == -1 || end_idx == -1 || start_idx > end_idx) {
                    fprintf(stderr, "Error: Unable to find standard sizes in range %ld-%ld\n", 
                            start_size, end_size);
                    fprintf(stderr, "Valid range is from %zu to %zu MB\n", 
                            STANDARD_SIZES[0], STANDARD_SIZES[NUM_STANDARD_SIZES-1]);
                    return -1;
                }
                
                // Add all sizes in the range
                for (int j = start_idx; j <= end_idx; j++) {
                    if (add_size_if_unique(params->sizes, &params->sizes_count, STANDARD_SIZES[j]) < 0) {
                        fprintf(stderr, "Error: Too many sizes specified (max %d)\n", MAX_SIZES);
                        return -1;
                    }
                }
            }
            // Check for comma-separated list (e.g., 128,512,1024)
            else if (strchr(size_str, ',')) {
                char *token;
                char *saveptr;
                char *size_list = strdup(size_str); // Create a copy to tokenize
                
                if (!size_list) {
                    fprintf(stderr, "Error: Memory allocation failed for size list\n");
                    return -1;
                }
                
                // Parse comma-separated list
                token = strtok_r(size_list, ",", &saveptr);
                while (token != NULL) {
                    char *endptr;
                    long parsed_size = strtol(token, &endptr, 10);
                    
                    if (*endptr != '\0' || parsed_size <= 0) {
                        fprintf(stderr, "Error: Invalid size value '%s' in list\n", token);
                        free(size_list);
                        return -1;
                    }
                    
                    if (add_size_if_unique(params->sizes, &params->sizes_count, (size_t)parsed_size) < 0) {
                        fprintf(stderr, "Error: Too many sizes specified (max %d)\n", MAX_SIZES);
                        free(size_list);
                        return -1;
                    }
                    
                    token = strtok_r(NULL, ",", &saveptr);
                }
                
                free(size_list);
            }
            // Single value format (e.g., 512)
            else {
                char *endptr;
                long parsed_size = strtol(size_str, &endptr, 10);
                
                if (*endptr != '\0' || parsed_size <= 0) {
                    fprintf(stderr, "Error: Invalid size value '%s'\n", size_str);
                    return -1;
                }
                
                if (add_size_if_unique(params->sizes, &params->sizes_count, (size_t)parsed_size) < 0) {
                    fprintf(stderr, "Error: Too many sizes specified (max %d)\n", MAX_SIZES);
                    return -1;
                }
            }
            
            continue;
        }
        
        // Unrecognized option
        fprintf(stderr, "Warning: Unrecognized option '%s'\n", argv[i]);
    }
    
    // If no size was specified, use the default
    if (params->sizes_count == 0) {
        params->sizes[0] = DEFAULT_ALLOC_SIZE_MB;
        params->sizes_count = 1;
    }
    
    return 0;
}

void print_results_table_header(int num_sizes, size_t *sizes) {
    // Calculate the width for latency section based on number of sizes
    const int column_width = 9; // Fixed width for each column (7 for value + 1 space + 1 separator)
    int latency_section_width = num_sizes * column_width + 1; // +1 for final separator
    
    // Calculate padding for centering "LATENCY (ns)"
    int latency_text_length = 13; // Length of "LATENCY (ns)"
    int padding_before = ((latency_section_width - latency_text_length) / 2) - 1;
    int padding_after = latency_section_width - latency_text_length - padding_before -1;
    
    // Calculate total width of the entire table
    int fixed_section_width = 38; // Width of MPI+CPU+MEMORY sections
    int total_width = fixed_section_width + latency_section_width+5;
    
    // Print top line
    printf("\n ");
    for (int i = 0; i < total_width; i++) {
        printf("=");
    }
    printf("\n");
    
    // Print header row with LATENCY centered
    printf("|  MPI  |        CPU     |         MEMORY    |");
    
    // Print padding before LATENCY
    for (int i = 0; i < padding_before; i++) {
        printf(" ");
    }
    
    printf("LATENCY (ns)");
    
    // Print padding after LATENCY
    for (int i = 0; i < padding_after; i++) {
        printf(" ");
    }
    printf("|\n");
    
    // Print first separator line
    printf("|-------|---------|------|-----------|-------|");
    for (int i = 0; i < num_sizes; i++) {
        printf("--------|");
    }
    printf("\n");
    
    // Print column headers row
    printf("| Ranks | Cores   | NUMA |  Address  | NUMA  |");
    
    // Print each size as a column header with fixed width, including MB unit
    for (int i = 0; i < num_sizes; i++) {
        // Format size with MB unit - always use MB
        char size_text[10];
        snprintf(size_text, sizeof(size_text), "%zuMB", sizes[i]);
        printf(" %-7s|", size_text);
    }
    printf("\n");
    
    // Print second separator line
    printf("|-------|---------|------|-----------|-------|");
    for (int i = 0; i < num_sizes; i++) {
        printf("--------|");
    }
    printf("\n");
    
    fflush(stdout);
}

void print_results_table_row(int rank, int world_size, int cpu_id, int core_numa, void *addr, int num_sizes, double *latencies) {
    // Suppress unused parameter warning
    (void)cpu_id;
    
    // Use strict ordering to print ranks sequentially
    for (int r = 0; r < world_size; r++) {
        if (rank == r) {
            char cpu_list[256] = "N/A";
            
            // Get CPU affinity information using hwloc
            hwloc_topology_t topology;
            hwloc_topology_init(&topology);
            hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM);
            hwloc_topology_load(topology);
            
            hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
            if (hwloc_get_cpubind(topology, cpuset, 0) == 0) {
                int first = -1;
                int last = -1;
                int count = 0;
                
                // Get the first and last CPU in the set
                first = hwloc_bitmap_first(cpuset);
                last = hwloc_bitmap_last(cpuset);
                count = hwloc_bitmap_weight(cpuset);
                
                if (count > 0) {
                    // For hyperthreading, we expect exactly 2 cores (physical + hyperthread)
                    if (count == 2) {
                        snprintf(cpu_list, sizeof(cpu_list), "%d,%d", first, last);
                    } else {
                        // Fallback to showing just the current CPU
                        snprintf(cpu_list, sizeof(cpu_list), "%d", first);
                    }
                }
            }
            
            hwloc_bitmap_free(cpuset);
            hwloc_topology_destroy(topology);
            
            // Get NUMA maps information
            char numa_maps_info[256] = "N/A";
            int node = get_numa_node_of_address(addr);
            if (node >= 0) {
                snprintf(numa_maps_info, sizeof(numa_maps_info), "%d", node);
            }
            
            // Print fixed part of the row
            printf("|  %03d  | %-7s |   %-2d | %-9p |   %-2s  |", 
                   rank, cpu_list, core_numa, addr, numa_maps_info);
            
            // Print each latency measurement with fixed width formatting (7 chars for value)
            for (int i = 0; i < num_sizes; i++) {
                printf(" %-6.2f |", latencies[i]);
            }
            
            printf("\n");
            fflush(stdout);
        }
        
        // Synchronize after each rank prints
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Final global barrier to ensure all ranks have completed printing
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_results_table_footer(int num_sizes) {
    // Calculate total width based on known dimensions
    const int column_width = 9; // Same as in header
    int latency_section_width = num_sizes * column_width + 1;
    int fixed_section_width = 38; // Width of MPI+CPU+MEMORY sections
    int total_width = fixed_section_width + latency_section_width + 5;
    
    // Ensure output buffer is flushed before printing footer
    fflush(stdout);
    
    // Print bottom line with correct width
    printf(" ");
    for (int i = 0; i < total_width; i++) {
        printf("=");
    }
    printf("\n");
    
    // Ensure footer is flushed immediately
    fflush(stdout);
}

int collect_mapping_info(int rank, int size, int cpu_id, int core_numa, void *addr) {
    struct mapping_info my_info;
    struct mapping_info *all_info = NULL;
    
    // Fill in local information
    my_info.rank = rank;
    my_info.cpu_id = cpu_id;
    my_info.cpu_numa = core_numa;
    my_info.memory_numa = get_numa_node_of_address(addr);
    
    // Allocate buffer on rank 0
    if (rank == 0) {
        all_info = malloc(size * sizeof(struct mapping_info));
        if (!all_info) {
            fprintf(stderr, "Error: Failed to allocate memory for mapping info\n");
            return -1;
        }
    }
    
    // Gather all information to rank 0
    MPI_Gather(&my_info, sizeof(struct mapping_info), MPI_BYTE,
               all_info, sizeof(struct mapping_info), MPI_BYTE,
               0, MPI_COMM_WORLD);
    
    // Rank 0 writes the mapping file
    if (rank == 0) {
        int ret = write_mapping_info(MAPPING_FILE, all_info, size);
        free(all_info);
        if (ret != 0) {
            return -1;
        }
    }
    
    return 0;
}

int write_mapping_info(const char *filename, struct mapping_info *info, int world_size) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Failed to open mapping file %s\n", filename);
        return -1;
    }

    fprintf(file, "rank,cpu_id,cpu_numa,memory_numa\n");
    for (int i = 0; i < world_size; i++) {
        fprintf(file, "%d,%d,%d,%d\n",
                info[i].rank,
                info[i].cpu_id,
                info[i].cpu_numa,
                info[i].memory_numa);
    }

    fclose(file);
    return 0; 
}

void print_help(const char *program_name) {
    printf("\nNuma-Mem-Bench: A comprehensive NUMA performance benchmarking suite\n");
    printf("Usage: %s [options]\n\n", program_name);
    printf("Options:\n");
    printf("  --help, -h             Display this help message and exit\n");
    printf("  --size=SIZE            Memory size(s) in MB. Can be:\n");
    printf("                           - Single value: --size=512\n");
    printf("                           - Range: --size=128-1024 (uses standard sizes)\n");
    printf("                           - List: --size=1,4,16,64,256\n");
    printf("                         Default is %d MB\n", DEFAULT_ALLOC_SIZE_MB);
    printf("  --serial               Run in serial mode (one rank at a time)\n");
    printf("                         Default is parallel mode (all ranks simultaneously)\n");
    printf("  --csv=FILE             Output results to CSV file\n");
    printf("\nStandard sizes (MB): ");
    for (int i = 0; i < NUM_STANDARD_SIZES; i++) {
        printf("%zu", STANDARD_SIZES[i]);
        if (i < NUM_STANDARD_SIZES - 1) {
            printf(", ");
        }
    }
    printf("\n\n");
    printf("Use external tools like numactl to control NUMA binding:\n");
    printf("Example: srun --nodes=1 --ntasks=4 numactl --membind=1 %s --size=16-512\n\n", program_name);
} 
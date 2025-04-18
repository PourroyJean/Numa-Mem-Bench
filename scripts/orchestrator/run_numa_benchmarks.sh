#!/bin/bash

# =============================================================================
# Script: run_numa_benchmarks.sh
# Author: Jean Pourroy
# Organization: HPE
# License: MIT (See LICENSE file for details)
# Version: 1.1
# =============================================================================
# This script runs NUMA-aware benchmarks with different memory allocation strategies
# and core binding configurations. It supports both sequential and interleaved
# memory allocation across specified NUMA domains.
#
# Features:
# - Supports sequential and interleaved memory allocation strategies
# - Configurable NUMA domain selection
# - Automatic core binding based on system topology
# - Memory size scaling
# - Dry-run mode for testing configurations
# - Custom job directory labels
#
# Usage:
#   ./run_numa_benchmarks.sh --arch <config> --wrapper <path> --benchmark <path> [options]
#
# Required:
#   --arch <config>         Path to architecture configuration file
#   --wrapper <path>        Path to wrapper_numa.sh script
#   --benchmark <path>      Path to numa_mem_bench executable
#
# Options:
#   --sequential <domains>  Run with sequential CPU allocation
#   --interleaved <domains> Run with interleaved CCDs
#   --memory-sizes <sizes>  Comma-separated list of memory sizes in MB to test
#   --dry-run               Show commands without executing
#   --label <string>        Add a label to the job directory name
#   --help                  Display help message
#
# Example:
#  sbatch ./run_numa_benchmarks.sh --arch architectures/lumi_partc --wrapper scripts/wrapper_numa.sh --benchmark bin/numa_mem_bench --interleaved 0,1,2 --label my_test
# =============================================


## SBATCH --account=${SLURM_ACCOUNT:-project_462000031}
## SBATCH --partition=${SLURM_PARTITION:-standard}
#SBATCH --nodes 1
#SBATCH --job-name numa_bench
#SBATCH --output=slurm_run_numa_benchmarks_%j.log
#SBATCH --time 00:60:00
#SBATCH --hint=nomultithread
set -e # Exit on any error 

VERSION="1.0"

# Configuration
MEMORY_SIZES="8" # Default memory size in MB if not specified
RESULTS_DIR="results_scaling"
JOB_DIR="$(\command pwd)/$RESULTS_DIR/job_${SLURM_JOB_ID}"


# Default values
DRY_RUN=false               # If true, show commands without executing
MODE=""                     # Mode to run the benchmark in : sequential or interleaved
DOMAINS=""                  # List of NUMA domains to run the benchmark on
ARCH_CONFIG=""              # Path to architecture configuration file
LABEL=""                    # Optional label for job directory
declare -A SKIP_CORES_MAP   # Associative array to track cores that should be skipped
SEP_LONG="================================================================================"
SEP_SHORT="--------------------------------------------------------------------------------"

# Get the directory where this script is located
# Handle both direct execution and SLURM submission
if [ -n "$SLURM_JOB_ID" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(\command cd "$(\command dirname "${BASH_SOURCE[0]}")" && \command pwd)"
    DRY_RUN=true
fi
WRAPPER_PATH=""       # Path to wrapper_numa.sh script
BENCHMARK_PATH=""     # Path to numa_mem_bench executable
WRAPPER_NAME=""       # Basename of wrapper script
BENCHMARK_NAME=""     # Basename of benchmark executable
REQUIRED_FILES=()     # Array of required file paths


# =============================================
# Function Definitions
# =============================================

# Function to display help message
show_help() {
    echo "Usage: $0 --arch <architecture_config> --wrapper <path> --benchmark <path> [options]"
    echo ""
    echo "Required:"
    echo "  --arch <config>            Path to architecture configuration file"
    echo "  --wrapper <path>           Path to wrapper_numa.sh script"
    echo "  --benchmark <path>         Path to numa_mem_bench executable"
    echo ""
    echo "Options:"
    echo "  --sequential <domains>     Run benchmark with sequential CPU allocation on specified domains"
    echo "                             (comma-separated list, e.g. 0,1,2)"
    echo "                             Example of cores given by sequential allocation for domain 0: 0,1,2,3,4,5,6,7..."
    echo ""
    echo "  --interleaved <domains>    Run benchmark with interleaved CCDs on specified domains"
    echo "                             (comma-separated list, e.g. 0,1,2)"
    echo "                             Example of cores given by interleaved allocation for 2 CCDs for domain 0: 0,8,1,9,2,10,3,11..."
    echo ""
    echo "  --memory-sizes <sizes>     Memory size(s) in MB to test. Can be:"
    echo "                             - Single value: --memory-sizes 512"
    echo "                             - Range: --memory-sizes 128-1024"
    echo "                             - List: --memory-sizes 1,4,16,64,256"
    echo "                             (default: 8)"
    echo ""
    echo "  --dry-run                  Show commands without executing"
    echo ""
    echo "  --label <string>           Add a label to the job directory name"
    echo "                             (e.g. 'my_numa_test' creates job_123_my_numa_test)"
    echo ""
    echo "  --help                     Display this help message"
    echo ""
    echo "Description:"
    echo "  This tool runs NUMA-aware benchmarks to measure memory latency across different"
    echo "  NUMA domains with various binding configurations. It supports both sequential"
    echo "  and interleaved memory allocation strategies."
    echo ""
    echo "Examples:"
    echo "  $0 --arch architectures/lumi_partc --wrapper scripts/wrapper_numa.sh --benchmark bin/numa_mem_bench --interleaved 0 --dry-run"
    echo "  $0 --arch architectures/lumi_partc --wrapper scripts/wrapper_numa.sh --benchmark bin/numa_mem_bench --sequential 0,1,2,3 --memory-sizes 1,4,16,64 --label my_test"
    echo "$SEP_LONG"
    echo ""
    exit 0
}

# Helper function to parse basic system information
# Sets: SYSTEM_NAME, PARTITION, CCDS_PER_NUMA
parse_basic_system_info() {
    local config_file="$1"
    local -n error_list=$2  # nameref to errors array, renamed to avoid circular reference
    
    # Extract system info
    SYSTEM_NAME=$(grep "^SYSTEM_NAME=" "$config_file" | cut -d'"' -f2)
    PARTITION=$(grep "^PARTITION=" "$config_file" | cut -d'"' -f2)
    CCDS_PER_NUMA=$(grep "^CCDS_PER_NUMA=" "$config_file" | awk -F'=' '{print $2}' | awk '{print $1}')
    
    # Validate CCD configuration
    if [ -z "$CCDS_PER_NUMA" ]; then
        error_list+=("CCD configuration missing in $config_file")
        return 1
    elif ! [[ "$CCDS_PER_NUMA" =~ ^[0-9]+$ ]]; then
        error_list+=("CCDS_PER_NUMA ($CCDS_PER_NUMA) is not a valid number")
        return 1
    fi
    
    return 0
}

# Helper function to parse NUMA topology information
# Sets: NUMA_NODES, NUMA_CPUS
parse_numa_topology() {
    local config_file="$1"
    local -n error_list=$2  # nameref to errors array, renamed to avoid circular reference
    
    # Extract NUMA node count using progressively more general patterns
    NUMA_NODES=$(grep -E "^NUMA node\(s\):|^  NUMA node\(s\):" "$config_file" | awk '{print $NF}' || true)
    
    if [ -z "$NUMA_NODES" ]; then
        NUMA_NODES=$(grep -E "NUMA node\(s\):" "$config_file" | \
            awk '{for(i=1;i<=NF;i++) if($i=="node(s):") print $(i+1)}' || true)
    fi
    
    if [ -z "$NUMA_NODES" ] || ! [[ "$NUMA_NODES" =~ ^[0-9]+$ ]]; then
        error_list+=("Could not detect valid NUMA node count from configuration file")
        return 1
    fi
    
    # Parse CPU ranges for each NUMA domain
    declare -g -A NUMA_CPUS
    local parse_failed=false
    
    for i in $(seq 0 $((NUMA_NODES-1))); do
        # Try different pattern formats
        local cpu_range=""
        for pattern in "^NUMA node$i CPU\(s\):" "^  NUMA node$i CPU\(s\):" "NUMA node$i CPU\(s\):"; do
            cpu_range=$(grep -E "$pattern" "$config_file" | sed -E "s/.*CPU\(s\)://g" | tr -d ' ' || true)
            [ -n "$cpu_range" ] && break
        done
        
        if [ -n "$cpu_range" ]; then
            NUMA_CPUS[$i]=$cpu_range
        else
            error_list+=("Could not find CPU range for NUMA domain $i")
            parse_failed=true
        fi
    done
    
    [ "$parse_failed" = true ] && return 1
    return 0
}

# Helper function to parse SKIP_CORES directive
# Sets: SKIP_CORES_MAP
parse_skip_cores() {
    local config_file="$1"
    local -n error_list=$2  # nameref to errors array, renamed to avoid circular reference
    
    # Initialize global associative array
    declare -g -A SKIP_CORES_MAP
    
    # Check for SKIP_CORES directive
    local skip_cores_line=$(grep "^SKIP_CORES=" "$config_file" || true)
    
    if [ -n "$skip_cores_line" ]; then
        local skip_cores_list=$(echo "$skip_cores_line" | cut -d'=' -f2 | tr -d ' ')
        local OLDIFS="$IFS"
        local IFS=','
        read -ra SKIP_CORES_ARRAY <<< "$skip_cores_list"
        IFS="$OLDIFS"
        
        local invalid_cores=false
        for core in "${SKIP_CORES_ARRAY[@]}"; do
            if [[ "$core" =~ ^[0-9]+$ ]]; then
                SKIP_CORES_MAP["$core"]=1
            else
                error_list+=("Invalid core ID in SKIP_CORES: $core")
                invalid_cores=true
            fi
        done
        
        [ "$invalid_cores" = true ] && return 1
    fi
    
    return 0
}

# Helper function to calculate derived metrics
# Sets: CORES_PER_CCD, CPUS_PER_DOMAIN
# Requires: NUMA_CPUS, CCDS_PER_NUMA to be set
calculate_derived_metrics() {
    local -n error_list=$1  # nameref to errors array, renamed to avoid circular reference
    
    if [ ${#NUMA_CPUS[@]} -gt 0 ]; then
        local first_domain_cores_range=$(echo "${NUMA_CPUS[0]}" | cut -d',' -f1)
        local cores_in_domain=$(echo "$first_domain_cores_range" | tr '-' ' ' | xargs -r seq | wc -l)
        
        if [ "$cores_in_domain" -eq 0 ]; then
            error_list+=("Could not calculate cores in domain - empty or invalid range")
            return 1
        elif [ "$CCDS_PER_NUMA" -eq 0 ]; then
            error_list+=("Invalid CCDS_PER_NUMA value - cannot be zero")
            return 1
        fi
        
        # Set global variables
        declare -g CORES_PER_CCD=$((cores_in_domain / CCDS_PER_NUMA))
        declare -g CPUS_PER_DOMAIN=$((CORES_PER_CCD * CCDS_PER_NUMA))
    else
        error_list+=("No NUMA CPU information available to calculate metrics")
        return 1
    fi
    
    return 0
}

# Main configuration parsing function
parse_arch_config() {
    local config_file="$1"
    local -a errors=()  # Define as array, not nameref
    local exit_status=0
    
    echo "# Architecture Configuration "
    
    # Parse each component, collecting any errors
    parse_basic_system_info "$config_file" errors || exit_status=1
    parse_numa_topology     "$config_file" errors || exit_status=1
    parse_skip_cores        "$config_file" errors || exit_status=1
    
    # Only calculate derived metrics if previous parsing was successful
    if [ $exit_status -eq 0 ]; then
        calculate_derived_metrics errors || exit_status=1
    fi
    
    # Output configuration summary
    echo "System: $SYSTEM_NAME"
    echo "Partition: $PARTITION"
    echo "CCDs per NUMA domain: $CCDS_PER_NUMA"
    echo "Detected $NUMA_NODES NUMA nodes"
    
    if [ ${#SKIP_CORES_MAP[@]} -gt 0 ]; then
        echo "Found SKIP_CORES directive: ${!SKIP_CORES_MAP[*]}"
    else
        echo "No SKIP_CORES directive found, using all cores"
    fi
    
    echo "$SEP_SHORT"
    
    # If any errors occurred, print them all and exit
    if [ ${#errors[@]} -gt 0 ]; then
        echo "Error: Architecture configuration file '$config_file' has the following issues:"
        for error in "${errors[@]}"; do
            echo "  - $error"
        done
        exit 1
    fi
}

# Parse domain list and validate
parse_domains() {
    local domain_list="$1"
    local parsed_domains=()
    
    # Parse comma-separated list
    # Save original IFS
    local OLDIFS="$IFS"
    local IFS=','
    read -ra DOMAIN_ARRAY <<< "$domain_list"
    # Restore original IFS
    IFS="$OLDIFS"
    
    for domain in "${DOMAIN_ARRAY[@]}"; do
        # Check if domain is a valid number
        if ! [[ "$domain" =~ ^[0-9]+$ ]]; then
            echo "Error: Domain must be a number, got '$domain'"
            exit 1
        fi
        
        # Check if domain is in range
        if [ "$domain" -ge "$NUMA_NODES" ]; then
            echo "Error: Domain must be between 0 and $((NUMA_NODES-1)), got '$domain'"
            exit 1
        fi
        
        parsed_domains+=("$domain")
    done
    
    # Set global DOMAIN_ARRAY variable
    declare -g -a DOMAIN_ARRAY=("${parsed_domains[@]}")
}

# Function to get CPU list for a domain, filtering out skipped cores
get_cpu_list() {
    local domain="$1"
    local num_ranks="$2"
    local total_ranks="$3"
    
    # Save original error handling state
    local original_error_state=$(set +o | grep errexit)
    
    # Temporarily disable exit on error to handle it ourselves
    set +e
    
    # Get CPU range for this domain
    local cpu_range="${NUMA_CPUS[$domain]}"
    if [ -z "$cpu_range" ]; then
        echo "[ ERROR ] No CPU range found for NUMA domain $domain" >&2
        # Restore original error handling state
        eval "$original_error_state"
        return 1
    fi
    
    # We only care about physical cores (first part before comma), not hyperthreads
    local cores_range=$(echo "$cpu_range" | cut -d',' -f1)
    
    # Convert range to list
    local all_cpu_list=($(echo "$cores_range" | tr '-' ' ' | xargs seq))
    
    # Filter out cores that should be skipped
    local filtered_cpu_list=()
    local skipped_count=0
    for cpu in "${all_cpu_list[@]}"; do
        if [ -z "${SKIP_CORES_MAP[$cpu]}" ]; then
            filtered_cpu_list+=("$cpu")
        else
            echo "[ $total_ranks RANKS ] - Skipping CPU $cpu in domain $domain as specified in SKIP_CORES" >&2
            skipped_count=$((skipped_count + 1))
        fi
    done
    
    # Check for CPU oversubscription
    if [ "$num_ranks" -gt "${#filtered_cpu_list[@]}" ]; then
        echo "[ $total_ranks RANKS ] Notice: Can only use ${#filtered_cpu_list[@]} cores on domain $domain (requested $num_ranks)" >&2
        echo "[ $total_ranks RANKS ] Domain $domain has ${#all_cpu_list[@]} total cores but $skipped_count were excluded by SKIP_CORES" >&2
        echo "[ $total_ranks RANKS ] Limiting to the maximum available cores (${#filtered_cpu_list[@]})" >&2
        
        # Restore original error handling state
        eval "$original_error_state"
        return 1
    fi
    
    # Take the first num_ranks CPUs from the filtered list and join them with commas
    local result=()
    for ((i=0; i<num_ranks; i++)); do
        result+=("${filtered_cpu_list[$i]}")
    done
    
    # Join array elements with commas
    local OLDIFS="$IFS"
    local IFS=','
    echo "${result[*]}"
    IFS="$OLDIFS"
    
    # Restore original error handling state
    eval "$original_error_state"
    return 0
}

# Function to get interleaved CPU list, filtering out skipped cores
get_interleaved_cpu_list() {
    local domain="$1"
    local num_ranks="$2"
    local total_ranks="$3"
    
    # Save original error handling state
    local original_error_state=$(set +o | grep errexit)
    
    # Temporarily disable exit on error to handle it ourselves
    set +e
    
    # Get CPU range for this domain
    local cpu_range="${NUMA_CPUS[$domain]}"
    if [ -z "$cpu_range" ]; then
        echo "[ ERROR ] No CPU range found for NUMA domain $domain" >&2
        # Restore original error handling state
        eval "$original_error_state"
        return 1
    fi
    
    # We only care about physical cores (first part before comma), not hyperthreads
    local cores_range=$(echo "$cpu_range" | cut -d',' -f1)
    
    # Convert range to list of CPU IDs
    local all_cores=($(echo "$cores_range" | tr '-' ' ' | xargs seq))
    
    # Filter out cores that should be skipped
    local filtered_cores=()
    local skipped_count=0
    for cpu in "${all_cores[@]}"; do
        if [ -z "${SKIP_CORES_MAP[$cpu]}" ]; then
            filtered_cores+=("$cpu")
        else
            echo "[ $total_ranks RANKS ] - Skipping CPU $cpu in domain $domain as specified in SKIP_CORES" >&2
            skipped_count=$((skipped_count + 1))
        fi
    done
    
    local total_cores=${#filtered_cores[@]}
    
    # Check for CPU oversubscription
    if [ "$num_ranks" -gt "$total_cores" ]; then
        echo "[ $total_ranks RANKS ] Notice: Can only use $total_cores cores on domain $domain (requested $num_ranks)" >&2
        echo "[ $total_ranks RANKS ] Domain $domain has ${#all_cores[@]} total cores but $skipped_count were excluded by SKIP_CORES" >&2
        echo "[ $total_ranks RANKS ] Limiting to the maximum available cores ($total_cores)" >&2
        
        # Restore original error handling state before return
        eval "$original_error_state"
        return 1
    fi
    
    # Create a properly interleaved list of cores based on CCDs
    local result=()
    
    # Create a list of core IDs for each CCD, skipping the ones that should be skipped
    local -a ccd_cores=()
    for ((ccd=0; ccd<CCDS_PER_NUMA; ccd++)); do
        ccd_cores[$ccd]=""
    done
    
    # Assign cores to their respective CCDs
    for cpu in "${filtered_cores[@]}"; do
        # Calculate which CCD this core belongs to
        # Find the original index in all_cores array
        local original_index=0
        for ((i=0; i<${#all_cores[@]}; i++)); do
            if [ "${all_cores[$i]}" -eq "$cpu" ]; then
                original_index=$i
                break
            fi
        done
        
        # Which CCD does this core belong to
        local ccd_index=$((original_index / CORES_PER_CCD))
        if [ "$ccd_index" -lt "$CCDS_PER_NUMA" ]; then
            if [ -n "${ccd_cores[$ccd_index]}" ]; then
                ccd_cores[$ccd_index]="${ccd_cores[$ccd_index]} $cpu"
            else
                ccd_cores[$ccd_index]="$cpu"
            fi
        fi
    done
    
    # Now take cores in an interleaved fashion from each CCD
    local idx=0
    while [ "$idx" -lt "$num_ranks" ]; do
        for ((ccd=0; ccd<CCDS_PER_NUMA; ccd++)); do
            # Check if this CCD has any cores left
            if [ -n "${ccd_cores[$ccd]}" ]; then
                # Get the first core from this CCD - use read to avoid subshell
                local next_core
                read -r next_core _ <<< "${ccd_cores[$ccd]}"
                
                # Remove it from the CCD's list with proper word boundary matching
                ccd_cores[$ccd]=$(echo "${ccd_cores[$ccd]}" | sed -E "s/^$next_core[[:space:]]?|[[:space:]]$next_core[[:space:]]?|[[:space:]]$next_core$/ /g" | tr -s ' ')
                
                # Add it to the result
                result+=("$next_core")
                
                # Increment the index
                idx=$((idx + 1))
                
                # Stop if we've got enough cores
                if [ "$idx" -eq "$num_ranks" ]; then
                    break
                fi
            fi
        done
        
        # If we went through all CCDs and couldn't get any more cores, break the loop
        local all_empty=true
        for ((ccd=0; ccd<CCDS_PER_NUMA; ccd++)); do
            if [ -n "${ccd_cores[$ccd]}" ]; then
                all_empty=false
                break
            fi
        done
        if [ "$all_empty" = true ]; then
            break
        fi
    done
    
    # Join array elements with commas
    if [ ${#result[@]} -gt 0 ]; then
        # Save original IFS
        local OLDIFS="$IFS"
        local IFS=','
        echo "${result[*]}"
        # Restore original IFS
        IFS="$OLDIFS"
    else
        echo ""
    fi
    
    # Restore original error handling state before return
    eval "$original_error_state"
    return 0
}

# Helper function to generate output filename based on benchmark configuration
# Returns: Output filename for the benchmark results
generate_output_filename() {
    local allocation_type="$1"
    local domains=("${@:2}")  # Get all arguments after the first one
    local total_domains=${#domains[@]}
    local ranks_per_domain="$ranks_per_domain"
    local total_ranks=$((ranks_per_domain * total_domains))
    local size="$size"  # These are from parent scope
    
    local output_file
    # Save original IFS for domain_list generation
    local OLDIFS="$IFS"
    local IFS=','
    local domain_list="${domains[*]}"
    IFS="$OLDIFS"
    
    if [ $total_domains -eq 1 ]; then
        output_file="${allocation_type}_domain${domains[0]}_${total_ranks}ranks_${size}MB.csv"
    else
        output_file="${allocation_type}_domains${domain_list}_${total_ranks}ranks_${size}MB.csv"
    fi
    
    echo "$output_file"
}

# Helper function to build NUMA binding string
# Returns: Comma-separated list of NUMA domains for each rank
build_numa_binding() {
    local ranks_per_domain="$1"
    shift
    local domains=("$@")
    
    local numa_binding=""
    for domain in "${domains[@]}"; do
        for ((i=0; i<ranks_per_domain; i++)); do
            [ -n "$numa_binding" ] && numa_binding+=","
            numa_binding+="$domain"
        done
    done
    
    echo "$numa_binding"
}

# Helper function to build CPU binding string
# Returns: Status code (0=success, 1=failure) and binding string via stdout
# Note: Handles temporary disabling of exit-on-error
build_cpu_binding() {
    local allocation_type="$1"
    local ranks_per_domain="$2"
    local total_ranks="$3"
    shift 3
    local domains=("$@")
    
    local cpu_bind=""
    local status=0
    
    # Save original error handling state
    local original_error_state=$(set +o | grep errexit)
    
    # Temporarily disable exit on error to handle it ourselves
    set +e
    
    for domain in "${domains[@]}"; do
        local domain_cpus=""
        if [ "$allocation_type" = "sequential" ]; then
            domain_cpus=$(get_cpu_list "$domain" "$ranks_per_domain" "$total_ranks")
        else
            domain_cpus=$(get_interleaved_cpu_list "$domain" "$ranks_per_domain" "$total_ranks")
        fi
        
        local exit_status=$?
        if [ $exit_status -ne 0 ]; then
            echo "[ $total_ranks RANKS ] Skipping configuration with $ranks_per_domain ranks on domain $domain." >&2
            if [ "$DRY_RUN" = true ]; then
                continue
            else
                status=1
                break
            fi
        fi
        
        # Add to the binding list if we got valid CPUs
        if [ -n "$domain_cpus" ]; then
            [ -n "$cpu_bind" ] && cpu_bind+=","
            cpu_bind+="$domain_cpus"
        fi
    done
    
    # Restore original error handling state
    eval "$original_error_state"
    
    # Return both status and binding string
    echo "$cpu_bind"
    return $status
}

# Main benchmark execution function
run_benchmark() {
    local allocation_type="$1"  # "sequential" or "interleaved"
    local ranks_per_domain="$2"
    local size="$3"
    shift 3
    local domains=("$@")
    local total_domains=${#domains[@]}
    local total_ranks=$((ranks_per_domain * total_domains))
    
    # Generate output filename
    local output_file
    output_file=$(generate_output_filename "$allocation_type" "${domains[@]}")
    
    # Format domain list for output
    local OLDIFS="$IFS"
    local IFS=','
    local domain_list="${domains[*]}"
    IFS="$OLDIFS"
    
    # Print benchmark configuration
    echo ""
    echo "$SEP_LONG"
    echo "[ $total_ranks RANKS ] Benchmark Configuration"
    echo "[ $total_ranks RANKS ] - Allocation Type  : ${allocation_type}"
    echo "[ $total_ranks RANKS ] - Domains          : $domain_list"
    echo "[ $total_ranks RANKS ] - Ranks per Domain : $ranks_per_domain"
    
    # Build NUMA and CPU binding strings
    local numa_binding
    numa_binding=$(build_numa_binding "$ranks_per_domain" "${domains[@]}")
    
    local cpu_bind
    cpu_bind=$(build_cpu_binding "$allocation_type" "$ranks_per_domain" "$total_ranks" "${domains[@]}")
    local binding_status=$?
    
    # Skip if we couldn't get a valid CPU binding
    if [ -z "$cpu_bind" ] && [ "$DRY_RUN" = false ]; then
        echo "[ $total_ranks RANKS ] Insufficient usable cores for this configuration. Skipping benchmark."
        echo "$SEP_LONG"
        return 1
    fi
    
    # Print binding configuration
    echo "[ $total_ranks RANKS ] NUMA binding: $numa_binding"
    echo "[ $total_ranks RANKS ] CPU binding : $cpu_bind"
    
    # Construct the command array
    local cmd=(
        "srun" "--unbuffered" "--nodes" "1" "--ntasks" "$total_ranks"
        "--cpu-bind=map_cpu:$cpu_bind"
        "--hint=nomultithread"
        "$WRAPPER_NAME" "--numa=$numa_binding" "--executable" "./$BENCHMARK_NAME" "--"
        "--size=$MEMORY_SIZES" "--csv=$output_file"
    )

    # Execute or simulate based on dry-run mode
    if [ -n "$cpu_bind" ]; then
        echo "[ $total_ranks RANKS ] Executing : ${cmd[*]}"
        if [ "$DRY_RUN" = false ]; then
            if ! "${cmd[@]}"; then
                echo "[ $total_ranks RANKS ] Benchmark failed"
                echo "$SEP_LONG"
                return 1
            fi
            echo "[ $total_ranks RANKS ] Benchmark completed successfully"
        fi
    else
        echo "[ $total_ranks RANKS ] Skipping benchmark due to insufficient usable cores"
    fi
    echo "$SEP_LONG"
    return $binding_status
}

# Function to create symbolic links
create_symlink() {
    local source_file="$1"
    local target_name=$(basename "$source_file")
    
    if [ -f "$source_file" ] && [ ! -f "$JOB_DIR/$target_name" ]; then
        if ln -s "$(realpath "$source_file")" "$JOB_DIR/$target_name"; then
            echo "[ OK ] Created symlink: $target_name -> $(realpath "$source_file")"
            return 0
        else
            echo "[ ERROR ] Failed to create symlink: $target_name"
            return 1
        fi
    else
        if [ ! -f "$source_file" ]; then
            echo "[ ERROR ] Source file not found: $source_file"
        elif [ -f "$JOB_DIR/$target_name" ]; then
            echo "[ ERROR ] Target already exists: $target_name"
        fi
        return 1
    fi
}

# Function to create run summary
create_run_summary() {
    echo ""
    echo "$SEP_LONG"
    echo "                     Benchmark Run Summary                       "
    echo "$SEP_LONG"
    echo "Date: $(date)"
    echo "Job Directory: $JOB_DIR"
    echo ""
    echo "Configuration:"
    echo "$SEP_SHORT"
    echo "  System: $SYSTEM_NAME"
    echo "  Partition: $PARTITION"
    echo "  NUMA Nodes: $NUMA_NODES"
    echo "  CCDs per NUMA domain: $CCDS_PER_NUMA"
    echo "  Cores per CCD: $CORES_PER_CCD (calculated)"
    
    # Add SKIP_CORES to summary if any are specified
    if [ ${#SKIP_CORES_MAP[@]} -gt 0 ]; then
        echo "  Skipped Cores: ${!SKIP_CORES_MAP[*]}"
    fi
    
    echo ""
    echo "  CPU ranges:"
    for i in "${!NUMA_CPUS[@]}"; do
        echo "    Domain $i: ${NUMA_CPUS[$i]}"
    done
    
    echo ""
    echo "  Mode: $MODE"
    echo "  Domains: ${DOMAINS}"
    echo "  Memory Sizes: $MEMORY_SIZES"
    
    echo "  Dry Run: $DRY_RUN"
    echo "Benchmarks completed successfully"
    echo "All results will be saved under: $JOB_DIR"
}

# Function to parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --arch)
                ARCH_CONFIG="$2"
                if [ ! -f "$ARCH_CONFIG" ]; then
                    echo "Error: Architecture configuration file '$ARCH_CONFIG' not found"
                    exit 1
                fi
                shift 2
                ;;
            --sequential|--interleaved)
                # Store the mode (removing the -- prefix)
                MODE="${1#--}"
                # Check if domain list is provided
                if [ $# -lt 2 ] || [[ "$2" == --* ]]; then
                    echo "Error: Domain list is required for --$MODE"
                    echo "       Example: --$MODE 0,1,2"
                    exit 1
                fi
                DOMAINS="$2"
                shift 2
                ;;
            --memory-sizes)
                # Check if memory sizes list is provided
                if [ $# -lt 2 ] || [[ "$2" == --* ]]; then
                    echo "Error: Memory sizes list is required for --memory-sizes"
                    echo "       Example: --memory-sizes 1,2,4,8,16 or --memory-sizes 1-64"
                    exit 1
                fi
                
                # Store the memory sizes string directly - parsing will be done by numa_mem_bench
                MEMORY_SIZES="$2"
                shift 2
                ;;
            --wrapper)
                WRAPPER_PATH="$2"
                if [ ! -f "$WRAPPER_PATH" ]; then
                    echo "Error: Wrapper script '$WRAPPER_PATH' not found"
                    exit 1
                fi
                WRAPPER_NAME=$(basename "$WRAPPER_PATH")
                shift 2
                ;;
            --benchmark)
                BENCHMARK_PATH="$2"
                if [ ! -f "$BENCHMARK_PATH" ]; then
                    echo "Error: Benchmark executable '$BENCHMARK_PATH' not found"
                    exit 1
                fi
                BENCHMARK_NAME=$(basename "$BENCHMARK_PATH")
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --label)
                LABEL="$2"
                shift 2
                ;;
            --help)
                show_help
                ;;
            *)
                echo "Error: Unknown option $1"
                show_help
                ;;
        esac
    done
    
    # Check required arguments
    if [ -z "$ARCH_CONFIG" ]; then
        echo "Error: Architecture configuration file is required (--arch)"
        show_help
    fi
    
    if [ -z "$WRAPPER_PATH" ]; then
        echo "Error: Wrapper script path is required (--wrapper)"
        show_help
    fi
    
    if [ -z "$BENCHMARK_PATH" ]; then
        echo "Error: Benchmark executable path is required (--benchmark)"
        show_help
    fi
    
    # Validate domain list
    if [ -z "$DOMAINS" ]; then
        echo "Error: No domains specified"
        echo "       Use --sequential or --interleaved with a domain list"
        echo "       Example: --sequential 0,1,2"
        show_help
    fi
    
    # Populate REQUIRED_FILES after parsing arguments
    REQUIRED_FILES=("$WRAPPER_PATH" "$BENCHMARK_PATH")
}

# Function to set up job directory and create necessary links
setup_job_directory() {
    if [ "$DRY_RUN" = false ]; then
        # Handle SLURM output file relocation first
        if [ -n "$LABEL" ]; then
            JOB_DIR="${JOB_DIR}_${LABEL}"
        fi
        mkdir -p "$JOB_DIR"
        echo "Job started on $(date) in directory: $JOB_DIR"    # Force the log file to be created in the job directory
        mv "slurm_run_numa_benchmarks_${SLURM_JOB_ID}.log" "$JOB_DIR/"          # So we can move it to the job directory

        # Create symbolic links to required files
        local links_created=0
        local links_failed=0
        
        for file in "${REQUIRED_FILES[@]}"; do
            if create_symlink "$file"; then
                links_created=$((links_created + 1))
            else
                links_failed=$((links_failed + 1))
                echo "ERROR: Failed to create required symlink for $(basename "$file")"
            fi
        done
        
        echo "Created $links_created symbolic links to required files"
        
        # If any required symlinks failed to be created, exit with error
        if [ $links_failed -gt 0 ]; then
            echo "ERROR: Failed to create $links_failed required symlinks. Exiting."
            exit 1
        fi
        
        # Change to job directory so output files are written there
        cd "$JOB_DIR"
    fi
}

# Function to clean up symbolic links
cleanup() {
    [ "$DRY_RUN" = true ] && return 0
    
    local cleanup_failed=0
    local links_removed=0
    
    # Find all symlinks in the job directory
    local all_symlinks=$(find "$JOB_DIR" -type l -maxdepth 1)
    
    # Create array of basenames for required files
    local -a required_basenames=()
    for file in "${REQUIRED_FILES[@]}"; do
        required_basenames+=("$(basename "$file")")
    done
    
    # Check each symlink found
    for symlink in $all_symlinks; do
        local symlink_basename=$(basename "$symlink")
        
        # Check if this symlink's basename matches any of our required files
        for req_basename in "${required_basenames[@]}"; do
            if [ "$symlink_basename" = "$req_basename" ]; then
                # Found a match, remove the symlink
                if ! rm -f "$symlink"; then
                    echo "Warning: Failed to remove symbolic link: $symlink"
                    cleanup_failed=1
                else
                    links_removed=$((links_removed + 1))
                fi
                break
            fi
        done
    done
    
    if [ $cleanup_failed -eq 0 ]; then
        echo "Successfully cleaned up symbolic links [total removed: $links_removed]"
    fi
    
    return $cleanup_failed
}

# =============================================
# Main Function
# =============================================

main() {
    echo "Starting benchmarking process at $(date)"
    echo ""
    echo "$SEP_LONG"
    echo "=                  NUMA-Aware Benchmark Configuration Tool  v $VERSION              ="
    echo "$SEP_LONG"
    
    # Parse and validate command line arguments
    # Sets: ARCH_CONFIG, MODE, DOMAINS, DRY_RUN
    parse_args "$@"
    
    echo ""
    
    # Parse architecture configuration file
    # Extracts system topology and CPU configuration
    # Calculates core counts and CPU ranges
    # Sets: SYSTEM_NAME, PARTITION, CCDS_PER_NUMA, NUMA_NODES, NUMA_CPUS, CORES_PER_CCD, CPUS_PER_DOMAIN
    parse_arch_config "$ARCH_CONFIG"
    
    # Parse and validate domain list
    # Converts comma-separated list to array
    # Validates domain numbers against system topology
    # Sets: DOMAIN_ARRAY
    parse_domains "$DOMAINS"
    
    # Set up job directory and create necessary links
    # Creates directory for benchmark results
    # Sets up symbolic links to required executables
    # Creates: JOB_DIR, symbolic links to required files
    setup_job_directory
    
    # Run benchmarks based on mode
    local benchmark_failed=0
    case "$MODE" in
        "sequential"|"interleaved")
            # Run for increasing rank counts
            for ((ranks_per_domain=1; ranks_per_domain<=CPUS_PER_DOMAIN; ranks_per_domain++)); do
                if ! run_benchmark "$MODE" "$ranks_per_domain" "$MEMORY_SIZES" "${DOMAIN_ARRAY[@]}"; then
                    benchmark_failed=1
                fi
            done
            ;;
        "")
            echo "Error: No mode specified. Use --sequential or --interleaved"
            exit 1
            ;;
        *)
            echo "Error: Unknown mode '$MODE'"
            exit 1
            ;;
    esac
    
    # ---------------------- SUMMARY ----------------------
    
    # Clean up symbolic links
    create_run_summary
    cleanup
    echo "Benchmark run completed at $(date)"
    echo "$SEP_LONG"

    if [ "$DRY_RUN" = false ]; then
        cd - >/dev/null
    fi
    
    # Exit with appropriate status
    [ $benchmark_failed -eq 0 ] || exit 1
}

# Execute main function with all script arguments
main "$@" 

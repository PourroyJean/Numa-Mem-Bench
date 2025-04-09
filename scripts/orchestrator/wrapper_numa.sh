#!/bin/bash
#
# =============================================================================
# Script: wrapper_numa.sh
# Author: Jean Pourroy
# Organization: HPE
# License: MIT (See LICENSE file for details)
# =============================================================================
# This script is a wrapper for numa_bench that provides flexible NUMA binding options
# for MPI ranks. It supports various NUMA binding strategies and execution modes.
#
# Features:
# - Flexible NUMA binding per rank
# - Automatic round-robin NUMA distribution
# - Serial execution mode
# - Verbose output for debugging
# - Dry-run mode for testing
#
# Usage:
#   srun --nodes=1 --ntasks=N ./wrapper_numa.sh [options] [-- <numa_bench_options>]
#
# Options:
#   --numa=VALUE      NUMA binding configuration
#                     - Single value: Bind all ranks to specified NUMA node
#                     - Comma-separated list: Bind each rank to corresponding node
#                     - 'auto': Automatically distribute ranks across NUMA nodes
#
#   --quiet           Disable verbose output (default is verbose)
#   --serial          Run in serial mode (one rank at a time)
#   --dry-run         Print commands without execution
#   --executable PATH Path to the numa_mem_bench executable
#   --help            Display help message
#
# Example:
#   srun --nodes=1 --ntasks=56 ./wrapper_numa.sh --numa=3 --executable ../build/numa_mem_bench -- --size=2048
#   srun --nodes=1 --ntasks=6 ./wrapper_numa.sh --numa=0,1,2,3,0,1 -- --size=1024
#   srun --nodes=1 --ntasks=56 ./wrapper_numa.sh --numa=auto -- --size=2048
# =============================================================================

# Exit on any error
set -e

# Default values
NUMA_BINDING="0"
NUMA_SPECIFIED=0
VERBOSE=1
FORWARDED_ARGS=()
DRY_RUN=0
EXECUTABLE_PATH="./numa_mem_bench" # Default executable path

# Function to display help message
show_help() {
    cat << EOF
Usage: $0 [options] [-- <numa_mem_bench_options>]

This script runs numa_mem_bench with flexible NUMA binding options.

Options:
  --numa=VALUE      Control NUMA domain binding
                    - Single value (e.g., --numa=3): Bind all ranks to that NUMA node
                    - Comma-separated list (e.g., --numa=0,1,2,3): Bind each rank according to the list
                    - 'auto': Automatically distribute ranks across all NUMA nodes (round-robin)
                    Default: No NUMA binding

  --quiet           Disable verbose output (default is verbose)

  --dry-run         Print commands that would be executed without running them

  --executable PATH Path to the numa_mem_bench executable
                    Default: $EXECUTABLE_PATH

  --help            Display this help message and exit

  --                Separator after which all arguments are passed directly to the benchmark executable

Examples:
  # Bind all MPI ranks to NUMA node 3 using a specific executable path
  srun --nodes=1 --ntasks=4 ./wrapper_numa.sh --numa=3 --executable ../build/numa_mem_bench -- --size=2048

  # Specify binding per rank
  srun --nodes=1 --ntasks=6 ./wrapper_numa.sh --numa=0,1,2,3,0,1 -- --size=1024

  # Automatic round-robin binding
  srun --nodes=1 --ntasks=8 ./wrapper_numa.sh --numa=auto -- --size=2048

  # Dry run to see what commands would be executed
  srun --nodes=1 --ntasks=4 ./wrapper_numa.sh --numa=0,1,2,3 --dry-run -- --size=512
EOF
}

# Function to get the number of NUMA nodes
get_numa_node_count() {
    numactl -H | grep "available:" | awk '{print $2}'
}

# Function to check if numactl and the benchmark executable exist
check_prerequisites() {
    if ! command -v numactl &> /dev/null; then
        echo "Error: numactl not found. Please install numactl package." >&2
        exit 1
    fi

    if [ ! -x "$EXECUTABLE_PATH" ]; then
        echo "Error: Executable not found or not executable at '$EXECUTABLE_PATH'." >&2
        echo "Please check the path or run 'make' to compile." >&2
        exit 1
    fi
}

# Parse command line arguments
parse_args() {
    # Flag to indicate if we've seen the -- separator
    local passthrough=0
    
    while [ $# -gt 0 ]; do
        # If we've seen --, add all remaining arguments to FORWARDED_ARGS
        if [ $passthrough -eq 1 ]; then
            FORWARDED_ARGS+=("$1")
            shift
            continue
        fi
        
        # Check if this argument is --
        if [ "$1" = "--" ]; then
            passthrough=1
            shift
            continue
        fi
        
        # Process script arguments
        case "$1" in
            --numa=*)
                NUMA_BINDING="${1#*=}"
                NUMA_SPECIFIED=1
                shift
                ;;
            --quiet)
                VERBOSE=0
                shift
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            --executable)
                if [ -z "$2" ] || [[ "$2" == --* ]]; then
                    echo "Error: --executable requires a path argument." >&2
                    exit 1
                fi
                EXECUTABLE_PATH="$2"
                shift 2 # Consume both --executable and its argument
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                # If it looks like --opt=val, keep processing
                if [[ "$1" == --*=* ]]; then 
                     case "$1" in 
                         --numa=*)
                             NUMA_BINDING="${1#*=}"
                             NUMA_SPECIFIED=1
                             ;;
                         *) 
                             echo "Unknown option format: $1" >&2
                             show_help
                             exit 1
                             ;;
                     esac
                     shift
                else
                    echo "Unknown option or missing value: $1" >&2
                    show_help
                    exit 1
                fi
                ;;
        esac
    done
}

# Function to handle automatic NUMA node distribution
handle_auto_numa() {
    local num_ranks=$1
    local num_nodes=$(get_numa_node_count)
    
    if [ $num_nodes -eq 0 ]; then
        echo "Error: Could not detect NUMA nodes." >&2
        exit 1
    fi
    
    # Create an array for round-robin NUMA assignment
    local numa_list=""
    for ((i=0; i<num_ranks; i++)); do
        numa_id=$((i % num_nodes))
        [ -n "$numa_list" ] && numa_list+=","
        numa_list+="$numa_id"
    done
    
    echo "$numa_list"
}

# Function to validate NUMA binding list
validate_numa_binding() {
    local binding=$1
    local num_ranks=$2
    
    # Count elements in the NUMA binding list
    local binding_count=$(echo "$binding" | tr ',' '\n' | wc -l)
    
    if [ "$binding_count" -ne "$num_ranks" ]; then
        echo "Error: Number of NUMA bindings ($binding_count) does not match number of MPI ranks ($num_ranks)." >&2
        exit 1
    fi
}

# Main execution function
run_with_numa_binding() {
    local rank=$SLURM_PROCID
    local total_ranks=$SLURM_NTASKS
    
    # If rank is not set, we're not running under SLURM
    if [ -z "$rank" ] || [ -z "$total_ranks" ]; then
        echo "Error: This script is designed to run under SLURM with srun." >&2
        exit 1
    fi
    
    # Construct the arguments for numa_mem_bench
    local bench_args=""
    if [ ${#FORWARDED_ARGS[@]} -gt 0 ]; then
        # Properly quote arguments to handle spaces, etc.
        bench_args=$(printf "'%s' " "${FORWARDED_ARGS[@]}")
    fi
    
    # Base command with the executable path
    local base_cmd="$EXECUTABLE_PATH $bench_args"
    
    # If no NUMA binding specified, run without numactl
    if [ "$NUMA_SPECIFIED" -eq 0 ]; then
        local cmd="$base_cmd"
        
        # Print verbose output if requested
        if [ "$VERBOSE" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
            echo "[Rank $rank] Running without NUMA binding"
            echo "[Rank $rank] Executing: $cmd"
        fi
        
        # Execute the command or print it in dry-run mode
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "[Rank $rank] Would execute: $cmd"
        else
            if ! eval "$cmd"; then
                echo "[Rank $rank] Error: Command failed: $cmd" >&2
                # Consider whether to exit immediately or let other ranks continue/fail
                # exit 2 # Uncomment if failure of one rank should stop all
            fi
        fi
        return
    fi
    
    # Handle auto NUMA binding
    if [ "$NUMA_BINDING" = "auto" ]; then
        NUMA_BINDING=$(handle_auto_numa "$total_ranks")
    fi
    
    # For comma-separated list, validate and get the correct NUMA node for this rank
    if [[ $NUMA_BINDING == *,* ]]; then
        validate_numa_binding "$NUMA_BINDING" "$total_ranks"
        # Extract NUMA domain for this rank
        numa_domain=$(echo "$NUMA_BINDING" | cut -d ',' -f $((rank+1)))
    else
        # Single value for all ranks
        numa_domain=$NUMA_BINDING
    fi
    
    # Check if NUMA domain is valid
    if ! [[ "$numa_domain" =~ ^[0-9]+$ ]]; then
        echo "[Rank $rank] Error: Invalid NUMA domain '$numa_domain'." >&2
        exit 1
    fi
    
    # Verify if specified NUMA node exists
    local max_node=$(($(get_numa_node_count) - 1))
    if [ "$numa_domain" -gt "$max_node" ]; then
        echo "[Rank $rank] Error: NUMA domain $numa_domain does not exist. Max domain is $max_node." >&2
        exit 1
    fi
    
    # Prepare the command with numactl
    local cmd="numactl --membind=$numa_domain $base_cmd"
    
    # Print verbose output if requested
    if [ "$VERBOSE" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
        echo "[Rank $rank] Binding to NUMA domain $numa_domain"
        echo "[Rank $rank] Executing: $cmd"
    fi
    
    # Execute the command or print it in dry-run mode
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[Rank $rank] Would execute: $cmd"
    else
        if ! eval "$cmd"; then
            echo "[Rank $rank] Error: Command failed: $cmd" >&2
            # exit 2 # Consider exit strategy
        fi
    fi
}

# Main script execution
main() {
    # Parse args before checking prerequisites, so --executable is known
    parse_args "$@"
    check_prerequisites
    run_with_numa_binding
}

# Call main function with all script arguments
main "$@" 
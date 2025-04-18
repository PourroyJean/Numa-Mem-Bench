#!/bin/bash
# ============================================================================
# Environment setup for Numa-Mem-Bench
# ============================================================================
# This script sets up the necessary environment variables and paths for running
# the Numa-Mem-Bench benchmarks and analysis tools.
#
# Usage: source ./env.sh
# ============================================================================

# Environment setup for Numa-Mem-Bench
# Determine the path to the sourced script (cross-shell: bash and zsh)
get_script_dir() {
  local script_path
  if [ -n "${BASH_SOURCE[0]}" ]; then
    script_path="${BASH_SOURCE[0]}"
  elif [ -n "${(%):-%x}" ]; then
    script_path="${(%):-%x}"
  else
    echo "Could not detect script path." >&2
    return 1
  fi

  # Resolve symlinks and convert to absolute path
  while [ -h "$script_path" ]; do
    local dir="$(cd -P "$(dirname "$script_path")" >/dev/null 2>&1 && pwd)"
    script_path="$(readlink "$script_path")"
    [[ $script_path != /* ]] && script_path="$dir/$script_path"
  done

  cd -P "$(dirname "$script_path")" >/dev/null 2>&1 && pwd
}

ROOT_PATH="$(get_script_dir)"
echo "Using ROOT_PATH: $ROOT_PATH"


# Set paths based on ROOT_PATH
export NUMA_BENCH=$ROOT_PATH/numa_mem_bench
export WRAPPER=$ROOT_PATH/scripts/orchestrator/wrapper_numa.sh
export PYTHON_PLOT=$ROOT_PATH/scripts/visualization/analyze_results.py
export RUN_BENCHMARKS=$ROOT_PATH/scripts/orchestrator/run_numa_benchmarks.sh
export PYTHON_ENV_PATH=$ROOT_PATH/scripts/python_venv/bin/activate
export ARCHITECTURE_G=$ROOT_PATH/architectures/lumi_partg
export ARCHITECTURE_C=$ROOT_PATH/architectures/lumi_partc
export MAPPING_PATH=./mapping_check.log

# SLURM settings
export SLURM_ACCOUNT=project_462000031
export SLURM_PARTITION=""
export SLURM_RESERVATION=""

# Partition names
export CPU_PARTITION=standard
export GPU_PARTITION=standard-g

# Activate Python environment if it exists
if [ -f "$PYTHON_ENV_PATH" ]; then
    echo "Activating Python environment..."
    source $PYTHON_ENV_PATH
else
    echo "WARNING: Python virtual environment not found at $PYTHON_ENV_PATH"
    echo "Visualization tools may not work properly. To create the Python environment:"
    echo ""
    echo "    python3 -m venv $ROOT_PATH/scripts/python_venv"
    echo "    source $ROOT_PATH/scripts/python_venv/bin/activate"
    echo "    pip install -r $ROOT_PATH/scripts/visualization/requirements.txt"
    echo ""
    echo "Alternatively, ensure the required packages are installed in your system Python:"
    echo "    pip install pandas numpy matplotlib seaborn"
    echo ""
fi

echo "Numa-Mem-Bench environment set up at $ROOT_PATH"

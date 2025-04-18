#! /bin/bash

# Source the environment setup script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPT_DIR/../env.sh

# Set the mapping path for examples
MAPPING_PATH=./mapping_check.log

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# Example 1 : 7 ranks on domain 0, with memory binding to domain 0
function run_example1() {
    EXAMPLE_NAME=example1_lumig_7ranks_1_domain
    rm -rf $EXAMPLE_NAME 2>/dev/null && mkdir $EXAMPLE_NAME
    cd $EXAMPLE_NAME
    MEMORY_SIZES=1-2048
    srun --nodes 1 --partition $GPU_PARTITION --ntasks 7  --cpu-bind=map_cpu:1,2,3,4,5,6,7 --hint=nomultithread \
                numactl --membind=0 $NUMA_BENCH --size=$MEMORY_SIZES --csv="$EXAMPLE_NAME"_7ranks.csv

    python3 $PYTHON_PLOT .
    cd ..
}

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# Example 2 : 14 ranks on domain 0, with memory binding to domain 0 and 1, using wrapper script
function run_example2() {
    EXAMPLE_NAME=example2_lumig_14ranks_2_domains
    rm -rf $EXAMPLE_NAME 2>/dev/null && mkdir $EXAMPLE_NAME
    cd $EXAMPLE_NAME
    MEMORY_SIZES=1-2048
    srun --nodes 1  --partition $GPU_PARTITION --ntasks 14 --cpu-bind=map_cpu:1,2,3,4,5,6,7,17,18,19,20,21,22,23   \
                --hint=nomultithread $WRAPPER --executable $NUMA_BENCH        \
                --numa=0,0,0,0,0,0,0,1,1,1,1,1,1,1 -- --size=1-2048 --csv="$EXAMPLE_NAME"_14ranks.csv

    python3 $PYTHON_PLOT .

    cd ..
}

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# Example 3 : full benchmark run on all cores with memory sizes 512 all domains
function run_example3() {
    EXAMPLE_NAME=example3_all_cores_512MB
    rm -rf $EXAMPLE_NAME 2>/dev/null && mkdir $EXAMPLE_NAME
    cd $EXAMPLE_NAME

    MEMORY_SIZES=512
    sbatch --account=$SLURM_ACCOUNT --partition $CPU_PARTITION $RUN_BENCHMARKS --wrapper $WRAPPER --benchmark $NUMA_BENCH --memory-sizes $MEMORY_SIZES --arch $ARCHITECTURE_C \
                                               --sequential 0,1,2,3,4,5,6,7  --label lumic_all_cores_sequential 

    sbatch --account=$SLURM_ACCOUNT --partition $CPU_PARTITION $RUN_BENCHMARKS --wrapper $WRAPPER --benchmark $NUMA_BENCH --memory-sizes $MEMORY_SIZES --arch $ARCHITECTURE_C \
                                               --interleaved 0,1,2,3,4,5,6,7  --label lumic_all_cores_interleaved 

    sbatch --account=$SLURM_ACCOUNT --partition $GPU_PARTITION $RUN_BENCHMARKS --wrapper $WRAPPER --benchmark $NUMA_BENCH --memory-sizes $MEMORY_SIZES --arch $ARCHITECTURE_G \
                                                --sequential 0,1,2,3  --label lumig_all_cores_sequential 

    sbatch --account=$SLURM_ACCOUNT --partition $GPU_PARTITION $RUN_BENCHMARKS --wrapper $WRAPPER --benchmark $NUMA_BENCH --memory-sizes $MEMORY_SIZES --arch $ARCHITECTURE_G \
                                               --interleaved 0,1,2,3  --label lumig_all_cores_interleaved 

    # python3 $PYTHON_PLOT results_scaling/*

    cd ..
}

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# Example 4 : full benchmark run on all cores with memory sizes 1- 512 all domains
function run_example4() {
    EXAMPLE_NAME=example4_all_cores_1_to_512MB
    rm -rf $EXAMPLE_NAME 2>/dev/null && mkdir $EXAMPLE_NAME
    cd $EXAMPLE_NAME

    MEMORY_SIZES=1-512
    sbatch  --account=$SLURM_ACCOUNT     --partition $CPU_PARTITION $RUN_BENCHMARKS --wrapper $WRAPPER --benchmark $NUMA_BENCH \
            --memory-sizes $MEMORY_SIZES --arch $ARCHITECTURE_C     --interleaved 0,1,2,3,4,5,6,7  \
            --label lumic_all_cores_interleaved_multiple_sizes

    MEMORY_SIZES=1-1024
    sbatch  --account=$SLURM_ACCOUNT     --partition $GPU_PARTITION $RUN_BENCHMARKS --wrapper $WRAPPER --benchmark $NUMA_BENCH \
            --memory-sizes $MEMORY_SIZES --arch $ARCHITECTURE_G     --interleaved 0,1,2,3  \
            --label lumig_all_cores_interleaved_multiple_sizes

    

    cd ..
}

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

main() {
    echo "Running example 1" ; run_example1
    echo "Running example 2" ; run_example2
    echo "Running example 3" ; run_example3
    echo "Running example 4" ; run_example4
    
    # python3 $PYTHON_PLOT example3_all_cores_512MB/* example4_all_cores_1_to_512MB/*
}

main
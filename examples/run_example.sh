#! /bin/bash

BENCH_PATH=../numa_mem_bench
WRAPPER_PATH=../scripts/orchestrator/wrapper_numa.sh
PLOT_PATH=../scripts/visualization/plot_scaling.py
CSV_PATH=./results
MAPPING_PATH=./mapping_check.log
OUTPUT_PATH=./plot
ENV_PATH=../scripts/python_venv/bin/activate

source $ENV_PATH
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# Example 1 : 7 ranks on domain 0, with memory binding to domain 0
srun --nodes 1  --ntasks 7  --cpu-bind=map_cpu:1,2,3,4,5,6,7 --hint=nomultithread \
                numactl --membind=0 $BENCH_PATH --size=1-2048 --csv="$CSV_PATH"_example1.csv

python3 $PLOT_PATH "$CSV_PATH"_example1.csv --mapping "$MAPPING_PATH" --output "$OUTPUT_PATH"_example1.png

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# Example 2 : 7 ranks on domain 0, with memory binding to domain 0, using wrapper script
srun --nodes 1  --ntasks 14 --cpu-bind=map_cpu:1,2,3,4,5,6,7,17,18,19,20,21,22,23   \
                --hint=nomultithread $WRAPPER_PATH --executable $BENCH_PATH        \
                --numa=0,0,0,0,0,0,0,1,1,1,1,1,1,1 -- --size=1-2048 --csv="$CSV_PATH"_example2.csv 

python3 $PLOT_PATH "$CSV_PATH"_example2.csv --mapping "$MAPPING_PATH" --output "$OUTPUT_PATH"_example2.png

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
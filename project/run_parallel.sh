#!/bin/bash
# Parallel simulation runner for IsoPEPS.jl

# Configuration
PROJECT_DIR="/home/yuqingrong/IsoPEPS.jl"
THREADS_PER_JOB=40  # Adjust based on total cores / number of jobs
TOTAL_CORES=176

# Array of row values to run
ROWS=(3 4 5 6)

# Create logs directory
mkdir -p "$PROJECT_DIR/project/logs"

# Function to run simulation
run_simulation() {
    local row=$1
    local core_start=$2
    local core_end=$3
    local n_threads=$4
    
    echo "Starting simulation for row=$row on cores $core_start-$core_end"
    
    taskset -c $core_start-$core_end \
        julia --project="$PROJECT_DIR" --threads=$n_threads \
        -e "using IsoPEPS; simulation(1.0, [2.0], $row, 2, 5; maxiter=100)" \
        > "$PROJECT_DIR/project/logs/row${row}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    
    echo "PID: $! - Row: $row - Log: logs/row${row}_*.log"
}

# Calculate cores per job
CORES_PER_JOB=$((TOTAL_CORES / ${#ROWS[@]}))

# Run all simulations in parallel
core_offset=0
for row in "${ROWS[@]}"; do
    core_start=$core_offset
    core_end=$((core_start + CORES_PER_JOB - 1))
    
    run_simulation $row $core_start $core_end $CORES_PER_JOB
    
    core_offset=$((core_offset + CORES_PER_JOB))
done

echo ""
echo "All simulations started!"
echo "Monitor progress with:"
echo "  tail -f project/logs/*.log"
echo ""
echo "Check running jobs:"
echo "  jobs"
echo "  ps aux | grep julia"

#!/bin/bash
# Safe simulation runner using tmux
# This script runs your simulation in a persistent tmux session
# that survives disconnections

SESSION_NAME="isopeps_sim"

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill old session and create new: tmux kill-session -t $SESSION_NAME"
    echo ""
    exit 1
fi

echo "Creating new tmux session: $SESSION_NAME"
echo "Running simulation with threading enabled..."
echo ""
echo "IMPORTANT:"
echo "  - To detach (leave running): Press Ctrl+b, then d"
echo "  - To reattach later: tmux attach -t $SESSION_NAME"
echo "  - To kill session: tmux kill-session -t $SESSION_NAME"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Create new session and run simulation
tmux new-session -d -s $SESSION_NAME -c /home/yuqingrong/IsoPEPS.jl
tmux send-keys -t $SESSION_NAME "julia -t auto --project=. src/InfPEPS/simulation.jl" C-m

echo "✓ Simulation started in tmux session: $SESSION_NAME"
echo ""
echo "To view the simulation, run:"
echo "    tmux attach -t $SESSION_NAME"

# Auto-attach to the session
echo ""
read -p "Attach to session now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    tmux attach -t $SESSION_NAME
fi


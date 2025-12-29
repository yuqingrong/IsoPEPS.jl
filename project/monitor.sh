#!/bin/bash
# Monitor parallel Julia simulations

echo "=== Julia Processes ==="
ps aux | grep "[j]ulia" | awk '{print "PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "TIME:", $10}'

echo ""
echo "=== Recent Log Updates ==="
if [ -d "project/logs" ]; then
    for log in project/logs/*.log; do
        if [ -f "$log" ]; then
            echo "--- $(basename $log) (last 3 lines) ---"
            tail -n 3 "$log"
            echo ""
        fi
    done
else
    echo "No logs directory found"
fi

echo ""
echo "=== System Resources ==="
echo "CPU Load: $(uptime | awk -F'load average:' '{print $2}')"
echo "Memory: $(free -h | grep Mem | awk '{print "Used:", $3, "/", $2}')"

echo ""
echo "=== Monitoring Commands ==="
echo "  Watch all logs: tail -f project/logs/*.log"
echo "  Kill all Julia: pkill julia"
echo "  CPU usage: htop"

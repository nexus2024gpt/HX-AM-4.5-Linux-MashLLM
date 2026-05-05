#!/bin/bash
PID=$(pgrep -f "mesh-llm serve")
if [ -n "$PID" ]; then
  kill -9 $PID
  echo "mesh-llm stopped (PID $PID)"
else
  echo "mesh-llm not running"
fi

#!/bin/bash
# start-mesh.sh — запуск mesh-llm с выбранной моделью

MODEL=${1:-"Qwen3-8B"}  # по умолчанию 8B

case "$MODEL" in
  "9B"|"qwen9b"|"Qwen3.5-9B")
    MODEL_PATH="/home/roman220877/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
    CTX=32768
    ;;
  "8B"|"qwen8b"|"Qwen3-8B")
    MODEL_PATH="/home/roman220877/.cache/huggingface/hub/models--unsloth--Qwen3-8B-GGUF/snapshots/a6adef130ffb23ddaf1a62fec9dced968c9bc482/Qwen3-8B-Q4_K_M.gguf"
    CTX=16384
    ;;
  "4B"|"qwen4b"|"Qwen3-4B")
    MODEL_PATH="/home/roman220877/.cache/huggingface/hub/models--unsloth--Qwen3-4B-GGUF/snapshots/22c9fc8a8c7700b76a1789366280a6a5a1ad1120/Qwen3-4B-Q4_K_M.gguf"
    CTX=16384
    ;;
  *)
    echo "Usage: start-mesh.sh [4B|8B|9B]"
    exit 1
    ;;
esac

echo "Starting mesh-llm with $MODEL ($CTX ctx)..."
nohup mesh-llm serve --model "$MODEL_PATH" --ctx-size $CTX --port 9337 > ~/mesh-llm.log 2>&1 &
echo "PID: $!"
echo "Log: ~/mesh-llm.log"

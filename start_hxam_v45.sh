#!/bin/bash
echo "========================================="
echo "   HX-AM v4.5.2 — Последовательный запуск"
echo "========================================="

cd /home/roman220877/hxam

echo "[1/3] Запуск llama-server..."
./start_llama_cpu.sh

echo -n "     Ожидание llama-server..."
for i in {1..30}; do
    if curl -s --max-time 2 http://127.0.0.1:11435/health > /dev/null; then
        echo " OK"
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

echo "[2/3] Запуск MashLLM..."
mesh-llm client --auto > mesh_llm.log 2>&1 &
MESH_PID=$!

echo -n "     Ожидание MashLLM (9337)..."
for i in {1..20}; do
    if curl -s --max-time 2 http://127.0.0.1:9337/v1/models > /dev/null; then
        echo " OK"
        break
    fi
    echo -n "."
    sleep 3
done
echo ""

echo "[3/3] Запуск HX-AM v4.5.2..."
source venv/bin/activate
python hxam_v_4_server.py 2>&1 | tee /tmp/hxam_last.log &

echo "========================================="
echo "Все сервисы запущены последовательно."
echo "HX-AM доступен на http://127.0.0.1:8000"

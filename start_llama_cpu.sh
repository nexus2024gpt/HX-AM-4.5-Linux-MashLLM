#!/bin/bash
echo "=== Запуск llama-server (оптимизировано под i5-6300U и требования HX-AM) ==="

# Мягко гасим старый процесс, если он завис
pkill -f llama-server 2>/dev/null
sleep 1

# Запуск с увеличенным контекстом, жесткой температурой и привязкой к физ. ядрам
./llama-server \
  --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --port 11435 \
  --host 127.0.0.1 \
  --ctx-size 4096 \
  --threads 2 \
  --n-predict 512 \
  --temp 0.2 \
  --repeat-penalty 1.1 \
  2>&1 | tee -a llama_cpu.log &

echo "--------------------------------------------------------"
echo "llama-server успешно запущен в фоне (PID: $!)"
echo "Порт: 11435 | Контекст: 4096 | Потоки CPU: 2"
echo "Лог пишется в файл: llama_cpu.log"
echo "Для мониторинга генерации выполни: tail -f llama_cpu.log"
echo "--------------------------------------------------------"
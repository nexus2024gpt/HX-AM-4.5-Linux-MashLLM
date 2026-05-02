#!/usr/bin/env python3
"""
update_graph_with_stability.py
Обновляет invariant_graph.json, добавляя результаты стресс-теста (stability_score, math_model)
и пересчитывает веса рёбер с учётом устойчивости узлов.
"""

import json
from pathlib import Path

# Конфигурация
GRAPH_PATH = Path("artifacts/invariant_graph.json")
SIM_RESULTS_DIR = Path("sim_results")
BACKUP_PATH = Path("artifacts/invariant_graph.backup.json")

def main():
    # 1. Резервное копирование
    if GRAPH_PATH.exists():
        BACKUP_PATH.write_bytes(GRAPH_PATH.read_bytes())
        print(f"✅ Резервная копия создана: {BACKUP_PATH}")

    # 2. Загрузка графа
    graph_data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("links", [])

    # 3. Загрузка результатов стресс-теста
    stability = {}
    for f in SIM_RESULTS_DIR.glob("*_stress.json"):
        art_id = f.stem.replace("_stress", "")
        data = json.loads(f.read_text(encoding="utf-8"))
        stability[art_id] = {
            "score": data.get("stability_score", 0.0),
            "model": data.get("model_used", "unknown")
        }
    print(f"📊 Загружено {len(stability)} результатов стресс-теста")

    # 4. Обновление узлов
    updated_nodes = 0
    for node in nodes:
        node_id = node.get("id")
        if node_id in stability:
            node["stability_score"] = stability[node_id]["score"]
            node["math_model"] = stability[node_id]["model"]
            updated_nodes += 1
        else:
            # Для отсутствующих ставим нейтральное значение
            node["stability_score"] = node.get("stability_score", 0.5)
            node["math_model"] = node.get("math_model", "unknown")
    print(f"🔄 Обновлено {updated_nodes} узлов")

    # 5. Пересчёт весов рёбер с учётом stability_score
    # Создаём словарь для быстрого доступа к stability_score узлов
    node_stability = {node["id"]: node.get("stability_score", 0.5) for node in nodes}

    updated_edges = 0
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        s_stab = node_stability.get(source, 0.5)
        t_stab = node_stability.get(target, 0.5)
        old_weight = edge.get("weight", 0.5)
        # Новый вес = старый вес * произведение устойчивостей узлов
        new_weight = round(old_weight * s_stab * t_stab, 3)
        if new_weight != old_weight:
            edge["weight"] = new_weight
            edge["stability_factor"] = round(s_stab * t_stab, 3)
            updated_edges += 1
    print(f"⚖️ Пересчитано {updated_edges} рёбер")

    # 6. Сохранение обновлённого графа
    GRAPH_PATH.write_text(json.dumps(graph_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Граф сохранён: {GRAPH_PATH}")

if __name__ == "__main__":
    main()
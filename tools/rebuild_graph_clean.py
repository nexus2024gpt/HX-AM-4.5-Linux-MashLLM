import json
import numpy as np
import networkx as nx
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = Path("artifacts/four_d_index.jsonl")
GRAPH_OUT = Path("artifacts/invariant_graph.json")
SIM_THRESHOLD = 0.7

def load_artifacts():
    ids = []
    vectors = []
    domains = []
    stability = []
    models = []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            ids.append(data["id"])
            vectors.append(data["vector"])
            domains.append(data.get("domain", "?"))
            stability.append(data.get("stability_score", 0.5))
            # math_model можно взять из стресс-теста или оставить unknown
            models.append(data.get("math_model", "unknown"))
    return ids, np.array(vectors), domains, stability, models

def main():
    ids, vectors, domains, stability, models = load_artifacts()
    print(f"Загружено {len(ids)} артефактов")

    # Строим матрицу сходства
    sim_matrix = cosine_similarity(vectors)

    # Создаём обычный граф (НЕ MultiGraph)
    G = nx.Graph()

    # Добавляем узлы с атрибутами
    for i, (id_, dom, stab, mod) in enumerate(zip(ids, domains, stability, models)):
        G.add_node(id_, domain=dom, stability_score=stab, math_model=mod, index=i)

    # Добавляем рёбра (без дубликатов)
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            sim = sim_matrix[i, j]
            if sim >= SIM_THRESHOLD:
                # Вес = similarity * произведение stability
                weight = round(sim * stability[i] * stability[j], 3)
                G.add_edge(ids[i], ids[j], similarity=round(sim, 3), weight=weight)

    # Сохраняем в формате node_link
    graph_data = nx.node_link_data(G)
    GRAPH_OUT.write_text(json.dumps(graph_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Граф сохранён: {GRAPH_OUT}")
    print(f"Узлов: {G.number_of_nodes()}, рёбер: {G.number_of_edges()}")

if __name__ == "__main__":
    main()
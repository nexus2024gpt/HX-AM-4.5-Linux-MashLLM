import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = Path("artifacts/four_d_index.jsonl")   # или .txt
GRAPH_PATH = Path("artifacts/invariant_graph.json")
SIMILARITY_THRESHOLD = 0.7   # можно подобрать

def load_artifacts():
    ids = []
    vectors = []
    domains = []
    stability = []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            ids.append(data["id"])
            vectors.append(data["vector"])
            domains.append(data.get("domain", "?"))
            stability.append(data.get("stability_score", 0.5))
    return ids, np.array(vectors), domains, stability

def main():
    ids, vectors, domains, stability = load_artifacts()
    print(f"Загружено {len(ids)} артефактов")

    # Вычисляем косинусную матрицу сходства
    sim_matrix = cosine_similarity(vectors)

    # Строим граф
    nodes = []
    for i, (id_, dom, stab) in enumerate(zip(ids, domains, stability)):
        nodes.append({
            "id": id_,
            "domain": dom,
            "stability_score": stab,
            "math_model": "unknown",   # можно потом заполнить из результатов стресс-теста
        })

    links = []
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            sim = sim_matrix[i, j]
            if sim >= SIMILARITY_THRESHOLD:
                # Вес ребра = similarity * (1 + domain_distance) * (stability_i * stability_j)
                # Для простоты пока используем только similarity и произведение stability
                weight = round(sim * stability[i] * stability[j], 3)
                links.append({
                    "source": i,   # индекс узла в списке nodes
                    "target": j,
                    "similarity": round(sim, 3),
                    "weight": weight,
                })

    graph_data = {
        "nodes": nodes,
        "links": links,
    }
    GRAPH_PATH.write_text(json.dumps(graph_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Граф сохранён: {GRAPH_PATH}")
    print(f"Узлов: {len(nodes)}, рёбер: {len(links)}")

if __name__ == "__main__":
    main()
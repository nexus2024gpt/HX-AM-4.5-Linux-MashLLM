import json
from pathlib import Path

graph = json.loads(Path("artifacts/invariant_graph.json").read_text())
print(f"Узлов: {len(graph.get('nodes', []))}")
print(f"Рёбер (links): {len(graph.get('links', []))}")
if graph.get("links"):
    print("Пример ребра:", graph["links"][0])
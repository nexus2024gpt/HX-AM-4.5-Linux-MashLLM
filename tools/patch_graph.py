# tools/patch_graph.py — HX-AM v4.2
"""
Патч invariant_graph.json для v4.2.

Добавляет новые атрибуты узлам и рёбрам без LLM-вызовов.
Полностью безопасно — ничего не удаляет.

CLI:
  python tools/patch_graph.py --dry-run   # показать что изменится
  python tools/patch_graph.py             # применить патч
  python tools/patch_graph.py --backup    # сделать бэкап перед патчем
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

GRAPH_PATH = Path("artifacts/invariant_graph.json")


def load_graph() -> dict:
    if not GRAPH_PATH.exists():
        print(f"❌ {GRAPH_PATH} не найден")
        sys.exit(1)
    return json.loads(GRAPH_PATH.read_text(encoding="utf-8"))


def patch_graph(data: dict, dry_run: bool = False) -> dict:
    # networkx node_link_data: узлы в data["nodes"], рёбра в data["links"] или data["edges"]
    nodes = data.get("nodes", [])
    links = data.get("links") or data.get("edges") or []

    nodes_patched = 0
    links_patched = 0

    # ── Узлы ─────────────────────────────────────────────────────
    for node in nodes:
        changed = False
        # Новые атрибуты v4.2
        if "has_four_d" not in node:
            node["has_four_d"] = False
            changed = True
        if "stress_stable" not in node:
            node["stress_stable"] = None
            changed = True
        if "stability_score" not in node:
            node["stability_score"] = 0.0
            changed = True
        if changed:
            nodes_patched += 1

    # ── Рёбра ────────────────────────────────────────────────────
    for link in links:
        changed = False
        if "four_d_resonance" not in link:
            link["four_d_resonance"] = 0.0
            changed = True
        if changed:
            links_patched += 1

    # Добавить мета-поле о миграции
    data["_v42_graph_patch"] = {
        "patched_at": datetime.now(timezone.utc).isoformat(),
        "nodes_patched": nodes_patched,
        "links_patched": links_patched,
    }

    print(f"  Узлов обновлено:  {nodes_patched}/{len(nodes)}")
    print(f"  Рёбер обновлено:  {links_patched}/{len(links)}")
    return data


def rebuild_four_d_index():
    """
    Строит four_d_index.jsonl из всех артефактов у которых есть four_d_matrix.
    Вызывается после migrate_to_v42.py.
    """
    index_path = Path("artifacts/four_d_index.jsonl")
    artifacts_dir = Path("artifacts")

    entries = []
    for f in sorted(artifacts_dir.glob("*.json")):
        if f.stem == "invariant_graph":
            continue
        try:
            art = json.loads(f.read_text(encoding="utf-8"))
            art_id = art.get("id", f.stem.split(".")[0])
            data = art.get("data", {})
            gen = data.get("gen", {})
            four_d = gen.get("four_d_matrix")

            # Для hyx-portal — four_d_matrix прямо в корне
            if four_d is None:
                four_d = art.get("four_d_matrix")

            if four_d is None:
                continue

            domain = data.get("domain") or art.get("domain", "general")
            stability_score = 0.0

            # Берём stability_score из simulation если есть
            sim = art.get("simulation", {})
            if sim:
                stability_score = sim.get("stability_score", 0.0)
            # Или из sim_results/
            sim_file = Path("sim_results") / f"{art_id}_stress.json"
            if sim_file.exists():
                try:
                    sim_data = json.loads(sim_file.read_text())
                    stability_score = sim_data.get("stability_score", stability_score)
                except Exception:
                    pass

            # Вычислить 4D-вектор
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from schemas.four_d_matrix import FourDMatrix
                matrix = FourDMatrix.from_raw(four_d)
                if matrix is None:
                    continue
                vec = matrix.to_vector().tolist()
            except Exception as e:
                print(f"  ⚠ {art_id}: vector failed — {e}")
                continue

            entries.append({
                "id": art_id,
                "domain": domain,
                "four_d": four_d,
                "vector": vec,
                "stability_score": stability_score,
                "added_at": art.get("created_at", ""),
            })

        except Exception as e:
            print(f"  ⚠ {f.name}: {e}")
            continue

    if not entries:
        print("  Нет артефактов с four_d_matrix")
        return 0

    with open(index_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  four_d_index.jsonl: записано {len(entries)} записей → {index_path}")
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description="HX-AM v4.2 Graph Patch Tool")
    parser.add_argument("--dry-run",       action="store_true", help="Показать изменения без сохранения")
    parser.add_argument("--backup",        action="store_true", help="Сделать бэкап перед патчем")
    parser.add_argument("--rebuild-index", action="store_true", help="Пересобрать four_d_index.jsonl из артефактов")
    args = parser.parse_args()

    print("\n🔧 HX-AM v4.2 Graph Patch Tool")
    print(f"   Graph: {GRAPH_PATH.absolute()}")

    # ── Пересборка 4D-индекса
    if args.rebuild_index:
        print("\n📑 Пересборка four_d_index.jsonl...")
        count = rebuild_four_d_index()
        print(f"✅ Готово: {count} записей")
        return

    # ── Бэкап
    if args.backup and not args.dry_run:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = GRAPH_PATH.parent / f"invariant_graph.{ts}.bak.json"
        shutil.copy2(GRAPH_PATH, backup_path)
        print(f"💾 Бэкап: {backup_path}")

    # ── Загрузка и патч
    data = load_graph()
    nodes_before = len(data.get("nodes", []))
    links_before = len(data.get("links") or data.get("edges") or [])
    print(f"\nГраф: {nodes_before} узлов, {links_before} рёбер")

    if args.dry_run:
        print("\n[dry-run] Изменения которые будут применены:")
        patch_graph(data, dry_run=True)
        print("\n(dry-run: файл не изменён)")
        return

    print("\nПрименяю патч...")
    patched = patch_graph(data)
    GRAPH_PATH.write_text(json.dumps(patched, ensure_ascii=False, indent=2))
    print(f"\n✅ invariant_graph.json обновлён")


if __name__ == "__main__":
    main()

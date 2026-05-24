#!/usr/bin/env python3
# tools/migrate_general_domains.py — HX-AM v4.6
"""
Миграция артефактов с domain='general' на конкретные домены.

Стратегия (без LLM, детерминированная):
  1. Keyword-матч по hypothesis+mechanism (быстро, ~80% покрытие)
  2. Семантический матч по архиву (для оставшихся)
  3. Опционально: LLM для сложных случаев (--llm флаг)

Что обновляется:
  - artifacts/*.json → data.domain
  - artifacts/semantic_index.jsonl → domain поле
  - artifacts/invariant_graph.json → node.domain + edges domain_distance
  - artifacts/four_d_index.jsonl → domain поле

Граф перестраивается ИНКРЕМЕНТАЛЬНО (не полностью):
  - Узел получает новый домен
  - Пересчитываются domain_distance для рёбер этого узла
  - specificity НЕ пересчитывается (требует rebuild всего пространства)

CLI:
  python tools/migrate_general_domains.py --dry-run
  python tools/migrate_general_domains.py
  python tools/migrate_general_domains.py --llm       # LLM для сложных
  python tools/migrate_general_domains.py --id <id>   # один артефакт
  python tools/migrate_general_domains.py --rebuild-index  # только индексы
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate_general")

ARTIFACTS_DIR   = Path("artifacts")
SEM_INDEX_PATH  = ARTIFACTS_DIR / "semantic_index.jsonl"
GRAPH_PATH      = ARTIFACTS_DIR / "invariant_graph.json"
FOUR_D_IDX_PATH = ARTIFACTS_DIR / "four_d_index.jsonl"


# ── Загрузка артефактов с general ────────────────────────────────────────────

def find_general_artifacts(
    target_id: Optional[str] = None,
) -> List[Dict]:
    """Находит артефакты с domain='general'."""
    result = []
    for f in sorted(ARTIFACTS_DIR.glob("*.json")):
        if f.stem == "invariant_graph" or ".hyx-portal" in f.name:
            continue
        try:
            art = json.loads(f.read_text(encoding="utf-8"))
            art_id = art.get("id", f.stem)

            if target_id and art_id != target_id:
                continue

            data   = art.get("data", {})
            domain = data.get("domain", "general")

            if domain == "general" or not domain:
                gen = data.get("gen", {})
                result.append({
                    "path":       f,
                    "id":         art_id,
                    "hypothesis": gen.get("hypothesis", ""),
                    "mechanism":  gen.get("mechanism", ""),
                    "domain_old": domain,
                })
        except Exception as e:
            logger.warning(f"Cannot read {f.name}: {e}")

    return result


# ── Определение нового домена ─────────────────────────────────────────────────

def classify_domain(
    artifact: Dict,
    space,           # SemanticSpace или None
    llm=None,
) -> Tuple[str, float, str]:
    """
    Определяет домен артефакта через SmartDomainResolver.
    Возвращает (domain, confidence, method).
    """
    from smart_domain_resolver import SmartDomainResolver
    resolver = SmartDomainResolver(space=space)

    domain, conf, method = resolver.resolve(
        text="",
        hypothesis=artifact["hypothesis"],
        mechanism=artifact["mechanism"],
    )

    # LLM как последний рубеж (только при --llm и conf < 0.5)
    if llm and conf < 0.5 and domain == "general":
        domain, conf, method = _llm_classify(
            artifact["hypothesis"],
            artifact["mechanism"],
            llm,
        )

    return domain, conf, method


def _llm_classify(
    hypothesis: str,
    mechanism: str,
    llm,
) -> Tuple[str, float, str]:
    """LLM-классификация для случаев с низкой уверенностью."""
    from domain_config import VALID_DOMAINS

    valid_str = " | ".join(d for d in VALID_DOMAINS if d != "general")
    prompt = (
        f"Определи научный домен гипотезы. Ответь ОДНИМ словом из списка:\n"
        f"{valid_str}\n\n"
        f"Гипотеза: {hypothesis[:300]}\n"
        f"Механизм: {mechanism[:200]}\n\n"
        f"Правила:\n"
        f"- 'general' использовать НЕЛЬЗЯ\n"
        f"- Выбери ближайший по механизму\n"
        f"Ответ:"
    )
    try:
        raw, model_used = llm.generate(prompt)
        candidate = raw.strip().lower().split()[0] if raw.strip() else ""
        from domain_config import VALID_DOMAINS, DOMAIN_ALIASES
        if candidate in DOMAIN_ALIASES:
            candidate = DOMAIN_ALIASES[candidate]
        if candidate in VALID_DOMAINS and candidate != "general":
            logger.info(f"LLM classified → '{candidate}' via {model_used}")
            return candidate, 0.75, f"llm:{model_used.split('/')[-1]}"
    except Exception as e:
        logger.warning(f"LLM classify failed: {e}")

    return "general", 0.1, "llm_failed"


# ── Обновление файлов ─────────────────────────────────────────────────────────

def update_artifact_file(path: Path, new_domain: str) -> bool:
    """Обновляет domain в артефакте."""
    try:
        art = json.loads(path.read_text(encoding="utf-8"))
        old = art.get("data", {}).get("domain", "general")
        if old == new_domain:
            return False
        art["data"]["domain"] = new_domain
        art.setdefault("migration_v46", {})["domain_updated"] = new_domain
        art.setdefault("migration_v46", {})["domain_was"] = old
        path.write_text(json.dumps(art, ensure_ascii=False, indent=2))
        return True
    except Exception as e:
        logger.error(f"update_artifact_file {path.name}: {e}")
        return False


def update_semantic_index(updates: Dict[str, str]) -> int:
    """
    Обновляет domain в semantic_index.jsonl.
    updates: {artifact_id: new_domain}
    Возвращает количество обновлённых записей.
    """
    if not SEM_INDEX_PATH.exists():
        return 0

    lines = SEM_INDEX_PATH.read_text(encoding="utf-8").splitlines()
    new_lines = []
    updated = 0

    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            art_id = entry.get("id", "")
            if art_id in updates:
                entry["domain"] = updates[art_id]
                updated += 1
            new_lines.append(json.dumps(entry, ensure_ascii=False))
        except Exception:
            new_lines.append(line)

    SEM_INDEX_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return updated


def update_four_d_index(updates: Dict[str, str]) -> int:
    """Обновляет domain в four_d_index.jsonl."""
    if not FOUR_D_IDX_PATH.exists():
        return 0

    lines = FOUR_D_IDX_PATH.read_text(encoding="utf-8").splitlines()
    new_lines = []
    updated = 0

    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            art_id = entry.get("id", "")
            if art_id in updates:
                entry["domain"] = updates[art_id]
                updated += 1
            new_lines.append(json.dumps(entry, ensure_ascii=False))
        except Exception:
            new_lines.append(line)

    FOUR_D_IDX_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return updated


def update_graph(updates: Dict[str, str]) -> Tuple[int, int]:
    """
    Инкрементальное обновление графа.
    Обновляет:
      - node.domain для каждого изменённого узла
      - edge.domain_distance для рёбер этих узлов

    Возвращает (обновлено_узлов, обновлено_рёбер).
    """
    if not GRAPH_PATH.exists():
        return 0, 0

    try:
        from sentence_transformers import SentenceTransformer
        from scipy.spatial.distance import cosine as scipy_cosine
        import numpy as np

        model_st = SentenceTransformer("all-MiniLM-L6-v2")

        # Кэш domain-эмбеддингов
        domain_vecs: Dict[str, np.ndarray] = {}

        def get_domain_vec(d: str) -> np.ndarray:
            if d not in domain_vecs:
                domain_vecs[d] = model_st.encode(d)
            return domain_vecs[d]

    except ImportError:
        logger.warning("sentence-transformers недоступен — domain_distance не пересчитается")
        model_st = None

    try:
        data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        nodes = data.get("nodes", [])
        links = data.get("links") or data.get("edges") or []

        # Строим быстрый поиск node_id → domain
        node_domains: Dict[str, str] = {}
        nodes_updated = 0

        for node in nodes:
            nid = node.get("id", "")
            if nid in updates:
                node["domain"] = updates[nid]
                nodes_updated += 1
            node_domains[nid] = node.get("domain", "general")

        # Пересчёт domain_distance для рёбер затронутых узлов
        affected_ids = set(updates.keys())
        edges_updated = 0

        if model_st:
            for link in links:
                src = link.get("source", "")
                tgt = link.get("target", "")

                if isinstance(src, dict):
                    src = src.get("id", "")
                if isinstance(tgt, dict):
                    tgt = tgt.get("id", "")

                # Обновляем только рёбра где хотя бы один узел изменился
                if src not in affected_ids and tgt not in affected_ids:
                    continue

                d_src = node_domains.get(src, "general")
                d_tgt = node_domains.get(tgt, "general")

                try:
                    v1 = get_domain_vec(d_src)
                    v2 = get_domain_vec(d_tgt)
                    new_dist = round(float(scipy_cosine(v1, v2)), 3)
                    old_dist = link.get("domain_distance", 0.0)

                    if abs(new_dist - old_dist) > 0.01:
                        link["domain_distance"] = new_dist
                        # Пересчитываем weight = sim × (1 + dist) × spec
                        sim  = link.get("similarity", 0.5)
                        spec = link.get("specificity", 0.5)
                        link["weight"] = round(sim * (1 + new_dist) * spec, 3)
                        edges_updated += 1
                except Exception:
                    pass

        GRAPH_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return nodes_updated, edges_updated

    except Exception as e:
        logger.error(f"update_graph failed: {e}", exc_info=True)
        return 0, 0


# ── Главная функция ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HX-AM v4.6 — Migrate 'general' domains"
    )
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--id",            type=str, default="")
    parser.add_argument("--llm",           action="store_true",
                        help="Использовать LLM для сложных случаев")
    parser.add_argument("--min-conf",      type=float, default=0.4,
                        help="Минимальный confidence для обновления")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Только пересобрать индексы из артефактов")
    args = parser.parse_args()

    print("\n🔄 HX-AM v4.6 — Domain Migration Tool")
    print(f"   mode={'DRY RUN' if args.dry_run else 'LIVE'} "
          f"llm={args.llm} min_conf={args.min_conf}")

    # ── Rebuild-only режим ────────────────────────────────────────────────────
    if args.rebuild_index:
        print("\n📑 Пересборка индексов из артефактов...")
        _rebuild_indexes_from_artifacts(dry_run=args.dry_run)
        return

    # ── Загрузка SemanticSpace ────────────────────────────────────────────────
    print("\n📦 Загрузка SemanticSpace...")
    try:
        from invariant_engine import SemanticSpace
        space = SemanticSpace()
        print(f"   {len(space.vectors)} векторов загружено")
    except Exception as e:
        logger.warning(f"SemanticSpace недоступен: {e}")
        space = None

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm = None
    if args.llm:
        try:
            from llm_client_v_4 import LLMClient
            llm = LLMClient()
            print("   LLM клиент готов")
        except Exception as e:
            logger.warning(f"LLM недоступен: {e}")

    # ── Поиск кандидатов ──────────────────────────────────────────────────────
    candidates = find_general_artifacts(target_id=args.id or None)
    if not candidates:
        print("\n✅ Нет артефактов с domain='general'")
        return

    print(f"\n   Найдено {len(candidates)} артефактов с 'general'\n")

    # ── Классификация ─────────────────────────────────────────────────────────
    results   = []
    updates   = {}   # {artifact_id: new_domain}
    skipped   = 0

    for art in candidates:
        new_domain, conf, method = classify_domain(art, space, llm)

        if new_domain == "general" or conf < args.min_conf:
            skipped += 1
            print(
                f"  ⚠ SKIP [{art['id']}] "
                f"→ '{new_domain}' conf={conf:.2f} method={method}"
            )
            continue

        print(
            f"  ✓ [{art['id']}] "
            f"general → '{new_domain}' "
            f"conf={conf:.2f} method={method}"
        )
        results.append({
            "id":         art["id"],
            "path":       art["path"],
            "new_domain": new_domain,
            "confidence": conf,
            "method":     method,
        })
        updates[art["id"]] = new_domain

    if not results:
        print(f"\n⚠ Нечего обновлять (пропущено {skipped})")
        print("  Попробуйте --llm для сложных случаев")
        return

    # ── Статистика новых доменов ──────────────────────────────────────────────
    domain_counts = Counter(r["new_domain"] for r in results)
    print(f"\n📊 Распределение новых доменов:")
    for d, c in domain_counts.most_common():
        print(f"   {d:25s} {c}")

    if args.dry_run:
        print(f"\n(dry-run: {len(results)} артефактов были бы обновлены)")
        return

    # ── Применение изменений ──────────────────────────────────────────────────
    print(f"\n⚙️ Применяю изменения...")

    # 1. Артефакты
    art_updated = 0
    for r in results:
        if update_artifact_file(r["path"], r["new_domain"]):
            art_updated += 1
    print(f"   ✓ Артефактов обновлено: {art_updated}")

    # 2. semantic_index.jsonl
    sem_updated = update_semantic_index(updates)
    print(f"   ✓ semantic_index: {sem_updated} записей")

    # 3. four_d_index.jsonl
    fd_updated = update_four_d_index(updates)
    print(f"   ✓ four_d_index: {fd_updated} записей")

    # 4. Граф (инкрементально)
    nodes_upd, edges_upd = update_graph(updates)
    print(f"   ✓ Граф: {nodes_upd} узлов, {edges_upd} рёбер пересчитано")

    # ── Пересчёт specificity ──────────────────────────────────────────────────
    # specificity зависит от domain_centroid, который изменился.
    # Полный пересчёт — только для затронутых артефактов.
    if space:
        _recalculate_specificity(updates, space)

    print(f"\n✅ Миграция завершена!")
    print(f"   Обновлено: {art_updated} артефактов")
    print(f"   Пропущено: {skipped} (confidence < {args.min_conf})")
    print("\n⚠️  Перезапустите сервер чтобы применить изменения в памяти.")


def _recalculate_specificity(
    updates: Dict[str, str],
    space,
) -> None:
    """
    Пересчитывает specificity для изменённых артефактов.
    Обновляет structural.specificity в файлах артефактов.
    """
    import numpy as np
    recalculated = 0

    for art_id, new_domain in updates.items():
        # Обновляем домен в памяти space.meta
        idx = space._id_to_idx.get(art_id)
        if idx is None:
            continue

        space.meta[idx]["domain"] = new_domain

        # Пересчитываем specificity
        vec  = space.vectors[idx]
        spec = space.specificity(np.array(vec), new_domain)

        # Записываем в артефакт
        art_path = ARTIFACTS_DIR / f"{art_id}.json"
        if art_path.exists():
            try:
                art = json.loads(art_path.read_text(encoding="utf-8"))
                if "structural" in art.get("data", {}):
                    art["data"]["structural"]["specificity"] = spec
                    art_path.write_text(
                        json.dumps(art, ensure_ascii=False, indent=2)
                    )
                    recalculated += 1
            except Exception as e:
                logger.warning(f"specificity update {art_id}: {e}")

    if recalculated:
        # Перезаписываем semantic_index с обновлёнными meta
        with open(SEM_INDEX_PATH, "w", encoding="utf-8") as f:
            for m in space.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    logger.info(f"specificity пересчитан для {recalculated} артефактов")


def _rebuild_indexes_from_artifacts(dry_run: bool = False) -> None:
    """
    Пересобирает semantic_index.jsonl и four_d_index.jsonl
    из актуальных данных артефактов.
    Используется после ручного редактирования или крупных миграций.
    """
    sem_entries = []
    fd_entries  = []
    total = 0

    for f in sorted(ARTIFACTS_DIR.glob("*.json")):
        if f.stem == "invariant_graph" or ".hyx-portal" in f.name:
            continue
        try:
            art    = json.loads(f.read_text(encoding="utf-8"))
            art_id = art.get("id", f.stem)
            data   = art.get("data", {})
            gen    = data.get("gen", {})
            domain = data.get("domain", "general")
            hyp    = gen.get("hypothesis", "")
            b_sync = float(gen.get("b_sync", 0))

            if hyp:
                sem_entries.append({
                    "id": art_id, "invariant": hyp,
                    "domain": domain, "b_sync": b_sync,
                })

            four_d = gen.get("four_d_matrix")
            if four_d:
                try:
                    from schemas.four_d_matrix import FourDMatrix
                    matrix = FourDMatrix.from_raw(four_d)
                    if matrix:
                        sim_path = Path("sim_results") / f"{art_id}_stress.json"
                        stab = 0.0
                        if sim_path.exists():
                            stab = json.loads(
                                sim_path.read_text()
                            ).get("stability_score", 0.0)
                        fd_entries.append({
                            "id": art_id, "domain": domain,
                            "four_d": matrix.to_dict(),
                            "vector": matrix.to_vector().tolist(),
                            "stability_score": stab,
                        })
                except Exception:
                    pass
            total += 1
        except Exception as e:
            logger.warning(f"Cannot read {f.name}: {e}")

    print(f"   Артефактов: {total}")
    print(f"   semantic_index: {len(sem_entries)} записей")
    print(f"   four_d_index: {len(fd_entries)} записей")

    if dry_run:
        print("   (dry-run: файлы не перезаписаны)")
        return

    SEM_INDEX_PATH.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in sem_entries) + "\n",
        encoding="utf-8",
    )
    FOUR_D_IDX_PATH.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in fd_entries) + "\n",
        encoding="utf-8",
    )
    print("   ✅ Индексы пересобраны")


if __name__ == "__main__":
    main()
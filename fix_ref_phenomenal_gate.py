#!/usr/bin/env python3
# fix_ref_phenomenal_gate.py — HX-AM v4.5
"""
Две вещи:

1. ПАТЧ для hxam_v_4_server.py — update_referenced_artifact() должна
   прогонять PHENOMENAL gate так же, как process_query().

2. РЕТРОАКТИВНОЕ исправление — запускает на текущем архиве:
   находит все PHENOMENAL артефакты без _downgraded_from
   у которых simulation.stability_score < 0.5 и снижает до NOVEL.

CLI:
  python fix_ref_phenomenal_gate.py --dry-run    # показать без изменений
  python fix_ref_phenomenal_gate.py              # применить
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fix_ref_gate")

ARTIFACTS_DIR     = Path("artifacts")
STABILITY_THRESHOLD = 0.5


# ══════════════════════════════════════════════════════════
# РЕТРОАКТИВНОЕ ИСПРАВЛЕНИЕ АРХИВА
# ══════════════════════════════════════════════════════════

def fix_archive(dry_run: bool = False) -> dict:
    """
    Находит все артефакты с PHENOMENAL + stability < 0.5 и снижает до NOVEL.
    Включает артефакты обновлённые через REF (у которых нет _downgraded_from).
    """
    files = sorted([
        f for f in ARTIFACTS_DIR.glob("*.json")
        if f.stem != "invariant_graph" and ".hyx-portal" not in f.name
    ])

    stats = {"checked": 0, "phenomenal_total": 0, "downgraded": 0, "kept": 0, "errors": 0}

    for f in files:
        stats["checked"] += 1
        try:
            art = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"  ERR read {f.name}: {e}")
            stats["errors"] += 1
            continue

        arch = art.get("archivist") or {}
        novelty = arch.get("novelty", "")
        if not novelty.startswith("PHENOMENAL"):
            continue

        stats["phenomenal_total"] += 1
        art_id = art.get("id", f.stem)

        # Получить stability_score
        sim   = art.get("simulation") or {}
        score = float(sim.get("stability_score", 1.0))  # default 1.0 — безопасно не трогать

        if score >= STABILITY_THRESHOLD:
            logger.info(f"  OK  {art_id}: PHENOMENAL kept (stability={score:.3f})")
            stats["kept"] += 1
            continue

        # Снижение
        reason = f"stability_score={score:.3f} < {STABILITY_THRESHOLD} (retroactive REF gate fix)"
        arch_copy = dict(arch)
        arch_copy["novelty"]           = "NOVEL"
        arch_copy["_downgraded_from"]  = "PHENOMENAL"
        arch_copy["_downgrade_reason"] = reason
        arch_copy["_downgrade_by"]     = "fix_ref_phenomenal_gate.py"

        tag = "[DRY] " if dry_run else ""
        logger.info(f"  {tag}FIX {art_id}: PHENOMENAL→NOVEL (stability={score:.3f})")

        if not dry_run:
            art["archivist"] = arch_copy
            f.write_text(json.dumps(art, ensure_ascii=False, indent=2))

        stats["downgraded"] += 1

    return stats


# ══════════════════════════════════════════════════════════
# ПАТЧ ДЛЯ hxam_v_4_server.py
# (добавить PHENOMENAL gate в update_referenced_artifact)
# ══════════════════════════════════════════════════════════

PATCH_INSTRUCTION = '''
╔══════════════════════════════════════════════════════════════════╗
║  ПАТЧ hxam_v_4_server.py — PHENOMENAL gate в REF обновлении    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  В функции update_referenced_artifact() найти строку:           ║
║    "archivist_result = archivist.process(ref_id)"               ║
║                                                                  ║
║  ПОСЛЕ этой строки добавить блок:                               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
'''

GATE_CODE_TO_INSERT = '''
        # ── PHENOMENAL GATE (REF update) ────────────────────────────────────
        # Применяем тот же gate что и в process_query()
        _arch_ref = archivist_result or {}
        if _arch_ref.get("novelty") == "PHENOMENAL":
            _sim_ref = artifact.get("simulation") or {}
            _score_ref = float(_sim_ref.get("stability_score", 1.0))
            if _score_ref < 0.5:
                _arch_ref["novelty"]           = "NOVEL"
                _arch_ref["_downgraded_from"]  = "PHENOMENAL"
                _arch_ref["_downgrade_reason"] = f"stability_score={_score_ref:.3f} < 0.5 (math gate, REF update)"
                archivist_result = _arch_ref
                try:
                    _art_path_ref = Path("artifacts") / f"{ref_id}.json"
                    if _art_path_ref.exists():
                        _art_data_ref = json.loads(_art_path_ref.read_text(encoding="utf-8"))
                        _art_data_ref["archivist"] = _arch_ref
                        _art_path_ref.write_text(json.dumps(_art_data_ref, ensure_ascii=False, indent=2))
                except Exception as _ge:
                    logger.warning(f"PHENOMENAL gate (REF) patch failed: {_ge}")
                logger.info(f"REF {ref_id}: PHENOMENAL→NOVEL (math gate, score={_score_ref:.3f})")
        # ─────────────────────────────────────────────────────────────────────
'''


def main():
    parser = argparse.ArgumentParser(description="HX-AM v4.5 — PHENOMENAL Gate REF Fix")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("\n=== PHENOMENAL Gate Retroactive Fix ===")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Threshold: stability_score < {STABILITY_THRESHOLD}")

    if not ARTIFACTS_DIR.exists():
        print(f"ERROR: {ARTIFACTS_DIR} not found")
        return

    stats = fix_archive(dry_run=args.dry_run)

    print(f"\nResults:")
    print(f"  Checked:            {stats['checked']}")
    print(f"  PHENOMENAL total:   {stats['phenomenal_total']}")
    print(f"  {'Would downgrade' if args.dry_run else 'Downgraded'}:  {stats['downgraded']}")
    print(f"  Kept PHENOMENAL:    {stats['kept']}")
    print(f"  Errors:             {stats['errors']}")

    if args.dry_run and stats['downgraded'] > 0:
        print("\nRun without --dry-run to apply.")

    print(PATCH_INSTRUCTION)
    print("Code to insert in update_referenced_artifact() after archivist.process():")
    print(GATE_CODE_TO_INSERT)


if __name__ == "__main__":
    main()

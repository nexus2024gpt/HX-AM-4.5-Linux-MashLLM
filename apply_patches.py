# apply_patches.py — HX-AM v4.2 patch applier
"""
Применяет точечные патчи к hxam_v_4_server.py и math_core.py.
Запускать из корня проекта:
  python apply_patches.py --dry-run   # проверить без записи
  python apply_patches.py             # применить
"""
import argparse
import re
import sys
from pathlib import Path


def patch_server(content: str) -> tuple[str, list]:
    """Добавляет PHENOMENAL gate после блока MathCore."""
    changes = []

    GATE_CODE = '''
        # ── PHENOMENAL GATE ──────────────────────────────────────────────────
        # Если Archivist дал PHENOMENAL, но MathCore показал нестабильность —
        # понижаем до NOVEL. Патчим и в памяти, и в сохранённом файле.
        _arch = result.get("archivist") or {}
        if _arch.get("novelty") == "PHENOMENAL":
            _sim = result.get("simulation") or {}
            _score = float(_sim.get("stability_score", 1.0))
            if _score < 0.5:
                _arch["novelty"] = "NOVEL"
                _arch["_downgraded_from"] = "PHENOMENAL"
                _arch["_downgrade_reason"] = f"stability_score={_score:.3f} < 0.5 (math gate)"
                result["archivist"] = _arch
                if save:
                    try:
                        _art_path = Path("artifacts") / f"{job_id}.json"
                        if _art_path.exists():
                            _art_data = json.loads(_art_path.read_text(encoding="utf-8"))
                            _art_data["archivist"] = _arch
                            _art_path.write_text(json.dumps(_art_data, ensure_ascii=False, indent=2))
                    except Exception as _pe:
                        logger.warning(f"PHENOMENAL gate patch failed: {_pe}")
                logger.info(f"Job {job_id}: PHENOMENAL→NOVEL (math gate, score={_score:.3f})")
        # ─────────────────────────────────────────────────────────────────────

'''

    # Найти место вставки: после блока MathCore, перед "# ── ИСТОРИЯ"
    insert_marker = "        # ── ИСТОРИЯ"
    if GATE_CODE.strip() not in content:
        if insert_marker in content:
            content = content.replace(insert_marker, GATE_CODE + insert_marker, 1)
            changes.append("hxam_v_4_server.py: added PHENOMENAL gate before history logging")
        else:
            changes.append("hxam_v_4_server.py: ERROR — could not find insertion marker '# ── ИСТОРИЯ'")
    else:
        changes.append("hxam_v_4_server.py: PHENOMENAL gate already present, skipped")

    return content, changes


def patch_math_core(content: str) -> tuple[str, list]:
    """
    1. GraphInvariantStability: нормализует r_final через proximity к p_c.
    2. ResonanceMatcher.find_similar: добавляет exclude_id параметр.
    3. MathCore.find_resonance: добавляет exclude_id параметр.
    """
    changes = []

    # --- Патч 1: GraphInvariantStability.run() ---
    OLD_GI = '''        order  = min(1.0, max(0.0, S + C * 0.15 - eta * 0.25))
        stable = p > p_c and order > 0.4'''
    NEW_GI = '''        # Normalize by proximity to critical threshold (prevents trivially high scores)
        proximity = (p - p_c) / max(1.0 - p_c, 1e-6)  # 0=at threshold, 1=well above
        order  = min(1.0, max(0.0, proximity * (1.0 + C * 0.15) - eta * 0.25))
        stable = p > p_c and order > 0.3'''
    if OLD_GI in content:
        content = content.replace(OLD_GI, NEW_GI, 1)
        changes.append("math_core.py: GraphInvariantStability normalized by p_c proximity")
    elif NEW_GI in content:
        changes.append("math_core.py: GraphInvariantStability patch already applied, skipped")
    else:
        changes.append("math_core.py: ERROR — GraphInvariantStability pattern not found")

    # --- Патч 2: ResonanceMatcher.find_similar signature ---
    OLD_RS = '''    def find_similar(self, query_vec: np.ndarray,
                     top_k: int = 5, threshold: float = 0.55) -> List[Dict]:'''
    NEW_RS = '''    def find_similar(self, query_vec: np.ndarray,
                     top_k: int = 5, threshold: float = 0.55,
                     exclude_id: str = "") -> List[Dict]:'''
    if OLD_RS in content:
        content = content.replace(OLD_RS, NEW_RS, 1)
        changes.append("math_core.py: ResonanceMatcher.find_similar added exclude_id param")
    elif NEW_RS in content:
        changes.append("math_core.py: find_similar exclude_id already present, skipped")
    else:
        changes.append("math_core.py: ERROR — find_similar signature not found")

    # --- Патч 3: ResonanceMatcher.find_similar body — add exclude_id filtering ---
    OLD_BODY = '''        results = []
        for i, vec in enumerate(self._vectors):
            r = self._res(query_vec, vec)
            if r >= threshold:
                results.append({**self._meta[i], "4d_resonance": r})
        return sorted(results, key=lambda x: -x["4d_resonance"])[:top_k]'''
    NEW_BODY = '''        results = []
        for i, vec in enumerate(self._vectors):
            if exclude_id and self._meta[i].get("id") == exclude_id:
                continue
            r = self._res(query_vec, vec)
            if r >= threshold:
                results.append({**self._meta[i], "4d_resonance": r})
        return sorted(results, key=lambda x: -x["4d_resonance"])[:top_k]'''
    if OLD_BODY in content:
        content = content.replace(OLD_BODY, NEW_BODY, 1)
        changes.append("math_core.py: find_similar body excludes self by exclude_id")
    elif NEW_BODY in content:
        changes.append("math_core.py: find_similar body already patched, skipped")
    else:
        changes.append("math_core.py: ERROR — find_similar body pattern not found")

    # --- Патч 4: MathCore.find_resonance signature ---
    OLD_FR = '''    def find_resonance(self, query_four_d: Dict, query_domain: str,
                       query_survival: str = "UNKNOWN",
                       target_domains: Optional[List[str]] = None,
                       top_k: int = 3) -> Dict[str, Any]:'''
    NEW_FR = '''    def find_resonance(self, query_four_d: Dict, query_domain: str,
                       query_survival: str = "UNKNOWN",
                       target_domains: Optional[List[str]] = None,
                       top_k: int = 3, exclude_id: str = "") -> Dict[str, Any]:'''
    if OLD_FR in content:
        content = content.replace(OLD_FR, NEW_FR, 1)
        changes.append("math_core.py: MathCore.find_resonance added exclude_id param")
    elif NEW_FR in content:
        changes.append("math_core.py: find_resonance exclude_id already present, skipped")
    else:
        changes.append("math_core.py: ERROR — find_resonance signature not found")

    # --- Патч 5: MathCore.find_resonance — pass exclude_id to find_similar ---
    OLD_SIM = '''        similar   = self.resonance_matcher.find_similar(query_vec, top_k=top_k)'''
    NEW_SIM = '''        similar   = self.resonance_matcher.find_similar(query_vec, top_k=top_k, exclude_id=exclude_id)'''
    if OLD_SIM in content:
        content = content.replace(OLD_SIM, NEW_SIM, 1)
        changes.append("math_core.py: find_resonance passes exclude_id to find_similar")
    elif NEW_SIM in content:
        changes.append("math_core.py: find_similar call already passes exclude_id, skipped")
    else:
        changes.append("math_core.py: ERROR — find_similar call pattern not found")

    return content, changes


def main():
    parser = argparse.ArgumentParser(description="Apply HX-AM v4.2 quality patches")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show changes without writing files")
    args = parser.parse_args()

    all_changes = []

    for filename, patcher in [
        ("hxam_v_4_server.py", patch_server),
        ("math_core.py", patch_math_core),
    ]:
        path = Path(filename)
        if not path.exists():
            print(f"[SKIP] {filename} not found")
            continue
        content = path.read_text(encoding="utf-8")
        patched, changes = patcher(content)
        all_changes.extend(changes)
        if not args.dry_run and patched != content:
            # Backup
            backup = path.with_suffix(".py.bak")
            backup.write_text(content, encoding="utf-8")
            path.write_text(patched, encoding="utf-8")
            print(f"[PATCHED] {filename} (backup: {backup.name})")
        elif patched == content:
            print(f"[UNCHANGED] {filename}")
        else:
            print(f"[DRY] {filename} would be patched")

    print("\nChanges summary:")
    for c in all_changes:
        status = "✅" if "ERROR" not in c else "❌"
        print(f"  {status} {c}")

    if args.dry_run:
        print("\nRun without --dry-run to apply.")


if __name__ == "__main__":
    main()

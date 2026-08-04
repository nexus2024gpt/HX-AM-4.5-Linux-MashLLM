#!/usr/bin/env python3
# tests/test_response_normalizer.py — HX-AM core pipeline
"""
Регрессионные тесты для response_normalizer.py — слоя, который спасает
LLM-ответы от малейшего отклонения от валидного JSON перед PipelineGuard.

Запуск из корня проекта:
  python tests/test_response_normalizer.py

Покрытие: is_garbage_text, clean_llm_artifacts, extract_json_multi,
normalize_gen, normalize_ver.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from response_normalizer import (
    is_garbage_text,
    clean_llm_artifacts,
    extract_json_multi,
    normalize_gen,
    normalize_ver,
)

PASS = "✅"
FAIL = "❌"

_pass_count = 0
_fail_count = 0


def check(name: str, condition: bool, detail: str = ""):
    global _pass_count, _fail_count
    if condition:
        print(f"  {PASS} {name}")
        _pass_count += 1
    else:
        print(f"  {FAIL} {name}  ← {detail}")
        _fail_count += 1


def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ═══════════════════════════════════════════════════════════
# 1. is_garbage_text
# ═══════════════════════════════════════════════════════════

def test_is_garbage_text():
    section("1. is_garbage_text")

    check("empty string is garbage", is_garbage_text(""))
    check("None-like empty is garbage", is_garbage_text(None))
    check(
        "bare <think> tag is garbage",
        is_garbage_text("<think>"),
    )
    check(
        "channel/thought leak is garbage",
        is_garbage_text("<|channel>thought"),
    )
    check(
        "special tokens stripped down to nothing is garbage",
        is_garbage_text("<|im_start|><|im_end|>"),
    )
    check(
        "real hypothesis text is not garbage",
        not is_garbage_text(
            "Synchronization of coupled oscillators depends on the "
            "critical coupling threshold K_c."
        ),
    )
    check(
        "short real text below min_meaningful_chars is garbage",
        is_garbage_text("too short", min_meaningful_chars=20),
    )


# ═══════════════════════════════════════════════════════════
# 2. clean_llm_artifacts
# ═══════════════════════════════════════════════════════════

def test_clean_llm_artifacts():
    section("2. clean_llm_artifacts")

    check(
        "empty string passes through",
        clean_llm_artifacts("") == "",
    )
    check(
        "think open/close tags stripped (content between them is kept)",
        clean_llm_artifacts("<think>reasoning</think>Actual answer")
        == "reasoningActual answer",
        repr(clean_llm_artifacts("<think>reasoning</think>Actual answer")),
    )
    check(
        "im_start/im_end tokens stripped",
        "<|im_start|>" not in clean_llm_artifacts("<|im_start|>Hello<|im_end|>"),
    )
    check(
        "leading newlines stripped",
        clean_llm_artifacts("\n\nHello") == "Hello",
    )
    check(
        "plain text unaffected",
        clean_llm_artifacts("Hello world") == "Hello world",
    )


# ═══════════════════════════════════════════════════════════
# 3. extract_json_multi
# ═══════════════════════════════════════════════════════════

def test_extract_json_multi():
    section("3. extract_json_multi")

    # Strategy: direct
    data, strategy = extract_json_multi('{"a": 1}')
    check("direct parse works", data == {"a": 1}, str(data))
    check("direct parse reports strategy='direct'", strategy == "direct", strategy)

    # Strategy: markdown fence stripping
    data, strategy = extract_json_multi('```json\n{"a": 1}\n```')
    check("markdown-fenced JSON extracted", data == {"a": 1}, str(data))

    # Strategy: outermost braces with leading chatter
    data, strategy = extract_json_multi(
        'Sure, here is the JSON:\n{"hypothesis": "test", "domain": "physics"}'
    )
    check(
        "JSON preceded by chatter text extracted",
        data == {"hypothesis": "test", "domain": "physics"},
        str(data),
    )

    # Strategy: truncated/unclosed JSON (common LLM failure: hit token limit)
    data, strategy = extract_json_multi('{"hypothesis": "test", "domain": "physics"')
    check(
        "truncated JSON (missing closing brace) still recovered",
        data is not None and data.get("hypothesis") == "test",
        str(data),
    )

    # Empty input
    data, strategy = extract_json_multi("")
    check("empty input returns None", data is None)
    check("empty input reports strategy='empty_input'", strategy == "empty_input", strategy)

    # Completely unparseable garbage
    data, strategy = extract_json_multi("not json at all, just prose.")
    check(
        "unparseable prose returns None (no crash)",
        data is None,
        str(data),
    )


# ═══════════════════════════════════════════════════════════
# 4. normalize_gen
# ═══════════════════════════════════════════════════════════

def test_normalize_gen():
    section("4. normalize_gen")

    # Clean, well-formed input round-trips with no repairs
    raw = (
        '{"hypothesis": "Coupled oscillators synchronize above a critical '
        'coupling threshold in scale-free networks.", '
        '"mechanism": "Kuramoto-type phase coupling", '
        '"domain": "physics", "b_sync": 0.8, '
        '"implication": "Predicts phase transition at K_c"}'
    )
    data, repairs, ok = normalize_gen(raw)
    check("well-formed generation input is accepted", ok is True)
    check("domain passed through unchanged", data.get("domain") == "physics", data.get("domain"))
    check("b_sync passed through unchanged", data.get("b_sync") == 0.8, data.get("b_sync"))

    # Russian domain gets translated via DOMAIN_MAP
    raw_ru_domain = (
        '{"hypothesis": "Coupled oscillators synchronize above a critical '
        'coupling threshold in scale-free networks.", '
        '"domain": "физика", "b_sync": 0.7}'
    )
    data, repairs, ok = normalize_gen(raw_ru_domain)
    check("Russian domain name normalized to English", data.get("domain") == "physics", data.get("domain"))
    check("domain normalization is logged as a repair", any("domain" in r for r in repairs), repairs)

    # Missing b_sync falls back to conservative default 0.55
    raw_no_bsync = (
        '{"hypothesis": "Coupled oscillators synchronize above a critical '
        'coupling threshold in scale-free networks.", "domain": "physics"}'
    )
    data, repairs, ok = normalize_gen(raw_no_bsync)
    check("missing b_sync defaults to 0.55", data.get("b_sync") == 0.55, data.get("b_sync"))

    # Hypothesis is pure garbage/special-token leakage and unrecoverable
    raw_garbage = '{"hypothesis": "<|channel>thought", "domain": "physics"}'
    data, repairs, ok = normalize_gen(raw_garbage)
    check(
        "garbage hypothesis with no recoverable fallback field is rejected",
        ok is False,
        f"ok={ok} data={data}",
    )

    # Hypothesis garbage but mechanism field is usable → recovered
    raw_recoverable = (
        '{"hypothesis": "<think>", '
        '"mechanism": "Coupled oscillators synchronize above a critical threshold.", '
        '"domain": "physics"}'
    )
    data, repairs, ok = normalize_gen(raw_recoverable)
    check(
        "garbage hypothesis recovered from mechanism field",
        ok is True and data.get("hypothesis", "").startswith("Coupled oscillators"),
        f"ok={ok} hypothesis={data.get('hypothesis')!r}",
    )

    # Completely unparseable input fails cleanly, no exception
    data, repairs, ok = normalize_gen("this is not json")
    check("unparseable input fails cleanly without raising", ok is False)


# ═══════════════════════════════════════════════════════════
# 5. normalize_ver
# ═══════════════════════════════════════════════════════════

def test_normalize_ver():
    section("5. normalize_ver")

    # Well-formed verifier output
    raw = '{"verdict": "VALID", "confidence": 0.9, "issues": []}'
    data, repairs, ok = normalize_ver(raw)
    check("well-formed verification input is accepted", ok is True)
    check("verdict passed through unchanged", data.get("verdict") == "VALID", data.get("verdict"))

    # Russian/synonym verdict gets mapped via VERDICT_MAP
    raw_ru = '{"verdict": "подтверждено", "confidence": 0.85}'
    data, repairs, ok = normalize_ver(raw_ru)
    check(
        "Russian verdict synonym normalized to VALID",
        data.get("verdict") == "VALID",
        data.get("verdict"),
    )

    # Missing verdict is inferred from confidence (conservative fallback)
    raw_no_verdict = '{"confidence": 0.9}'
    data, repairs, ok = normalize_ver(raw_no_verdict)
    check(
        "missing verdict inferred as VALID from high confidence",
        data.get("verdict") == "VALID",
        data.get("verdict"),
    )

    raw_low_conf = '{"confidence": 0.1}'
    data, repairs, ok = normalize_ver(raw_low_conf)
    check(
        "missing verdict with low confidence defaults to conservative WEAK",
        data.get("verdict") == "WEAK",
        data.get("verdict"),
    )

    # Confidence out of range gets clamped into [0, 1]
    raw_over = '{"verdict": "VALID", "confidence": 5}'
    data, repairs, ok = normalize_ver(raw_over)
    check(
        "out-of-range confidence clamped to 1.0",
        data.get("confidence") == 1.0,
        data.get("confidence"),
    )

    # issues as a single string gets split into a list
    raw_issues_str = '{"verdict": "WEAK", "confidence": 0.5, "issues": "unclear scope; no citation"}'
    data, repairs, ok = normalize_ver(raw_issues_str)
    check(
        "issues string split into list",
        data.get("issues") == ["unclear scope", "no citation"],
        data.get("issues"),
    )

    # Missing translation gets a safe default with survival=UNKNOWN
    raw_no_translation = '{"verdict": "VALID", "confidence": 0.8}'
    data, repairs, ok = normalize_ver(raw_no_translation)
    check(
        "missing translation defaults to survival=UNKNOWN",
        data.get("translation", {}).get("survival") == "UNKNOWN",
        data.get("translation"),
    )

    # Unparseable input fails cleanly
    data, repairs, ok = normalize_ver("garbage, not json")
    check("unparseable verification input fails cleanly without raising", ok is False)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  response_normalizer.py — Regression Tests")
    print("=" * 55)

    tests = [
        ("is_garbage_text",     test_is_garbage_text),
        ("clean_llm_artifacts", test_clean_llm_artifacts),
        ("extract_json_multi",  test_extract_json_multi),
        ("normalize_gen",       test_normalize_gen),
        ("normalize_ver",       test_normalize_ver),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n  {FAIL} {name} crashed: {e}")
            traceback.print_exc()
            global _fail_count
            _fail_count += 1

    print(f"\n{'=' * 55}")
    print(f"  Results: {PASS} {_pass_count} passed  |  {FAIL} {_fail_count} failed")
    print(f"{'=' * 55}\n")

    sys.exit(0 if _fail_count == 0 else 1)


if __name__ == "__main__":
    main()

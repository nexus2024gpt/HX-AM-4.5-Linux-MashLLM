# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HX-AM Proxy v4.5 — a dual-LLM hypothesis-formalization system. A generator LLM proposes a
4D-formalized hypothesis (structure/factors/dynamics/time parameters), a verifier LLM stress-tests
it via translation + adversarial critique, an Invariant Engine builds a semantic graph with
4D-weighted edges and detects phase transitions, an Archivist rates novelty
(PHENOMENAL | NOVEL | KNOWN | REPHRASING), and MathCore validates dynamical stability with actual
simulations (Kuramoto, Ising, percolation, delay, Lotka-Volterra, graph_invariant). MGAP then maps
validated invariants onto real industry monitoring systems (WMS, EEGlab, OpenFOAM, etc.) via a
UNESCO-taxonomy domain classifier and an η/τ/K metric-gap calculation. See `README.md` for the
full provider-chain diagram (Russian).

## Setup & running

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements_v42.txt
```

The FastAPI app (`hxam_v_4_server.py`) expects two local LLM services to already be listening,
falling back to cloud providers per `config/providers.json` priority when they're not:

```bash
./start_llama_cpu.sh        # llama-server (llama.cpp), port 11435 — local Qwen2.5-1.5B gen/ver
mesh-llm client --auto      # mesh-llm P2P client, port 9337 — Qwen3.5-9B/Qwen3-8B/Qwen3-4B
python hxam_v_4_server.py   # FastAPI app → http://localhost:8000
```

`start_hxam_v45.sh` sequences all three with health-check polling and is the normal way to boot
the whole stack. `config/providers.json` (gitignored, holds live API keys) is the actual provider
registry read by `api_usage_tracker.py`/`smart_router.py` at runtime — `.env` only carries legacy
base URLs/model names and is not the source of truth for provider priority/enable state.

## Tests

There is no pytest config (no `pytest.ini`/`pyproject.toml`) and no lint config in the repo. Tests
are standalone scripts, run directly and from the project root:

```bash
python mgap_lib/tests/test_integration.py        # MGAP: no LLM, no DB required
python mgap_lib/tests/test_hybrid_components.py
python tests/test_response_normalizer.py         # LLM-JSON repair layer, pure functions
```

`test_keys/test_*.py` are one-off connectivity probes for individual provider API keys (Groq,
Gemini/google, HuggingFace, NVIDIA, OpenRouter), not part of a suite.

## Architecture

**Request pipeline** — `process_query()` in `hxam_v_4_server.py` is the one function to read to
understand the system; everything else is a component it orchestrates:

1. REF-detection (`extract_ref_id`) — if the query references an existing artifact ID, the job
   reuses that `job_id` and updates it in place instead of creating a new one.
2. RAG context — `SemanticSpace.nearest()` (`invariant_engine.py`) pulls structurally-similar prior
   invariants, filtered for domain diversity (`filter_rag_diversity`) so the prompt doesn't get
   flooded by one overrepresented domain.
3. Generation — `QuestionGenerator.build_generation_context()` makes 4 sequential LLM calls (one
   per field) rather than asking for the whole hypothesis in one shot.
4. **Programmatic enrichment** — the 4D parameter matrix (`four_d_matrix`) and `b_sync` are computed
   by `FourDBuilder`/`compute_b_sync`, not asked of the LLM. Only `hypothesis`/`mechanism`/
   `implication`/`domain` come from the generator model; the numeric structure is deterministic.
5. `PipelineGuard.validate_gen()` — rejects malformed generation output into `QuarantineLog` before
   spending a verification call on it.
6. Verification — `LLMClient.verify()` with `RetryManager`-driven retry on validation failure.
7. `response_normalizer.py` cleans/repairs LLM JSON (garbage-text detection, bracket-balancing,
   alias resolution) — expect LLM output to arrive malformed and need this layer, not to be
   trusted as valid JSON directly.
8. `Archivist` novelty rating → `MathCore` stress test/resonance → MGAP match → artifact persisted
   under `artifacts/`.

**Provider routing** is a separate concern from the pipeline logic: `llm_client_v_4.py`
(`LLMClient`) makes the HTTP calls; `api_usage_tracker.py` (`APIUsageTracker`) and
`smart_router.py` (`SmartRouter`, `CircuitBreaker`, `HealthChecker`) decide which provider/account
to use per role (`generator`/`verifier`) based on configured priority in
`config/providers.json`, circuit-breaker trip state, and health-check results. Local mesh-llm/
llama-server entries have priority 1-3; Groq/Gemini/OpenRouter/NVIDIA NIM/HuggingFace are the
fallback chain, split across two accounts ("Nexus"/"Roman") for quota headroom.

**MGAP is a facade over selected `mgap_lib` pieces, not a duplicate implementation.**
`hxam_v_4_server.py` imports `MGAPMatcher` from the top-level `mgap_matcher.py` (1077 lines,
v4.7), which loads `mgap_registry.json` itself and calls `match_artifact()`/`match_batch()`/
`get_registry_summary()` directly for the `/mgap/*` endpoints. It genuinely reuses four newer
`mgap_lib.engine` modules (`dimensional_normalizer`, `threshold_calculator`, `topology_validator`,
`falsification_engine`) — those get real fixes and are live. The *rest* of `mgap_lib/` —
`engine/matcher.py` (`MGAPEngine`), `registry.py`, `domain_classifier.py`, `gap_calculator.py`,
`models/database.py` (SQLAlchemy), `api/routes.py`, `cli/mgap_cli.py` — is a separate, complete,
DB-backed engine that is **not** wired into the server; it's reachable only via
`mgap_lib/cli/mgap_cli.py` or direct import. `mgap_lib/README_MGAP.md` documents both halves
clearly (updated 2026-08-04) — read it before assuming either "the whole of mgap_lib is dead" or
"the whole of mgap_lib is live"; it's split down the middle.

**Generated/output directories are not source** and can be inspected but shouldn't be hand-edited:
`artifacts/` (per-hypothesis JSON + `four_d_index.jsonl` + `invariant_graph.json` +
`semantic_index.jsonl`), `insights/`, `sim_results/`, `mgap_results/`, `chat_history/`, `logs/`.
None of `artifacts/` is git-tracked (confirmed via `git ls-files artifacts/` — empty); its only
backup is the Google Drive sync mirror of the whole working directory, with no version history.

**If `invariant_graph.json` ever needs rebuilding from scratch**, use
`tools/rebuild_graph_knn.py` only. Two earlier attempts (`tools/archive/rebuild_graph_clean.py`,
`tools/archive/restore_graph.py`) are archived and documented as superseded in their own headers —
both produce a degenerate near-complete-clique graph on a dense corpus; `rebuild_graph_knn.py`'s
k-NN capping is what avoids that.

**Edge weight must stay consistent across both edge-creation paths.** Edges get
`weight = similarity × (1 + domain_distance) × specificity × (1 + four_d_resonance × 0.2)`.
Two places compute this and they must not drift: `InvariantGraph.add_edge()` in
`invariant_engine.py` (live pipeline) and `build_knn_graph()` in `tools/rebuild_graph_knn.py`
(bulk rebuild). The rebuild tool used to hardcode `four_d_resonance: 0.0`, which silently cost
those edges up to 20% of their weight — fixed 2026-08-05, with `tools/repair_edge_resonance.py`
available to recompute resonance/weight on an existing graph without touching topology.

**Deleting an artifact touches four stores, and the API only handles three.**
`DELETE /artifact/{id}` moves the file to `trash/` and cleans the graph + `semantic_index.jsonl`,
but leaves `four_d_index.jsonl` behind. Use `tools/purge_artifacts.py` for a consistent removal
(`--orphans` also sweeps graph/index entries whose artifact file is already gone).

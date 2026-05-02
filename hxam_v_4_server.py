# HX-AM v4.2 Server — 4D-граф формализации + MathCore
# Исправления v4.2.1:
#   - process_query: MathCore перемещён ПОСЛЕ save_artifact (fix: _patch_artifact)
#   - process_query: domain diversity guard (подавление overrepresented+low-spec)
#   - result dict инициализируется с simulation/resonance=None до сохранения

import os, json, time, hashlib, logging, re, shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from llm_client_v_4 import LLMClient
from archivist import Archivist
from invariant_engine import SemanticSpace, InvariantGraph, PhaseDetector, process_with_invariants
from pipeline_guard import PipelineGuard, RollbackManager, QuarantineLog, FailureCode
from question_generator import QuestionGenerator
from api_usage_tracker import tracker
from response_normalizer import normalize_gen, normalize_ver, repairs_summary

from math_core import MathCore
from mgap_matcher import MGAPMatcher

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HXAM.v4.2")

app = FastAPI(title="HX-AM v4.2")

Path("artifacts").mkdir(exist_ok=True)
Path("trash").mkdir(exist_ok=True)
Path("chat_history").mkdir(exist_ok=True)
Path("sim_results").mkdir(exist_ok=True)
Path("insights").mkdir(exist_ok=True)

logger.info("Загрузка семантического индекса...")
semantic_space = SemanticSpace()
logger.info("Загрузка графа инвариантов...")
invariant_graph = InvariantGraph()
phase_detector = PhaseDetector()
logger.info("Invariant Engine готов.")
archivist = Archivist(space=semantic_space, graph=invariant_graph)

guard = PipelineGuard()
quarantine = QuarantineLog()
question_gen = QuestionGenerator(space=semantic_space, graph=invariant_graph)

logger.info("Инициализация MathCore...")
math_core = MathCore(artifacts_dir="artifacts", four_d_index="artifacts/four_d_index.jsonl")
logger.info("MathCore готов.")

logger.info("Инициализация MGAPMatcher...")
mgap_matcher = MGAPMatcher(
    registry_path="mgap_registry.json",
    artifacts_dir="artifacts",
)
logger.info(f"MGAPMatcher готов. Моделей в реестре: {len(mgap_matcher.registry)}")


# ════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    text: str
    domain: str = "general"
    x_coordinate: float = 500.0

class ProvidersUpdateRequest(BaseModel):
    providers: List[Dict[str, Any]]

class ProviderAddRequest(BaseModel):
    id: str
    provider: str
    account: str
    label: str
    api_key: str
    api_base: str
    model: str
    roles: List[str]
    enabled: bool = True
    priority: int = 99

class ResetRequest(BaseModel):
    scope: str = "today"


# ════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════

def load_prompt(name: str) -> str:
    path = Path("prompts") / name
    if not path.exists():
        logger.warning(f"Prompt file not found: {path}")
        return ""
    return path.read_text(encoding="utf-8")

GEN_PROMPT = load_prompt("generator_prompt.txt")
VER_PROMPT = load_prompt("verifier_prompt.txt")


def resolve_domain(gen: dict, req_domain: str) -> str:
    gen_domain = gen.get("domain", "").strip().lower()
    if gen_domain and gen_domain not in ("general", ""):
        return gen_domain
    if req_domain and req_domain not in ("general", ""):
        return req_domain
    return "general"


def save_artifact(job_id: str, data: Dict[str, Any]) -> Path:
    path = Path("artifacts")
    path.mkdir(exist_ok=True)
    obj = {
        "id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
        "history": [],
    }
    file = path / f"{job_id}.json"
    file.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    return file


def log_history(entry: Dict[str, Any]):
    path = Path("chat_history")
    path.mkdir(exist_ok=True)
    file = path / "history.jsonl"
    with open(file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _rejected_response(job_id: str, code: str, reason: str, stage: str) -> dict:
    return {
        "job_id": job_id,
        "rejected": True,
        "stage": stage,
        "failure_code": code,
        "reason": reason,
        "generation": None,
        "verification": None,
        "saved": False,
        "artifact": None,
        "domain": None,
    }


def filter_rag_diversity(
    similar: List[dict],
    max_per_domain: int = 1,
    sim_cap: float = 0.88,
) -> List[dict]:
    seen_domains: dict = {}
    result = []
    dropped = 0
    for s in similar:
        sim = s.get("similarity", 0.0)
        domain = s.get("domain", "general")
        if sim >= sim_cap:
            dropped += 1
            continue
        if seen_domains.get(domain, 0) >= max_per_domain:
            dropped += 1
            continue
        seen_domains[domain] = seen_domains.get(domain, 0) + 1
        result.append(s)
    if dropped:
        logger.info(f"RAG filter: {len(similar)} → {len(result)} (dropped {dropped})")
    return result


def extract_ref_id(text: str) -> Optional[str]:
    match = re.search(r'\[REF:([a-f0-9]{8,20})\]', text)
    return match.group(1) if match else None


def _reload_semantic_index():
    try:
        with open(semantic_space.index_path, "w", encoding="utf-8") as f:
            for m in semantic_space.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to rewrite semantic_index: {e}")


def update_referenced_artifact(ref_id: str, result: dict, query: str):
    artifact_path = Path("artifacts") / f"{ref_id}.json"
    if not artifact_path.exists():
        logger.warning(f"REF artifact {ref_id} not found — skipping update")
        return

    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        data = artifact.get("data", {})

        history_entry = {
            "revision": len(artifact.get("history", [])) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ref_query": query,
            "gen": data.get("gen", {}),
            "ver": data.get("ver", {}),
            "structural": data.get("structural", {}),
            "archivist": artifact.get("archivist"),
        }
        if "history" not in artifact:
            artifact["history"] = []
        artifact["history"].append(history_entry)

        new_gen = result.get("generation", {})
        new_ver = result.get("verification", {})
        new_structural = result.get("structural", {})
        new_domain = result.get("domain", data.get("domain", "general"))

        artifact["data"]["gen"] = new_gen
        artifact["data"]["ver"] = new_ver
        artifact["data"]["structural"] = new_structural
        artifact["data"]["domain"] = new_domain
        artifact["data"]["normalization"] = result.get("repairs", {})
        artifact["last_updated"] = datetime.now(timezone.utc).isoformat()
        artifact["ref_query"] = query

        new_invariant = new_gen.get("hypothesis", "")
        if new_invariant:
            idx = semantic_space._id_to_idx.get(ref_id)
            if idx is not None and idx < len(semantic_space.vectors):
                new_vec = semantic_space.encode(new_invariant)
                semantic_space.vectors[idx] = new_vec
                semantic_space.meta[idx] = {
                    "id": ref_id,
                    "invariant": new_invariant,
                    "domain": new_domain,
                    "b_sync": float(new_gen.get("b_sync", 0)),
                }
                _reload_semantic_index()
                logger.info(f"Semantic space updated for REF {ref_id}")

        if ref_id in invariant_graph.G:
            translation = new_ver.get("translation", {})
            survival = (translation.get("survival", "UNKNOWN")
                        if isinstance(translation, dict) else "UNKNOWN")
            invariant_graph.G.nodes[ref_id].update({
                "domain": new_domain,
                "b_sync": float(new_gen.get("b_sync", 0)),
                "stability": new_structural.get("stability", "unknown"),
                "specificity": new_structural.get("specificity", 0.5),
                "survival": survival,
            })
            invariant_graph._save()
            logger.info(f"Graph node updated for REF {ref_id}")

        artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))

        try:
            archivist_result = archivist.process(ref_id)
            
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
            
            logger.info(f"Archivist re-ran for REF {ref_id}: {archivist_result.get('novelty')}")
        except Exception as e:
            logger.warning(f"Archivist re-run failed for REF {ref_id}: {e}")

        logger.info(f"REF artifact {ref_id} updated. History entries: {len(artifact['history'])}")

    except Exception as e:
        logger.error(f"update_referenced_artifact failed for {ref_id}: {e}", exc_info=True)


# ════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ════════════════════════════════════════════════════════════════

# Порог domain diversity: если домен занимает больше DOMAIN_CAP доли архива
# И specificity ниже SPEC_FLOOR — сохранение подавляется.
_DOMAIN_CAP = 0.30
_SPEC_FLOOR = 0.15
_DOMAIN_MIN_NODES = 15   # не применяем до накопления базы


def _is_domain_overrepresented(domain: str, spec: float) -> tuple[bool, str]:
    """Возвращает (подавить, причина) для domain diversity guard."""
    G = invariant_graph.G
    total = G.number_of_nodes()
    if total < _DOMAIN_MIN_NODES:
        return False, ""
    dom_count = sum(1 for _, a in G.nodes(data=True) if a.get("domain") == domain)
    ratio = dom_count / total
    if ratio > _DOMAIN_CAP and spec < _SPEC_FLOOR:
        return True, f"domain='{domain}' {dom_count}/{total} ({ratio:.0%}), spec={spec:.3f}"
    return False, ""


def process_query(req: QueryRequest):
    client = LLMClient()
    job_id = hashlib.md5(f"{req.text}{time.time()}".encode()).hexdigest()[:12]
    rollback = RollbackManager()
    gen_model = "unknown"
    ver_model = "unknown"
    gen_repairs: List[str] = []
    ver_repairs: List[str] = []

    try:
        # ── RAG ─────────────────────────────────────────────────────
        rag_raw = semantic_space.nearest(req.text, top_k=5, threshold=0.55)
        rag_similar = filter_rag_diversity(rag_raw, max_per_domain=1, sim_cap=0.88)
        rag_block = ""
        if rag_similar:
            rag_block = "\n\nRAG context (structural inspiration only — do NOT copy phrases):\n"
            for s in rag_similar:
                rag_block += f"- [{s['domain']}] {s['invariant']} (sim:{s['similarity']})\n"

        # ── ГЕНЕРАЦИЯ ───────────────────────────────────────────────
        gen_input = f"{GEN_PROMPT}{rag_block}\n\nX: {req.x_coordinate}\n\nUser input:\n{req.text}"
        logger.info(f"Job {job_id}: generating... rag_raw={len(rag_raw)} rag_filtered={len(rag_similar)}")
        gen_raw, gen_model = client.generate(gen_input)

        vr = guard.validate_gen_raw(gen_raw, gen_model)
        if not vr:
            quarantine.record(job_id, req.text, vr.code, vr.reason, "generation", gen_model=gen_model)
            return _rejected_response(job_id, vr.code, vr.reason, "generation")

        gen, gen_repairs, gen_ok = normalize_gen(gen_raw)
        if not gen_ok:
            reason = f"Generator output unrecoverable: {'; '.join(gen_repairs[-3:])}"
            quarantine.record(job_id, req.text, FailureCode.GEN_UNRECOVERABLE, reason,
                              "generation", gen_model=gen_model, gen_repairs=gen_repairs)
            return _rejected_response(job_id, FailureCode.GEN_UNRECOVERABLE, reason, "generation")

        vr = guard.validate_gen(gen, gen_model)
        if not vr:
            quarantine.record(job_id, req.text, vr.code, vr.reason, "generation",
                              gen_model=gen_model, gen_repairs=gen_repairs)
            return _rejected_response(job_id, vr.code, vr.reason, "generation")

        domain = resolve_domain(gen, req.domain)
        logger.info(f"Job {job_id}: gen OK → b_sync={gen.get('b_sync')} domain={domain}")

        # ── ВЕРИФИКАЦИЯ ─────────────────────────────────────────────
        ver_input = f"{VER_PROMPT}\n\nHypothesis:\n{json.dumps(gen, ensure_ascii=False)}"
        ver_raw, ver_model = client.verify(ver_input, context=req.text)

        vr = guard.validate_ver_raw(ver_raw, ver_model)
        if not vr:
            quarantine.record(job_id, req.text, vr.code, vr.reason, "verification",
                              gen_model=gen_model, ver_model=ver_model, gen_repairs=gen_repairs)
            return _rejected_response(job_id, vr.code, vr.reason, "verification")

        ver, ver_repairs, ver_ok = normalize_ver(ver_raw)
        if not ver_ok:
            reason = f"Verifier output unrecoverable: {'; '.join(ver_repairs[-3:])}"
            quarantine.record(job_id, req.text, FailureCode.VER_UNRECOVERABLE, reason,
                              "verification", gen_model=gen_model, ver_model=ver_model,
                              gen_repairs=gen_repairs, ver_repairs=ver_repairs)
            return _rejected_response(job_id, FailureCode.VER_UNRECOVERABLE, reason, "verification")

        vr = guard.validate_ver(ver, ver_model)
        if not vr:
            quarantine.record(job_id, req.text, vr.code, vr.reason, "verification",
                              gen_model=gen_model, ver_model=ver_model,
                              gen_repairs=gen_repairs, ver_repairs=ver_repairs)
            return _rejected_response(job_id, vr.code, vr.reason, "verification")

        verdict = ver.get("verdict", "WEAK")
        confidence = ver.get("confidence", 0.5)
        logger.info(f"Job {job_id}: ver OK → verdict={verdict} conf={confidence}")

        # ── РЕШЕНИЕ О СОХРАНЕНИИ ────────────────────────────────────
        save = False
        if verdict == "VALID" and confidence > 0.6:
            save = True
        elif verdict == "WEAK" and float(gen.get("b_sync", 0)) > 0.7:
            save = True

        # ── INVARIANT ENGINE ────────────────────────────────────────
        rollback.snapshot_space(semantic_space)
        rollback.register_graph_node(job_id)

        result = process_with_invariants(
            result={
                "job_id": job_id,
                "generation": gen,
                "verification": ver,
                "saved": save,
                "artifact": None,
                "domain": domain,
                "rag_context": rag_similar,
                "rag_dropped": len(rag_raw) - len(rag_similar),
                "gen_model": gen_model,
                "ver_model": ver_model,
                "repairs": repairs_summary(gen_repairs, ver_repairs),
            },
            job_id=job_id,
            space=semantic_space,
            graph=invariant_graph,
            detector=phase_detector,
        )
        structural = result.get("structural", {})
        phase_signal = structural.get("phase_signal", {})
        spec = structural.get("specificity", 1.0)
        logger.info(
            f"Job {job_id}: engine OK → type={structural.get('artifact_type')} "
            f"phase={phase_signal.get('signal')} spec={spec:.3f}"
        )

        # ── DOMAIN DIVERSITY GUARD ──────────────────────────────────
        # Если домен перенасыщен И гипотеза банальна — не сохранять,
        # но продолжать: граф уже обновлён, артефакт останется как weak_pattern.
        suppress, suppress_reason = _is_domain_overrepresented(domain, spec)
        if save and suppress:
            save = False
            result["save_skipped_reason"] = suppress_reason
            logger.info(f"Job {job_id}: save suppressed by domain diversity guard — {suppress_reason}")

        # Обновляем флаг сохранения в result
        result["saved"] = save

        # 4D и stress_test — доступны сразу из gen/ver, не требуют сохранённого файла
        four_d = gen.get("four_d_matrix")
        result["four_d_matrix"] = four_d
        result["stress_test"] = ver.get("stress_test")
        result["simulation"] = None
        result["resonance"] = None

        # ── СОХРАНЕНИЕ ──────────────────────────────────────────────
        # ВАЖНО: portal и artifact сохраняются ДО вызова MathCore,
        # чтобы _patch_artifact() внутри stress_test() нашёл файл.

        if structural.get("is_bridge"):
            portal_path = Path("artifacts") / f"{job_id}.hyx-portal.json"
            portal_data = {
                "id": job_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "type": "hyx-portal",
                "domain": domain,
                "hypothesis": gen.get("hypothesis", ""),
                "centrality": structural.get("centrality", 0),
                "similar_invariants": structural.get("similar_invariants", []),
                "phase_signal": phase_signal,
            }
            portal_path.write_text(json.dumps(portal_data, indent=2, ensure_ascii=False))
            rollback.register_file(portal_path)

        if save:
            artifact_file = save_artifact(
                job_id, {
                    "gen": result["generation"],
                    "ver": ver,
                    "domain": domain,
                    "structural": structural,
                    "normalization": repairs_summary(gen_repairs, ver_repairs),
                }
            )
            rollback.register_file(artifact_file)
            result["artifact"] = str(artifact_file)

            try:
                archivist_result = archivist.process(job_id)
                result["archivist"] = archivist_result
            except Exception as e:
                logger.warning(f"Job {job_id}: archivist failed - {e}")
                result["archivist"] = None

        # ── MATHCORE ────────────────────────────────────────────────
        # Вызывается ПОСЛЕ save_artifact, чтобы _patch_artifact нашёл файл.
        # Для несохранённых артефактов stress_test всё равно запускается
        # (результат в sim_results/), но патчить артефакт некуда — это ожидаемо.
        if four_d:
            try:
                sim_result = math_core.stress_test(job_id, four_d)
                result["simulation"] = sim_result

                math_core.index_artifact(
                    artifact_id=job_id,
                    four_d=four_d,
                    domain=domain,
                    stability_score=sim_result.get("stability_score", 0.5),
                )

                survival = ver.get("translation", {}).get("survival", "UNKNOWN")
                if isinstance(ver.get("translation"), dict):
                    survival = ver["translation"].get("survival", "UNKNOWN")

                resonance_result = math_core.find_resonance(
                    query_four_d=four_d,
                    query_domain=domain,
                    query_survival=survival,
                    top_k=3,
                    exclude_id=job_id,   # ← ДОБАВИТЬ ЭТУ СТРОКУ
                )
                result["resonance"] = resonance_result
                logger.info(
                    f"Job {job_id}: MathCore OK → stability={sim_result.get('stability_score')} "
                    f"top_resonance={resonance_result.get('top_resonance')}"
                )
            except Exception as e:
                logger.warning(f"Job {job_id}: MathCore failed — {e}")


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

        # ── ИСТОРИЯ ─────────────────────────────────────────────────
        log_history({
            "time": time.time(),
            "job_id": job_id,
            "query": req.text,
            "domain": domain,
            "gen": result["generation"],
            "ver": ver,
            "saved": save,
            "artifact_id": job_id if save else None,
            "structural": structural,
            "rag_context": rag_similar,
            "rag_dropped": len(rag_raw) - len(rag_similar),
            "repairs": repairs_summary(gen_repairs, ver_repairs),
            "save_skipped_reason": result.get("save_skipped_reason"),
        })

        # ── AUTO-UPDATE REF ──────────────────────────────────────────
        ref_id = extract_ref_id(req.text)
        if ref_id:
            logger.info(f"Job {job_id}: detected REF:{ref_id} — updating referenced artifact")
            update_referenced_artifact(ref_id, result, req.text)
            result["ref_updated"] = ref_id

        rollback.clear()
        return result

    except Exception as exc:
        logger.error(f"Job {job_id}: unexpected exception - {exc}", exc_info=True)
        actions = rollback.rollback(semantic_space, invariant_graph)
        quarantine.record(
            job_id, req.text, FailureCode.PIPELINE_EXCEPTION, str(exc), "unknown",
            gen_model=gen_model, ver_model=ver_model,
            rollback_actions=actions, gen_repairs=gen_repairs, ver_repairs=ver_repairs,
        )
        return _rejected_response(job_id, FailureCode.PIPELINE_EXCEPTION, str(exc), "unknown")


# ════════════════════════════════════════════════════════════════
# ENDPOINTS — CORE
# ════════════════════════════════════════════════════════════════

@app.post("/query")
def query(req: QueryRequest):
    return process_query(req)

@app.get("/quarantine")
def get_quarantine(limit: int = 20):
    return {"quarantine": quarantine.recent(limit)}

@app.delete("/quarantine/{job_id}")
def delete_quarantine_entry(job_id: str):
    path = quarantine.path
    if not path.exists():
        raise HTTPException(404, "Quarantine log not found")
    lines = path.read_text(encoding="utf-8").splitlines()
    new_lines, found = [], False
    for line in lines:
        try:
            if json.loads(line).get("job_id") == job_id:
                found = True
                continue
        except Exception:
            pass
        new_lines.append(line)
    if not found:
        raise HTTPException(404, f"Entry {job_id} not found in quarantine")
    path.write_text(("\n".join(new_lines) + "\n") if new_lines else "", encoding="utf-8")
    logger.info(f"Quarantine entry deleted: {job_id}")
    return {"ok": True, "deleted": job_id}

@app.get("/rag/context")
def rag_context(text: str, top_k: int = 3):
    similar = semantic_space.nearest(text, top_k=top_k, threshold=0.55)
    filtered = filter_rag_diversity(similar, max_per_domain=1, sim_cap=0.88)
    return {
        "similar": filtered,
        "similar_raw": similar,
        "count": len(filtered),
        "dropped": len(similar) - len(filtered),
    }

@app.get("/history")
def history():
    file = Path("chat_history/history.jsonl")
    if not file.exists():
        return {"history": []}
    lines = file.read_text(encoding="utf-8").splitlines()[-50:]
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except Exception:
            continue
    return {"history": result}

@app.get("/")
def ui():
    for name in ("index_v_4.html", "index.html"):
        html_path = Path(name)
        if html_path.exists():
            return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


# ════════════════════════════════════════════════════════════════
# ENDPOINTS — ARTIFACTS
# ════════════════════════════════════════════════════════════════

def _trashed_ids() -> set:
    trash_path = Path("trash")
    if not trash_path.exists():
        return set()
    ids = set()
    for f in trash_path.glob("*.json"):
        try:
            art = json.loads(f.read_text(encoding="utf-8"))
            ids.add(art.get("id", f.stem.split(".")[0]))
        except Exception:
            pass
    return ids


@app.get("/artifacts")
def artifacts():
    path = Path("artifacts")
    if not path.exists():
        return {"artifacts": []}
    trashed = _trashed_ids()
    files = sorted(
        [f for f in path.glob("*.json") if f.stem != "invariant_graph"],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    result = []
    for f in files:
        try:
            art = json.loads(f.read_text(encoding="utf-8"))
            art_id = art.get("id", f.stem.split(".")[0])
            if art_id in trashed:
                continue
            data = art.get("data", {})
            gen = data.get("gen", {})
            ver = data.get("ver", {})
            structural = data.get("structural", {})
            result.append({
                "file": f.name,
                "id": art_id,
                "domain": data.get("domain", "general"),
                "hypothesis_short": gen.get("hypothesis", "")[:80],
                "verdict": ver.get("verdict", ""),
                "confidence": ver.get("confidence", 0),
                "b_sync": gen.get("b_sync", 0),
                "artifact_type": structural.get("artifact_type", ""),
                "stability": structural.get("stability", ""),
                "novelty": art.get("archivist", {}).get("novelty", "") if art.get("archivist") else "",
                "created_at": art.get("created_at", ""),
                "history_count": len(art.get("history", [])),
                "is_portal": ".hyx-portal" in f.name,
            })
        except Exception:
            continue
    return {"artifacts": result[:50]}


@app.get("/artifacts/list")
def artifacts_list_all():
    path = Path("artifacts")
    if not path.exists():
        return {"artifacts": []}
    trashed = _trashed_ids()
    result = []
    for f in sorted(path.glob("*.json"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        if f.stem in ("invariant_graph",):
            continue
        if ".hyx-portal" in f.name:
            continue
        try:
            art = json.loads(f.read_text(encoding="utf-8"))
            art_id = art.get("id", f.stem)
            if art_id in trashed:
                continue
            data = art.get("data", {})
            gen = data.get("gen", {})
            ver = data.get("ver", {})
            structural = data.get("structural", {})
            result.append({
                "id": art_id,
                "domain": data.get("domain", "general"),
                "hypothesis_short": gen.get("hypothesis", "")[:80],
                "verdict": ver.get("verdict", ""),
                "confidence": ver.get("confidence", 0),
                "b_sync": gen.get("b_sync", 0),
                "issues_count": len(ver.get("issues", [])),
                "specificity": structural.get("specificity", 0.5),
                "artifact_type": structural.get("artifact_type", ""),
                "stability": structural.get("stability", ""),
                "novelty": art.get("archivist", {}).get("novelty", "") if art.get("archivist") else "",
                "history_count": len(art.get("history", [])),
            })
        except Exception:
            continue
    return {"artifacts": result}


@app.get("/artifact/{name}")
def artifact(name: str):
    file = Path("artifacts") / name
    if not file.exists():
        raise HTTPException(404)
    return json.loads(file.read_text(encoding="utf-8"))


@app.delete("/artifact/{artifact_id}")
def soft_delete_artifact(artifact_id: str):
    art_file = Path("artifacts") / f"{artifact_id}.json"
    portal_file = Path("artifacts") / f"{artifact_id}.hyx-portal.json"
    if not art_file.exists() and not portal_file.exists():
        raise HTTPException(404, f"Artifact {artifact_id} not found")
    trash_path = Path("trash")
    trash_path.mkdir(exist_ok=True)
    moved = []
    for src in [art_file, portal_file]:
        if src.exists():
            dst = trash_path / src.name
            shutil.move(str(src), str(dst))
            moved.append(src.name)
    if artifact_id in invariant_graph.G:
        invariant_graph.G.remove_node(artifact_id)
        invariant_graph._save()
    idx = semantic_space._id_to_idx.get(artifact_id)
    if idx is not None:
        semantic_space.meta.pop(idx)
        semantic_space.vectors.pop(idx)
        del semantic_space._id_to_idx[artifact_id]
        semantic_space._id_to_idx = {m["id"]: i for i, m in enumerate(semantic_space.meta)}
        _reload_semantic_index()
    logger.info(f"Artifact {artifact_id} moved to trash: {moved}")
    return {"ok": True, "moved": moved}


# ════════════════════════════════════════════════════════════════
# ENDPOINTS — TRASH
# ════════════════════════════════════════════════════════════════

@app.get("/trash")
def get_trash_list():
    trash_path = Path("trash")
    if not trash_path.exists():
        return {"trash": []}
    result = []
    for f in sorted(trash_path.glob("*.json"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            art = json.loads(f.read_text(encoding="utf-8"))
            data = art.get("data", {})
            gen = data.get("gen", {})
            ver = data.get("ver", {})
            result.append({
                "file": f.name,
                "id": art.get("id", f.stem),
                "domain": data.get("domain", "general"),
                "hypothesis_short": gen.get("hypothesis", "")[:80],
                "verdict": ver.get("verdict", ""),
                "created_at": art.get("created_at", ""),
                "deleted_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
        except Exception:
            continue
    return {"trash": result}

@app.delete("/trash/clear")
def clear_trash_all():
    trash_path = Path("trash")
    if not trash_path.exists():
        return {"ok": True, "deleted_count": 0}
    files = list(trash_path.glob("*.json"))
    for f in files:
        f.unlink()
    logger.info(f"Trash cleared: {len(files)} files")
    return {"ok": True, "deleted_count": len(files)}

@app.post("/trash/{artifact_id}/restore")
def restore_from_trash(artifact_id: str):
    trash_path = Path("trash")
    art_file = trash_path / f"{artifact_id}.json"
    portal_file = trash_path / f"{artifact_id}.hyx-portal.json"
    if not art_file.exists():
        raise HTTPException(404, f"Artifact {artifact_id} not found in trash")
    arts_path = Path("artifacts")
    restored = []
    for src in [art_file, portal_file]:
        if src.exists():
            dst = arts_path / src.name
            shutil.move(str(src), str(dst))
            restored.append(src.name)
    art = json.loads((arts_path / f"{artifact_id}.json").read_text(encoding="utf-8"))
    data = art.get("data", {})
    gen = data.get("gen", {})
    hypothesis = gen.get("hypothesis", "")
    domain = data.get("domain", "general")
    b_sync = float(gen.get("b_sync", 0))
    if hypothesis and artifact_id not in semantic_space._id_to_idx:
        semantic_space.add(artifact_id, hypothesis, domain, b_sync)
    if artifact_id not in invariant_graph.G:
        structural = data.get("structural", {})
        translation = data.get("ver", {}).get("translation", {})
        survival = (translation.get("survival", "UNKNOWN")
                    if isinstance(translation, dict) else "UNKNOWN")
        invariant_graph.add_node(
            artifact_id,
            domain=domain,
            b_sync=b_sync,
            stability=structural.get("stability", "unknown"),
            specificity=structural.get("specificity", 0.5),
            survival=survival,
        )
        invariant_graph._save()
    logger.info(f"Artifact {artifact_id} restored from trash")
    return {"ok": True, "restored": restored}


@app.delete("/trash/{artifact_id}/permanent")
def permanent_delete(artifact_id: str):
    trash_path = Path("trash")
    deleted = []
    for f in [trash_path / f"{artifact_id}.json",
              trash_path / f"{artifact_id}.hyx-portal.json"]:
        if f.exists():
            f.unlink()
            deleted.append(f.name)
    if not deleted:
        raise HTTPException(404, f"Artifact {artifact_id} not found in trash")
    logger.info(f"Artifact {artifact_id} permanently deleted")
    return {"ok": True, "deleted": deleted}


# ════════════════════════════════════════════════════════════════
# ENDPOINTS — GRAPH & PHASE
# ════════════════════════════════════════════════════════════════

@app.get("/graph/data")
def graph_data():
    G = invariant_graph.G
    trashed = _trashed_ids()
    nodes = []
    for node_id, attrs in G.nodes(data=True):
        if node_id in trashed:
            continue
        domain = attrs.get("domain", "")
        if not domain:
            continue
        nodes.append({
            "id": node_id,
            "domain": domain,
            "b_sync": attrs.get("b_sync", 0.0),
            "stability": attrs.get("stability", "unknown"),
            "specificity": attrs.get("specificity", 0.5),
            "survival": attrs.get("survival", "UNKNOWN"),
        })
    valid_ids = {n["id"] for n in nodes}
    links = []
    for u, v, attrs in G.edges(data=True):
        if u not in valid_ids or v not in valid_ids:
            continue
        links.append({
            "source": u, "target": v,
            "weight": attrs.get("weight", 0.0),
            "similarity": attrs.get("similarity", 0.0),
            "domain_distance": attrs.get("domain_distance", 0.0),
            "four_d_resonance": attrs.get("four_d_resonance", 0.0),
            "specificity": attrs.get("specificity", 0.0),
        })
    clusters = [list(c) for c in invariant_graph.get_invariant_clusters()]
    bridge_nodes = {n for e in invariant_graph.get_bridges() for n in e}
    cluster_map = {nid: i for i, cluster in enumerate(clusters) for nid in cluster}
    for node in nodes:
        node["cluster"] = cluster_map.get(node["id"], -1)
        node["is_bridge"] = node["id"] in bridge_nodes
    return {
        "nodes": nodes,
        "links": links,
        "meta": {
            "total_nodes": len(nodes),
            "total_edges": len(links),
            "cluster_count": len(clusters),
            "bridge_count": len(bridge_nodes),
        },
    }


@app.get("/graph")
def graph():
    clusters = [list(c) for c in invariant_graph.get_invariant_clusters()]
    bridges = invariant_graph.get_bridges()
    return {
        "nodes": len(invariant_graph.G.nodes),
        "edges": len(invariant_graph.G.edges),
        "clusters": clusters,
        "bridges": bridges,
        "cluster_count": len(clusters),
    }

@app.get("/phase")
def phase():
    return phase_detector.detect_phase_transition(semantic_space)


# ════════════════════════════════════════════════════════════════
# ENDPOINTS — QUESTION GENERATOR
# ════════════════════════════════════════════════════════════════

@app.get("/question/suggest")
def suggest_question():
    try:
        return question_gen.suggest_novel()
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/question/clarify/{artifact_id}")
def clarify_artifact(artifact_id: str):
    try:
        return question_gen.suggest_clarification(artifact_id)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/question/candidates")
def clarification_candidates():
    return {"candidates": question_gen.list_clarification_candidates()}


# ════════════════════════════════════════════════════════════════
# ENDPOINTS — TRACKER
# ════════════════════════════════════════════════════════════════

@app.get("/tracker/stats")
def tracker_stats():
    return tracker.get_stats()

@app.get("/tracker/providers")
def tracker_providers_get():
    return {
        "providers": tracker.get_providers(),
        "known_models": tracker.get_known_models(),
    }

@app.post("/tracker/providers")
def tracker_providers_update(req: ProvidersUpdateRequest):
    ok = tracker.update_providers(req.providers)
    if not ok:
        raise HTTPException(400, "Ошибка сохранения конфига")
    return {"ok": True, "count": len(req.providers)}

@app.post("/tracker/providers/add")
def tracker_provider_add(req: ProviderAddRequest):
    ok = tracker.add_provider(req.dict())
    if not ok:
        raise HTTPException(400, "Ошибка добавления провайдера")
    return {"ok": True}

@app.post("/tracker/save-to-env")
def tracker_save_to_env(req: ProvidersUpdateRequest):
    try:
        ok = tracker.update_providers(req.providers)
        if not ok:
            raise HTTPException(400, "Ошибка сохранения провайдеров")

        env_lines = ["# === HX-AM Proxy v4.2 Environment (auto-generated) ==="]
        for p in tracker._providers:
            prefix = p.provider.upper()
            env_lines.append(f"{prefix}_API_KEY={p.api_key}")
            env_lines.append(f"{prefix}_API_BASE={p.api_base}")
            env_lines.append(f"{prefix}_MODEL={p.model}")

        with open(".env", "w", encoding="utf-8") as f:
            f.write("\n".join(env_lines) + "\n")

        return {"ok": True, "message": "Saved to .env"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.delete("/tracker/providers/{provider_id}")
def tracker_provider_delete(provider_id: str):
    ok = tracker.delete_provider(provider_id)
    if not ok:
        raise HTTPException(404, f"Провайдер {provider_id} не найден")
    return {"ok": True}

@app.post("/tracker/reset")
def tracker_reset_stats(req: ResetRequest):
    if req.scope == "all":
        tracker.reset_all()
    else:
        tracker.reset_today()
    return {"ok": True, "scope": req.scope}


# ════════════════════════════════════════════════════════════════
# v4.2 ENDPOINTS — MathCore, Insights
# ════════════════════════════════════════════════════════════════

@app.post("/math/stress/{artifact_id}")
def math_stress_test(artifact_id: str):
    # Убираем .json если UI передал имя файла вместо ID
    artifact_id = artifact_id.replace(".json", "").replace(".hyx-portal", "")
    """Запускает MathCore стресс-тест для артефакта."""
    try:
        result = math_core.stress_test(artifact_id)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/insights/feed")
def insights_feed(status: str = "all", limit: int = 20):
    """Список вероятностных инсайтов."""
    insights_dir = Path("insights")
    if not insights_dir.exists():
        return {"insights": [], "total": 0}
    files = sorted(insights_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    result = []
    for f in files[:limit]:
        try:
            insight = json.loads(f.read_text(encoding="utf-8"))
            if status != "all" and insight.get("status") != status:
                continue
            result.append(insight)
        except Exception:
            continue
    return {"insights": result, "total": len(result)}


@app.get("/math/stats")
def math_stats():
    """Статистика MathCore симуляций."""
    sim_dir = Path("sim_results")
    insights_dir = Path("insights")
    four_d_idx = Path("artifacts/four_d_index.jsonl")
    stats = {
        "sim_results": len(list(sim_dir.glob("*.json"))) if sim_dir.exists() else 0,
        "insights": len(list(insights_dir.glob("*.json"))) if insights_dir.exists() else 0,
        "indexed_4d": 0,
    }
    if four_d_idx.exists():
        try:
            stats["indexed_4d"] = sum(1 for line in four_d_idx.read_text().splitlines() if line.strip())
        except Exception:
            pass
    return stats


@app.get("/mgap/match/{artifact_id}")
def mgap_match(
    artifact_id: str,
    top_k: int = 3,
    all_types: bool = False,
    model_id: str = "",
):
    """
    Поиск топ-K отраслевых моделей, резонирующих с артефактом.

    Fallback: если math_type_only=True даёт пустой результат (нет моделей данного
    math_type в реестре) — автоматически повторяет поиск по всем math_type.
    Результат fallback-поиска помечается полем _fallback=True.

    Сохранение: один файл mgap_results/{artifact_id}.json — перезапись при каждом вызове.
    """
    try:
        math_type_only = not all_types
        results = mgap_matcher.match_artifact(
            artifact_id=artifact_id,
            top_k=top_k,
            math_type_only=math_type_only,
            model_id=model_id or None,
        )

        # ── Fallback: нет моделей с точным math_type → ищем по всем ─────────
        first_is_error = bool(results and results[0].get("error"))
        if first_is_error and math_type_only:
            original_error = results[0].get("error", "no exact math_type match")
            logger.info(f"MGAP fallback triggered for {artifact_id}: {original_error}")
            results_fb = mgap_matcher.match_artifact(
                artifact_id=artifact_id,
                top_k=top_k,
                math_type_only=False,
                model_id=model_id or None,
            )
            if results_fb and not results_fb[0].get("error"):
                for r in results_fb:
                    r["_fallback"] = True
                    r["_fallback_reason"] = original_error
                results = results_fb
                logger.info(
                    f"MGAP fallback success: {artifact_id} → "
                    f"{[r.get('model_id') for r in results[:2]]}"
                )

        response = {"artifact_id": artifact_id, "matches": results, "total": len(results)}

        # ── Сохранение: без timestamp → один файл на артефакт ────────────────
        try:
            os.makedirs("mgap_results", exist_ok=True)
            filename = f"mgap_results/{artifact_id}.json"   # ← перезапись
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            logger.info(f"MGAP result saved: {filename}")
        except Exception as save_err:
            logger.error(f"Failed to save MGAP result: {save_err}")

        return response
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/mgap/registry")
def mgap_registry():
    """Сводка моделей реестра MGAP."""
    return {
        "models": mgap_matcher.get_registry_summary(),
        "total": len(mgap_matcher.registry),
        "version": "v4.5",
        "math_types": list({m.get("math_type") for m in mgap_matcher.registry}),
    }


@app.post("/mgap/batch")
def mgap_batch_endpoint(
    top_k: int = 2,
    all_types: bool = False,
    min_resonance: float = 0.3,
):
    """
    Batch: прогнать все артефакты через MGAPMatcher.
    Fallback: артефакты без exact math_type матча автоматически получают
    cross-type поиск, их результаты помечаются _fallback=True.
    """
    try:
        results = mgap_matcher.match_batch(
            top_k=top_k,
            math_type_only=not all_types,
            min_resonance=min_resonance,
        )

        # ── Fallback для артефактов без exact матча ───────────────────────────
        if not all_types:
            artifacts_dir = Path("artifacts")
            for f in sorted(artifacts_dir.glob("*.json")):
                if f.stem == "invariant_graph" or ".hyx-portal" in f.name:
                    continue
                art_id = f.stem
                if art_id not in results:
                    try:
                        fb = mgap_matcher.match_artifact(
                            art_id, top_k=top_k, math_type_only=False
                        )
                        ok = [
                            m for m in fb
                            if not m.get("error") and m.get("resonance", 0) >= min_resonance
                        ]
                        if ok:
                            for m in ok:
                                m["_fallback"] = True
                            results[art_id] = ok
                    except Exception:
                        pass

        return {
            "results": results,
            "artifacts_count": len(results),
            "models_count": len(mgap_matcher.registry),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

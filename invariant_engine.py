# invariant_engine.py — HX-AM v4.2 (replaces v4.0 version)
# v4.2 changes:
#   - FourDSemanticSpace: stores 4D vectors alongside text embeddings
#   - process_with_invariants(): computes 4d_resonance if four_d_matrix present
#   - compute_4d_resonance injected into structural result
#   - All v4.0 fixes retained (domain normalisation, batch encode, etc.)

from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import logging
import networkx as nx
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import fclusterdata

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_logger = logging.getLogger("HXAM.engine")


# ── Domain normalisation ──────────────────────────────────────────────────────
_DOMAIN_MAP: dict[str, str] = {
    "социология": "sociology", "психология": "psychology",
    "физика": "physics", "биология": "biology",
    "математика": "mathematics", "химия": "chemistry",
    "лингвистика": "linguistics", "экономика": "economics",
    "экология": "ecology", "нейронаука": "neuroscience",
    "геология": "geology", "медицина": "medicine",
    "астрономия": "astronomy",
    "social": "sociology", "psych": "psychology",
    "neuro": "neuroscience", "bio": "biology",
    "chem": "chemistry", "math": "mathematics",
    "econ": "economics", "sociolinguistics": "linguistics",
}


def _normalize_domain(domain: str) -> str:
    d = domain.strip().lower()
    return _DOMAIN_MAP.get(d, d)


# ══════════════════════════════════════════════
# СЛОЙ 1 — Семантическое пространство (v4.2)
# ══════════════════════════════════════════════

class SemanticSpace:
    """
    v4.2: добавлена поддержка 4D-векторов.
    Хранит параллельно: text-embeddings (512-dim) + 4D-vectors (13-dim).
    """

    def __init__(self, index_path: str = "artifacts/semantic_index.jsonl"):
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(exist_ok=True)
        self.vectors: list = []        # text embeddings
        self.meta: list = []
        self._id_to_idx: dict[str, int] = {}

        # v4.2: 4D-векторы
        self._four_d_vecs: dict[str, np.ndarray] = {}  # artifact_id → 13-dim vec

        self._load()

    def encode(self, text: str) -> np.ndarray:
        return _embedder.encode(text)

    def add(self, artifact_id: str, invariant: str, domain: str, b_sync: float,
            four_d_vec: np.ndarray | None = None):
        domain = _normalize_domain(domain)
        vec = _embedder.encode(invariant)
        idx = len(self.vectors)
        self.vectors.append(vec)
        self._id_to_idx[artifact_id] = idx
        entry = {"id": artifact_id, "invariant": invariant, "domain": domain, "b_sync": b_sync}
        self.meta.append(entry)
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # v4.2: сохранить 4D если передан
        if four_d_vec is not None:
            self._four_d_vecs[artifact_id] = four_d_vec

    def vector_by_id(self, artifact_id: str) -> np.ndarray | None:
        idx = self._id_to_idx.get(artifact_id)
        if idx is not None and idx < len(self.vectors):
            return self.vectors[idx]
        return None

    def four_d_vec_by_id(self, artifact_id: str) -> np.ndarray | None:
        """v4.2: возвращает 4D-вектор по ID."""
        return self._four_d_vecs.get(artifact_id)

    def store_four_d_vec(self, artifact_id: str, vec: np.ndarray):
        """v4.2: добавляет 4D-вектор для уже существующего узла."""
        self._four_d_vecs[artifact_id] = vec

    def _load(self):
        if not self.index_path.exists():
            return
        entries = []
        with open(self.index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    e["domain"] = _normalize_domain(e.get("domain", "general"))
                    entries.append(e)
                except Exception:
                    continue
        if not entries:
            return
        texts = [e["invariant"] for e in entries]
        _logger.info(f"SemanticSpace: batch-encoding {len(texts)} invariants...")
        vecs = _embedder.encode(texts, show_progress_bar=False)
        for i, (e, vec) in enumerate(zip(entries, vecs)):
            self.meta.append(e)
            self.vectors.append(vec)
            self._id_to_idx[e["id"]] = i
        _logger.info(f"SemanticSpace: loaded {len(self.vectors)} vectors")
        self._load_four_d_index()

    def _load_four_d_index(self):
        """Загружает 4D-векторы из four_d_index.jsonl в память при старте."""
        idx_path = Path("artifacts/four_d_index.jsonl")
        if not idx_path.exists():
            return
        loaded = 0
        with open(idx_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    art_id = e.get("id", "")
                    vec_list = e.get("vector")
                    if art_id and vec_list:
                        self._four_d_vecs[art_id] = np.array(vec_list, dtype=np.float64)
                        loaded += 1
                except Exception:
                    continue
        _logger.info(f"SemanticSpace: loaded {loaded} 4D vectors from four_d_index")

    def nearest(self, invariant: str, top_k: int = 5, threshold: float = 0.65) -> list:
        if not self.vectors:
            return []
        query_vec = _embedder.encode(invariant)
        return self._nearest_by_vec(query_vec, top_k=top_k, threshold=threshold)

    def nearest_by_vec(self, vec: np.ndarray, top_k: int = 8, threshold: float = 0.0) -> list:
        return self._nearest_by_vec(vec, top_k=top_k, threshold=threshold)

    def _nearest_by_vec(self, query_vec: np.ndarray, top_k: int = 5, threshold: float = 0.65) -> list:
        if not self.vectors:
            return []
        matrix = np.array(self.vectors)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
        norms = np.where(norms == 0, 1e-9, norms)
        similarities = matrix @ query_vec / norms
        results = []
        for i, sim in enumerate(similarities):
            if sim >= threshold:
                results.append({**self.meta[i], "similarity": round(float(sim), 3)})
        return sorted(results, key=lambda x: -x["similarity"])[:top_k]

    def domain_centroid(self, domain: str):
        domain = _normalize_domain(domain)
        domain_vecs = [
            self.vectors[i]
            for i, m in enumerate(self.meta)
            if m.get("domain") == domain
        ]
        if len(domain_vecs) < 2:
            return None
        centroid = np.mean(domain_vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid

    def specificity(self, vec: np.ndarray, domain: str) -> float:
        centroid = self.domain_centroid(domain)
        if centroid is None:
            return 0.5
        try:
            vec_norm = vec / np.linalg.norm(vec)
            dist = float(cosine(vec_norm, centroid))
            return round(min(dist, 1.0), 3)
        except Exception:
            return 0.5


# ══════════════════════════════════════════════
# СЛОЙ 2 — Структурный граф (v4.2)
# ══════════════════════════════════════════════

class InvariantGraph:
    """
    v4.2: узлы дополнительно хранят four_d_resonance.
    """

    def __init__(self, graph_path: str = "artifacts/invariant_graph.json"):
        self.path = Path(graph_path)
        self.path.parent.mkdir(exist_ok=True)
        self.G = self._load()

    def add_node(self, artifact_id: str, **attrs):
        self.G.add_node(artifact_id, **attrs)

    def add_edge(self, id1: str, id2: str, similarity: float,
                 domain_distance: float, specificity: float = 0.5,
                 four_d_resonance: float = 0.0):
        # v4.2: four_d_resonance усиливает вес если > 0
        base_weight = round(similarity * (1 + domain_distance) * specificity, 3)
        if four_d_resonance > 0:
            # Буст: кросс-доменный 4D-изоморфизм увеличивает вес до 20%
            boost = 1.0 + four_d_resonance * 0.2
            weight = round(base_weight * boost, 3)
        else:
            weight = base_weight
        self.G.add_edge(id1, id2,
                        similarity=similarity,
                        domain_distance=domain_distance,
                        specificity=specificity,
                        four_d_resonance=four_d_resonance,
                        weight=weight)
        self._save()

    def get_invariant_clusters(self) -> list:
        components = list(nx.connected_components(self.G))
        return [c for c in components if len(c) >= 2]

    def get_bridges(self) -> list:
        try:
            return list(nx.bridges(self.G))
        except Exception:
            return []

    def node_centrality(self, artifact_id: str) -> float:
        centrality = nx.betweenness_centrality(self.G)
        return round(centrality.get(artifact_id, 0.0), 3)

    def get_similar_nodes(self, embedding: np.ndarray, space: "SemanticSpace",
                          top_k: int = 8, exclude_id: str = None) -> list:
        candidates = space.nearest_by_vec(embedding, top_k=top_k * 2, threshold=0.0)
        results = []
        for c in candidates:
            if exclude_id and c["id"] == exclude_id:
                continue
            node_id = c["id"]
            node_attrs = self.G.nodes.get(node_id, {})
            results.append({
                "id": node_id,
                "domain": c.get("domain", node_attrs.get("domain", "general")),
                "similarity": c["similarity"],
                "domain_distance": 0.0,
                "weight": 0.0,
                "b_sync": c.get("b_sync", node_attrs.get("b_sync", 0.0)),
                "stability": node_attrs.get("stability", "unknown"),
                "survival": node_attrs.get("survival", "UNKNOWN"),
                "invariant": c.get("invariant", ""),
                # v4.2
                "four_d_resonance": node_attrs.get("four_d_resonance", 0.0),
                "stability_score": node_attrs.get("stability_score", 0.0),
            })
            if len(results) >= top_k:
                break
        return results

    def get_subgraph(self, node_id: str, depth: int = 2) -> dict:
        if node_id not in self.G:
            return {"nodes": [], "edges": [], "clusters": [], "bridges": []}
        ego = nx.ego_graph(self.G, node_id, radius=depth)
        clusters = [list(c) for c in nx.connected_components(ego) if len(c) >= 2]
        try:
            bridges = list(nx.bridges(ego))
        except Exception:
            bridges = []
        nodes = [
            {"id": n, **{k: v for k, v in self.G.nodes[n].items()
                         if k in ("domain", "b_sync", "stability", "survival",
                                  "specificity", "four_d_resonance", "stability_score")}}
            for n in ego.nodes
        ]
        edges = [
            {"source": u, "target": v,
             "weight": d.get("weight", 0),
             "similarity": d.get("similarity", 0),
             "four_d_resonance": d.get("four_d_resonance", 0.0)}
            for u, v, d in ego.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges, "clusters": clusters,
                "bridges": [list(b) for b in bridges]}

    def update_with_archivist(self, node_id: str, archivist_result: dict):
        if node_id not in self.G:
            return
        novelty_score = archivist_result.get("novelty_score", 0.6)
        novelty = archivist_result.get("novelty", "KNOWN")
        math_ver = archivist_result.get("mathematical_verification", "TERMINOLOGICAL")
        self.G.nodes[node_id]["novelty"] = novelty
        self.G.nodes[node_id]["novelty_score"] = novelty_score
        self.G.nodes[node_id]["math_verification"] = math_ver
        tags = archivist_result.get("suggested_tags") or []
        self.G.nodes[node_id]["suggested_tags"] = [
            "hyx-portal" if t == "hyx_portal" else t for t in tags
        ]
        linked = [l for l in (archivist_result.get("linked_to") or []) if l != node_id]
        self.G.nodes[node_id]["linked_to"] = linked
        multipliers = {"PHENOMENAL": 0.9, "NOVEL": 0.6, "KNOWN": 0.3}
        mul = next((v for k, v in multipliers.items() if novelty.startswith(k)), 0.1)
        for u, v, edge_data in list(self.G.edges(node_id, data=True)):
            edge_data["novelty_weight"] = round(edge_data.get("weight", 0.5) * mul, 3)
        self._save()
        _logger.info(f"Archivist updated node {node_id}: novelty={novelty} score={novelty_score}")

    def save_graph(self):
        self._save()

    def _save(self):
        data = nx.node_link_data(self.G)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _load(self) -> nx.Graph:
        if not self.path.exists():
            return nx.Graph()
        try:
            data = json.loads(self.path.read_text())
            has_links = "links" in data
            has_edges = "edges" in data
            if has_edges and not has_links:
                data["links"] = data["edges"]
            elif has_links and not has_edges:
                data["edges"] = data["links"]
            G = nx.node_link_graph(data)
            if G.is_multigraph():
                _logger.warning("InvariantGraph: converting MultiGraph → Graph on load")
                G = nx.Graph(G)
            _logger.info(
                f"InvariantGraph: loaded {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges"
            )
            return G
        except Exception as e:
            _logger.error(f"InvariantGraph._load failed: {e} — starting fresh")
            return nx.Graph()


# ══════════════════════════════════════════════
# СЛОЙ 3 — Детектор фазовых переходов (v4.2)
# ══════════════════════════════════════════════

class PhaseDetector:

    def is_stable_invariant(
        self,
        invariant: str,
        similar_artifacts: list,
        graph: InvariantGraph,
        specificity: float = 0.5,
        similar_vecs: list | None = None,
    ) -> tuple:
        if len(similar_artifacts) == 0:
            return False, "isolated"
        if len(similar_artifacts) == 1:
            sim = similar_artifacts[0]["similarity"]
            return sim > 0.80, "weak_pattern"

        similar_specs = [a.get("specificity", 0.5) for a in similar_artifacts]
        avg_specificity = (specificity + sum(similar_specs)) / (len(similar_artifacts) + 1)
        if avg_specificity < 0.3:
            return False, "low_specificity_cluster"

        if similar_vecs and len(similar_vecs) == len(similar_artifacts):
            vectors = np.array(similar_vecs)
        else:
            vectors = np.array([_embedder.encode(a["invariant"]) for a in similar_artifacts])

        try:
            labels = fclusterdata(vectors, t=0.35, criterion="distance", metric="cosine")
            unique_clusters = len(set(labels))
        except Exception:
            return False, "mixed_patterns"

        if unique_clusters == 1:
            return True, "stable_cluster"
        return False, "mixed_patterns"

    def detect_phase_transition(self, space: SemanticSpace, window: int = 10) -> dict:
        if len(space.vectors) < window:
            return {
                "transition": False, "density": 0.0, "window": window,
                "signal": "noise", "unique_domains": 0, "domain_entropy": 0.0,
            }
        recent = np.array(space.vectors[-window:])
        norms = np.linalg.norm(recent, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normalized = recent / norms
        sim_matrix = normalized @ normalized.T
        np.fill_diagonal(sim_matrix, 0)
        density = sim_matrix.sum() / (window * (window - 1))
        recent_meta = space.meta[-window:]
        domains = [_normalize_domain(m.get("domain", "general")) for m in recent_meta]
        unique_domains = len(set(domains))
        domain_entropy = round(unique_domains / window, 3)
        if density > 0.60:
            if unique_domains >= 3:
                signal = "sigma_primitive_candidate"
                transition = True
            else:
                signal = "template_loop"
                transition = False
                _logger.warning(
                    f"PhaseDetector: template_loop — density={density:.3f} "
                    f"unique_domains={unique_domains} (domains: {set(domains)})"
                )
        else:
            signal = "noise"
            transition = False
        return {
            "transition": transition,
            "density": round(float(density), 3),
            "window": window,
            "signal": signal,
            "unique_domains": unique_domains,
            "domain_entropy": domain_entropy,
        }

    def log_phenomenal(self, node_id: str, reason: str):
        _logger.info(f"PHENOMENAL link detected: node={node_id} reason={reason}")


# ══════════════════════════════════════════════
# DOMAIN VECTOR CACHE
# ══════════════════════════════════════════════
_domain_vec_cache: dict[str, np.ndarray] = {}


def _get_domain_vec(domain: str) -> np.ndarray:
    domain = _normalize_domain(domain)
    if domain not in _domain_vec_cache:
        _domain_vec_cache[domain] = _embedder.encode(domain)
    return _domain_vec_cache[domain]


# ══════════════════════════════════════════════
# ОРКЕСТРАТОР (v4.2)
# ══════════════════════════════════════════════

def process_with_invariants(
    result: dict,
    job_id: str,
    space: SemanticSpace,
    graph: InvariantGraph,
    detector: PhaseDetector,
) -> dict:
    gen = result.get("generation", {})
    ver = result.get("verification", {})

    invariant = gen.get("hypothesis")
    if not invariant:
        result["structural"] = {"error": "no hypothesis in generation"}
        return result

    b_sync = float(gen.get("b_sync", 0.0))
    domain = _normalize_domain(result.get("domain", "general"))

    current_vec = _embedder.encode(invariant)
    spec = space.specificity(current_vec, domain)

    translation = ver.get("translation", {})
    survival = (translation.get("survival", "UNKNOWN")
                if isinstance(translation, dict) else "UNKNOWN")
    if survival == "UNKNOWN":
        _logger.warning(f"Job {job_id}: verifier did not return translation — Step 0 skipped")

    # v4.2: извлекаем 4D-вектор если есть
    four_d_matrix = gen.get("four_d_matrix")
    four_d_vec: np.ndarray | None = None
    if four_d_matrix:
        try:
            from schemas.four_d_matrix import FourDMatrix
            matrix_obj = FourDMatrix.from_raw(four_d_matrix)
            if matrix_obj is not None:
                four_d_vec = matrix_obj.to_vector()
        except Exception as e:
            _logger.warning(f"Job {job_id}: 4D vector extraction failed — {e}")

    # v4.2: стресс-тест из верификатора (дополнительный сигнал устойчивости)
    stress_test = ver.get("stress_test") or {}
    stress_dynamics_stable = stress_test.get("stress_dynamics_stable", None)

    similar = space.nearest(invariant, threshold=0.65)
    similar_vecs = [space.vector_by_id(s["id"]) for s in similar]

    is_stable, stability_type = detector.is_stable_invariant(
        invariant, similar, graph,
        specificity=spec,
        similar_vecs=[v for v in similar_vecs if v is not None],
    )

    if survival == "TERMINOLOGICAL":
        is_stable = False
        stability_type = "terminological"

    # v4.2: если верификатор явно сказал нестабильно — понижаем
    if stress_dynamics_stable is False and stability_type == "stable_cluster":
        stability_type = "weak_pattern"
        is_stable = False
        _logger.info(f"Job {job_id}: stability downgraded — stress_test says unstable")

    space.add(job_id, invariant, domain, b_sync, four_d_vec=four_d_vec)
    graph.add_node(
        job_id,
        domain=domain,
        b_sync=b_sync,
        stability=stability_type,
        specificity=spec,
        survival=survival,
        translation=(translation.get("translated_mechanism", "")
                     if isinstance(translation, dict) else ""),
        # v4.2 extra attrs
        has_four_d=four_d_vec is not None,
        stress_stable=stress_dynamics_stable,
    )

    domain_vec = _get_domain_vec(domain)

    for s in similar:
        try:
            neighbor_domain = _normalize_domain(s.get("domain", "general"))
            neighbor_domain_vec = _get_domain_vec(neighbor_domain)
            dist = round(float(cosine(domain_vec, neighbor_domain_vec)), 3)
        except Exception:
            dist = 0.0
        neighbor_spec = graph.G.nodes.get(s["id"], {}).get("specificity", 0.5)
        edge_spec = round((spec + neighbor_spec) / 2, 3)

        # v4.2: вычислить 4D-резонанс для этого ребра
        edge_4d_resonance = 0.0
        if four_d_vec is not None:
            neighbor_4d_vec = space.four_d_vec_by_id(s["id"])
            if neighbor_4d_vec is not None:
                try:
                    from schemas.four_d_matrix import compute_4d_resonance
                    edge_4d_resonance = compute_4d_resonance(four_d_vec, neighbor_4d_vec)
                except Exception:
                    pass

        graph.add_edge(job_id, s["id"],
                       similarity=s["similarity"],
                       domain_distance=dist,
                       specificity=edge_spec,
                       four_d_resonance=edge_4d_resonance)

    phase = detector.detect_phase_transition(space)
    bridge_nodes = {n for e in graph.get_bridges() for n in e}
    centrality = graph.node_centrality(job_id)

    if phase["signal"] == "sigma_primitive_candidate":
        artifact_type = "sigma_primitive_candidate"
    elif job_id in bridge_nodes:
        artifact_type = "hyx-portal"
    elif stability_type == "stable_cluster":
        artifact_type = "hyx-artifact"
    elif stability_type == "weak_pattern":
        artifact_type = "weak_pattern"
    else:
        artifact_type = "noise"

    if (not is_stable or survival == "TERMINOLOGICAL") and b_sync > 0.50:
        result["generation"] = {**gen, "b_sync": round(b_sync * 0.75, 2)}

    # v4.2: вычислить top 4D-резонанс среди соседей
    top_4d_resonance = 0.0
    if four_d_vec is not None:
        for s in similar[:5]:
            n_vec = space.four_d_vec_by_id(s["id"])
            if n_vec is not None:
                try:
                    from schemas.four_d_matrix import compute_4d_resonance
                    r = compute_4d_resonance(four_d_vec, n_vec)
                    top_4d_resonance = max(top_4d_resonance, r)
                except Exception:
                    pass

    result["structural"] = {
        "similar_invariants": similar,
        "stability": stability_type,
        "is_stable": is_stable,
        "specificity": spec,
        "centrality": centrality,
        "is_bridge": job_id in bridge_nodes,
        "artifact_type": artifact_type,
        "phase_signal": phase,
        "translation": translation if isinstance(translation, dict) else {},
        # v4.2 additions
        "has_four_d": four_d_vec is not None,
        "top_4d_resonance": round(top_4d_resonance, 3),
        "stress_stable": stress_dynamics_stable,
    }

    return result

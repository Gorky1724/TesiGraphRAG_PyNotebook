"""
Orchestratore del benchmark RAGAS (cap. sperimentazione).

Due fasi separate, entrambe riprendibili (resume) perche' LLM locale e giudice API sono lenti/limitati:
  1) generate(run_id)        -> usa PipelineHandler (retrieval, reranking, risposta LLM) per ogni query x configurazione
                                e salva contesti, risposta, reference in  <out_dir>/<run_id>_generation.jsonl
  2) evaluate_ragas(run_id)  -> calcola le metriche RAGAS con un giudice LLM esterno, una configurazione alla volta
  3) summary / paired_vs_baseline -> tabelle per la tesi (medie per categoria, differenze con IC bootstrap, win/tie/loss)

run_id serve a distinguere i passaggi: es. "pre_edit" (sidecar senza ritocchi) e "post_edit" (dopo le modifiche manuali).
A parita' di query le configurazioni usano lo STESSO numero di chunk nel contesto (top_j).
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ablation: C0 baseline vettoriale, C1 solo tag, C2 solo distanze, C3 distanze + tag
DEFAULT_CONFIGS: List[Dict[str, Any]] = [
    {"name": "C0_baseline", "mode": "vector", "auto_tags": False},
    {"name": "C1_tags", "mode": "rerank", "order_by": ["TAGS"], "auto_tags": False, "manual_tags": True, "conv_tags": True},
    {"name": "C2_dist", "mode": "rerank", "order_by": ["DIST"], "auto_tags": False, "manual_tags": True, "conv_tags": True},
    {"name": "C3_dist_tags", "mode": "rerank", "order_by": ["DIST", "TAGS"], "auto_tags": False, "manual_tags": True, "conv_tags": True},
    {"name": "C4_dist_tags_joinconv", "mode": "rerank", "order_by": ["DIST", "TAGS", "JOIN_CONV_TAG"], "auto_tags": False, "manual_tags": True, "conv_tags": True},
    {"name": "C5_dist_tags_joinall", "mode": "rerank", "order_by": ["DIST", "TAGS", "JOIN_ALL"], "auto_tags": False, "manual_tags": True, "conv_tags": True},
]

ENGLISH_PROMPT = (
    "You are a research assistant. Answer the question using ONLY the context below. "
    "If the context is insufficient or only partial, say so clearly without inventing facts.\n\n"
    "Question: {q}\n\nContext ({n} chunks):\n{ctx}\n\nAnswer:"
)

BASE_COLS = {"user_input", "retrieved_contexts", "response", "reference", "reference_contexts"}


def _rec_info(rec: Any, idx: int):
    """(chunk_id, testo) da uno ScoredPoint Qdrant o da un dict del reranker (stessa logica di PipelineHandler)."""
    if isinstance(rec, dict):
        payload, cid = rec.get("payload", rec), rec.get("chunk_id")
    else:
        payload, cid = getattr(rec, "payload", {}) or {}, None
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else payload
    cid = cid or payload.get("chunk_id") or meta.get("chunk_id") or f"chunk_{idx}"
    text = (payload.get("text") or meta.get("text") or payload.get("page_content")
            or meta.get("page_content") or "")
    return cid, text.strip()


def _bootstrap_ci(diffs: np.ndarray, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _extract_conv_tags(q: Dict[str, Any]) -> List[str]:
    """Estrae i tag conversazionali dalla voce 'expected_tag' (o 'expected_tags')."""
    raw = q.get("expected_tag") if "expected_tag" in q else q.get("expected_tags", [])
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        return list(raw)
    return []


def _extract_manual_tags(q: Dict[str, Any]) -> List[str]:
    """Estrae i tag manuali dalla voce 'tags'."""
    raw = q.get("tags", [])
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        return list(raw)
    return []


class RagasBenchmark:
    def __init__(
        self,
        handler,                      # PipelineHandler gia' inizializzato (sidecar + queries_path)
        predetermined_path: str,      # queries_predetermined.json
        normal_path: str,             # queries_normal.json
        out_dir: str,
        configs: Optional[List[Dict[str, Any]]] = None,
        top_j: int = 5,               # chunk nel contesto, uguale per tutte le configurazioni
        top_k: int = 10,              # candidati del retrieval vettoriale
        top_n: int = 7,
        gold_path: Optional[str] = None,   # opzionale: {query_id: [chunk_id, ...]} per le metriche di retrieval
        use_handler_prompt: bool = False,  # True = prompt italiano di handler.generate_rag_response
    ):
        self.h = handler
        self.configs = configs or DEFAULT_CONFIGS
        self.top_j, self.top_k, self.top_n = top_j, top_k, top_n
        self.out_dir = Path(out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.use_handler_prompt = use_handler_prompt

        pred = json.load(open(Path(predetermined_path).expanduser(), encoding="utf-8"))
        norm = json.load(open(Path(normal_path).expanduser(), encoding="utf-8"))
        self.queries: List[Dict[str, Any]] = (
            [{**q, "is_predetermined": True} for q in pred] + [{**q, "is_predetermined": False} for q in norm]
        )
        self.gold = json.load(open(Path(gold_path).expanduser(), encoding="utf-8")) if gold_path else {}
        self.pred_path = predetermined_path
        self.norm_path = normal_path

    # ------------------------------------------------------------------ setup
    def register_predetermined(self, which: str = "pred") -> List[str]:
        """Importa query nel sidecar come predeterminate (id con prefisso predq_).
        which: "pred" (21 predeterminate) | "normal" (le altre 21) | "all" (tutte e 42)."""
        ids = []
        if which in ("pred", "all"):
            ids += self.h.add_predetermined_queries_from_file(self.pred_path)
        if which in ("normal", "all"):
            ids += self.h.add_predetermined_queries_from_file(self.norm_path)
        return ids

    def _registered(self) -> set:
        return set(self.h.sidecar_manager.load_data().get("predetermined_queries", {}))

    def edit_status(self) -> pd.DataFrame:
        """Per ogni query: registrata come predeterminata? quante modifiche query<->chunk e tag grafici ha nel sidecar."""
        pq = self.h.sidecar_manager.load_data().get("predetermined_queries", {})
        rows = []
        for q in self.queries:
            e = pq.get(f"predq_{q['id']}")
            rows.append({"query_id": q["id"], "registered": e is not None,
                         "n_dist_edits": len((e or {}).get("adjacencies", {})),
                         "n_graph_tags": len((e or {}).get("graphically_assigned_tags", []))})
        return pd.DataFrame(rows)

    def prepare(
        self,
        query_id: str,
        order_by: Optional[List[str]] = None,
        auto_tags: bool = False,
        manual_tags: bool = True,
        conv_tags: bool = True,
    ) -> Dict[str, Any]:
        """Porta l'handler sulla query (retrieval + reranking): poi si usa il widget come in handler.update_widget_view(...)."""
        q = next(q for q in self.queries if q["id"] == query_id)
        h = self.h
        h.set_parameters(top_j=self.top_j, top_k=self.top_k, top_n=self.top_n, order_by=order_by or ["DIST", "TAGS"])
        h.update_query(new_text=q["query"])

        conv_list = _extract_conv_tags(q) if conv_tags else []
        ass_list = _extract_manual_tags(q) if manual_tags else []

        h.update_query_tags(new_conv_list=conv_list, new_ass_list=ass_list, auto_ass_flag=auto_tags, print_debug=True)
        h.run_vectorial_retrieval()
        h.run_top_j_distance_tag_reranking()
        return q

    def check_similarity(self, warn_above: float = 0.80) -> pd.DataFrame:
        """Per ogni normale, la massima similarita' coseno con le predeterminate (>= soglia handler = collasso)."""
        pred = [q for q in self.queries if q["is_predetermined"]]
        norm = [q for q in self.queries if not q["is_predetermined"]]
        e = self.h.embeddings
        P = np.array(e.embed_documents([q["query"] for q in pred]))
        N = np.array(e.embed_documents([q["query"] for q in norm]))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        S = N @ P.T
        rows = [{"normal_id": q["id"], "paired_with": q["paired_with"],
                 "max_sim": float(S[i].max()), "closest_pred": pred[int(S[i].argmax())]["id"],
                 "sim_with_partner": float(S[i][[p["id"] for p in pred].index(q["paired_with"])])}
                for i, q in enumerate(norm)]
        df = pd.DataFrame(rows).sort_values("max_sim", ascending=False)
        bad = df[df.max_sim >= warn_above]
        if len(bad):
            logger.warning(f"!> {len(bad)} query normali troppo simili a una predeterminata: {bad.normal_id.tolist()}")
        return df

    # ------------------------------------------------------------- generation
    def _gen_path(self, run_id: str) -> Path:
        return self.out_dir / f"{run_id}_generation.jsonl"

    def load_generation(self, run_id: str) -> pd.DataFrame:
        p = self._gen_path(run_id)
        return pd.read_json(p, lines=True) if p.exists() else pd.DataFrame()

    def _answer(self, query: str, records: List[Any]) -> str:
        if self.use_handler_prompt:
            return self.h.generate_rag_response(records=records)["response"]
        ctx = "\n\n".join(f"[{i}] {_rec_info(r, i)[1]}" for i, r in enumerate(records, 1))
        out = self.h.llm.invoke(ENGLISH_PROMPT.format(q=query, n=len(records), ctx=ctx))
        return (out.content if hasattr(out, "content") else str(out)).strip()

    def _retrieval_metrics(self, qid: str, ids: List[str]) -> Dict[str, Any]:
        gold = set(self.gold.get(qid, []))
        if not gold:
            return {}
        hits = [i for i, c in enumerate(ids) if c in gold]
        return {"gold_hit": int(bool(hits)),
                "gold_recall": len(gold & set(ids)) / len(gold),
                "gold_precision": len(gold & set(ids)) / max(len(ids), 1),
                "gold_mrr": 1.0 / (hits[0] + 1) if hits else 0.0}

    def generate(self, run_id: str, query_ids: Optional[List[str]] = None, with_answers: bool = True) -> pd.DataFrame:
        """Esegue query x configurazioni. Riprende da dove era arrivato (chiave: query_id + config)."""
        path = self._gen_path(run_id)
        done = set()
        if path.exists():
            done = {(r["query_id"], r["config"]) for r in map(json.loads, open(path, encoding="utf-8"))}
        qs = [q for q in self.queries if query_ids is None or q["id"] in query_ids]
        registered = self._registered()

        with open(path, "a", encoding="utf-8") as f:
            for n, q in enumerate(qs, 1):
                todo = [c for c in self.configs if (q["id"], c["name"]) not in done]
                if not todo:
                    continue
                h = self.h
                h.set_parameters(top_j=self.top_j, top_k=self.top_k, top_n=self.top_n)
                h.update_query(new_text=q["query"])
                predq_ok = bool(h.is_predq) == (f"predq_{q['id']}" in registered)
                if not predq_ok:
                    logger.warning(f"!> {q['id']}: riconoscimento predetermined inatteso (is_predq={h.is_predq})")
                h.run_vectorial_retrieval()

                for c in todo:
                    t0 = time.perf_counter()
                    if c["mode"] == "vector":
                        records = list(h.raw_records)[: self.top_j]
                    else:
                        h.set_parameters(order_by=c.get("order_by", []))

                        conv_list = _extract_conv_tags(q) if c.get("conv_tags", False) else []
                        ass_list = _extract_manual_tags(q) if c.get("manual_tags", False) else []

                        h.update_query_tags(new_conv_list=conv_list, new_ass_list=ass_list,
                                            auto_ass_flag=c.get("auto_tags", False))
                        h.run_top_j_distance_tag_reranking()
                        records = list(h.top_j_reranked)
                    info = [_rec_info(r, i) for i, r in enumerate(records)]
                    ids = [i for i, _ in info]
                    row = {
                        "run_id": run_id, "query_id": q["id"], "category": q["category"],
                        "subtype": q.get("subtype"), "is_predetermined": q["is_predetermined"],
                        "predq_ok": predq_ok, "registered": f"predq_{q['id']}" in registered, "config": c["name"],
                        "user_input": q["query"], "reference": q["reference"],
                        "retrieved_ids": ids, "retrieved_contexts": [t for _, t in info],
                        "response": self._answer(q["query"], records) if with_answers else "",
                        "seconds": round(time.perf_counter() - t0, 2),
                        **self._retrieval_metrics(q["id"], ids),
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                logger.info(f"?> [{n}/{len(qs)}] {q['id']} completata ({len(todo)} config)")
        return self.load_generation(run_id)

    # ------------------------------------------------------------------ RAGAS
    def evaluate_ragas(self, run_id: str, judge_llm, max_workers: int = 2, timeout: int = 300,
                       metrics: Optional[List[Any]] = None, configs: Optional[List[str]] = None) -> pd.DataFrame:
        """
        judge_llm: modello chat LangChain del giudice (es. ChatGoogleGenerativeAI / ChatGroq / ChatOpenAI).
        Embedding per ResponseRelevancy: quelli locali dell'handler (nessun costo API).
        Salva un CSV per configurazione: se esiste gia' viene saltata (riprendibile).
        """
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                                   LLMContextRecall, ResponseRelevancy)
        from ragas.run_config import RunConfig

        gen = self.load_generation(run_id)
        if gen.empty:
            raise RuntimeError("Nessuna generazione trovata: esegui prima generate().")
        llm = LangchainLLMWrapper(judge_llm)

        # Poiché si usa un embedding locale: assegna una stringa fittizia all'attributo model dell'embedding
        # per evitare il ValidationError di Pydantic (Tracking dei costi di Ragas)
        self.h.embeddings.model = "fastembed-local"

        emb = LangchainEmbeddingsWrapper(self.h.embeddings)
        metrics = metrics or [Faithfulness(), ResponseRelevancy(),
                              LLMContextPrecisionWithReference(), LLMContextRecall()]

        for cname in (configs or [c["name"] for c in self.configs]):
            out = self.out_dir / f"{run_id}_{cname}_ragas.csv"
            if out.exists():
                logger.info(f"?> {cname}: gia' valutata, salto")
                continue
            sub = gen[gen.config == cname].reset_index(drop=True)
            ds = EvaluationDataset.from_list(sub[["user_input", "retrieved_contexts", "response", "reference"]]
                                             .to_dict("records"))
            # Isola Ragas dal loop di Jupyter tramite un Thread isolato permettendone l'esecuzione senza problemi di contesti
            import concurrent.futures

            def run_eval():
                return evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb,
                                run_config=RunConfig(max_workers=max_workers, timeout=timeout),
                                raise_exceptions=False, show_progress=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                res = executor.submit(run_eval).result()

            df = res.to_pandas()
            mcols = [c for c in df.columns if c not in BASE_COLS]
            df = pd.concat([sub[["run_id", "query_id", "category", "subtype", "is_predetermined", "config"]],
                            df[mcols]], axis=1)
            df.to_csv(out, index=False)
            logger.info(f"?> {cname}: salvato {out.name} (NaN: {int(df[mcols].isna().sum().sum())})")
        return self.load_ragas(run_id)

    def load_ragas(self, run_id: str) -> pd.DataFrame:
        fs = sorted(self.out_dir.glob(f"{run_id}_*_ragas.csv"))
        return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

    # ---------------------------------------------------------------- reports
    @staticmethod
    def _metric_cols(df: pd.DataFrame) -> List[str]:
        meta = {"run_id", "query_id", "category", "subtype", "is_predetermined", "config"}
        return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]

    def summary(self, df: pd.DataFrame, by: str = "category") -> pd.DataFrame:
        """Media per configurazione x (category | is_predetermined | subtype), con n e NaN ignorati."""
        return df.groupby(["config", by])[self._metric_cols(df)].mean().round(3)

    def paired_vs_baseline(self, df: pd.DataFrame, baseline: str = "C0_baseline", tol: float = 0.01,
                           subset: Optional[pd.Series] = None) -> pd.DataFrame:
        """Differenza media appaiata (config - baseline) con IC95% bootstrap e win/tie/loss per query."""
        d = df if subset is None else df[subset]
        rows = []
        for m in self._metric_cols(d):
            piv = d.pivot_table(index="query_id", columns="config", values=m, aggfunc="first")
            for c in [c for c in piv.columns if c != baseline]:
                pair = piv[[baseline, c]].dropna()
                if len(pair) < 3:
                    continue
                diff = (pair[c] - pair[baseline]).to_numpy()
                lo, hi = _bootstrap_ci(diff)
                rows.append({"metric": m, "config": c, "n": len(pair), "n_nan": int(piv[c].isna().sum()),
                             "mean_diff": round(float(diff.mean()), 3), "ci95_low": round(lo, 3),
                             "ci95_high": round(hi, 3), "win": int((diff > tol).sum()),
                             "tie": int((np.abs(diff) <= tol).sum()), "loss": int((diff < -tol).sum())})
        return pd.DataFrame(rows)

    def retrieval_summary(self, run_id: str) -> pd.DataFrame:
        """Metriche di retrieval su gold chunk (nessun LLM): hit, recall, precision, MRR per configurazione."""
        g = self.load_generation(run_id)
        cols = [c for c in ["gold_hit", "gold_recall", "gold_precision", "gold_mrr"] if c in g.columns]
        return g.groupby(["config", "category"])[cols].mean().round(3) if cols else pd.DataFrame()


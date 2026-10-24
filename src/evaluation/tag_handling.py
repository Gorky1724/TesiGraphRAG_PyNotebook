"""
Tag handling "specifico" per il benchmark (dataset del professore o simili), SENZA modificare TagAssigner / PreTagger.

Idea: al posto dell'espansione LLM del tag (generica => tag assegnati a quasi tutti i chunk) ogni tag ha un'espansione
SCRITTA A MANO come liste di termini del dominio:
  - "strong": termini/locuzioni specifiche (peso 2)
  - "weak"  : termini piu' generici (peso 1)
Un tag e' assegnato a un testo se la somma dei pesi dei termini DISTINTI trovati e' >= MIN_SCORE (default 2: basta un
termine strong oppure due weak). Per testo si tengono al massimo MAX_TAGS tag (i piu' punteggiati): meglio qualche tag
in meno che in piu'. Nessun embedding e nessuna soglia da calibrare: il risultato e' deterministico e si puo' verificare
a mano (vedi tag_report / explain).

Uso (notebook 09_, kernel con `handler` gia' creato):
    from src.evaluation.tag_handling import DSP_TAG_SPECS, pretag, tag_report, explain
    report = pretag(handler)                  # backup + azzera i tag dei chunk + pre-tagging con le espansioni a mano
    display(report)
    explain(handler, "Didimo_SeriesParallel_chunk_9")   # perche' un chunk ha (o non ha) certi tag

`pretag` installa un assegnatore specifico in handler.tag_assigner: lo usano sia il pre-tagging sia l'assegnazione
automatica dei tag alle query (config con auto_tags=True). Tag non presenti negli specs (es. aggiunti dal widget)
ricadono sul TagAssigner originale. Per un altro dataset basta passare altri specs (stesso formato).
"""
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

from src.tag_assigner import TagAssigner

MIN_SCORE = 2     # soglia sul punteggio (strong=2, weak=1)
MAX_TAGS = 3      # massimo di tag per testo

# --------------------------------------------------------------------------------------------------------------------
# Espansioni dei 7 tag del dataset del professore (orthogonal graph drawing). Sono regex (case-insensitive) applicate
# al testo normalizzato (ligature, trattini e a-capo gia' sistemati).
# --------------------------------------------------------------------------------------------------------------------
DSP_TAG_SPECS: Dict[str, Dict[str, List[str]]] = {
    "Orthogonal Representation & Embedding": {
        "strong": [
            r"orthogonal representations?", r"angle (assignment|labell?ing)", r"planar (embedding|representation)s?",
            r"(fixed|variable|given|prescribed) embedding", r"embedding[- ]preserving", r"plane (4-)?graphs?",
            r"dimensionless", r"region[- ]preserving", r"rotation system",
        ],
        "weak": [
            r"sequence of bends", r"left and right turns?", r"outer face", r"\bfaces?\b", r"angles? (around|at) (a |each |every )?vert",
            r"\bshape\b", r"reflex", r"turn[- ]regular",
        ],
    },
    "Bend Minimization & Bounds": {
        "strong": [
            r"bend[- ]?minim\w*", r"bend[- ]optimal", r"(minimum|minimal|fewest|optimal) (number (of )?)?bends?",
            r"number of bends", r"bends? per edge", r"bend cost", r"(lower|upper) bounds? on (the )?(total |minimum |maximum )?(number of )?bends?",
            r"at most [^.]{0,40}bends", r"at least [^.]{0,40}bends",
        ],
        "weak": [
            r"\bbends?\b", r"subdivision vertices", r"dummy vert", r"(lower|upper) bounds?",
        ],
    },
    "Network Flow & Tamassia": {
        "strong": [
            r"network[- ]?flows?", r"flows?[- ]?networks?", r"min(imum)?[- ]?cost[- ]?flow", r"cost[- ]?flow", r"tamassia", r"\bTam87\b",
            r"out[- ]of[- ]kilter", r"supply and demand", r"flow model",
        ],
        "weak": [
            r"\bflows?\b", r"\bsupply\b", r"\bdemand\b", r"\barcs?\b", r"face nodes?", r"commodity", r"unit cost",
        ],
    },
    "Topology-Shape-Metrics Pipeline": {
        "strong": [
            r"topology[- ]shape[- ]metrics", r"\bTSM\b", r"shape[- ]metrics", r"shape[- ]first", r"planari[sz]ation",
            r"orthogonali[sz]ation", r"compaction", r"\bmetrics\b",
        ],
        "weak": [
            r"crossing minimi[sz]ation", r"minimi[sz]\w* (the )?(number of )?crossings", r"orthogonal crossings?",
            r"dummy vert\w* [^.]{0,40}crossings?", r"coordinates", r"edge crossings?",
        ],
    },
    "Rectilinear Planarity & Complexity": {
        "strong": [
            r"rectilinear (planar\w*|drawings?|graphs?)", r"HV[- ](rectilinear|restricted|drawings?|graphs?)",
            r"NP[- ](hard\w*|complete\w*)", r"fixed[- ]parameter", r"\bFPT\b", r"rectilinear planarity",
            r"(polynomial|linear)[- ]time (algorithm|solvable)", r"longstanding", r"open (problem|question)",
        ],
        "weak": [
            r"complexity", r"polynomial", r"linear[- ]time", r"\bSAT\b", r"reduction", r"4-planar", r"4-graphs?", r"degree[- ](four|4)",
        ],
    },
    "Series-Parallel & Cubic Graphs": {
        "strong": [
            r"series[- ]?parallel", r"\bSPQ\*?", r"spirality", r"\bcubic\b", r"tri-?connected", r"3-connected",
            r"rectangular drawings?", r"(bad|legged|maximal) cycles?", r"thomassen",
        ],
        "weak": [
            r"decomposition tree", r"\bpoles?\b", r"bi-?connected", r"[SPQR]\*?-nodes?", r"3-graphs?", r"degree[- ](three|3)\b",
        ],
    },
    "Applications & Aesthetics": {
        "strong": [
            r"aesthetics?", r"readability", r"\bVLSI\b", r"floor[- ]?plans?", r"entity[- ]relationship", r"data[- ]?flow diagrams?",
            r"organi[sz]ation charts?", r"circuit (layout|schematics?)", r"machine[- ]learning", r"three[- ]dimensional", r"\b3D\b",
            r"applications? (of|in|to|include)", r"software engineering",
        ],
        "weak": [
            r"angular resolution", r"geometric uniformity", r"\barea\b", r"transportation", r"\bdatabases?\b", r"diagrams?",
        ],
    },
}


# --------------------------------------------------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Ligature (fl, fi), trattini unicode, parole spezzate a fine riga e spazi: cosi' le regex non perdono match."""
    t = unicodedata.normalize("NFKC", text or "")
    t = re.sub(r"[‐‑‒–—−]", "-", t)
    t = re.sub(r"-\s*\n\s*", "-", t)
    return re.sub(r"\s+", " ", t)


def compile_specs(specs: Dict[str, Dict[str, List[str]]]):
    return {tag: [(2, re.compile(p, re.I)) for p in s.get("strong", [])] + [(1, re.compile(p, re.I)) for p in s.get("weak", [])]
            for tag, s in specs.items()}


def score_text(text: str, compiled) -> Dict[str, int]:
    """Punteggio di ogni tag degli specs per il testo (somma dei pesi dei termini distinti trovati)."""
    t = normalize_text(text)
    return {tag: sum(w for w, rx in pats if rx.search(t)) for tag, pats in compiled.items()}


class SpecificTagAssigner(TagAssigner):
    """
    TagAssigner con le espansioni a mano.
    Fa il controllo dei termini presenti nel testo del chunk, e somma il punteggio relativo per i termini presenti allo score finale.
    De il punteggio di termini trovati nel chunk supera la soglia assegna il tag di cui ha preso la lista. Può assegnarne fino a un max di 3.
    Così facendo non dipende più dalla casualità dell'LLM.
    """

    @classmethod
    def from_assigner(cls, base: TagAssigner, specs=None, min_score: int = MIN_SCORE, max_tags: int = MAX_TAGS):
        new = cls.__new__(cls)
        new.__dict__.update(base.__dict__)
        new.specs = specs or DSP_TAG_SPECS
        new.compiled = compile_specs(new.specs)
        new.min_score, new.max_tags = min_score, max_tags
        return new

    def assign_tags(self, text, candidate_tags, vector=None) -> List[str]:
        known = [t for t in candidate_tags if t in self.compiled]
        scores = score_text(text, {t: self.compiled[t] for t in known})
        # a parita' di punteggio vale l'ordine dello spec; si tengono i migliori max_tags
        order = {t: i for i, t in enumerate(self.compiled)}
        ranked = sorted((t for t in known if scores[t] >= self.min_score), key=lambda t: (-scores[t], order[t]))
        assigned = ranked[: self.max_tags]
        unknown = [t for t in candidate_tags if t not in self.compiled]
        if unknown:   # tag non negli specs (es. aggiunti dal widget): comportamento originale
            assigned += super().assign_tags(text, unknown, vector)
        return assigned


def install(handler, specs=None, min_score: int = MIN_SCORE, max_tags: int = MAX_TAGS) -> SpecificTagAssigner:
    """Sostituisce handler.tag_assigner (usato da pretag_sidecar e dall'auto-tagging delle query)."""
    handler.tag_assigner = SpecificTagAssigner.from_assigner(handler.tag_assigner, specs, min_score, max_tags)
    return handler.tag_assigner


def pretag(handler, specs=None, batch_size: int = 100, min_score: int = MIN_SCORE, max_tags: int = MAX_TAGS,
           reset: bool = True):
    """Installa l'assegnatore specifico e fa il pre-tagging di tutta la collezione con PreTagger (invariato).
    reset=True: copia di backup del sidecar (<nome>_pre_tag_handling.json) e azzeramento di tag_overrides/global_tags
    (il pre-tagging aggiunge soltanto); modified_adjacencies e predetermined_queries NON si toccano.
    ATTENZIONE: i tag dei chunk modificati a mano prima di questa chiamata vanno persi (restano nel backup)."""
    specs = specs or DSP_TAG_SPECS
    install(handler, specs, min_score, max_tags)
    if reset:
        p = Path(handler.sidecar_path)
        if p.exists():
            shutil.copy(p, p.with_name(p.stem + "_pre_tag_handling" + p.suffix))
        data = handler.sidecar_manager.load_data()
        data["tag_overrides"], data["global_tags"] = {}, {}
        handler.sidecar_manager.save_data(data)
    handler.pretag_sidecar(candidate_tags=list(specs), batch_size=batch_size)
    return tag_report(handler)


def tag_report(handler):
    """Tabella: per ogni tag quanti chunk (e quale %) lo hanno, piu' chunk senza tag e tag medi per chunk."""
    import pandas as pd
    from src.config import COLLECTION_NAME
    data = handler.sidecar_manager.load_data()
    total = handler.qdrant_client.count(collection_name=COLLECTION_NAME).count
    counts = {t: v.get("count", 0) for t, v in data.get("global_tags", {}).items()}
    per_chunk = [len(v.get("user_tags", [])) for v in data.get("tag_overrides", {}).values()]
    rows = [{"tag": t, "chunk": n, "%": round(100 * n / max(total, 1), 1)} for t, n in sorted(counts.items(), key=lambda x: -x[1])]
    rows.append({"tag": "(chunk senza tag)", "chunk": total - sum(1 for n in per_chunk if n), "%": round(100 * (total - sum(1 for n in per_chunk if n)) / max(total, 1), 1)})
    rows.append({"tag": "(tag medi per chunk)", "chunk": round(sum(per_chunk) / max(total, 1), 2), "%": None})
    return pd.DataFrame(rows)


def explain(handler, chunk_id: str):
    """Punteggi per tag di un chunk, con i termini che hanno fatto match (per capire perche' un tag c'e' o manca)."""
    from src.config import COLLECTION_NAME
    recs, off = [], None
    while not recs:      # scansione in Python (2-3k chunk): evita dipendenze da indici/filtri Qdrant
        page, off = handler.qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=256, offset=off, with_payload=True)
        for r in page:
            pl = r.payload or {}
            m = pl.get("metadata") if isinstance(pl.get("metadata"), dict) else pl
            if (m.get("chunk_id") or pl.get("chunk_id")) == chunk_id:
                recs = [r]; break
        if off is None: break
    if not recs:
        print("chunk non trovato:", chunk_id); return
    p = recs[0].payload or {}
    text = p.get("page_content") or p.get("text") or ""
    ta = handler.tag_assigner
    t = normalize_text(text)
    print(text[:300].replace("\n", " "), "...\n")
    for tag, pats in ta.compiled.items():
        hits = [rx.pattern for w, rx in pats if rx.search(t)]
        sc = sum(w for w, rx in pats if rx.search(t))
        print(f"{'*' if sc >= ta.min_score else ' '} {sc:2d}  {tag:40s} {hits[:5]}")
    print("\nassegnati:", ta.assign_tags(text, list(ta.compiled)))


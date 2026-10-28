"""
colosseum_controls.py - controlli SOLO RETRIEVAL (nessun LLM, nessun giudice) per il benchmark Colosseo.
Salvare come ~/tesi_graphrag/src/evaluation/colosseum_controls.py

Quattro controlli, tutti riprendibili (stesso run_id = riprende da dove era arrivato, come bench.generate):
  1. wrong_tag      - controllo NEGATIVO: stesse query con un conv_tag SBAGLIATO. Se il tag agisce in modo causale il recall deve calare.
  2. two_readings   - la stessa query con due letture (tag A / tag B, gold A / gold B): il sistema segue il tag o lo ignora?
  3. degradation    - degradazione dei tag separata (mancanti / sbagliati / solo primo tag / combinata) su piu' seed.
  4. distance       - le distanze servono davvero? sidecar CON distanze contro sidecar SENZA distanze (stessi tag), al variare di top_k.

PERCHE' IL CONTROLLO 4 (letto da distance_tag_reranker.py / distance_retriever.py):
  - il 2° retrieval (distanze) parte SOLO se esistono chunk di categoria 1 (chunk del 1° retrieval con il conv_tag) e,
    senza flag JOIN_*, riempie solo i posti rimasti: top_m = top_j - n_categoria1. Con top_k=15 e un tag frequente la
    categoria 1 riempie gia' i top_j e il 2° retrieval non parte (nei tuoi dati C1 e C3 danno lo stesso insieme in 26 query normali su 30).
  - le categorie 1-4 sono definite dal conv_tag a prescindere da order_by: C2_dist NON e' "solo distanze", e' "conv_tag + ordine per DIST".
  - con JOIN_* il distance_factor delle categorie 2-3 viene azzerato (=1.0): la classe di distanza decide solo CHI ENTRA nel pool.
    Quindi C5 senza DIST sarebbe identico a C5: l'ablazione pulita e' "sidecar senza distanze" (controllo 4).
"""
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

BASE_KEYS = ["query_id", "config"]

# ------------------------------------------------------------------------------------------------ utilita'
def _boot(diffs, n_boot=5000, seed=0):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@contextmanager
def _patched(bench, queries=None, configs=None, top_k=None, top_j=None):
    """Sostituisce temporaneamente query / configurazioni / top_k / top_j del benchmark e li ripristina sempre."""
    old = (bench.queries, bench.configs, bench.top_k, bench.top_j)
    try:
        if queries is not None:
            bench.queries = queries
        if configs is not None:
            bench.configs = [c for c in old[1] if c["name"] in configs]
        if top_k is not None:
            bench.top_k = top_k
        if top_j is not None:
            bench.top_j = top_j
        yield bench
    finally:
        bench.queries, bench.configs, bench.top_k, bench.top_j = old


def run(bench, run_id, queries=None, configs=None, top_k=None, top_j=None):
    """Retrieval di tutte le query x config (senza LLM). Restituisce il DataFrame di generazione del run."""
    with _patched(bench, queries, configs, top_k, top_j):
        return bench.generate(run_id, with_answers=False)


def run_regime(bench, run_id, top_k, top_j, queries=None, configs=None, with_answers=False):
    """Un REGIME (top_k, top_j) con confronto alla pari: le config di reranking usano (top_k, top_j), mentre C0 usa top_k = top_j
    (vettoriale puro a top_j chunk). Senza questo, con top_k < top_j C0 vedrebbe solo top_k chunk invece di top_j.
    Salva due file (<run_id>_rr, <run_id>_c0) e restituisce il DataFrame unito."""
    names = [c["name"] for c in bench.configs if configs is None or c["name"] in configs]
    rr = [n for n in names if n != "C0_baseline"]
    parts = []
    with _patched(bench, queries, rr, top_k, top_j):
        parts.append(bench.generate(f"{run_id}_rr", with_answers=with_answers))
    if "C0_baseline" in names:
        with _patched(bench, queries, ["C0_baseline"], max(top_k, top_j), top_j):
            parts.append(bench.generate(f"{run_id}_c0", with_answers=with_answers))
    return pd.concat(parts, ignore_index=True)


def short_context_quota(gen, top_j):
    """Quota di query (per config) con meno di top_j chunk nel contesto: categoria 1 vuota, il 2° retrieval non e' partito
    e il contesto resta quello dei soli top_k vettoriali. Con top_k < top_j e' il caso da contare."""
    n = gen.retrieved_ids.apply(len)
    return (n < top_j).groupby(gen.config).mean().round(2)


def gold_metrics(ids, gold_ids):
    g = set(gold_ids)
    hits = [i for i, c in enumerate(ids) if c in g]
    return {"recall": len(g & set(ids)) / len(g), "mrr": 1.0 / (hits[0] + 1) if hits else 0.0}


def _mask(df, name):
    normal = ~df.is_predetermined.astype(bool)
    return {"normali": normal,
            "ambigue normali": normal & (df.category == "ambiguous"),
            "tutte": pd.Series(True, index=df.index)}[name]


SUBSETS = ("normali", "ambigue normali", "tutte")


def paired_between_runs(gen_a, gen_b, metrics=("gold_recall", "gold_mrr"), subsets=SUBSETS, tol=0.01):
    """Differenza appaiata (B - A) sulle STESSE query e config, con IC95% bootstrap e win/tie/loss."""
    m = gen_a.merge(gen_b, on=BASE_KEYS, suffixes=("_a", "_b"))
    m["is_predetermined"], m["category"] = m["is_predetermined_a"], m["category_a"]
    rows = []
    for s in subsets:
        sub = m[_mask(m, s)]
        for cfg, g in sub.groupby("config"):
            for met in metrics:
                d = (g[f"{met}_b"] - g[f"{met}_a"]).to_numpy(float)
                lo, hi = _boot(d)
                rows.append({"subset": s, "config": cfg, "metric": met, "n": len(d),
                             "mean_A": round(g[f"{met}_a"].mean(), 3), "mean_B": round(g[f"{met}_b"].mean(), 3),
                             "diff_B_minus_A": round(d.mean(), 3), "ci95_low": round(lo, 3), "ci95_high": round(hi, 3),
                             "win": int((d > tol).sum()), "tie": int((abs(d) <= tol).sum()), "loss": int((d < -tol).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ 1. tag sbagliato
MONUMENT_TAGS = ("Colosseum", "DomusAurea", "ColossusOfNero", "LudusMagnus", "ArchOfConstantine")   # come in add_monument_tags.py


def conv_of(q):
    """Lista dei conv tag della query (expected_tag puo' essere stringa o lista): il primo e' il monumento, gli altri aspetti."""
    e = q.get("expected_tag")
    return [e] if isinstance(e, str) else list(e or [])


def wrong_tag_queries(queries, gold, chunks, tags, seed=0, mode="monument"):
    """Copia delle query con un tag SBAGLIATO su uno dei due assi (si sceglie quello meno presente nei chunk gold; a parita' random col seed):
      mode="monument": il monumento (primo conv_tag) e' un ALTRO monumento; gli aspetti restano giusti;
      mode="aspect"  : il monumento resta giusto; l'aspetto (conv_tag aggiuntivo o, se non c'e', tag manuale) diventa un aspetto sbagliato.
    Confronto corretto: sulle query NORMALI (le predq hanno comunque le adiacenze query->chunk, che non dipendono dal tag)."""
    rng = random.Random(seed)
    mons = [t for t in tags if t in MONUMENT_TAGS]; asp = [t for t in tags if t not in MONUMENT_TAGS]
    if not mons:
        raise ValueError("TAGS non contiene tag monumento: stai usando i chunk v1 (09C_06)? Esegui prima le celle 0-4 di 09C_08 "
                         "nello STESSO kernel (chunks/gold/TAGS/bench devono essere quelli v2), poi riesegui la cella 0 di 09C_07.")
    out = []
    for q in queries:
        conv = conv_of(q)
        gt = [x for cid in gold[q["id"]] for x in chunks[cid]["tags"]]
        if mode == "monument":
            cands = sorted(set(mons) - {conv[0]}, key=lambda x: (gt.count(x), rng.random()))
            out.append({**q, "orig_expected_tag": conv, "expected_tag": [cands[0]] + conv[1:]})
        elif len(conv) > 1:
            cands = sorted(set(asp) - set(conv), key=lambda x: (gt.count(x), rng.random()))
            out.append({**q, "orig_expected_tag": conv, "expected_tag": [conv[0], cands[0]]})
        else:
            cands = sorted(set(asp) - set(q["tags"]), key=lambda x: (gt.count(x), rng.random()))
            out.append({**q, "orig_tags": list(q["tags"]), "tags": [cands[0]] if q["tags"] else []})
    return out


def wrong_tag_leak(gen, queries, chunks, mode="monument"):
    """Quota dei top-J che porta il tag (sbagliato) usato: se alta, il sistema SEGUE il tag."""
    use = {q["id"]: (conv_of(q)[:1] if mode == "monument" else conv_of(q)[1:] + list(q["tags"])) for q in queries}
    g = gen.copy()
    g["quota_con_tag_usato"] = [np.mean([bool(set(use[r.query_id]) & set(chunks[i]["tags"])) for i in r.retrieved_ids])
                                for r in g.itertuples()]
    return g.groupby("config").quota_con_tag_usato.mean().round(2)


# ------------------------------------------------------------------------------------------------ 2. due letture
# BOZZA da rivedere leggendo i chunk: per ogni ambigua una seconda lettura (tag B, gold B). Il gold A e' quello di gold_chunks.json.
TWO_READINGS = {
    "AMB_14": {"tag_b": "Spectacles",   "gold_b": ["wiki_colosseum_27", "wiki_colosseum_10", "wiki_colosseum_29"]},      # oggi (Culture) vs giochi antichi
    "AMB_12": {"tag_b": "Archaeology",  "gold_b": ["web_underground_04", "web_underground_03", "wiki_colosseum_24"]},    # resoconti (Spectacles) vs tracce fisiche
    "AMB_07": {"tag_b": "Conservation", "gold_b": ["wiki_domus_aurea_06", "wiki_domus_aurea_20"]},                      # influenza sul Rinascimento vs degrado
    "AMB_09": {"tag_b": "History",      "gold_b": ["wiki_ludus_magnus_04", "wiki_ludus_magnus_05", "wiki_ludus_magnus_01"]},  # scavi (Archaeology) vs storia
    "AMB_08": {"tag_b": "Spectacles",   "gold_b": ["wiki_ludus_magnus_03", "wiki_ludus_magnus_10", "wiki_ludus_magnus_01"]},  # edificio (Architecture) vs vita/addestramento
}


def validate_two_readings(pairs, gold, chunks):
    """Controlli di coerenza: chunk esistenti, tag B presente nei gold B, nessuna sovrapposizione gold A / gold B."""
    for qid, p in pairs.items():
        assert all(c in chunks for c in p["gold_b"]), f"{qid}: gold_b con chunk inesistenti"
        assert not set(gold[qid]) & set(p["gold_b"]), f"{qid}: gold A e gold B si sovrappongono"
        miss = [c for c in p["gold_b"] if p["tag_b"] not in chunks[c]["tags"]]
        assert not miss, f"{qid}: tag_b '{p['tag_b']}' assente in {miss}"
    return True


def two_readings_queries(queries, pairs):
    by_id = {q["id"]: q for q in queries}
    out = []
    for qid, p in pairs.items():
        q = by_id[qid]
        out.append({**q, "id": f"{qid}__A", "is_predetermined": False})
        conv = conv_of(q)
        if len(conv) > 1:   # l'aspetto e' un conv tag: la lettura B lo sostituisce (stesso monumento)
            out.append({**q, "id": f"{qid}__B", "expected_tag": [conv[0], p["tag_b"]], "is_predetermined": False})
        else:               # l'aspetto e' un tag manuale
            out.append({**q, "id": f"{qid}__B", "tags": [p["tag_b"]], "is_predetermined": False})
    return out


def two_readings_report(gen, pairs, gold):
    """Per config: recall del gold della lettura VOLUTA (own), recall del gold dell'altra lettura (other), specificita'.
    specificita' = media di [(R_own - R_other)] sulle due direzioni. 0 = il tag non decide la lettura (C0 vale 0 per costruzione)."""
    rows = []
    for cfg, g in gen.groupby("config"):
        ids = {r.query_id: list(r.retrieved_ids) for r in g.itertuples()}
        for qid, p in pairs.items():
            ga, gb = gold[qid], p["gold_b"]
            for side, own, other in (("A", ga, gb), ("B", gb, ga)):
                r = ids[f"{qid}__{side}"]
                rows.append({"config": cfg, "query": qid, "lettura": side,
                             "R_own": gold_metrics(r, own)["recall"], "R_other": gold_metrics(r, other)["recall"]})
    d = pd.DataFrame(rows)
    d["specificita"] = d.R_own - d.R_other
    return d.groupby("config")[["R_own", "R_other", "specificita"]].mean().round(3), d


# ------------------------------------------------------------------------------------------------ 3. degradazione tag
# (parametri ASPETTO, parametri MONUMENTO). Valori arbitrari ma dichiarati: il monumento e' facile da assegnare (si legge dal documento),
# quindi nel caso "utente medio" ha errori piu' bassi dell'aspetto.
VARIANTS = {
    "tag_esperti":           (dict(), dict()),
    "aspetti_mancanti30":    (dict(missing_frac=0.30), dict()),
    "aspetti_sbagliati15":   (dict(wrong_frac=0.15), dict()),
    "aspetti_sbagliati30":   (dict(wrong_frac=0.30), dict()),
    "solo_primo_aspetto":    (dict(drop_secondary=True), dict()),
    "monumento_mancante20":  (dict(), dict(missing_frac=0.20)),
    "monumento_sbagliato10": (dict(), dict(wrong_frac=0.10)),
    "combinato_user":        (dict(missing_frac=0.30, wrong_frac=0.15, drop_secondary=True), dict(missing_frac=0.10, wrong_frac=0.05)),
}
DEG_CONFIGS = ("C0_baseline", "C1_tags", "C4_dist_tags_joinconv", "C5_dist_tags_joinall")


def degrade_two_axes(chunks, topical_tags, monuments=MONUMENT_TAGS, topical=None, monument=None, seed=0):
    """Degrada i tag dei chunk SEPARATAMENTE sui due assi. topical: kwargs di degrade_tags (aspetto); monument: missing_frac / wrong_frac."""
    from src.evaluation.degrade_tags import degrade_tags
    mons = set(monuments); topical, monument = dict(topical or {}), dict(monument or {})
    top_in = [{**c, "tags": [t for t in c["tags"] if t not in mons]} for c in chunks]
    top_out, log = degrade_tags(top_in, list(topical_tags), seed=seed,
                                missing_frac=topical.get("missing_frac", 0.0), wrong_frac=topical.get("wrong_frac", 0.0),
                                drop_secondary=topical.get("drop_secondary", False))
    rng = random.Random(seed + 1000)
    out = []
    for c, t in zip(chunks, top_out):
        m = [x for x in c["tags"] if x in mons]
        if m and rng.random() < monument.get("missing_frac", 0.0):
            m = []; log["monument_missing"] = log.get("monument_missing", 0) + 1
        elif m and rng.random() < monument.get("wrong_frac", 0.0):
            m = [rng.choice([x for x in monuments if x != m[0]])]; log["monument_wrong"] = log.get("monument_wrong", 0) + 1
        out.append({**c, "tags": m + t["tags"]})
    return out, log


def degradation_study(chunks, tags, base_sidecar, work_dir, make_bench, variants=None, seeds=(0, 1, 2),
                      configs=DEG_CONFIGS, prefix="deg"):
    """Per ogni variante x seed: copia del sidecar BASE (tag + predq, nessuna distanza), tag degradati sui due assi, nuovo handler/bench, retrieval.
    make_bench(sidecar_path) -> RagasBenchmark (a te la costruzione: vedi la cella del notebook).
    Distanze assenti di proposito: si isola l'effetto dei tag. tag_esperti non dipende dal seed (1 solo run)."""
    import shutil
    from src.evaluation.json_ingest import load_tags_into_sidecar
    work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    topical_tags = [t for t in tags if t not in MONUMENT_TAGS]
    frames = []
    for name, (kt, km) in (variants or VARIANTS).items():
        for seed in ((0,) if name == "tag_esperti" else seeds):
            sp = work_dir / f"{prefix}_{name}_s{seed}.json"
            shutil.copy(base_sidecar, sp)
            deg, log = degrade_two_axes(list(chunks.values()), topical_tags, topical=kt, monument=km, seed=seed)
            load_tags_into_sidecar(deg, sp).release_path()
            bench = make_bench(sp)
            g = run(bench, f"{prefix}_{name}_s{seed}", configs=configs)
            frames.append(g.assign(variant=name, seed=seed))
            print(f"[{name} s{seed}] {log} | senza monumento: {sum(not set(c['tags']) & set(MONUMENT_TAGS) for c in deg)}/{len(deg)}")
    return pd.concat(frames, ignore_index=True)


def summarize_vs_c0(df, metrics=("gold_recall", "gold_mrr"), subsets=("normali", "ambigue normali")):
    """Differenza di ogni config rispetto a C0 (stessa variante e seed), media e dev.std. sui seed."""
    rows = []
    for s in subsets:
        d = df[_mask(df, s)]
        for met in metrics:
            t = d.groupby(["variant", "seed", "config"])[met].mean().unstack("config")
            diff = t.sub(t["C0_baseline"], axis=0).drop(columns="C0_baseline")
            agg = diff.groupby(level="variant").agg(["mean", "std"]).round(3)
            agg.columns = [f"{c}|{a}" for c, a in agg.columns]
            agg.insert(0, "metric", met); agg.insert(0, "subset", s)
            rows.append(agg.reset_index())
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------------------------ 4. distanze
DIST_CONFIGS = ("C0_baseline", "C1_tags", "C3_dist_tags", "C4_dist_tags_joinconv", "C5_dist_tags_joinall")


REGIMES = ((15, 5), (10, 5), (7, 5), (5, 8), (5, 10))   # (top_k, top_j); i regimi con top_k < top_j fanno sempre partire le distanze


def distance_ablation(bench_dist, bench_nodist, regimes=REGIMES, configs=DIST_CONFIGS, prefix="dist"):
    """bench_dist: sidecar con le distanze esperte. bench_nodist: copia del sidecar BASE (stessi tag, nessuna distanza).
    Per ogni regime (top_k, top_j): effetto delle distanze = run(con) - run(senza), appaiato per query; C0 sempre a top_j chunk
    (run_regime). Restituisce (tabella, generazioni {(k, j): (senza, con)})."""
    gens, tabs = {}, []
    for k, j in regimes:
        a = run_regime(bench_nodist, f"{prefix}_nodist_k{k}_j{j}", k, j, configs=configs)
        b = run_regime(bench_dist, f"{prefix}_dist_k{k}_j{j}", k, j, configs=configs)
        gens[(k, j)] = (a, b)
        t = paired_between_runs(a, b, subsets=("normali", "ambigue normali"))
        pa = a.pivot(index="query_id", columns="config", values="retrieved_ids")
        pb = b.pivot(index="query_id", columns="config", values="retrieved_ids")
        pred = a.drop_duplicates("query_id").set_index("query_id").is_predetermined.astype(bool)
        act = {c: float(np.mean([set(x) != set(y) for x, y in zip(pa.loc[~pred, c], pb.loc[~pred, c])])) for c in pa.columns}
        t["top_k"], t["top_j"] = k, j
        t["quota_query_cambiate"] = t.config.map(act).round(2)   # normali in cui le distanze cambiano l'insieme dei top-J
        tabs.append(t)
    return pd.concat(tabs, ignore_index=True), gens


def vs_c0_by_topk(gens, side=1, metric="gold_recall", subsets=("normali", "ambigue normali")):
    """Per ogni top_k: differenza di ogni config con C0, per il sidecar CON (side=1) o SENZA (side=0) distanze."""
    rows = []
    for (k, j), pair in gens.items():
        g = pair[side]
        for s in subsets:
            d = g[_mask(g, s)]
            t = d.groupby("config")[metric].mean()
            rows += [{"top_k": k, "top_j": j, "subset": s, "config": c, f"{metric}": round(v, 3), "diff_vs_C0": round(v - t["C0_baseline"], 3)}
                     for c, v in t.items()]
    return pd.DataFrame(rows)

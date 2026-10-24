"""
Pre-istanziazione delle classi di distanza (sidecar) per il benchmark Armstrong.
Salvare come ~/tesi_graphrag/src/evaluation/distance_presets.py

Due passi, separati apposta per poter RILEGGERE E RITOCCARE il piano prima di applicarlo:
  plan = build_plan(...)  -> dict serializzabile (salvabile in JSON, modificabile a mano)
  apply_plan(sm, plan)    -> scrive le adiacenze nel sidecar

Regole (solo sulle PREDETERMINATE; le normali non vengono mai toccate, cosi' la generalizzazione resta misurabile):
  1. predq -> chunk gold:  il primo gold = classe forte di avvicinamento, gli altri = classe lieve.
  2. predq -> chunk sbagliati (solo se si passa `bench`): i chunk NON gold che il retrieval vettoriale mette nei top_j
     della predq -> classe lieve di allontanamento; i primi 3 del ranking -> classe forte di allontanamento.
  3. chunk <-> chunk: tutte le coppie dei gold di una stessa predq -> classe lieve di avvicinamento.
     Sono le uniche modifiche che raggiungono le query normali (il 2° retrieval parte dai chunk di categoria 1).
     Con cluster_union=True (versione ESPERTA) si collega l'intero cluster tematico: gold della predq + gold della normale
     abbinata (paired_with); le coppie in cui ENTRAMBI i chunk sono gold di entrambe le query (il nucleo) usano la classe forte.
     ATTENZIONE: cosi' l'esperto "conosce" gli argomenti delle normali -> e' un tetto (upper bound), non una misura di generalizzazione.
     Le coppie chunk<->chunk di ALLONTANAMENTO non hanno effetto sul ranking (il 2° retrieval considera solo fattori > 1): non vengono generate.
Versione UTENTE MEDIO: degrade_plan(plan, keep=0.6, seed=...) tiene a caso una frazione degli stessi interventi (da scrivere in un altro sidecar).
Le classi sono scelte dalla scala attiva nel sidecar (get_class_factors): funziona anche con le classi custom di 08_01.
"""
import itertools, json
from pathlib import Path


def pick_classes(sm):
    f = sm.get_class_factors()
    up = sorted(((v, k) for k, v in f.items() if v > 1.0), reverse=True)
    down = sorted((v, k) for k, v in f.items() if v < 1.0)
    if not up:
        raise ValueError("La scala non ha classi di avvicinamento (fattore > 1).")
    return {"strong_up": up[0][1], "mild_up": up[1][1] if len(up) > 1 else up[0][1],
            "strong_down": down[0][1] if down else None, "mild_down": down[-1][1] if down else None}


def build_plan(queries, gold, sm, bench=None, top_j=None, top_k=None, penalize_wrong=True, link_gold_pairs=True, cluster_union=True):
    """queries: lista di eval_queries.json; gold: {query_id: [chunk_id,...]} con il chunk piu' importante per primo.
    bench: RagasBenchmark (opzionale) per individuare i chunk sbagliati col retrieval vettoriale reale."""
    from src.evaluation.benchmark_runner import _rec_info
    cls = pick_classes(sm)
    plan = {"classes": cls, "predq": {}, "pairs": []}
    by_id = {q["id"]: q for q in queries}
    seen_pairs = set()
    for q in queries:
        if not q.get("is_predetermined"):
            continue
        g = list(gold[q["id"]])
        edits = {c: (cls["strong_up"] if i == 0 else cls["mild_up"]) for i, c in enumerate(g)}
        if bench is not None and penalize_wrong and cls["mild_down"]:
            bench.h.set_parameters(top_j=top_j or bench.top_j, top_k=top_k or bench.top_k, top_n=bench.top_n)
            bench.h.update_query(new_text=q["query"])
            bench.h.run_vectorial_retrieval()
            ranked = [_rec_info(r, i)[0] for i, r in enumerate(bench.h.raw_records)]
            for rank, cid in enumerate(ranked[: (top_j or bench.top_j)]):
                if cid not in g:
                    edits[cid] = cls["strong_down"] if rank < 3 else cls["mild_down"]
        plan["predq"][f"predq_{q['id']}"] = edits
        if link_gold_pairs:
            partner = by_id.get(q.get("paired_with"))
            g2 = list(gold.get(partner["id"], [])) if (cluster_union and partner) else []
            core = set(g) & set(g2)
            for a, b in itertools.combinations(list(dict.fromkeys(g + g2)), 2):
                key = tuple(sorted((a, b)))
                c = cls["strong_up"] if (a in core and b in core) else cls["mild_up"]
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    plan["pairs"].append([key[0], key[1], c])
                else:   # coppia gia' vista da un'altra predq: vince la classe piu' forte
                    for p in plan["pairs"]:
                        if (p[0], p[1]) == key and c == cls["strong_up"]:
                            p[2] = c
    return plan


def degrade_plan(plan, keep=0.6, seed=0, class_noise=0.0):
    """Versione "utente medio": tiene a caso una frazione `keep` degli interventi dell'esperto.
    - predq->chunk: per ogni predq si tiene round(keep*n) interventi (almeno 1), scelti a caso;
    - coppie chunk<->chunk: si tiene a caso una frazione `keep` del totale;
    - class_noise: probabilita' di scambiare classe forte/lieve (stessa direzione) in un intervento tenuto (default 0 = solo omissioni).
    Riproducibile con `seed`. Il piano originale non viene modificato."""
    import random
    rng = random.Random(seed)
    cls = plan["classes"]
    swap = {cls["strong_up"]: cls["mild_up"], cls["mild_up"]: cls["strong_up"]}
    if cls.get("strong_down") and cls.get("mild_down"):
        swap.update({cls["strong_down"]: cls["mild_down"], cls["mild_down"]: cls["strong_down"]})
    noisy = lambda c: swap.get(c, c) if rng.random() < class_noise else c
    out = {"classes": dict(cls), "predq": {}, "pairs": [],
           "meta": {"variant": "average_user", "keep": keep, "seed": seed, "class_noise": class_noise}}
    for pid, m in plan["predq"].items():
        items = list(m.items())
        k = max(1, round(keep * len(items))) if items else 0
        out["predq"][pid] = {c: noisy(v) for c, v in rng.sample(items, k)}
    k = round(keep * len(plan["pairs"]))
    out["pairs"] = [[p[0], p[1], noisy(p[2])] for p in rng.sample(plan["pairs"], k)]
    return out


def plan_stats(plan):
    from collections import Counter
    return {"predq_edits": sum(len(m) for m in plan["predq"].values()), "chunk_pairs": len(plan["pairs"]),
            "classes": dict(Counter([c for m in plan["predq"].values() for c in m.values()] + [p[2] for p in plan["pairs"]]))}


def build_expert_plan(queries, gold, sm, curation, bench=None, top_j=None, top_k=None, include_cross=True, penalize_wrong=True):
    """Piano "esperto di dominio": il grafo e' tarato su cio' che i chunk DICONO (curation.GROUPS/CROSS), non sul gold.
    - chunk<->chunk: per ogni gruppo, coppie del nucleo in classe forte, tutte le altre in classe lieve (query-indipendenti);
    - predq->chunk: gold (il primo forte, gli altri lievi) + `support` della curation (lievi) + `away` della curation;
      con `bench`, anche i chunk sbagliati che il retrieval vettoriale mette nei top_j (la curation, se presente, vince).
    Il gold serve solo per sapere quali chunk sono la risposta della predq; non genera le coppie chunk<->chunk."""
    from src.evaluation.benchmark_runner import _rec_info
    cls = pick_classes(sm)
    plan = {"classes": cls, "predq": {}, "pairs": [], "pair_notes": {}, "meta": {"variant": "expert_curated", "include_cross": include_cross}}
    rank = {"s": cls["strong_down"], "m": cls["mild_down"]}
    for q in queries:
        if not q.get("is_predetermined"):
            continue
        g = list(gold[q["id"]]); cur = curation.PREDQ.get(q["id"], {"support": [], "away": {}})
        e = {c: (cls["strong_up"] if i == 0 else cls["mild_up"]) for i, c in enumerate(g)}
        for c in cur["support"]:
            e.setdefault(c, cls["mild_up"])
        if bench is not None and penalize_wrong and cls["mild_down"]:
            bench.h.set_parameters(top_j=top_j or bench.top_j, top_k=top_k or bench.top_k, top_n=bench.top_n)
            bench.h.update_query(new_text=q["query"]); bench.h.run_vectorial_retrieval()
            ranked = [_rec_info(r, i)[0] for i, r in enumerate(bench.h.raw_records)]
            for r_, cid in enumerate(ranked[: (top_j or bench.top_j)]):
                if cid not in e:
                    e[cid] = cls["strong_down"] if r_ < 3 else cls["mild_down"]
        for c, lvl in cur["away"].items():
            if c not in g and rank.get(lvl):
                e[c] = rank[lvl]
        plan["predq"][f"predq_{q['id']}"] = e
    pairs = {}
    groups = dict(curation.GROUPS)
    if include_cross:
        groups.update(curation.CROSS)
    for name, (core, others) in groups.items():
        members = list(dict.fromkeys(core + others))
        for a, b in itertools.combinations(members, 2):
            key = tuple(sorted((a, b)))
            c = cls["strong_up"] if (a in core and b in core) else cls["mild_up"]
            if key not in pairs or (c == cls["strong_up"] and pairs[key] != cls["strong_up"]):
                pairs[key] = c
            plan["pair_notes"].setdefault("|".join(key), []).append(name)
    plan["pairs"] = [[a, b, c] for (a, b), c in sorted(pairs.items())]
    return plan


def save_plan(plan, path):
    Path(path).expanduser().write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")


def load_plan(path):
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def reset_distance_edits(sm):
    """Azzera TUTTE le adiacenze (chunk<->chunk e predq<->chunk). Tag, predq registrate e scala delle classi restano."""
    d = sm.load_data(force_reload=True)
    d["modified_adjacencies"] = {}
    for p in d.get("predetermined_queries", {}).values():
        p["adjacencies"] = {}
    sm.save_data(d)


def apply_plan(sm, plan, reset_first=True):
    """Scrive il piano nel sidecar. Le predq devono essere gia' registrate (bench.register_predetermined())."""
    registered = set(sm.load_data(force_reload=True).get("predetermined_queries", {}))
    missing = [p for p in plan["predq"] if p not in registered]
    if missing:
        raise RuntimeError(f"Predq non registrate nel sidecar: {missing[:3]}... Esegui prima bench.register_predetermined().")
    if reset_first:
        reset_distance_edits(sm)
    edits = [{"chunk_1": pid, "chunk_2": cid, "distance_class": c} for pid, m in plan["predq"].items() for cid, c in m.items()]
    edits += [{"chunk_1": p[0], "chunk_2": p[1], "distance_class": p[2]} for p in plan["pairs"]]
    sm.save_pairwise_class_edits_batch(edits)
    return {"predq_edits": sum(len(m) for m in plan["predq"].values()), "chunk_pairs": len(plan["pairs"])}

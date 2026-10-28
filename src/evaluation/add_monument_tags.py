#!/usr/bin/env python3
"""add_monument_tags.py - struttura a DUE ASSI per il benchmark Colosseo.
Salvare come ~/tesi_graphrag/src/evaluation/add_monument_tags.py

  - ogni CHUNK riceve un tag MONUMENTO (derivato da doc_id, mappa MONUMENT sotto) in testa alla lista, piu' i tag tematici di prima;
  - ogni QUERY ha SEMPRE un conversation tag = il MONUMENTO della conversazione (expected_tag); l'ASPETTO (ex expected_tag
    tematico) passa nei tag manuali (`tags`), insieme a quelli che c'erano gia'.
  - regola per le query il cui gold attraversa piu' monumenti: monumento = quello del PRIMO chunk gold (il piu' importante);
    gli altri monumenti restano raggiungibili solo con tag manuali / distanze (e' proprio dove le distanze servono).
  - PIU' conv tag: le query AMBIGUE (aspetto unico, stabilito dalla conversazione) hanno conv_tag = [monumento, aspetto]; la categoria 1
    richiede TUTTI i conv tag, quindi il pool e' ristretto a (monumento AND aspetto). Eccezioni in MULTI_CONV_EXCLUDE: query il cui gold
    attraversa monumenti/aspetti (AMB_04, AMB_05): conv_tag = [monumento], aspetto nei tag manuali. Specialistiche e multi-dominio:
    conv_tag = [monumento], aspetti nei tag manuali (le multi-dominio hanno piu' aspetti: un conv_tag in piu' escluderebbe gold dal pool).
Predeterminate e normali: stessa estrazione di make_queries.py (stesso seed), quindi gli stessi id del piano di distanze.

Uso: python add_monument_tags.py --chunks data/queries/colosseum/chunks_colosseum_final.json --out data/queries/colosseum_v2 [--seed 42]
"""
import argparse, json, random
from collections import Counter
from pathlib import Path

# NOMI SENZA SPAZI (stessa convenzione dei tag precedenti): cambiali qui se vuoi etichette diverse nell'interfaccia.
MONUMENT = {
    "wiki_colosseum": "Colosseum", "paper_arena_floor": "Colosseum", "web_underground": "Colosseum",
    "wiki_velarium": "Colosseum", "wiki_venatio": "Colosseum", "wiki_naumachia": "Colosseum",
    "wiki_domus_aurea": "DomusAurea", "wiki_colossus_of_nero": "ColossusOfNero",
    "wiki_ludus_magnus": "LudusMagnus", "wiki_arch_of_constantine": "ArchOfConstantine",
}
MONUMENTS = sorted(set(MONUMENT.values()))
MULTI_CONV_CATEGORIES = ("ambiguous",)       # categorie con conv_tag = [monumento, aspetto]
MULTI_CONV_EXCLUDE = {"AMB_04", "AMB_05"}    # ambigue il cui gold attraversa monumenti/aspetti: un solo conv_tag (modificabile)


def tag_chunks(chunks):
    out = []
    for c in chunks:
        m = MONUMENT[c["doc_id"]]                       # KeyError = doc_id nuovo da mappare: voluto
        out.append({**c, "tags": [m] + [t for t in c["tags"] if t != m]})
    return out


def restructure_queries(queries, gold, chunk_monument, multi_cats=MULTI_CONV_CATEGORIES, exclude=MULTI_CONV_EXCLUDE):
    """expected_tag := [monumento del primo gold] (+ [aspetto] per le categorie multi_cats, tranne exclude);
    tags := aspetto precedente + manuali precedenti (senza duplicati), tolti quelli gia' passati nei conv tag."""
    out = []
    for q in queries:
        g = gold[q["id"]]
        mon = chunk_monument[g[0]]
        aspects = list(dict.fromkeys(([q["expected_tag"]] if q.get("expected_tag") else []) + list(q.get("tags", []))))
        spread = sorted({chunk_monument[x] for x in g})
        conv = [mon]
        if q["category"] in multi_cats and q["id"] not in exclude and q.get("expected_tag"):
            conv.append(q["expected_tag"]); aspects = [x for x in aspects if x not in conv]
        out.append({**q, "expected_tag": conv, "tags": aspects, "aspect_tag": q.get("expected_tag"),
                    "gold_monuments": spread})
    return out


def main():
    import importlib.util, sys
    ap = argparse.ArgumentParser(); ap.add_argument("--chunks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--per_cat", type=int, default=4)
    ap.add_argument("--make_queries", default=str(Path(__file__).with_name("make_queries.py")))
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("make_queries", a.make_queries)
    mq = importlib.util.module_from_spec(spec); spec.loader.exec_module(mq)

    chunks = json.load(open(Path(a.chunks).expanduser(), encoding="utf-8"))
    chunks2 = tag_chunks(chunks)
    cm = {c["chunk_id"]: c["tags"][0] for c in chunks2}
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)

    qs, gold = [], {}
    for qid, cat, sub, text, exp, tags, gd, ref in mq.Q:
        assert all(x in cm for x in gd), f"{qid}: gold inesistenti"
        qs.append({"id": qid, "category": cat, "subtype": sub, "query": text, "expected_tag": exp, "tags": tags, "reference": ref})
        gold[qid] = gd
    qs = restructure_queries(qs, gold, cm)

    rng = random.Random(a.seed); pred_ids = set()                       # stessa estrazione di make_queries.main()
    for cat in ("ambiguous", "specialistic", "multi_domain"):
        pred_ids |= set(rng.sample([q["id"] for q in qs if q["category"] == cat], a.per_cat))
    pred = [q for q in qs if q["id"] in pred_ids]; norm = [q for q in qs if q["id"] not in pred_ids]
    dump = lambda name, obj: json.dump(obj, open(out / name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    dump("chunks_colosseum_final.json", chunks2)
    dump("queries_predetermined.json", pred); dump("queries_normal.json", norm)
    dump("eval_queries.json", [{**q, "is_predetermined": q["id"] in pred_ids} for q in qs]); dump("gold_chunks.json", gold)

    print("predeterminate:", sorted(pred_ids))
    print("chunk per monumento:", dict(Counter(cm.values())))
    print("monumento (primo conv_tag):", dict(Counter(q["expected_tag"][0] for q in qs)))
    print("query con piu' conv tag:", sum(len(q["expected_tag"]) > 1 for q in qs), "di", len(qs))
    print("\nid       conv_tag                            tag manuali                  pool cat.1  gold su altri monumenti")
    for q in qs:
        other = [m for m in q["gold_monuments"] if m != q["expected_tag"][0]]
        pool = sum(set(q["expected_tag"]) <= set(c["tags"]) for c in chunks2)
        print(f"{q['id']:7s}  {'+'.join(q['expected_tag']):34s}  {','.join(q['tags']):26s}  {pool:4d}      {','.join(other) or '-'}")


if __name__ == "__main__":
    main()

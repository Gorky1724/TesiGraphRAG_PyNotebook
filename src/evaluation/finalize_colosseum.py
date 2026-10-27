#!/usr/bin/env python3
"""finalize_colosseum.py - applica la revisione manuale (tag + chunk scartati) a chunks_colosseum.json.
Uso:  python finalize_colosseum.py --chunks data/queries/colosseum/chunks_colosseum.json --review colosseum_review.json
Scrive accanto al file di input: chunks_colosseum_final.json (da ingerire), tag_overrides.json, review_report.txt.
Gli id non cambiano (restano 'buchi' nella numerazione dei chunk scartati). Gli chunk non presenti nella revisione
(es. testi aggiunti dopo) tengono i tag provvisori e sono elencati nel report."""
import argparse, json
from pathlib import Path
from collections import Counter
TAGS = ["History", "Architecture", "Spectacles", "Archaeology", "Conservation", "Culture"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--chunks", required=True); ap.add_argument("--review", required=True)
    a = ap.parse_args()
    cp = Path(a.chunks).expanduser(); out = cp.parent
    chunks = json.load(open(cp, encoding="utf-8")); rev = json.load(open(Path(a.review).expanduser(), encoding="utf-8"))
    drop, tags = set(rev["drop"]), rev["tags"]
    for k, v in tags.items():
        assert v and all(x in TAGS for x in v), f"{k}: tag non validi {v}"
    final, pending = [], []
    for c in chunks:
        if c["chunk_id"] in drop: continue
        c = dict(c)
        if c["chunk_id"] in rev.get("text", {}): c["text"] = rev["text"][c["chunk_id"]]   # pulizia manuale (es. intestazioni/didascalie del PDF)
        if c["chunk_id"] in tags: c["tags"] = list(tags[c["chunk_id"]])
        else: pending.append(c["chunk_id"])
        c["true_domain"] = c["tags"][0]
        final.append(c)
    json.dump(final, open(out / "chunks_colosseum_final.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({c["chunk_id"]: c["tags"] for c in final}, open(out / "tag_overrides.json", "w", encoding="utf-8"), indent=1)
    rep = [f"chunk finali: {len(final)} (scartati {len(chunks) - len(final)})", "", "copertura per tag:"]
    for t in TAGS:
        n = sum(t in c["tags"] for c in final); rep.append(f"  {t:13s} {n:3d}  {100 * n / len(final):4.0f}%")
    rep += ["", "chunk per documento:"] + [f"  {d:28s} {n}" for d, n in Counter(c["doc_id"] for c in final).items()]
    rep += ["", f"SENZA revisione manuale (tag provvisori): {pending}"]
    (out / "review_report.txt").write_text("\n".join(rep), encoding="utf-8"); print("\n".join(rep))

if __name__ == "__main__": main()

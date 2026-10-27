#!/usr/bin/env python3
"""
build_colosseum_corpus.py - costruisce il corpus "Colosseo" (chunk + tag provvisori) da Wikipedia e da testi locali.

Uso (sull'hub o su Colab: serve Internet):
    python build_colosseum_corpus.py --out ~/tesi_graphrag/data/queries/colosseum \
        --papers ~/tesi_graphrag/data/raw/colosseum_papers --contact "tuo_nome / universita"

Produce in --out:
    chunks_colosseum.json   formato atteso da json_ingest.ingest_chunks_json
    chunks_preview.tsv      revisione veloce: id, tag, parole, sezione, inizio del testo
    sources_colosseum.json  provenienza: URL, revisione Wikipedia, data di scarico, licenza
    corpus_report.txt       chunk per documento e copertura per tag
    raw/<doc_id>.txt        testo grezzo scaricato (cache: i rilanci non riscaricano; --refetch per forzare)

Revisione dei tag: i tag sono PROVVISORI (regole su titolo di sezione, vedi RULES). Dopo la revisione si possono fissare
chunk per chunk con --overrides tag_overrides.json  ({"chunk_id": ["Tag1", "Tag2"], ...}).

Licenze: il testo di Wikipedia e' CC BY-SA 4.0 (citare titolo, revisione e data: sono in sources_colosseum.json).
Gli altri testi vanno citati in tesi; la cartella data/ e i file .json/.pdf/.txt sono gia' esclusi da .gitignore.
"""
import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- configurazione
TAGS = ["History", "Architecture", "Spectacles", "Archaeology", "Conservation", "Culture"]

# (titolo Wikipedia, doc_id)
WIKI_ARTICLES = [
    ("Colosseum", "wiki_colosseum"),
    ("Domus Aurea", "wiki_domus_aurea"),
    ("Colossus of Nero", "wiki_colossus_of_nero"),
    ("Ludus Magnus", "wiki_ludus_magnus"),
    ("Venatio", "wiki_venatio"),
    ("Naumachia", "wiki_naumachia"),
    ("Velarium", "wiki_velarium"),
    ("Arch of Constantine", "wiki_arch_of_constantine"),
]

# testi locali (PDF leggibili senza OCR, oppure .txt/.md): nome file -> parametri. Adatta i nomi ai tuoi file.
LOCAL_DOCS = {
    "cerone_arena_floor.pdf": {"doc_id": "paper_arena_floor", "tags": ["Archaeology", "Conservation"],
                               "title": "Cerone, Viskovic, Fumagalli, Rea - The Re-formation of Arena Floor Deck in the Colosseum in Roma",
                               "skip_first_words": 40, "max_chunks": 10},
    "firerisk_post_earthquake_fire.txt": {"doc_id": "web_post_earthquake_fire", "tags": ["Conservation", "Architecture"],
                                          "title": "FireRiskHeritage.net - Did Rome's Colosseum Suffer a Post-Earthquake Fire in 271 A.D.?",
                                          "skip_first_words": 0, "max_chunks": 6},
    "colosseum_net_underground.txt": {"doc_id": "web_underground", "tags": ["Archaeology", "Architecture"],
                                      "title": "the-colosseum.net - Underground (the hypogea)",
                                      "skip_first_words": 0, "max_chunks": 8},
    "mordanova_defelice_2020.pdf": {"doc_id": "paper_seismic_dem", "tags": ["Conservation", "Architecture"],
                                    "title": "Mordanova, de Felice - Seismic Assessment of Archaeological Heritage Using Discrete Element Method",
                                    "skip_first_words": 80, "max_chunks": 10},
}

# regole PROVVISORIE per i tag dei chunk Wikipedia: la prima regola che combacia con il percorso della sezione
# (es. "history > construction") decide; altrimenti vale il tag di default del documento.
DEFAULT_TAGS = {
    "wiki_colosseum": ["History"],
    "wiki_domus_aurea": ["History"],
    "wiki_colossus_of_nero": ["History"],
    "wiki_ludus_magnus": ["Spectacles"],
    "wiki_venatio": ["Spectacles"],
    "wiki_naumachia": ["Spectacles"],
    "wiki_velarium": ["Architecture"],
    "wiki_arch_of_constantine": ["History"],
}
RULES = {
    "wiki_colosseum": [
        (r"hypogeum|underground|arena floor|\barena\b", ["Archaeology", "Architecture"]),
        (r"religio|christian|martyr|church|stations of the cross|pilgrim", ["Culture"]),
        (r"flora|plant|botan|vegetation", ["Conservation"]),
        (r"earthquake|stripp|spoil|restor|preserv|conserv|damage|pollution|ruin", ["Conservation"]),
        (r"gladiat|games|event|show|hunt|execution|naval|naumach|spectacle|venatio|entertain|inaugur", ["Spectacles"]),
        (r"exterior|fa[cç]ade|interior|seating|access|architect|design|structure|material|dimension|velarium|awning", ["Architecture"]),
        (r"cultural|influence|legacy|popular|symbol|death penalty|in culture", ["Culture"]),
    ],
    "wiki_domus_aurea": [
        (r"rediscover|excavat|archaeolog", ["Archaeology"]),
        (r"preserv|restor|collapse|conserv|damage|closure|water", ["Conservation"]),
        (r"architect|design|decorat|room|octagon|painting|fresco|description|layout|plan", ["Architecture"]),
    ],
    "wiki_ludus_magnus": [
        (r"excavat|archaeolog|remains|discover|ruin|rediscover", ["Archaeology"]),
        (r"architect|structure|design|building|cells|training arena|layout|description", ["Architecture"]),
    ],
    "wiki_arch_of_constantine": [
        (r"restor|preserv|conserv", ["Conservation"]),
        (r"architect|design|decorat|relief|spolia|description|sculpt|inscription", ["Architecture"]),
    ],
}

SKIP_SECTIONS = {"see also", "references", "notes", "citations", "bibliography", "further reading", "external links",
                 "sources", "footnotes", "gallery", "works cited", "explanatory notes", "footnotes and references"}
MAX_WORDS = 200        # parole massime per chunk
MIN_WORDS = 25         # sotto questa soglia un chunk si unisce al precedente dello stesso documento
MIN_TAIL = 40          # una coda piu' corta di cosi' si unisce al chunk precedente della stessa sezione

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEAD = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")


# --------------------------------------------------------------------------- Wikipedia
def fetch_wikipedia(title, contact=""):
    """Testo semplice di una voce (con titoli di sezione '== X ==') + revisione e data. Una voce per chiamata."""
    import requests
    ua = f"TesiGraphRAG-corpus-builder/1.0 (university thesis project{'; ' + contact if contact else ''})"
    params = {"action": "query", "format": "json", "formatversion": 2, "redirects": 1, "titles": title,
              "prop": "extracts|revisions", "explaintext": 1, "exsectionformat": "wiki", "rvprop": "ids|timestamp"}
    r = requests.get(WIKI_API, params=params, headers={"User-Agent": ua}, timeout=60)
    r.raise_for_status()
    page = r.json()["query"]["pages"][0]
    if page.get("missing"):
        raise LookupError(f"voce Wikipedia inesistente: {title}")
    rev = page["revisions"][0]
    return {"title": page["title"], "text": page["extract"], "revid": rev["revid"], "timestamp": rev["timestamp"]}


# --------------------------------------------------------------------------- parsing e chunking
def parse_sections(text):
    """-> [(percorso: [titoli], testo)]. Il testo prima del primo titolo ha percorso ['Lead']."""
    sections, buf, stack, cur = [], [], [], ["Lead"]

    def flush():
        t = "\n".join(buf).strip()
        if t:
            sections.append((list(cur), t))

    for line in text.splitlines():
        m = HEAD.match(line.strip())
        if m:
            flush()
            buf.clear()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur = [t for _, t in stack]
        else:
            buf.append(line)
    flush()
    return [(p, t) for p, t in sections if not any(x.lower() in SKIP_SECTIONS for x in p)]


def words(s):
    return len(s.split())


def paragraphs(text):
    out = []
    for p in re.split(r"\n\s*\n", text):
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(p)
    return out


def split_sentences(par):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“‘(])", par)


def chunk_paragraphs(pars, max_words=MAX_WORDS):
    """Raggruppa paragrafi fino a max_words; un paragrafo piu' lungo viene spezzato sui confini di frase."""
    units = []
    for p in pars:
        if words(p) <= max_words:
            units.append(p)
            continue
        buf, k = [], 0
        for s in split_sentences(p):
            if buf and k + words(s) > max_words:
                units.append(" ".join(buf))
                buf, k = [], 0
            buf.append(s)
            k += words(s)
        if buf:
            units.append(" ".join(buf))
    chunks, cur, n = [], [], 0
    for u in units:
        w = words(u)
        if cur and n + w > max_words:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(u)
        n += w
    if cur:
        chunks.append("\n\n".join(cur))
    if len(chunks) > 1 and words(chunks[-1]) < MIN_TAIL:       # coda troppo corta: si attacca al chunk precedente
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + "\n\n" + tail
    return chunks


def assign_tags(doc_id, path):
    heading = " > ".join(path).lower()
    for pattern, tags in RULES.get(doc_id, []):
        if re.search(pattern, heading):
            return list(tags)
    return list(DEFAULT_TAGS.get(doc_id, ["History"]))


def build_wiki_chunks(doc_id, text):
    """-> lista di dict {section, tags, text} (senza chunk_id)."""
    out = []
    for path, body in parse_sections(text):
        tags = assign_tags(doc_id, path)
        for ch in chunk_paragraphs(paragraphs(body)):
            if out and words(ch) < MIN_WORDS:                    # frammento minuscolo: si unisce al precedente
                out[-1]["text"] += "\n\n" + ch
            else:
                out.append({"section": " > ".join(path), "tags": tags, "text": ch})
    return out


# --------------------------------------------------------------------------- testi locali (PDF / txt)
def normalize_pdf_text(raw):
    """Ripulisce il testo estratto da un PDF: sillabazione, a capo dentro i paragrafi, riferimenti bibliografici."""
    t = raw.replace("\r", "")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)                           # parole spezzate a fine riga
    m = re.search(r"\n\s*(references|bibliography|acknowledg(?:e)?ments?)\s*\n", t, flags=re.I)
    if m and m.start() > 0.3 * len(t):                               # taglia i riferimenti (solo se nella parte finale)
        t = t[:m.start()]
    lines = [ln.strip() for ln in t.split("\n")]
    lens = [len(ln) for ln in lines if len(ln) > 20]
    med = statistics.median(lens) if lens else 80
    paras, cur = [], []
    for ln in lines:
        if not ln or re.fullmatch(r"\d{1,4}", ln):                   # riga vuota o solo numero di pagina
            if cur:
                paras.append(" ".join(cur)); cur = []
            continue
        cur.append(ln)
        if re.search(r"[.!?:]$", ln) and len(ln) < 0.65 * med:       # riga corta che chiude una frase: fine paragrafo
            paras.append(" ".join(cur)); cur = []
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(paras)


def read_local_text(path):
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        import fitz  # pymupdf (gia' in requirements.txt)
        with fitz.open(p) as doc:
            raw = "\n".join(page.get_text("text") for page in doc)
        return normalize_pdf_text(raw)
    return normalize_pdf_text(p.read_text(encoding="utf-8", errors="ignore"))


def build_local_chunks(text, spec):
    pars = paragraphs(text)
    skip = spec.get("skip_first_words", 0)
    while skip > 0 and pars:                                          # salta titolo/autori/affiliazioni all'inizio
        w = words(pars[0])
        if w > skip and skip < 0.5 * w:
            break
        skip -= w
        pars.pop(0)
    chunks = chunk_paragraphs(pars)
    chunks = [c for c in chunks if words(c) >= MIN_WORDS][: spec.get("max_chunks", 10)]
    return [{"section": "full text", "tags": list(spec["tags"]), "text": c} for c in chunks]


# --------------------------------------------------------------------------- orchestrazione
def validate_tags(tags, where):
    bad = [t for t in tags if t not in TAGS]
    if bad:
        raise ValueError(f"{where}: tag non validi {bad}; validi: {TAGS}")


def assemble(out_dir, papers_dir=None, contact="", refetch=False, overrides_path=None, fetcher=fetch_wikipedia):
    out_dir = Path(out_dir).expanduser()
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    chunks, sources, seen_titles, warnings = [], [], set(), []

    for title, doc_id in WIKI_ARTICLES:
        cache, meta = raw_dir / f"{doc_id}.txt", raw_dir / f"{doc_id}.meta.json"
        if cache.exists() and meta.exists() and not refetch:
            info = json.loads(meta.read_text(encoding="utf-8"))
            text = cache.read_text(encoding="utf-8")
        else:
            try:
                got = fetcher(title, contact)
            except Exception as e:                                    # voce mancante / rete: si prosegue con le altre
                warnings.append(f"{doc_id}: non scaricata ({type(e).__name__}: {e})")
                continue
            text = got["text"]
            info = {"title": got["title"], "revid": got["revid"], "timestamp": got["timestamp"],
                    "retrieved": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
            cache.write_text(text, encoding="utf-8")
            meta.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
            time.sleep(1)                                             # cortesia verso l'API
        if info["title"] in seen_titles:                              # un redirect puo' riportare a una voce gia' presa
            warnings.append(f"{doc_id}: '{title}' porta a '{info['title']}' gia' inclusa: saltata")
            continue
        seen_titles.add(info["title"])
        parts = build_wiki_chunks(doc_id, text)
        if not parts:
            warnings.append(f"{doc_id}: nessun chunk prodotto")
            continue
        for i, c in enumerate(parts, 1):
            chunks.append({"doc_id": doc_id, "chunk_index": i, "source_title": info["title"], **c})
        sources.append({"doc_id": doc_id, "title": info["title"], "type": "wikipedia",
                        "url": "https://en.wikipedia.org/wiki/" + info["title"].replace(" ", "_"),
                        "revision_id": info["revid"], "revision_timestamp": info["timestamp"],
                        "retrieved": info["retrieved"], "license": "CC BY-SA 4.0", "n_chunks": len(parts)})

    if papers_dir and Path(papers_dir).expanduser().exists():
        pdir = Path(papers_dir).expanduser()
        for fname, spec in LOCAL_DOCS.items():
            f = pdir / fname
            if not f.exists():
                warnings.append(f"{spec['doc_id']}: file '{fname}' non trovato in {pdir} (saltato)")
                continue
            parts = build_local_chunks(read_local_text(f), spec)
            for i, c in enumerate(parts, 1):
                chunks.append({"doc_id": spec["doc_id"], "chunk_index": i, "source_title": spec["title"], **c})
            sources.append({"doc_id": spec["doc_id"], "title": spec["title"], "type": "local", "file": fname,
                            "license": "da verificare (citare in tesi; non ridistribuire)", "n_chunks": len(parts)})
        known = set(LOCAL_DOCS)
        for f in pdir.iterdir():
            if f.is_file() and f.name not in known:
                warnings.append(f"file ignorato in {pdir}: '{f.name}' (non e' in LOCAL_DOCS)")

    overrides = {}
    if overrides_path:
        overrides = json.loads(Path(overrides_path).expanduser().read_text(encoding="utf-8"))
    for c in chunks:
        c["chunk_id"] = f"{c['doc_id']}_{c['chunk_index']:02d}"
        validate_tags(c["tags"], c["chunk_id"])
    ids = {c["chunk_id"] for c in chunks}
    unknown = [k for k in overrides if k not in ids]
    if unknown:
        raise ValueError(f"overrides per chunk inesistenti: {unknown[:10]}")
    for c in chunks:
        if c["chunk_id"] in overrides:
            validate_tags(overrides[c["chunk_id"]], c["chunk_id"])
            c["tags"] = list(overrides[c["chunk_id"]])
        c["true_domain"] = c["tags"][0]
        c["text"] = c["text"].strip()

    final = [{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "true_domain": c["true_domain"],
              "chunk_index": c["chunk_index"], "tags": c["tags"], "text": c["text"],
              "section": c["section"], "source_title": c["source_title"]} for c in chunks]
    (out_dir / "chunks_colosseum.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "sources_colosseum.json").write_text(json.dumps(sources, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out_dir / "chunks_preview.tsv", "w", encoding="utf-8") as f:
        f.write("chunk_id\ttags\twords\tsection\tstart\n")
        for c in final:
            f.write(f"{c['chunk_id']}\t{'+'.join(c['tags'])}\t{words(c['text'])}\t{c['section']}\t"
                    f"{c['text'][:140].replace(chr(10), ' ')}\n")

    rep = [f"chunk totali: {len(final)}  (obiettivo ~80-110)", "", "per documento:"]
    for s in sources:
        rep.append(f"  {s['doc_id']:28s} {s['n_chunks']:3d}  {s['title']}")
    rep += ["", "copertura per tag (quota di chunk che hanno il tag; obiettivo ~12-30% ciascuno):"]
    for t in TAGS:
        n = sum(t in c["tags"] for c in final)
        rep.append(f"  {t:13s} {n:3d} chunk  {100 * n / max(len(final), 1):4.0f}%")
    wl = [words(c["text"]) for c in final]
    if wl:
        rep += ["", f"parole per chunk: min {min(wl)}, mediana {int(statistics.median(wl))}, max {max(wl)}"]
    if warnings:
        rep += ["", "AVVISI:"] + [f"  - {w}" for w in warnings]
    (out_dir / "corpus_report.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    return final


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="cartella di output (es. ~/tesi_graphrag/data/queries/colosseum)")
    ap.add_argument("--papers", default=None, help="cartella con i PDF/txt locali (vedi LOCAL_DOCS)")
    ap.add_argument("--contact", default="", help="contatto da mettere nello User-Agent verso Wikipedia (nome/universita')")
    ap.add_argument("--refetch", action="store_true", help="riscarica Wikipedia anche se in cache")
    ap.add_argument("--overrides", default=None, help="JSON {chunk_id: [tag, ...]} che sostituisce i tag provvisori")
    a = ap.parse_args(argv)
    assemble(a.out, a.papers, a.contact, a.refetch, a.overrides)


if __name__ == "__main__":
    sys.exit(main())


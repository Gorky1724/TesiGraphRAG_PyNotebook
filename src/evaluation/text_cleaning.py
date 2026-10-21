import re
from collections import Counter

CJK = re.compile(r"[぀-ヿ一-鿿가-힯]")
REF = re.compile(r"^\s*(references|bibliography)\s*$", re.I)
CAP = re.compile(r"^\s*(figure|fig\.)\s*\d+\s*[:.]", re.I)
PAGENUM = re.compile(r"\d{1,5}")

# --- regole aggiuntive (attive solo con extended=True, vedi clean_pages) -------------------------------------------
ACK = re.compile(r"^\s*acknowledg(e)?ments?\b", re.I)                       # "Acknowledgment(s)" -> inizio della coda
PAGELABEL = re.compile(r"\d{1,3}:\d{1,3}")                                  # etichette di pagina tipo LIPIcs "35:2"
BOILERPLATE = re.compile(r"^Downloaded .*Redistribution subject to SIAM", re.I)   # riga di copyright SIAM (>120 car.)
TOC_LEADER = re.compile(r"(\.\s){5,}")                                      # righe d'indice "Introduction . . . . 223"
LICENSE = re.compile(r"(?ms)^Funding Open access.*?licenses/by/4\.0/\.[ \t]*$")   # blocco licenza Springer


def clean_pages(pages, min_chars=200, extended=False, first_page_start=None):
    """Pulisce le pagine caricate (Document LangChain): header/footer ripetuti, numeri di pagina,
    didascalie 'Figure N:', righe con CJK e coda 'References'.

    extended=False (default) = comportamento originale: la coda 'References' taglia solo il resto della
    pagina in cui compare (le pagine successive restano).
    extended=True aggiunge:
      - dopo 'References'/'Bibliography'/'Acknowledgment(s)' scarta anche TUTTE le pagine successive;
      - righe-indice con puntini di guida, etichette di pagina 'NN:NN', riga di copyright SIAM, blocco licenza Springer.
    first_page_start: regex; nella prima pagina si scartano le righe che precedono la prima riga che la soddisfa
    (serve per le pagine d'apertura dei capitoli di handbook, che iniziano con indice e affiliazioni).
    """
    cnt = Counter(l.strip() for p in pages for l in set(p.page_content.split("\n")) if l.strip())
    repeated = {l for l, n in cnt.items() if n >= 3 and len(l) < 120}
    start_re = re.compile(first_page_start) if first_page_start else None
    out, tail_reached = [], False
    for n, p in enumerate(pages):
        if tail_reached:
            continue
        text = LICENSE.sub("", p.page_content) if extended else p.page_content
        lines = text.split("\n")
        if start_re and n == 0:
            idx = next((i for i, l in enumerate(lines) if start_re.match(l.strip())), None)
            if idx is not None:
                lines = lines[idx:]
        kept, in_caption = [], False
        for line in lines:
            s = line.strip()
            if REF.match(s) or (extended and ACK.match(s)):
                tail_reached = extended
                break
            if CAP.match(s):
                in_caption = True
            if in_caption:
                if s.endswith("."):
                    in_caption = False
                continue
            if not s or s in repeated or CJK.search(s) or PAGENUM.fullmatch(s):
                continue
            if extended and (PAGELABEL.fullmatch(s) or BOILERPLATE.match(s) or TOC_LEADER.search(s)):
                continue
            kept.append(line)
        p.page_content = "\n".join(kept)
        if len(p.page_content) >= min_chars:
            out.append(p)
    return out


# --- DataSet "dst" (12 PDF: libro + handbook + paper di Tamassia, Rahman, Didimo, Bekos, ...) ---------------------
# pages: intervalli 1-based inclusivi di pagine PDF da TENERE (metadata["page"] resta quello originale, 0-based).
# first_page_start: vedi clean_pages. I documenti non elencati si puliscono con la sola clean_pages(extended=True).
DST_RULES = {
    # Libro di 400 pagine: si tengono solo le parti sull'orthogonal drawing/bend (prefazione con le applicazioni,
    # cap. 1-2, sez. 4.9, cap. 5, 7, 8 senza esercizi, Appendice A "Bounds"); fuori: copertina/indici, cap. 3, 6, 9, 10, 11,
    # resto del cap. 4, bibliografia e indice analitico.
    "Di Batista, Eades, Tamassia, Tollis - Graph Drawing": {
        "pages": [(7, 8), (12, 49), (138, 141), (145, 177), (222, 244), (246, 269), (345, 362)],
    },
    "handbook-Planar-Orthogonal-and-Polyline-Drawing-Algorithms": {"first_page_start": r"One can assess"},
    "handbook-rectangular-drawing-algorithms": {"first_page_start": r"A rectangular drawing of a plane graph"},
}


def clean_dst_pages(pages, filename):
    """Selezione pagine (se prevista) + pulizia estesa per un PDF del dataset dst. filename: nome senza '.pdf'."""
    rule = DST_RULES.get(filename, {})
    ranges = rule.get("pages")
    if ranges:
        pages = [p for p in pages if any(a <= int(p.metadata.get("page", 0)) + 1 <= b for a, b in ranges)]
    return clean_pages(pages, extended=True, first_page_start=rule.get("first_page_start"))

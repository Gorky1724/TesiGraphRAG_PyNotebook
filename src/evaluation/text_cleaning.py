import re
from collections import Counter

CJK = re.compile(r"[぀-ヿ一-鿿가-힯]")
REF = re.compile(r"^\s*(references|bibliography)\s*$", re.I)
CAP = re.compile(r"^\s*(figure|fig\.)\s*\d+\s*[:.]", re.I)
PAGENUM = re.compile(r"\d{1,5}")


def clean_pages(pages, min_chars=200):
    """Pulisce le pagine caricate (Document LangChain): header/footer ripetuti, numeri di pagina,
    didascalie 'Figure N:', righe con CJK e coda 'References' (taglia solo il resto della pagina)."""
    cnt = Counter(l.strip() for p in pages for l in set(p.page_content.split("\n")) if l.strip())
    repeated = {l for l, n in cnt.items() if n >= 3 and len(l) < 120}
    out = []
    for p in pages:
        kept, in_caption = [], False
        for line in p.page_content.split("\n"):
            s = line.strip()
            if REF.match(s):
                break
            if CAP.match(s):
                in_caption = True
            if in_caption:
                if s.endswith("."):
                    in_caption = False
                continue
            if not s or s in repeated or CJK.search(s) or PAGENUM.fullmatch(s):
                continue
            kept.append(line)
        p.page_content = "\n".join(kept)
        if len(p.page_content) >= min_chars:
            out.append(p)
    return out

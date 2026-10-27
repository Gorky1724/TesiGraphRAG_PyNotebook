"""degrade_tags - versione "utente medio" dei TAG (il corrispettivo di distance_presets.degrade_plan per i tag).
Parte dai tag dell'esperto (chunks_colosseum.json) e produce una copia con errori realistici, riproducibile col seed:
  - missing_frac : quota di chunk lasciati SENZA tag (l'utente non ha fatto in tempo a taggarli);
  - wrong_frac   : tra i chunk taggati, quota con un tag SBAGLIATO (sostituito con un altro tag a caso);
  - drop_secondary: nei chunk con piu' tag, l'utente tiene di solito solo il primo.
Il piano di distanze si degrada a parte con degrade_plan(); i due si combinano nel run 'post_edit_user'."""
import random
from collections import Counter


def degrade_tags(chunks, all_tags, missing_frac=0.30, wrong_frac=0.15, drop_secondary=True, seed=0):
    rng = random.Random(seed)
    out, log = [], Counter()
    for c in chunks:
        d = dict(c)
        tags = list(c["tags"])
        if rng.random() < missing_frac:
            d["tags"] = []; log["missing"] += 1
        else:
            if drop_secondary and len(tags) > 1:
                tags = tags[:1]; log["secondary_dropped"] += 1
            if tags and rng.random() < wrong_frac:
                i = rng.randrange(len(tags))
                tags[i] = rng.choice([t for t in all_tags if t not in tags]); log["wrong"] += 1
            d["tags"] = tags
        out.append(d)
    return out, dict(log)

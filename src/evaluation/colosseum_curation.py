"""
Taratura "da esperto di dominio" del grafo Colosseo (scritta a mano leggendo i chunk, non derivata dal gold).
Salvare come ~/tesi_graphrag/src/evaluation/colosseum_curation.py   (stessa struttura di armstrong_curation.py)

GROUPS: gruppi di chunk che un gestore del sito collega perche' raccontano lo stesso episodio/tema, INDIPENDENTEMENTE dalle query.
        "core" = nucleo (coppie in classe FORTE); gli altri "members" si collegano a tutti in classe LIEVE.
CROSS : gruppi trasversali ai documenti (stesso tipo di informazione: restauri, dimensioni, scavi...). Disattivabili.
PREDQ : per ogni predeterminata, interventi del gestore guardando la query:
        support = chunk non gold ma utili (avvicinati lievi); away = chunk di una lettura DIVERSA dell'ambiguita' (s = forte, m = lieve).
Prefissi: wc Colosseum, da Domus Aurea, cn Colossus of Nero, lm Ludus Magnus, vn Venatio, nm Naumachia, vl Velarium,
          ac Arch of Constantine, pa articolo Cerone (pavimento dell'arena), ug sotterranei (the-colosseum.net)
"""
_P = {"wc": "wiki_colosseum", "da": "wiki_domus_aurea", "cn": "wiki_colossus_of_nero", "lm": "wiki_ludus_magnus", "vn": "wiki_venatio",
      "nm": "wiki_naumachia", "vl": "wiki_velarium", "ac": "wiki_arch_of_constantine", "pa": "paper_arena_floor", "ug": "web_underground"}
def K(*s): return [f"{_P[x.split('_')[0]]}_{x.split('_')[1]}" for x in s]

GROUPS = {
 # --- storia del Colosseo
 "name_origin":        (K("wc_03","wc_04"),                      K("cn_04","da_09","wc_02")),
 "site_and_flavians":  (K("wc_07","wc_08","da_03"),              K("da_02","wc_09","cn_01")),
 "building_inaug":     (K("wc_09","wc_10"),                      K("wc_01","wc_08")),
 "late_antiquity":     (K("wc_10","wc_11"),                      K("wc_12","ug_05","pa_03")),
 "medieval_reuse":     (K("wc_11","wc_12"),                      K("wc_13","wc_33","wc_02")),
 "christian_site":     (K("wc_32","wc_33","wc_34","wc_35"),      K("wc_13","wc_02")),
 "modern_restoration": (K("wc_14","wc_31"),                      K("wc_26","wc_30","wc_12","pa_07")),
 "plants_and_decay":   (K("wc_36","wc_14"),                      K("wc_17")),
 "modern_use":         (K("wc_15","wc_30","wc_35"),              K("wc_02","wc_31","wc_37")),
 # --- architettura
 "exterior":           (K("wc_16","wc_17","wc_18"),              K("wc_19","vl_01","vl_02")),
 "velarium":           (K("wc_18","vl_01","vl_02"),              K("wc_16")),
 "crowd_and_seating":  (K("wc_19","wc_20","wc_21","wc_22"),      K("wc_01")),
 "arena_hypogeum":     (K("wc_23","wc_24","ug_02","ug_05"),      K("ug_01","ug_03","ug_04","pa_01","pa_05")),
 # --- spettacoli
 "games_and_hunts":    (K("wc_27","vn_02","vn_03"),              K("vn_04","vn_05","wc_29","wc_10")),
 "executions":         (K("wc_29","vn_05","wc_27"),              K("vn_04")),
 "arena_flooding":     (K("wc_28","wc_24","ug_03","ug_04","nm_09"), K("nm_08","nm_10","nm_11")),
 "naumachia_roman":    (K("nm_02","nm_03","nm_05","nm_06"),      K("nm_01","nm_07","nm_08")),
 "naumachia_amph":     (K("nm_08","nm_09","nm_10","nm_11"),      K("nm_12","nm_13","nm_14","wc_28")),
 "naumachia_decline":  (K("nm_12","nm_13","nm_14"),              K("nm_09")),
 "gladiator_school":   (K("lm_01","lm_03","lm_02","wc_25"),      K("lm_04","lm_08","lm_10","wc_23")),
 "ludus_remains":      (K("lm_06","lm_11","lm_07","lm_08","lm_09"), K("lm_02","lm_04")),
 # --- Nerone e Domus Aurea
 "colossus":           (K("cn_01","cn_02","cn_03"),              K("da_09","da_08","wc_04","cn_04")),
 "domus_construction": (K("da_02","da_07","da_08","da_11"),      K("da_01","da_10","da_13")),
 "domus_decoration":   (K("da_13","da_14","da_19","da_20"),      K("da_05","da_18")),
 "domus_rediscovery":  (K("da_04","da_05"),                      K("da_06","da_14")),
 "domus_decay":        (K("da_06","da_20"),                      K("da_04")),
 "octagonal_room":     (K("da_15","da_16","da_17"),              K("da_11")),
 "sphinx_room":        (K("da_18","da_19","da_20"),              []),
 # --- Arco di Costantino
 "arch_commission":    (K("ac_01","ac_03","ac_04","ac_06","ac_07"), K("ac_05","ac_08")),
 "arch_sculpture":     (K("ac_02","ac_09","ac_10","ac_13","ac_14"), K("ac_16","ac_18","ac_19","ac_20")),
 "arch_decoration":    (K("ac_16","ac_18","ac_19","ac_20","ac_25"), K("ac_14")),
 # --- pavimento dell'arena (Cerone et al.)
 "arena_reconstruction": (K("pa_01","pa_02","pa_07","pa_08","pa_10"), K("pa_03","pa_06","wc_31")),
 "arena_foundations":    (K("pa_03","pa_04","pa_05","pa_06"),         K("ug_02","ug_03","wc_23")),
}

CROSS = {
 "X_restorations":  (K("wc_31","da_06","ac_05"),         K("wc_14","pa_07")),
 "X_dimensions":    (K("wc_16","ac_01","lm_07"),         K("cn_01","wc_23")),
 "X_excavations":   (K("da_04","lm_06","cn_03"),         K("pa_01","wc_14")),
 "X_sun_statue":    (K("cn_02","da_09"),                 K("ac_08","wc_04")),
 "X_mussolini":     (K("wc_14","cn_03"),                 K("lm_06")),
 "X_ancient_texts": (K("da_07","nm_08"),                 K("lm_10","da_14","wc_10")),
 "X_hadrian":       (K("cn_02","da_09"),                 K("ac_19","lm_04")),
}

PREDQ = {
 "AMB_01": {"support": K("pa_02"),               "away": {**{c: "s" for c in K("wc_23")}, **{c: "m" for c in K("ug_03")}}},
 "AMB_02": {"support": K("wc_10"),               "away": {c: "m" for c in K("wc_30")}},
 "AMB_03": {"support": K("nm_09"),               "away": {c: "m" for c in K("wc_28","nm_10")}},
 "AMB_04": {"support": K("cn_01"),               "away": {**{c: "s" for c in K("wc_04")}, **{c: "m" for c in K("cn_04")}}},
 "AMB_05": {"support": K("wc_03"),               "away": {c: "m" for c in K("cn_03","cn_02")}},
 "AMB_06": {"support": K("wc_11","wc_10"),       "away": {}},
 "AMB_07": {"support": K("da_13"),               "away": {**{c: "s" for c in K("da_06")}, **{c: "m" for c in K("da_20")}}},
 "AMB_08": {"support": K("lm_02","lm_01"),       "away": {c: "m" for c in K("lm_11","lm_06","wc_25")}},
 "AMB_09": {"support": K("lm_07","lm_08"),       "away": {c: "m" for c in K("lm_01","lm_03")}},
 "AMB_10": {"support": K("wc_26","ac_05"),       "away": {}},
 "AMB_11": {"support": K("wc_21"),               "away": {c: "m" for c in K("lm_03")}},
 "AMB_12": {"support": K("nm_10","nm_11","ug_04"), "away": {c: "m" for c in K("vl_02","wc_18")}},
 "AMB_13": {"support": [],                       "away": {**{c: "s" for c in K("wc_28")}, **{c: "m" for c in K("nm_09")}}},
 "AMB_14": {"support": K("wc_35","wc_31"),       "away": {**{c: "s" for c in K("wc_27")}, **{c: "m" for c in K("wc_29")}}},
 "SPE_01": {"support": K("wc_10"),               "away": {}},
 "SPE_02": {"support": K("wc_21","wc_20"),       "away": {}},
 "SPE_03": {"support": K("vn_02","vn_04"),       "away": {}},
 "SPE_04": {"support": K("pa_04","ug_02"),       "away": {}},
 "SPE_05": {"support": K("wc_14"),               "away": {}},
 "SPE_06": {"support": K("ug_03"),               "away": {}},
 "SPE_07": {"support": K("ac_03"),               "away": {}},
 "SPE_08": {"support": K("cn_03","cn_01"),       "away": {}},
 "SPE_09": {"support": K("nm_05","nm_06"),       "away": {}},
 "SPE_10": {"support": K("wc_31"),               "away": {}},
 "SPE_11": {"support": K("wc_14"),               "away": {}},
 "SPE_12": {"support": [],                       "away": {}},
 "SPE_13": {"support": K("da_20"),               "away": {}},
 "SPE_14": {"support": K("da_02"),               "away": {}},
 "MUL_01": {"support": K("wc_10","wc_26"),       "away": {}},
 "MUL_02": {"support": K("nm_10","nm_08","wc_23"), "away": {}},
 "MUL_03": {"support": K("wc_19"),               "away": {}},
 "MUL_04": {"support": K("wc_35","wc_02"),       "away": {}},
 "MUL_05": {"support": K("pa_02","pa_01","pa_06"), "away": {}},
 "MUL_06": {"support": K("wc_30","ug_02"),       "away": {}},
 "MUL_07": {"support": K("cn_01","da_02","da_08"), "away": {}},
 "MUL_08": {"support": K("vn_05","wc_29","wc_28"), "away": {}},
 "MUL_09": {"support": K("wc_14"),               "away": {}},
 "MUL_10": {"support": K("lm_01","wc_24"),       "away": {}},
 "MUL_11": {"support": K("wc_30","wc_31"),       "away": {}},
 "MUL_12": {"support": K("da_01","da_07"),       "away": {}},
 "MUL_13": {"support": K("ac_07","ac_08","ac_10"), "away": {}},
 "MUL_14": {"support": K("da_11"),               "away": {}},
}

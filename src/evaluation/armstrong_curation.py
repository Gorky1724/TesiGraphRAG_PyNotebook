"""
Taratura "da esperto di dominio" del grafo Armstrong (scritta a mano leggendo TUTTI i chunk, non derivata dal gold).
Salvare come ~/tesi_graphrag/src/evaluation/armstrong_curation.py

GROUPS: gruppi di chunk che un esperto collega perche' raccontano la stessa storia/evento/periodo, INDIPENDENTEMENTE dalle query.
        "core" = nucleo del gruppo (coppie in classe FORTE); gli altri "members" si collegano a tutti in classe LIEVE.
CROSS : gruppi trasversali ai tre Armstrong (stesso tipo di informazione: onorificenze, nascita, squadra...). Disattivabili.
PREDQ : per ogni predeterminata, interventi che un esperto farebbe guardando la query:
        support = chunk non gold ma utili (avvicinati lievi); away = chunk simili ma fuorvianti (s = forte, m = lieve).
Tutto modificabile: il piano generato (JSON) resta rivedibile prima di applicarlo.
"""
L = lambda *n: [f"louis_armstrong_{i:02d}" for i in n]
C = lambda *n: [f"lance_armstrong_{i:02d}" for i in n]
N = lambda *n: [f"neil_armstrong_{i:02d}" for i in n]

# ---- storie interne a ciascun Armstrong: nome -> (core, altri membri)
GROUPS = {
 # Louis
 "L_childhood":      (L(1, 2),          L(3, 23)),          # infanzia; l'autobiografia 'My Life in New Orleans' racconta l'infanzia
 "L_apprenticeship": (L(3, 4),          L(5, 6)),           # Oliver, riverboat, Chicago, New York
 "L_20s_recordings": (L(6, 7, 8),       L(4, 5, 9)),        # Hot Five/Seven, scat, West End Blues
 "L_innovation":     (L(8, 9),          L(6, 10)),          # il solista al centro del jazz
 "L_voice":          (L(7, 10),         L(24)),             # scat e stile vocale
 "L_expansion":      (L(11, 12),        L(13, 14)),         # anni 30-50: tour, All Stars, film, diplomazia
 "L_sixties_hits":   (L(15, 16, 17),    L(21)),             # Hello Dolly, What a Wonderful World, anni 60
 "L_race_politics":  (L(18, 19),        L(14)),             # Little Rock, critiche, tour cancellato
 "L_end_legacy":     (L(20, 22),        L(21, 23, 24)),     # morte, casa-museo, premi, libri, soprannomi
 # Lance
 "C_youth":          (C(1, 2),          C(3)),
 "C_early_pro":      (C(3, 4),          C(2)),
 "C_cancer_return":  (C(5, 23, 7),      C(6, 4)),           # diagnosi, Cofidis/Postal, 1999
 "C_foundation":     (C(5, 6),          C(19)),             # fondazione Livestrong (+ dimissioni da presidente)
 "C_tour_wins":      (C(8, 22),         C(7, 9, 12, 14, 24)),
 "C_tactics_team":   (C(10, 11),        C(9, 12, 8)),       # preparazione, squadra, 'the Look', rivali
 "C_doping_arc":     (C(16, 17, 18),    C(15, 19, 20, 24, 13, 21)),
 "C_stripped":       (C(17, 13),        C(20, 16)),         # titoli e medaglia revocati
 # Neil
 "N_youth":          (N(2, 3),          N(1)),
 "N_test_pilot":     (N(4, 5),          N(3, 22, 6)),
 "N_selection":      (N(5, 6),          N(7, 4)),
 "N_gemini8":        (N(7, 8),          N(6, 9, 22)),
 "N_emergencies":    (N(8, 9),          N(11, 22)),         # gemini 8, LLRV, allarmi dell'allunaggio: sangue freddo
 "N_apollo11":       (N(11, 12, 13),    N(10, 14, 24, 23)),
 "N_apollo_prep":    (N(10, 24),        N(9, 22)),
 "N_postwar_life":   (N(16, 18),        N(17, 19)),
 "N_nasa_advocacy":  (N(17, 19),        N(16)),
 "N_death_legacy":   (N(20, 21),        N(15, 23, 18)),
 "N_honors":         (N(15, 23),        N(12)),
}

# ---- gruppi trasversali (stesso tipo di informazione nei tre Armstrong)
CROSS = {
 "X_birth":      (L(1) + C(1) + N(1),        []),
 "X_start":      (L(2) + C(2) + N(2),        L(3) + N(3)),
 "X_honors":     (L(21) + C(20) + N(15),     []),
 "X_memorials":  (L(22) + N(21, 23) + C(21), []),
 "X_health":     (C(5) + L(20) + N(20),      []),
 "X_danger":     (C(5) + N(8, 9),            []),
 "X_records":    (L(15) + C(22) + N(12, 7),  []),
 "X_teams":      (L(6, 12) + C(11) + N(10, 7), L(4) + C(10)),
 "X_sixties":    (L(17) + N(7, 12, 10),      L(15, 16) + C(1)),
}

# ---- interventi per predeterminata: support (non gold, utili) e away (simili ma fuorvianti; s = forte, m = lieve)
PREDQ = {
 "AMB_P01": {"support": L(3),        "away": {**{c: "s" for c in N(1) + C(1)}, **{c: "m" for c in N(2) + C(2)}}},
 "AMB_P02": {"support": C(22),       "away": {c: "m" for c in C(14, 3)}},
 "AMB_P03": {"support": N(9),        "away": {c: "m" for c in N(11, 5)}},
 "AMB_P04": {"support": L(18),       "away": {c: "m" for c in C(15) + N(19, 18)}},
 "AMB_P05": {"support": C(15, 19),   "away": {c: "m" for c in L(18, 19) + N(19)}},
 "AMB_P06": {"support": [],          "away": {c: "s" for c in L(20)}},
 "SPE_P01": {"support": L(4, 5),     "away": {c: "m" for c in L(12)}},
 "SPE_P02": {"support": L(17, 21),   "away": {c: "m" for c in L(16, 13)}},
 "SPE_P03": {"support": C(9),        "away": {}},
 "SPE_P04": {"support": C(7),        "away": {c: "m" for c in L(20) + N(20, 9)}},
 "SPE_P05": {"support": N(12, 10),   "away": {c: "m" for c in N(8, 9)}},
 "SPE_P06": {"support": N(4, 6),     "away": {c: "m" for c in N(3)}},
 "MUL_P01": {"support": C(17),       "away": {}},
 "MUL_P02": {"support": L(3) + N(3), "away": {}},
 "MUL_P03": {"support": [],          "away": {}},
 "MUL_P04": {"support": C(8) + N(10), "away": {}},
 "MUL_P05": {"support": L(16) + N(9, 11, 15), "away": {}},
 "MUL_P06": {"support": L(12),       "away": {}},
}

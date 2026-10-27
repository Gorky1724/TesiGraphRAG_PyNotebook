#!/usr/bin/env python3
"""make_queries.py - genera dal corpus Colosseo: queries_predetermined.json (~30%), queries_normal.json, eval_queries.json, gold_chunks.json.
Uso: python make_queries.py --chunks data/queries/colosseum/chunks_colosseum_final.json --out data/queries/colosseum [--seed 42]
Le 42 query sono definite qui sotto (testo, categoria, tag di conversazione, gold in ordine di importanza, risposta di riferimento)."""
import argparse, json, random
from pathlib import Path
P = {"wc": "wiki_colosseum", "da": "wiki_domus_aurea", "cn": "wiki_colossus_of_nero", "lm": "wiki_ludus_magnus", "vn": "wiki_venatio",
     "nm": "wiki_naumachia", "vl": "wiki_velarium", "ac": "wiki_arch_of_constantine", "pa": "paper_arena_floor", "ug": "web_underground"}
def g(*s): return [f"{P[x.split('_')[0]]}_{x.split('_')[1]}" for x in s]
H, A, S, R, C, U = "History", "Architecture", "Spectacles", "Archaeology", "Conservation", "Culture"

# id, categoria, sottotipo, query, expected_tag (tag di conversazione), tags (interessi del visitatore), gold, reference
Q = [
# ---------------- AMBIGUE: la stessa formulazione ha piu' letture; il tag di conversazione ne seleziona una
("AMB_01","ambiguous","modern_vs_ancient","What is the arena floor made of?",C,[],g("pa_10","pa_08","pa_07"),
 "The partially rebuilt arena floor uses laminated wood, reinforced with carbon-fibre and aramid-fibre where needed, assembled with stainless steel plates and bolts so that it can be dismounted (reversibility). The ancient floor was wood covered with sand."),
("AMB_02","ambiguous","build_vs_use","Who built the part under the arena, and when?",R,[],g("ug_03","ug_02","wc_23"),
 "The hypogeum was ordered by Domitian; by the dates stamped on the bricks the walls supporting the arena were first built about ten years after the inauguration (80 AD), and were later modified in four stages."),
("AMB_03","ambiguous","evidence_vs_accounts","How did water get into the arena?",R,[],g("ug_04","ug_03","wc_24"),
 "Before Domitian's hypogeum, the arena may have been filled with water; traces of 'locks' to make it waterproof up to about 5 feet and two spaces ('darsene') where ships were kept have been found. The accounts suggest a connection to a nearby aqueduct."),
("AMB_04","ambiguous","statue_vs_name","What happened to the Colossus?",H,[],g("cn_03","cn_02","da_09"),
 "The statue was moved by Hadrian next to the Colosseum, remained until at least the 4th-century sources, and then disappeared, possibly in the Sack of Rome in 410 or a 5th-century earthquake; only the pedestal foundations remain (pedestal removed in 1936, foundations excavated in 1986)."),
("AMB_05","ambiguous","name_meaning","Why is it called the Colosseum?",U,[],g("wc_04","cn_04"),
 "The name is believed to derive from the colossal statue of Nero (the Colossus) beside the amphitheatre; Bede's epigram about the Colossus is often mistranslated as referring to the Colosseum."),
("AMB_06","ambiguous","damage_cause","How was the building damaged over the centuries?",C,[],g("wc_12","wc_14","wc_17"),
 "Stone robbers stripped the interior and marble (burned for quicklime) and pried out the iron clamps, leaving pockmarks; vegetation, pollution and general deterioration threatened the structure, so the wall was shored up with brick wedges and restored."),
("AMB_07","ambiguous","paintings","What happened to the frescoes of the Golden House?",U,[],g("da_05","da_04","da_14"),
 "Rediscovered at the end of the 15th century, they profoundly influenced Renaissance artists (Raphael, Michelangelo studied them); the grotesque decorations came back in Neoclassicism, making the painter Famulus very influential."),
("AMB_08","ambiguous","living_quarters","Where did the gladiators live and train?",A,[],g("lm_08","lm_07","lm_09"),
 "At the Ludus Magnus: small rectangular chambers opening onto a two-storey portico served as living quarters, around an elliptical practice arena (about 63 x 42 m) with a small cavea."),
("AMB_09","ambiguous","site_today","Tell me about the Ludus Magnus.",R,[],g("lm_11","lm_06","lm_09"),
 "Rediscovered in 1937 and excavated in 1957-61, the remains represent less than half of the building and mostly reflect Trajan's rebuilding; the plan is completed with the Severan Marble Plan."),
("AMB_10","ambiguous","which_restoration","Tell me about the restoration.",C,[],g("wc_31","wc_14","pa_07"),
 "The Colosseum had a major 1993-2000 restoration and a 2013-2016 cleaning and repair financed by Tod's (EUR 25 million); a modern arena floor is being rebuilt to protect the hypogeum and give back the performance floor."),
("AMB_11","ambiguous","capacity","How many people could it hold?",A,[],g("wc_20","wc_01","wc_19"),
 "The Codex-Calendar of 354 gives 87,000, but modern estimates are around 50,000; other estimates range 50,000-80,000 with an average of about 65,000."),
("AMB_12","ambiguous","ships","Were there ships in the Colosseum?",S,[],g("wc_28","nm_09","nm_08"),
 "Ancient writers describe naumachiae (mock naval battles) at the inaugural games of 80 AD and under Domitian, but it is debated how the arena could be waterproofed and whether real ships could manoeuvre; stage-props representing ships may have been used."),
("AMB_13","ambiguous","sailors","What did the sailors do at the Colosseum?",A,[],g("vl_02","wc_18","vl_01"),
 "Sailors from the Misenum fleet, expert in sailmaking and rigging, built, maintained and operated the velarium, the retractable awning over the seats."),
("AMB_14","ambiguous","events_now","What kind of events take place at the Colosseum?",U,[],g("wc_15","wc_30","wc_02"),
 "Today only a few hundred spectators fit in temporary seating, so big concerts are held outside using it as a backdrop; religious ceremonies such as the Good Friday Way of the Cross take place there."),
# ---------------- SPECIALISTICHE: un solo dominio, domande su dettagli
("SPE_01","specialistic","dates","When did construction of the Colosseum begin and when were the inaugural games held?",H,[H],g("wc_09","wc_01"),
 "Construction began under Vespasian around 70-72 AD, the third storey was complete at his death in 79, Titus finished the top level in 80, and the inaugural games were held in 80 or 81 AD."),
("SPE_02","specialistic","crowd_flow","How many entrances did the Colosseum have and how did spectators find their seats?",A,[A],g("wc_19","wc_22"),
 "Eighty numbered entrances (76 for ordinary spectators); tickets were numbered pottery shards directing people to a section and row, and vomitoria let them reach seats and leave within minutes; each seat was designated by gradus, cuneus and number."),
("SPE_03","specialistic","animals","How many animals were killed at the inaugural games and which species appeared in the hunts?",S,[S],g("wc_10","vn_03","wc_27"),
 "Over 9,000 wild animals according to Dio Cassius; species included lions, elephants, bears, tigers, leopards, crocodiles, giraffes, hippos and ostriches."),
("SPE_04","specialistic","foundations","How deep are the foundations of the hypogeum walls and what were they built on?",R,[R],g("pa_05","pa_03"),
 "The foundations were dug between about 3.25 and 5.10 m below the hypogeal base floor, on a levelled concrete structure and a concrete platform 0.10-1.60 m thick, with the base rows of tufa blocks laid on top."),
("SPE_05","specialistic","sponsor","Who sponsored the 2011 restoration of the Colosseum and how much did it cost?",C,[C],g("wc_31"),
 "Diego Della Valle, head of Tod's, sponsored a EUR 25 million restoration; work began in 2013 and was completed on 1 July 2016."),
("SPE_06","specialistic","survey","What did the 1996 survey of the hypogea reveal about their history?",R,[R],g("ug_02","ug_05"),
 "Italian and German archaeologists surveyed the walls and divided the history of the hypogea in four stages: thin pillars, strengthening, further reinforcement, and finally division by the walls seen today, built with materials fallen from the top of the building."),
("SPE_07","specialistic","dimensions","How tall and wide is the Arch of Constantine?",A,[A],g("ac_01"),
 "21 m high, 25.9 m wide and 7.4 m deep; the central bay is 11.5 m high and 6.5 m wide."),
("SPE_08","specialistic","statue_move","When and how was the Colossus of Nero moved next to the Colosseum?",H,[H],g("cn_02","da_09"),
 "Around 128 AD Hadrian had it moved, by the architect Decrianus using 24 elephants, to make room for the Temple of Venus and Roma."),
("SPE_09","specialistic","augustus_naumachia","How many men and ships fought in Augustus's naumachia?",S,[S],g("nm_02"),
 "According to the Res Gestae, 3,000 men (not counting rowers) fought in 30 vessels with rams plus smaller boats, in a battle between Greeks and Persians staged in 2 BC."),
("SPE_10","specialistic","restoration_90s","Why was a major restoration carried out between 1993 and 2000 and how much did it cost?",C,[C],g("wc_14"),
 "The effects of pollution and general deterioration over time prompted it; it cost 40 billion lire (about EUR 20.6 million at 2000 prices)."),
("SPE_11","specialistic","lights","What does the colour of the Colosseum's night lighting mean?",U,[U],g("wc_15"),
 "The lighting changes from white to gold when someone condemned to death is released or a death sentence is commuted, or when a jurisdiction abolishes the death penalty (most recently Connecticut, November 2012)."),
("SPE_12","specialistic","velarium","How did the velarium work?",A,[A],g("wc_18","vl_02","vl_01"),
 "A retractable canvas awning, a net-like rope structure with a central hole, supported by 240 mast corbels at the top of the attic; it covered part of the arena, sloped towards the centre to catch the wind and create a breeze, and was operated by sailors from Misenum."),
("SPE_13","specialistic","sphinx_room","What was discovered in the Domus Aurea in 2018?",R,[R],g("da_18","da_19"),
 "A barrel-vaulted room (room 72), the Sphinx Room, decorated with panthers, centaurs, the god Pan and a sphinx in Fourth Style paintings, built between 65 and 68 AD."),
("SPE_14","specialistic","site_choice","Why did Vespasian build the amphitheatre on the site of Nero's lake?",H,[H],g("wc_08","wc_07","da_03"),
 "It was a populist gesture returning to the people an area Nero had appropriated; unlike most amphitheatres, built on city outskirts, it was placed at the heart of Rome."),
# ---------------- MULTI-DOMINIO: servono chunk di piu' domini/documenti
("MUL_01","multi_domain","damage_and_protection","How did earthquakes and stone robbing damage the Colosseum, and what has been done to protect it?",H,[H,C],g("wc_11","wc_12","wc_14","wc_31"),
 "The 1349 earthquake collapsed the south side and the stone was reused elsewhere; stripping removed marble and iron clamps. Later popes added brick wedges (1807, 1827), removed vegetation and repaired it; major restorations were done in 1993-2000 and 2013-2016."),
("MUL_02","multi_domain","flooding_evidence","Could the arena really be flooded for naval battles, and what evidence is there?",S,[S,R],g("wc_28","wc_24","ug_03","ug_04","nm_09"),
 "Ancient accounts describe flooding at the inaugural games, and traces of locks and 'docks' (darsene) were found, but there is no physical evidence of naumachiae and the arena was small; Domitian's hypogeum ended the practice."),
("MUL_03","multi_domain","social_seating","How were seats assigned to different social classes at the Colosseum?",A,[A,U],g("wc_20","wc_21","wc_22"),
 "Seating reflected Roman social hierarchy: emperor and Vestal Virgins in special boxes, senators on the podium, knights in the maenianum primum, citizens above (wealthy lower, poor upper), a top gallery for the poor, slaves and women; some groups were banned."),
("MUL_04","multi_domain","christian_view","How did the Christian view of the Colosseum change from the Middle Ages to the 18th century?",U,[H,U],g("wc_33","wc_34","wc_13","wc_32"),
 "In the Middle Ages it was treated as a quarry and not a sacred site; in the 17th-18th centuries, after Martinelli's book, popes declared it a sanctuary, Benedict XIV consecrated it and erected Stations of the Cross, though evidence of martyrdoms there is lacking."),
("MUL_05","multi_domain","floor_vs_hypogeum","How are the ancient hypogeum and the new arena floor reconciled in the reconstruction?",C,[R,C],g("pa_03","pa_07","pa_10","pa_08"),
 "The new wooden floor at the Flavian level rests on the original walls through interposed reversible elements, without occupying the hypogeal spaces, and protects the structures from weathering."),
("MUL_06","multi_domain","machinery","How did the underground machinery support the shows?",S,[S,A],g("wc_23","wc_24","ug_01","ug_05"),
 "Lifts, pulleys, trapdoors and hinged platforms raised animals and scenery from the hypogeum; the pegmata were prepared in the Summum Choragium and emerged in the arena to create instant scenery."),
("MUL_07","multi_domain","nero_to_flavian","How are the Domus Aurea, the Colossus and the Colosseum connected?",H,[H,A],g("wc_07","da_03","da_09","cn_02"),
 "After the fire of 64 Nero built the Domus Aurea with an artificial lake and the Colossus; Vespasian filled the lake and built the amphitheatre there, and Hadrian moved the Colossus next to it, giving the Colosseum its name."),
("MUL_08","multi_domain","games_evolution","How did the games at the Colosseum evolve from the inauguration to the last mentions?",S,[S,H],g("wc_10","wc_27","vn_02"),
 "From the 80 AD inaugural games with over 9,000 animals killed, through gladiators, hunts and executions, to Honorius's bans in 399 and 404; gladiatorial fights are last mentioned around 435 and animal hunts continued until at least 523."),
("MUL_09","multi_domain","vulnerability","What features of the Colosseum's structure and materials make it vulnerable to damage?",A,[A,C],g("wc_16","wc_12","wc_11","wc_17"),
 "The outer wall was travertine set without mortar and held by iron clamps (300 tons), so stripping the clamps and earthquakes caused large collapses, especially on the south side on alluvial terrain."),
("MUL_10","multi_domain","gladiator_route","Where did gladiators train and how did they reach the arena?",S,[S,R],g("lm_02","lm_03","wc_23","wc_25"),
 "They trained at the Ludus Magnus east of the Colosseum and reached the arena through an underground passage connecting it to the hypogeum."),
("MUL_11","multi_domain","plants_and_use","How do plants and modern events affect the preservation of the Colosseum?",C,[C,U],g("wc_36","wc_14","wc_15"),
 "Vegetation (684 species recorded) threatened the masonry and was removed, though much returned; because of the ruined interior only a few hundred spectators can be seated, so large concerts are held outside."),
("MUL_12","multi_domain","lost_and_found","What do we know about how the Domus Aurea was lost and then found again?",H,[H,R],g("da_02","da_03","da_04","da_06"),
 "Stripped of marble within a decade of Nero's death and built over by the Flavians, it survived underground; found in the late 15th century, it then suffered moisture, collapses and closures."),
("MUL_13","multi_domain","arch_spolia","How did the Arch of Constantine reuse earlier monuments and why is it controversial?",A,[A,U],g("ac_02","ac_06","ac_09","ac_14"),
 "Many reliefs and statues come from monuments of Trajan, Hadrian and Marcus Aurelius with portrait heads replaced; scholars debate whether it was built for Constantine, Maxentius or earlier, and art historians discuss the mixed styles."),
("MUL_14","multi_domain","octagonal_room","Why is the Octagonal Room important for archaeoastronomy and later architecture?",A,[R,A],g("da_15","da_17","da_16"),
 "Its domed octagonal design with a central oculus may have been a model for the Pantheon, and its openings are calibrated so that on the equinoxes the beam of light strikes the northern doorway."),
]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--chunks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--per_cat", type=int, default=4)
    a = ap.parse_args()
    ids = {c["chunk_id"] for c in json.load(open(Path(a.chunks).expanduser(), encoding="utf-8"))}
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    qs, gold = [], {}
    for qid, cat, sub, text, exp, tags, gd, ref in Q:
        missing = [c for c in gd if c not in ids]
        assert not missing, f"{qid}: gold inesistenti {missing}"
        assert len(gd) == len(set(gd)), f"{qid}: gold duplicati"
        qs.append({"id": qid, "category": cat, "subtype": sub, "query": text, "expected_tag": exp, "tags": tags, "reference": ref})
        gold[qid] = gd
    assert len({q["id"] for q in qs}) == len(qs)
    rng = random.Random(a.seed); pred_ids = set()
    for cat in ("ambiguous", "specialistic", "multi_domain"):
        pool = [q["id"] for q in qs if q["category"] == cat]
        pred_ids |= set(rng.sample(pool, a.per_cat))
    pred = [q for q in qs if q["id"] in pred_ids]; norm = [q for q in qs if q["id"] not in pred_ids]
    dump = lambda name, obj: json.dump(obj, open(out / name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    dump("queries_predetermined.json", pred); dump("queries_normal.json", norm)
    dump("eval_queries.json", [{**q, "is_predetermined": q["id"] in pred_ids} for q in qs]); dump("gold_chunks.json", gold)
    print(f"query: {len(qs)} (predeterminate {len(pred)} = {100*len(pred)/len(qs):.0f}%, normali {len(norm)}), seed={a.seed}")
    print("predeterminate:", sorted(pred_ids))
    from collections import Counter
    print("chunk gold usati:", len({c for v in gold.values() for c in v}), "di", len(ids))
    print("tag di conversazione:", dict(Counter(q["expected_tag"] for q in qs)))

if __name__ == "__main__": main()

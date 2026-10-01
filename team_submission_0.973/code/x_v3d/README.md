# Business Entity Resolution — v3d (two-stage, generator-aware)

All scores are the official macro-F0.5 (per Source-1 entity; singletons and zero-candidate
entities included). Train S1 is split by hashed id into 80% fit / 10% dev / 10% locked holdout; dev
and holdout samples are 20,000 S1 each, and every S1 is searched against the full 10.3M-record train
S2+S3 catalogue.

## Results

| system | dev F0.5 | locked holdout F0.5 | precision (dev) | recall (dev) | retrieval ceiling |
|---|---|---|---|---|---|
| v2 (previous submission, leaderboard 0.898) | 0.9209 | 0.9183 | 0.978 | 0.846 | 0.974 |
| v3c (leaderboard ≈0.97) | 0.9774 | 0.9767 | 0.995 | 0.947 | 0.993 |
| **v3d** | **0.9794** | **0.9785** | 0.995 | 0.952 | 0.994 |

Test-condition estimate (dev re-weighted to the test catalogue's neighbourhood mix, see
work/audit/DIAGNOSIS.md): India/US 0.978 (before the one-owner rule). Test predictions: 3.31 links per S1.

## v3d preprocessing (on top of v3c)

- Dotted acronyms are collapsed before punctuation removal: "S.A.R.L." used to become four junk tokens
  and never matched the legal-form list (5.4% of France and 4.3% of US S2/S3 names).
- OCR digits inside name words are folded (0->o 1->l 5->s 8->b); PO box / PMB / BP numbers are removed.
- Per-country address-component aliases (`learn_components.py`): US/India state names and codes learned
  on fit links ("north carolina" -> nc, "उत्तर प्रदेश" -> uttar pradesh); France
  department -> region learned label-free from confident test matches (gironde -> nouvelle aquitaine,
  loire atlantique -> pays de la loire, nord / pas de calais -> hauts de france). The table is
  country-specific because codes collide ("tn" = Tamil Nadu / Tennessee).
- A space-free name key in retrieval (domain-form names).

## What changed from v2, and why (each step measured on dev)

1. **Normalization that keeps the generator's evidence** (`norm.py`). The data generator builds
   negative "sibling" businesses by adding words such as holdings / group / industries to a real
   name. v2 deleted these as legal suffixes; they occur on 0 of 64,600 true links and on 12% of
   orphan candidates. Only legal forms are stripped now (plus French SAS/SASU/EURL/SCI/SNC). House
   numbers lose leading zeros (4.7% of true links are zero-padded), NULL/N-A/"CITY"/"No."/"bis" noise
   is removed, ordinals are normalized, French street types are unified, and honorifics, "(ID: n)"
   tags, "trading as" prefixes and glued domain names are cleaned.
2. **Evidence features** (`features.py`, 56 in total). Word evidence is the log-odds that a word a
   candidate adds to (or drops from) the S1 name marks a false candidate. It is learned by
   `word_evidence.py` on fit-split S1 that the matcher never trains on. Also: legal forms
   added/removed, main house-number relation (equal / one digit dropped / off by 1–3 / other), added
   number prefix, space-free name similarity, street-only address similarity.
3. **Retrieval** (`shared_index.py`): char 3-gram name TF-IDF, a word uni/bigram name TF-IDF (char
   3-grams of generic names are all pruned — 4.8% of India and 7.8% of France names had no feature),
   address TF-IDF, address-number postings, an exact normalized-name key, and a (house number,
   street word) key that works for 1–2 digit numbers. The dev retrieval ceiling rises from 0.974 to
   0.993.
4. **Stage 2** (`stage2.py`). Each candidate is re-scored from its S1's whole candidate list. The
   generator gives each entity one corrupted address per source; siblings carry their own number.
   Stage 2 uses per-source number support and conflict, the size of the sibling crowd, S1-name
   counts and a few stage-1 features.
5. **Decision**: per-entity expected-F0.5 selection, plus the one-owner rule over all S1 (a record
   linked by several S1 goes to the most probable one). The one-owner rule adds +0.0018 on dev.

Leakage control: aliases are learned on fit links; word evidence on fit set B; the pair model on fit
set A (150k S1); stage 2 on fit set C (150k S1). A, B and C are disjoint from each other and from dev
and holdout. France has no labels: four French sibling words (groupe, développement, participations,
snc) are added to the word-evidence table for test only (`word_evidence_test.pkl`). They come
label-free from the test catalogue; groupe/développement are set to their English analog's weight.

## Reproduce (data → blocking → matching → output)

    pip install -r requirements.txt
    cd src
    set D=<dataset dir>        # contains train/ and test/
    # 0. split (fit/dev/holdout by hashed S1 id) -> splits.pkl  (work/make_splits.py in the project)
    # 1. aliases (fit links only)
    python learn_aliases.py --data-dir %D%/train --splits splits.pkl --out aliases.pkl
    # 2. train index + S1 name counts
    python build_index.py --data-dir %D%/train --prefix train --out ../idx_train
    python s1_counts.py --s1 %D%/train/train_source1.tsv --index ../idx_train
    # 3. word evidence (needs forward scores of fit S1 from a previous model's run; see word_evidence.py)
    python word_evidence.py --data-dir %D%/train --splits splits.pkl --shards <score shards> --out word_evidence.pkl
    # 4. stage 1 + stage 2 training, dev scoring
    python train2.py --data-dir %D%/train --splits splits.pkl --index ../idx_train --work ../work --tag v3d
    # 5. test: index, S1 name counts, French evidence, inference (+ one-owner rule, output checks)
    python build_index.py --data-dir %D%/test --prefix test --out ../idx_test
    python s1_counts.py --s1 %D%/test/test_source1.tsv --index ../idx_test
    python word_evidence_fr.py --test-dir %D%/test --shards <v2 test score shards> --table word_evidence.pkl --out word_evidence_test.pkl
    set ER_WORD_EVIDENCE=%CD%\word_evidence_test.pkl
    python learn_components.py labelled --data-dir %D%/train --splits splits.pkl --out comp_alias.pkl   (before step 2)
    python learn_components.py unlabelled --data-dir %D%/test --shards <previous test score shards> --country France --table comp_alias.pkl --out comp_alias_test.pkl
    set ER_COMP_ALIAS=%CD%\comp_alias_test.pkl   (also when building the test index)
    python predict_parallel.py --data-dir %D%/test --model ../work/model_v3d.pkl --out-dir ../output --index-dir ../idx_test

## Cost (i7-13700HX, 24 threads, 32 GB)

| step | time |
|---|---|
| index build, 10M S2+S3 | ~17 min |
| stage-1 pairs, 150k S1 (20.1M pairs) + training | ~9 + 5 min |
| stage-2 rows, 150k S1 + training | ~9 min + 40 s |
| dev scoring, 20k S1 | ~1.5 min |
| full test inference, 1.73M S1 | ~70 min |

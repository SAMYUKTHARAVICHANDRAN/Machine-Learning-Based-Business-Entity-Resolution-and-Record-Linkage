# TRIDENT — business entity resolution

Source-1 → Source-2/3 matching for the Amazon ML Challenge 2026.

## Layout

```
TRIDENT_pipeline.ipynb            # the full pipeline (project root); embeds src/trident_core.py
code/business_entity_resolution/
├── src/trident_core.py           # all logic shared by training and inference
├── build_notebook.py             # regenerates the notebook from src/ (keeps them identical)
├── run_notebook.py               # headless execution, saves after every cell
├── requirements.txt
└── README.md
```

## Reproduce end to end

1. Put the challenge data at `student_resource/dataset/{train,test}/` under the project root.
2. `python -m pip install -r code/business_entity_resolution/requirements.txt`
3. From the project root, either open `TRIDENT_pipeline.ipynb` and *Run All*, or run it headless:

   ```
   python code/business_entity_resolution/run_notebook.py
   ```

4. Outputs: `submission/output/matching_results.tsv` and `submission/output/candidate_pairs.tsv`
   (validated in the notebook with its own checks and the official `validate_submission.py`).

Paths and switches live in the notebook's §1 configuration cell. All intermediate artefacts
(indexes, feature blocks, lexicon, model, inference parts) are cached under `work/trident_v2/` with
signatures; an interrupted run resumes from the last completed batch.

## Data flow

TSV → normalisation (spec NFKD/casefold + Indic transliteration + learned translit lexicon +
abbreviation canonicalisation) → Parquet → DuckDB `records` + frequency-filtered `postings`
(routes N, A, n, a, P, Q, X, C, Y; country-partitioned) → per-batch hash-join retrieval, evidence
`Σ w/(1+ln f)`, JW-based priority, top-60 → 46 pairwise features (29 spec + 17) → CatBoost (GPU when
faster) → calibrated threshold + one-owner-per-target resolution → TSVs.

No external data or services are used; the only learned artefacts come from the training files.

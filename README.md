# Visual-Memory Reliability

**Completed research pilot · dynamic visual-memory evaluation · Python · DynaMem / DynaBench**

Can evidence about **when an object was last observed** and **how well it matches a text query** help an existing visual-memory system avoid incorrect location answers, without merely declining more questions?

This repository contains a reproducible **offline diagnostic study**, not a newly trained vision model or a complete robot. We replay recorded RGB-D observations through the existing Stretch AI / DynaMem memory pipeline using **SigLIP + YOLOE**. We then compare the baseline, an **observation-age-only** answer filter, an **age-plus-similarity** filter, and an **always-decline** control. The models and candidate locations stay fixed; the filters only retain or reject existing answers.

## Results at a glance

One development scene (**env1, 54 queries**) and one held-out scene (**env2, 66 queries**) were run. All **120 primary queries** produced predictions or legitimate declines with **zero recorded inference errors**. A separate boundary-frame sensitivity branch adds another 120 prediction records; it is not a new independent test set.

At the preregistered 75%-retention **retrospective matched-answer quota** on the **42 known-location held-out questions**:

| Rule | Scorable locations returned | Geometric passes | Geometric errors | Error among returned locations |
|---|---:|---:|---:|---:|
| Baseline (more answers; for context) | 23 | 19 | 4 | 17.4% |
| Observation age only | 18 | 16 | 2 | 11.1% |
| Age + similarity | 18 | 17 | 1 | 5.6% |
| Always decline | 0 | 0 | 0 | Undefined |

**Interpretation:** At the same 18-answer quota, the combined rule had **one fewer geometric error**. This was **tie-sensitive**: an alternate valid ordering of equally old age-only candidates would eliminate the difference. We **do not claim robust superiority, statistical significance, a calibrated confidence model, or an official reproduction of the DynaMem paper**. On the other tested quotas, the methods tied. A retrospective fixed-answer quota is **not** an online operating threshold; see the separately reported frozen-threshold results.

**Other important finding:** For a “toy owl” query, the saved detector box selected the nearby **green toy** yet passed the owl's 30 cm geometric tolerance. A correct distance score need not imply a correct object identity. This was an assistant-assisted visual review, not a human-certified relabeling.

**Limitations:** 42/120 questions had unresolved blank labels and were excluded from primary correctness scoring; the authors' precise checkpoint convention was not verified; evaluation used only one held-out scene. The active **YOLOE + SigLIP** configuration is not the paper's verified OWL-based setup. See [detailed results and caveats](docs/RESULTS.md).

## Repository contents

| Path | Purpose |
|---|---|
| [`notebooks/DynaBench_research.ipynb`](notebooks/DynaBench_research.ipynb) | Complete Colab workflow (outputs cleared; includes an embedded source bundle and held-out schema adapter) |
| [`vmresearch/`](vmresearch/) | Dataset preparation, chronological replay, evidence tracing, scoring, rule freezing, comparisons, and reports |
| [`scripts/prepare_env2_schema.py`](scripts/prepare_env2_schema.py) | Audited non-destructive four-stream adapter for the held-out scene |
| [`tests/`](tests/) | Synthetic/structural tests; these are **not** benchmark scores |
| [`bootstrap_colab.sh`](bootstrap_colab.sh), [`bootstrap_guard.py`](bootstrap_guard.py) | Recorded Python 3.10 / CUDA Colab setup and version checks |
| [`results/`](results/) | Small, independently checked metrics, frozen rules, protocol and provenance (no raw scene images) |
| [`docs/SETUP.md`](docs/SETUP.md) | Environment, dataset, reproduction, and test instructions |
| [`docs/METHOD.md`](docs/METHOD.md) | Study design, metrics, cutoff and label handling |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Independent detailed review of what the experiments support |

## Quick start (Google Colab, GPU)

1. Open [`notebooks/DynaBench_research.ipynb`](notebooks/DynaBench_research.ipynb) in [Google Colab](https://colab.research.google.com/).
2. Select a **hosted GPU runtime** (the completed run used a Tesla T4) and run the numbered cells **1–11 in order**. Section 1 writes the embedded source into `/content/DynaBench_Research_v3`; its setup checks use `/content/stretch310/bin/python` (Python 3.10), not Colab's default interpreter.
3. The notebook pins [Stretch AI commit `6272b28`](https://github.com/hello-robot/stretch_ai/tree/6272b28e3ff7be482af3b427a4551d1772ebf7d7) and [DynaBench dataset revision `1d561e2`](https://huggingface.co/datasets/hello-peiqi/DynaMem-DynaBench/tree/1d561e2de8718addf398e5e967f2c44dc015c366). It downloads the small scenes as needed; do **not** upload raw scenes to this repository.
4. At the end, download the generated results ZIP to keep the full private evidence archive. The notebook has no active cloud-run outputs committed here.

See [SETUP](docs/SETUP.md) for full instructions, a headless CLI example, required environment caveats and testing.

## Reproducibility and responsible claims

- Same pretrained models and input prefixes across methods; only the answer-rejection rule changes.
- No future frames intentionally used in the primary cutoff branch; alternative included separately because the official convention is unresolved.
- Predictions are saved before scoring labels are read. Frozen thresholds are selected on `env1` before running `env2`.
- Fixed-threshold comparisons **change coverage**; exactly matched answer quotas are retrospective analyses on known-label questions.
- Unresolved blank annotation rows are **unknown**, not assumed negative.
- We did **not** upload the 335 MB evidence archive, raw DynaBench images/depth, model weights, tokens, or screenshots. Dataset-derived imagery has unresolved redistribution terms.
- The public source here is the experimental **harness**. It uses (but does not relicense or include) the upstream Stretch AI/DynaMem, SigLIP and YOLOE implementations.

The code was developed with AI assistance and the numerical results were subsequently checked using independent audit code; the full evidence is retained privately. A negative or tie-sensitive result is reported as such.

## References

- Liu *et al.*, [DynaMem: Online Dynamic Spatio-Semantic Memory for Open World Mobile Manipulation](https://arxiv.org/abs/2411.04999v2), 2025.
- [Hello Robot Stretch AI](https://github.com/hello-robot/stretch_ai) (source commit pinned above).
- [DynaBench public dataset](https://huggingface.co/datasets/hello-peiqi/DynaMem-DynaBench) (data revision pinned above).

**Status:** Finished diagnostic pilot, not an official benchmark or statistically validated new method. No license to redistribute third-party datasets or models is implied.

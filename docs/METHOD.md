# Experimental method

## Question

Can rejecting answers using **observation age + maximum query-to-memory similarity** reduce incorrect returned object locations compared with an **age-only** rule, when both retain the same number of answers?

## System and data

- Existing **DynaMem-style dynamic voxel memory** via Hello Robot Stretch AI commit `6272b28e3ff7be482af3b427a4551d1772ebf7d7`.
- Pretrained SigLIP visual/text encoder and YOLOE detector. **No new foundation model or model fine-tuning.** This is **not** the verified original DynaMem-paper configuration.
- Recorded RGB, depth, camera calibration and poses; 54 `env1` development questions and 66 `env2` held-out questions at scene-specific checkpoints.
- Primary cutoff: frame indices strictly earlier than checkpoint. Alternative includes boundary frame, measured separately because official dataset interpretation is unverified.

## Evidence and score

For each baseline returned location, record the evidence frame and its age in **observation steps**, the maximum query-to-memory similarity, detector return path/box, and input/provenance hashes. Labels are read **after** predictions have been persisted. Inference exceptions are errors, not declines.

An answer passes geometric scoring if the Euclidean distance from the annotated target is within the annotated tolerance. A pass does **not** guarantee the detector selected the requested object. Blank coordinate rows remain **unknown** in the main analysis rather than becoming presumed negatives.

## Rules

- **Baseline:** preserve original answer or decline.
- **Age-only:** prefer newer evidence (larger `-age`).
- **Age + similarity:** average the development percentile ranks of `-age` and maximum similarity with **equal weights**. Detector support was constant on development answers and was **not used**. Detector confidence was logged but **not used** in the combined score.
- **Always decline:** return no location; zero positive localization recall and undefined conditional error.

Frozen operating thresholds were derived from `env1` at 25%, 50%, 75% and 100% baseline-answer retention targets and applied to all 66 held-out questions. Those thresholds can produce unequal held-out coverage. A **separate retrospective equal-answer-quota comparison** ranks the same labeled held-out candidates by evidence and retains exactly the same number of returned locations, using deterministic ID-hash tie-breaking. It is not a deployable threshold policy that knows future label availability.

## Guardrails and interpretation

The inference stream, checkpoint conventions, candidate locations and model configuration are kept fixed among rejection rules. Training/development and held-out scenes are disjoint. Policies are saved and hash-checked before held-out execution. Evidence is auditable via frame IDs and detector traces, with the large image archive retained separately.

**Major limits:** one held-out scene; 42/120 unresolved blank labels; different-object selection may pass a tolerance radius; cutoff convention unverified; tie-dependent primary result; no robust or generalizable improvement demonstrated. See [RESULTS](RESULTS.md).

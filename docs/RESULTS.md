# Visual-memory reliability — independently reviewed pilot results

**Review date:** October 27, 2026  
**Input:** `DynaBench_Research_v3_results.zip` (335,374,566 bytes)  
**Scope:** saved-output recomputation, provenance checks, source review, and selected image inspection. No GPU model was rerun for this review.

## Executive conclusion

The real-data pipeline completed on one development scene (`env1`) and one held-out scene (`env2`). The saved geometric scores, policy thresholds, and comparison tables reproduce correctly. At the predeclared primary matched-answer target, the age-plus-similarity rule retained **18 answers with one geometric error**, versus **18 answers with two geometric errors** for age-only: **5.6% versus 11.1% conditional error**.

That is a small observed advantage, **not a robust demonstrated improvement**. The extra age-only error depends on the prespecified tie-breaking order. Two answers with exactly the same observation age compete for its final slot; one passes the distance rule and the other fails. Other ways of ordering that tie would make the methods equal. The frozen result has not been changed.

A separate, visually supported finding is that **passing the location-radius test does not guarantee the requested object was selected**. The original “toy owl” prediction selected the nearby green toy, yet remained within the owl annotation's 30 cm acceptance radius.

This is a completed small diagnostic pilot under recorded assumptions—not an official benchmark score, a faithful reproduction of the paper's configuration, a statistically established gain, or a verified novel method.

## 1. What was checked

The results archive contains **1,390 files**. Its ZIP CRC check passed, and **859 PNG files** passed file-integrity validation. These checks establish file readability, not semantic correctness of every image.

An independent, standard-library audit recomputed the numeric results without importing the experiment's scoring or policy modules. **150 consistency/recomputation checks passed**, including:

- Complete unique query IDs and agreement between individual prediction files and the aggregate prediction logs.
- Recomputed distances, correct/wrong/decline outcomes, summary metrics, frozen thresholds, applied policy decisions, and exactly matched selected-query sets.
- Recorded frame-prefix, observation-ID and age consistency; absence of target/score fields from model prediction records; presence of supporting images and matching detector-return traces.
- Agreement of the protocol, development data, frozen-policy and experiment-code hashes, and identical recorded package/source/model configurations across scenes.
- Agreement of the `env1` labels with the original, previously uploaded CSV files. The `env2` raw files are excluded from this results archive, so its saved labels were recomputed but not independently reread from original CSV bytes.

The source package supplied earlier matches the results' recorded harness hash exactly. The schema-repair audit's protected files also retain their recorded hashes. Archive modification times place the frozen policies before the held-out run; these are corroborating file timestamps, not independently trusted proof of execution order. Code inspection and hashes support the intended separation of inference, development selection, and scoring, but cannot prove every fact about the original remote process.

**Sources:** `independent_audit.json`, `archive_integrity.json`; original archive `plan_lock.json`, `preflight.json`, `model_lock.json`, per-scene manifests, and prediction/score logs.

## 2. Baseline execution is complete

The primary branch excludes the checkpoint-indexed frame. A separate sensitivity branch includes it. The branches are not pooled.

| Scene | Role | Frames in scene | Checkpoints | Primary questions | Additional sensitivity records | Recorded inference errors |
|---|---|---:|---|---:|---:|---:|
| env1 | Development | 42 | 14, 26, 41 | 54 | 54 | 0 |
| env2 | Held out | 55 | 18, 36, 54 | 66 | 66 | 0 |

There are **120 distinct primary questions and 240 baseline prediction records** across the two cutoff interpretations—not 240 independent test questions. Policy comparison reuses these predictions; it is not a further set of GPU experiments.

| Primary baseline | All answers | All declines | Correct scored locations | Wrong scored locations | Answers with unknown labels |
|---|---:|---:|---:|---:|---:|
| env1: 54 questions | 28 | 26 | 19 | 0 | 9 |
| env2: 66 questions | 37 | 29 | 19 | 4 | 14 |

`env1` has 36 known-location labels and 18 blank-label questions. `env2` has 42 known-location labels and 24 blank-label questions. Therefore **42 of 120 questions (35%) are excluded from primary correctness scoring**. They remain in all-question answer/decline counts.

On `env2`, baseline error among its **23 scorable returned locations** is **4/23 = 17.4%**. This is not an error rate over all 37 returned locations. Positive localization recall is **19/42 = 45.2%**. On development data, all 19 scorable answers pass the distance rule; that does not make the system error-free, given unknown labels and object-identity ambiguity.

**Sources:** `baseline_metrics.json`; original archive `prepared/*/manifest.json`, `runs/*/completion.json`, and `scoring/*/scored.jsonl`.

## 3. Which evidence was actually used

The frozen combined score is the equal-weight average of development-derived percentile ranks for:

1. Observation recency (negative age, measured in **observations**, not seconds).
2. The recorded maximum query-to-memory similarity.

The binary “detector supported this answer” feature was constant for the development answers and was excluded. **Detector confidence was logged, but was not used by the frozen combined score.** The evaluated method should therefore be described as **age plus similarity**, not a learned confidence model using all available evidence.

The transforms and thresholds regenerate from the 19 scorable development answers alone. They were chosen for retention targets, not trained as a predictor of correctness; no calibration claim is supported.

**Sources:** original `analysis/frozen_policies.json`; `independent_audit.json`.

## 4. Primary comparison at equal answer counts

This is a **retrospective ranking comparison** on the same 42 known-label held-out questions. It is not a deployable rule that knows which future queries will be labeled. It ranks baseline-returned answers by evidence, not by correctness, using a fixed query-ID hash to break ties.

The predeclared 75% retention target rounds up to 18 of the 23 scorable baseline answers: 78.3% of baseline answers, or 42.9% of the 42 known-label questions. It does **not** mean answering 75% of all questions.

| Method | Answers on the 42-question scored subset | Geometric passes | Geometric errors | Error among answers |
|---|---:|---:|---:|---:|
| Unchanged baseline (context; more answers) | 23 | 19 | 4 | 17.4% |
| Age-only, primary matched quota | 18 | 16 | 2 | 11.1% |
| Age plus similarity, same quota | 18 | 17 | 1 | 5.6% |
| Always decline | 0 | 0 | 0 | Undefined |

Only the two 18-answer rows form the matched-coverage comparison. Always-decline has zero positive recall; its undefined error is not evidence of better performance.

At the other predeclared quotas the observed methods tie: 0 versus 0 errors at 6 answers, 1 versus 1 at 12, and 4 versus 4 at 23.

**Sources:** `matched_quota_metrics.json`; original `analysis/matched_selected_ids.json` and `analysis/matched_coverage_comparison.csv`.

### New audit finding: the one-error advantage is tie-sensitive

At the age-only quota boundary, 17 answers rank strictly above age 9. One final slot is available, but two candidates have age 9:

| Candidate | Saved query ID | Geometric outcome |
|---|---|---|
| green sugar package | env2_q0018_003 | Wrong: 0.3220 m error versus 0.3000 m tolerance |
| trash can | env2_q0036_001 | Correct: 0.1154 m error versus 0.5000 m tolerance |

The prespecified hash tie-break selects the green-sugar-package answer. Choosing the other equally aged candidate would give age-only one error instead of two. This is **not** a recommendation to select the correct candidate using its label. It is a posthoc sensitivity check demonstrating that this result is not invariant to otherwise valid tie-breaking choices.

Across all boundary-tie orderings, age-only has **1–2 errors** at 18 answers; combined has **1 error**. Combined also has a boundary tie, but both tied candidates pass the distance rule. Its error count is unaffected by that particular tie.

The actual difference in selected sets is: combined drops the age-only selections “joystick” (correct) and “green sugar package” (wrong), and adds “trash can” (correct) and “toy bear” (correct). The discarded wrong answer misses the 30 cm radius by about **2.2 cm**. The remaining “toy drill” error misses its 30 cm radius by about **1.1 cm**. Keep the official annotation tolerances unchanged; neither proximity to the threshold nor this audit authorizes relabeling.

**Sources:** `tie_sensitivity.json`, `primary_selection_disagreements.json`, and the individual scored rows.

## 5. Frozen operational thresholds: a different comparison

These rules use fixed thresholds selected on development data and apply to all 66 held-out questions. Their answer rates differ, so their conditional errors must not be presented as an equal-coverage comparison.

| Rule at the primary development retention target | Answers / 66 | Scorable answers | Correct / wrong among scorable answers | Conditional error |
|---|---:|---:|---|---:|
| Baseline | 37 | 23 | 19 / 4 | 17.4% |
| Age-only | 31 | 19 | 17 / 2 | 10.5% |
| Age plus similarity | 25 | 17 | 16 / 1 | 5.9% |
| Always decline | 0 | 0 | 0 / 0 | Undefined |

Combined has fewer observed errors here, but it also supplies fewer answers and fewer correct locations. It cannot be called unconditionally better from this table. The frozen policy predictions and all 660 associated scoring records were independently reproduced.

**Source:** `frozen_threshold_metrics.json`; original `analysis/heldout_policy_predictions.jsonl` and `analysis/heldout_policy_scores.jsonl`.

## 6. Object identity: the original owl result

I inspected the saved unannotated RGB image, selected box, and projected annotation image. **The selected box encloses the small green toy on the left; the white owl toy is outside it on the right.** The green-toy annotation projects inside the box, while the owl annotation projects outside. The detector-only replay reproduces the saved returned coordinate.

The returned point is 27.51 cm from the owl annotation and 5.38 cm from the green-toy annotation. Consequently, the saved owl answer remains a **pass under the 30 cm distance rule**, while the image supports a **different-object selection**. The unchanged official-style distance test and this posthoc semantic observation are different assessments; they must not be silently merged.

This review is an assistant image assessment, not a human-certified relabeling. I did not edit the original `manual_review.json`, model outputs, labels, frozen rules, or scores. The annotation projection describes query-time targets, so it is supporting evidence rather than an independent guarantee of historical pixel identity.

Three development geometric passes are flagged as closer to another annotation. Only the first owl case has been resolved by this review's direct visual inspection; the other nearest-object flags should not be automatically treated as misidentifications.

**Evidence:** `evidence/evidence_rgb.png`, `evidence/selected_box.png`, `evidence/projected_label_anchors.png`, and `posthoc_visual_review.json`.

## 7. Remaining limits

**Checkpoint convention remains unresolved.** The supplied code follows its recorded frame-prefix assumptions consistently, but neither frame counts nor a good score establish the authors' intended convention. Adding the boundary frame changed 2 of 54 outputs in `env1` and 11 of 66 in `env2`; scored outcome categories changed for 1 and 6 questions, respectively. The sensitivity branch is not selected as a replacement or pooled with the primary branch.

**Blank-label meaning remains unresolved.** No verified negative examples enter primary correctness scoring. The saved alternative analysis treats blank rows as negatives only as an explicit sensitivity scenario. Under that assumption, the held-out baseline would have 18 wrong answers among 37 answers (48.6%), rather than 4 among 23 (17.4%). Neither interpretation can be declared official from this archive alone. The primary conclusions are about known-location questions, not reliable detection of absent objects.

**Small and correlated evidence.** There is one held-out scene, repeated objects, and four baseline geometric errors in the scorable primary subset. No across-scene confidence interval, statistical significance, broad reliability improvement, or novelty claim follows.

**Current-code configuration.** The archived path is YOLOE + SigLIP, not a verified recreation of the DynaMem paper's configuration. It preserves the recorded BGR/PIL channel path and bounding-box depth aggregation. Those may warrant separate work, but changing them now would define a different experiment.

**Environment and provenance.** Core package/source/model records match across the scenes. The two documented package-metadata conflicts remain. The `env2` adapter preserved the four required streams according to its logged fingerprints and left the protected study files unchanged; original `env2` stream bytes are not in this export for independent rechecking.

**Publication scope.** Dataset/image redistribution permissions were not verified by this review. Do not treat the exported evidence images as already cleared for public redistribution.

## 8. Final project wording and next action

A supported summary is:

> Built and evaluated an auditable offline visual-memory pipeline on one development and one held-out recorded scene. An age-plus-similarity rule produced one fewer geometric error than age-only at an equal 18-answer quota, but the advantage depended on tie-breaking and did not establish robust superiority. Detector-image review also showed that the distance acceptance radius can conceal a wrong-object selection.

This is sufficient to **close the bounded pilot as a diagnostic/evaluation project** with the limitations above. No installation, threshold change, or expensive rerun is needed to report the completed work. Stronger claims would require resolving dataset conventions and testing additional independent evidence; that is separate future work, not a reason to misstate this completed pilot.

### Session handoff

Completed: archive/content integrity checks; independent numerical recomputation; frozen-rule/hash checks; tie-sensitivity analysis; selected-object image review; this reviewed report.  
Output: `DynaBench_Reviewed_Report.md` and `DynaBench_Independent_Review.zip`.  
Changed conclusion: replace an unqualified combined-rule improvement claim with a small, tie-sensitive observed advantage; record the owl case as a visually supported wrong-object selection that still passes distance scoring.  
Unchanged: all original predictions, labels, thresholds, configuration, and development/evaluation split.  
Remaining limits: official cutoff/blank-label semantics and beyond-one-scene generalization.  
Next action: use this reviewed conclusion in the project write-up; retain the original results ZIP and the audit bundle together.

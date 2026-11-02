"""Write a short measured-results report. Missing experiments stay explicitly pending."""
from __future__ import annotations
import html
from pathlib import Path
from .core import REFERENCE, atomic_json, protocol, read_json


def fmt(x,percent=False):
    if x is None:return "not defined"
    return f"{100*x:.1f}%" if percent else str(x)

def table(rows,columns):
    if not rows:return "No completed records."
    lines=["| "+" | ".join(label for _,label in columns)+" |",
           "| "+" | ".join("---" for _ in columns)+" |"]
    for r in rows:
        vals=[]
        for key,label in columns:
            v=r.get(key)
            if key.startswith(("coverage_","conditional_","positive_")):v=fmt(v,True)
            elif v is None:v="not defined"
            vals.append(str(v).replace("|","/"))
        lines.append("| "+" | ".join(vals)+" |")
    return "\n".join(lines)

def write_report(base:Path) -> Path:
    p=protocol(base);base.mkdir(parents=True,exist_ok=True)
    old=read_json(REFERENCE/"result.json")
    geometry=read_json(base/"prior_audit/geometry_review.json") if (base/"prior_audit/geometry_review.json").exists() else None
    replay=read_json(base/"prior_audit/detector_replay.json") if (base/"prior_audit/detector_replay.json").exists() else None
    comparison=read_json(base/"analysis/comparison.json") if (base/"analysis/comparison.json").exists() else None
    lines=["# Visual-memory reliability: small real-data pilot", "",
        "## Question and scope",
        "Can a rule based on stored evidence reduce wrong object-location answers without merely answering fewer questions? "
        "This is an offline, current-code **YOLOE + SigLIP diagnostic**, not a verified DynaMem-paper reproduction. "
        "There is no robot navigation, active re-observation, manipulation, model training, or commercial API in this run.","",
        "## Completed evidence versus pending work",
        f"The previously uploaded single-query run returned a real location for **toy owl**, env1/checkpoint 14. "
        f"Its saved geometric error is **{old['distance_error_m']*100:.2f} cm**, against a **{old['tolerance_m']*100:.0f} cm** radius. "
        "That is a distance-rule pass, not proof of correct object identity.",""]
    if geometry:
        nearest=geometry["nearest_annotated_object"]
        lines += [f"Posthoc review: the saved point is {nearest['distance_from_saved_prediction_m']*100:.2f} cm from the "
                  f"**{nearest['query_label']}** annotation. This flags a possible wrong-object match hidden by tolerance; "
                  "nearest-coordinate identity is not semantic ground truth."]
    lines += ["Detector selection audit: "+(f"supporting-frame replay completed; returned location matches the saved point at 1e-6 m: "
                  f"**{replay['matches_saved_location_at_1e_6m']}**. See `prior_audit/selected_box.png`, its crop, "
                  "`actual_detector_input.png`, and `projected_label_anchors.png`." if replay else
                  "**pending GPU replay**. The earlier saved images do not contain a certified selected box."),""]
    review=base/"prior_audit/manual_review.json"
    if review.exists():
        mr=read_json(review);lines += [f"Optional human identity assessment: **{mr['semantic_identity']}**. "
                                       "This assessment is not used to train/tune either rule.",""]
    lines += ["## Protocol and unresolved conventions",
              f"Development scenes: `{', '.join(p['development_scenes'])}`. Held-out scenes: `{', '.join(p['evaluation_scenes'])}`. "
              "The split is declared before evaluation. Policies freeze after development and before the held-out run.",
              f"Primary cutoff: **{p['primary_cutoff']}**; author/exporter confirmation: **{p['checkpoint_semantics']}**. "
              "The default passes frame indices < checkpoint. A separately labelled inclusive sensitivity run also uses the "
              "boundary frame; it may leak a future frame if the exclusive convention is correct. These branches are never pooled "
              "and the better-scoring branch is not selected.",
              "One-based upstream observation IDs can be verified from source, but that does not establish what CSV checkpoint names mean. "
              "The source scanner saves evidence candidates; it never treats search hits or agreement with labels as proof.",
              f"Blank-coordinate semantics: **{p['blank_label_semantics']}**. Unless confirmed, fully blank rows are unscored unknown labels, "
              "not negative ground truth. A separate assumed-blank-negative sensitivity table is saved. Partial or nonfinite labels fail validation.",
              "Evidence age is the count of observations since the supporting frame, **not seconds or elapsed real-world time**.",""]
    lines += ["## Baseline batches", "Each question uses only its allowed frame prefix. Errors are separate from declines. "
              "Prediction JSON is saved before scoring. The model and its thresholds are unchanged; read-only profiling captures the selected "
              "bounding box, its confidence, the existing similarity score, and the return route.",""]
    all_scenes=p["development_scenes"]+p["evaluation_scenes"]
    for scene in all_scenes:
        f=base/"scoring"/scene/"metrics.json"
        if not f.exists():lines += [f"**{scene}: pending.** No batch-performance claim.",""];continue
        metric=read_json(f)[p["primary_cutoff"]]
        lines += [f"### {scene} — primary branch",
                  table([metric],[("queries","Questions"),("answers","Answers"),("declines","Declines"),
                                 ("correct_locations","Correct locations"),("wrong_locations","Wrong locations"),
                                 ("unscored_label_queries","Unknown labels"),("inference_errors","Errors")]),
                  f"All-question coverage: {fmt(metric['coverage_all_queries'],True)}. Conditional error among returned locations "
                  f"with known labels: {fmt(metric['conditional_error_scorable'],True)}. "
                  "Unknown-label answers are not silently counted as correct or wrong.",""]
    lines += ["## Rejection-rule comparison", "Age-only ranks newer supporting observations higher. Combined evidence averages "
              "development-set percentile ranks for recency and observed query similarity, adding detector support only when that signal "
              "varies and is available. Equal weights are fixed in advance; no probability-calibration claim is made. "
              "The combined rule is skipped if required evidence is missing.",
              "Two evaluations are kept distinct: frozen numeric thresholds selected on development data (coverage can shift on a new scene), "
              "and a retrospective equal-quota ranking comparison on the same scorable held-out cohort. The latter keeps identical answer counts "
              "using scores and a query-ID hash for ties, without consulting correctness to choose answers. It is not an online threshold policy.",""]
    if comparison:
        fixed=[r for r in comparison["fixed_threshold_metrics"] if r["scene"]=="pooled"]
        lines += ["### Frozen thresholds",table(fixed,[("policy","Policy"),("answers","Answers"),
                  ("coverage_all_queries","Coverage, all"),("conditional_error_scorable","Conditional error, scored"),
                  ("positive_localization_recall","Positive recall")]),"",
                  "### Exactly matched answer quotas (known-label cohort)",
                  table(comparison["matched_coverage_metrics"],[("method","Rule"),
                     ("retained_fraction_of_baseline_answers","Retained baseline fraction"),
                     ("exact_answer_quota","Answer quota"),("conditional_error_scorable","Conditional error")]),""]
        primary=[r for r in comparison["matched_coverage_metrics"]
                 if r["retained_fraction_of_baseline_answers"]==p["primary_retention_fraction"]]
        by={r["method"]:r for r in primary}
        a=by.get("age_only",{}).get("conditional_error_scorable")
        b=by.get("combined",{}).get("conditional_error_scorable")
        if a is not None and b is not None:
            delta=a-b
            if abs(delta)<1e-12:claim="The two rules tied at the predeclared primary quota."
            elif delta>0:claim=f"Combined evidence had {100*delta:.1f} percentage points lower observed conditional error than age-only at the primary quota."
            else:claim=f"Combined evidence had {100*(-delta):.1f} percentage points higher observed conditional error than age-only at the primary quota; it did not improve this pilot."
            lines += [claim+" This is a measured single-pilot comparison, not a generalization or statistical-significance claim.",""]
        else:lines += ["No supported primary age-versus-combined benefit statement is available (missing method or no returned scorable locations).",""]
        lines += ["Skipped methods/missing held-out signals: `"+str(comparison["skipped_methods_missing_heldout_features"])+"`.",""]
    else:lines += ["**Held-out comparison pending. No improvement, failure-to-improve, or completed-project result is claimed yet.**",""]
    lines += ["## Limitations and interpretation",
        "Always-decline is included as a sanity comparator: zero positive localization recall and undefined conditional error when no locations are returned. "
        "A decline never proves global object absence.",
        "One development scene and one held-out scene form an exploratory pilot. Frames and repeat-object questions are correlated; no query-level "
        "confidence interval or significance test is presented as evidence of across-scene generalization. Additional held-out scenes would be needed for that.",
        "The upstream path passes a BGR-converted array into a detector routine that wraps it as a PIL image; this channel path is recorded and preserved, "
        "not silently corrected. The selected location uses a bounding-box depth aggregation, not an object mask. Both deserve separate future ablations.",
        "The working environment has two recorded metadata conflicts (Transformers 4.49.0 vs Stretch >=5.3.0; setuptools 69.5.1 vs Torch >=77.0.3). "
        "The notebook preserves the empirically working path, not a clean validated installation of every robot feature. No new seed was recorded in the original "
        "one-query result; the new batches record their seed and may differ because upstream memory subsamples points.",
        "Dataset reuse/redistribution terms were not independently established by this pilot. Result archives omit raw scenes and model weights; do not publish "
        "source images or redistribute the dataset without checking the applicable permissions.","",
        "## Reproducibility and outputs",
        f"Source commit: `{p['code_commit']}`. Dataset revision: `{p['dataset_revision']}`. "
        "See `protocol.json`, `plan_lock.json`, `preflight.json`, `working_environment.txt`, per-scene manifests, model hashes, "
        "per-query predictions/traces, `frozen_policies.json`, and CSV comparisons. No external API keys or paid compute are invoked by the harness.","",
        "## Sources",
        "- Uploaded diagnostic: `reference/prediction.json`, `reference/result.json`, `reference/compatibility_pass.json`.",
        "- Pinned memory/query implementation: https://github.com/hello-robot/stretch_ai/blob/6272b28e3ff7be482af3b427a4551d1772ebf7d7/src/stretch/mapping/voxel/voxel_dynamem.py",
        "- Pinned detector selection: https://github.com/hello-robot/stretch_ai/blob/6272b28e3ff7be482af3b427a4551d1772ebf7d7/src/stretch/perception/detection/yoloe.py",
        "- Dataset: https://huggingface.co/datasets/hello-peiqi/DynaMem-DynaBench",""]
    dest=base/"report.md";dest.write_text("\n".join(lines),encoding="utf-8")
    return dest

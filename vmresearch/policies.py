"""Development-only rule selection, immutable freeze, and held-out comparison."""
from __future__ import annotations
from bisect import bisect_left,bisect_right
import hashlib
import math
from pathlib import Path
from .core import (atomic_json, digest, finite, init_project, protocol, read_json, read_jsonl,
                   score_prediction, sha256, summary, write_csv, write_jsonl)


def primary_rows(base:Path, scenes:list[str], scored:bool=True) -> list[dict]:
    p=protocol(base);all_rows=[]
    for scene in scenes:
        comp=read_json(base/"runs"/scene/"completion.json")
        if comp["errors"]:raise RuntimeError("Inference errors are not declines; resolve them before policy comparison.")
        path=base/"scoring"/scene/"scored.jsonl" if scored else base/"runs"/scene/"predictions.jsonl"
        rows=[r for r in read_jsonl(path) if r["cutoff"]==p["primary_cutoff"]]
        if len(rows)!=comp["primary_queries"]:raise ValueError("Incomplete primary batch")
        all_rows+=rows
    return all_rows

def feature_value(row:dict,name:str) -> float | None:
    if name=="recency":
        a=row.get("age_observation_steps")
        return -float(a) if finite(a) and float(a)>=0 else None
    if name=="similarity":
        a=row.get("similarity_max");return float(a) if finite(a) else None
    if name=="detector_support":
        a=row.get("detector_supported");return float(a) if isinstance(a,bool) else None
    raise ValueError("Unknown feature")

def cdf(value:float,sorted_training:list[float]) -> float:
    return (bisect_left(sorted_training,value)+bisect_right(sorted_training,value))/(2*len(sorted_training))

def reliability(row:dict,method:str,model:dict) -> float | None:
    if row["prediction_xyz"] is None or row["status"]=="error":return None
    if method=="age_only":return feature_value(row,"recency")
    if method!="combined":raise ValueError("Unknown method")
    if model["combined"]["status"]!="available":return None
    values=[]
    for name,train in model["combined"]["transforms"].items():
        v=feature_value(row,name)
        if v is None:return None
        values.append(cdf(v,train))
    return sum(values)/len(values)

def choose_threshold(scores:list[float],fraction:float) -> float | None:
    if not scores or not 0<fraction<=1:raise ValueError("Invalid retention threshold inputs")
    if fraction==1:return None  # Explicit full-retention anchor, including held-out ranges.
    k=max(1,math.ceil(fraction*len(scores)))
    return sorted(scores,reverse=True)[k-1]

def _tie(row:dict) -> str:
    return hashlib.sha256(row["query_id"].encode()).hexdigest()

def freeze(base:Path) -> dict:
    p=init_project(base);rows=primary_rows(base,p["development_scenes"])
    source_hashes={s:sha256(base/"scoring"/s/"scored.jsonl") for s in p["development_scenes"]}
    file=base/"analysis/frozen_policies.json"
    if file.exists():
        old=load_frozen(base)
        if old["development_files_sha256"]!=source_hashes:raise ValueError("Development predictions changed after freeze")
        return old
    known=[r for r in rows if r["correct"] is not None]
    answers=[r for r in known if r["prediction_xyz"] is not None]
    if len(answers)<p["minimum_development_answers"]:
        raise RuntimeError("Insufficient scorable development answers; no pretend calibration.")
    if any(feature_value(r,"recency") is None for r in answers):raise ValueError("Missing evidence ages")
    model={"schema_version":3,"protocol_sha256":digest(p),"development_scenes":p["development_scenes"],
           "evaluation_scenes":p["evaluation_scenes"],"development_files_sha256":source_hashes,
           "scorable_development_queries":len(known),"development_answers":len(answers),
           "selection":"Only development rows; no held-out labels. Fixed equal weights, quantile retention targets; not calibrated probabilities.",
           "combined":{"status":"unavailable","transforms":{},"reason":None},"thresholds":{}}
    usable={};missing=[]
    for name in p["combined_features"]:
        vals=[feature_value(r,name) for r in answers]
        if any(v is None for v in vals):missing.append(name);continue
        if name=="detector_support" and len(set(vals))<2:continue  # Constant evidence contributes no ranking information.
        usable[name]=sorted(vals)
    if "recency" in usable and "similarity" in usable and len(set(usable["similarity"]))>1:
        model["combined"]={"status":"available","transforms":usable,"reason":None,
                           "missing_or_constant_optional_features":[x for x in p["combined_features"] if x not in usable]}
    else:model["combined"]["reason"]="Need finite, varying query similarity plus age; absent signals are not fabricated."
    dev_metrics=[]
    for method in ["age_only"]+(["combined"] if model["combined"]["status"]=="available" else []):
        scores=[reliability(r,method,model) for r in answers]
        model["thresholds"][method]={}
        for f in p["retention_fractions"]:
            threshold=choose_threshold(scores,f)
            model["thresholds"][method][str(f)]=threshold
            kept={r["query_id"] for r in answers if threshold is None or reliability(r,method,model)>=threshold}
            applied=[score_prediction(gate_prediction(r,r["query_id"] in kept),r) for r in known]
            dev_metrics.append({"method":method,"target_retained_fraction":f,"threshold":threshold,
                                "actual_retained_fraction":len(kept)/len(answers),**summary(applied)})
    model["frozen_sha256"]=digest(model)
    atomic_json(file,model);write_csv(base/"analysis/development_operating_points.csv",dev_metrics)
    return model

def load_frozen(base:Path) -> dict:
    m=read_json(base/"analysis/frozen_policies.json");h=m.pop("frozen_sha256")
    if digest(m)!=h:raise ValueError("Frozen policy file was modified")
    m["frozen_sha256"]=h
    if m["protocol_sha256"]!=digest(protocol(base)):raise ValueError("Protocol changed after freeze")
    for s,h in m["development_files_sha256"].items():
        if sha256(base/"scoring"/s/"scored.jsonl")!=h:raise ValueError("Development data changed after freeze")
    return m

def gate_prediction(row:dict,keep:bool) -> dict:
    # Strip truth/score fields before an applied-policy record is persisted.
    excluded={"target_xyz","tolerance_m","label_status","correct","outcome","distance_error_m",
              "nearest_annotation_query","nearest_annotation_distance_m","possible_wrong_object_match"}
    out={k:v for k,v in row.items() if k not in excluded}
    out["baseline_prediction_xyz"]=row["prediction_xyz"]
    if not keep:out["prediction_xyz"]=None
    out["policy_declined"]=not keep and row["prediction_xyz"] is not None
    out["decision_reason"]=("additional_evidence_rule_rejection_not_proof_of_absence" if out["policy_declined"]
                            else "baseline_decline_not_proof_of_absence" if row["prediction_xyz"] is None
                            else "baseline_location_retained")
    return out

def evaluate_heldout(base:Path) -> dict:
    p=protocol(base);m=load_frozen(base)
    raw=primary_rows(base,p["evaluation_scenes"],scored=False)
    methods=["age_only"]+(["combined"] if m["combined"]["status"]=="available" else [])
    missing={}
    for method in list(methods):
        bad=[r["query_id"] for r in raw if r["prediction_xyz"] is not None and reliability(r,method,m) is None]
        if bad:missing[method]=bad;methods.remove(method)
    # First save label-free operational predictions using frozen thresholds.
    applied=[]
    for r in raw:
        applied.append({**gate_prediction(r,True),"policy":"baseline"})
        applied.append({**gate_prediction(r,False),"policy":"always_decline"})
        for method in methods:
            val=reliability(r,method,m)
            for frac,threshold in m["thresholds"][method].items():
                keep=r["prediction_xyz"] is not None and val is not None and (threshold is None or val>=threshold)
                applied.append({**gate_prediction(r,keep),"policy":f"{method}@{frac}",
                                "reliability_score":val,"frozen_threshold":threshold})
    out=base/"analysis";write_jsonl(out/"heldout_policy_predictions.jsonl",applied)
    # No scoring labels were consulted above. Join labels only after saving predictions.
    scored=primary_rows(base,p["evaluation_scenes"],scored=True)
    labels={r["query_id"]:{k:r[k] for k in ("target_xyz","tolerance_m","label_status")} for r in scored}
    policy_scores=[score_prediction(r,labels[r["query_id"]]) for r in applied]
    write_jsonl(out/"heldout_policy_scores.jsonl",policy_scores)
    fixed=[]
    for policy_name in sorted({r["policy"] for r in policy_scores}):
        these=[r for r in policy_scores if r["policy"]==policy_name]
        fixed.append({"policy":policy_name,"scene":"pooled",**summary(these)})
        for scene in p["evaluation_scenes"]:
            fixed.append({"policy":policy_name,"scene":scene,**summary([r for r in these if r["scene"]==scene])})
    write_csv(out/"fixed_threshold_comparison.csv",fixed)
    # Equal quota is a retrospective, label-free RANKING test, not an online threshold policy.
    # Cohort is fixed by label availability, never by correctness.
    known=[r for r in scored if r["correct"] is not None]
    candidates=[r for r in known if r["prediction_xyz"] is not None]
    matched=[];chosen=[]
    for fraction in p["retention_fractions"]:
        k=math.ceil(len(candidates)*fraction)
        for method in methods:
            ranked=sorted(candidates,key=lambda r:(-reliability(r,method,m),_tie(r)))
            keep={r["query_id"] for r in ranked[:k]}
            rows=[score_prediction(gate_prediction(r,r["query_id"] in keep),labels[r["query_id"]]) for r in known]
            matched.append({"method":method,"retained_fraction_of_baseline_answers":fraction,
                            "exact_answer_quota":k,"evaluation":"retrospective equal-quota ranking; no test-label threshold tuning",
                            **summary(rows)})
            chosen.append({"method":method,"fraction":fraction,"selected_query_ids":sorted(keep)})
    write_csv(out/"matched_coverage_comparison.csv",matched);atomic_json(out/"matched_selected_ids.json",chosen)
    result={"status":"heldout_comparison_computed", "frozen_policies_sha256":m["frozen_sha256"],
            "development_scenes":p["development_scenes"],"evaluation_scenes":p["evaluation_scenes"],
            "fixed_threshold_metrics":fixed,"matched_coverage_metrics":matched,
            "skipped_methods_missing_heldout_features":missing,
            "combined_status":m["combined"]["status"],"cutoff":p["primary_cutoff"],
            "checkpoint_semantics":p["checkpoint_semantics"],"blank_label_semantics":p["blank_label_semantics"],
            "uncertainty":"No inferential confidence interval: default study has one held-out scene and correlated repeated-object queries.",
            "claim_limit":"Exploratory scene-disjoint pilot only; no official benchmark, novelty, calibration, or general improvement claim."}
    atomic_json(out/"comparison.json",result)
    return result

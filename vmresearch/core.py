"""Shared, model-independent contracts. All generated JSON is strict and atomic."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

COMMIT = "6272b28e3ff7be482af3b427a4551d1772ebf7d7"
REVISION = "1d561e2de8718addf398e5e967f2c44dc015c366"
EXPERIMENT = "current_code_yoloe_diagnostic_not_paper_reproduction"
REFERENCE = Path(__file__).resolve().parent.parent / "reference"
DEFAULT_PROTOCOL = {
    "schema_version": 3, "code_commit": COMMIT, "dataset_revision": REVISION,
    "experiment_type": EXPERIMENT,
    "development_scenes": ["env1"], "evaluation_scenes": ["env2"],
    "primary_cutoff": "exclusive", "cutoff_sensitivity": True,
    "checkpoint_semantics": "unverified", "checkpoint_evidence": None,
    "blank_label_semantics": "unverified", "blank_label_evidence": None,
    "seed": 20261027,
    "retention_fractions": [0.25, 0.5, 0.75, 1.0],
    "primary_retention_fraction": 0.75,
    "combined_features": ["recency", "similarity", "detector_support"],
    "minimum_development_answers": 5,
    "policy_selection": "equal weights; development-only rank transforms and thresholds",
    "age_unit": "number of observations since supporting frame, NOT elapsed seconds",
    "source_config": "upstream defaults; no model, threshold, or color-path edits",
}

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def code_digest() -> str:
    return digest({p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})

def finite(value: Any) -> bool:
    try: return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError): return False

def xyz(value: Any) -> list[float] | None:
    if value is None: return None
    if hasattr(value, "detach"): value = value.detach().cpu().numpy()
    if hasattr(value, "reshape"): value = value.reshape(-1).tolist()
    if len(value) != 3 or not all(finite(v) for v in value):
        raise ValueError(f"Expected three finite coordinates, got {value!r}")
    return [float(v) for v in value]

def protocol(base: Path) -> dict:
    p = read_json(base / "protocol.json")
    dev, test = p["development_scenes"], p["evaluation_scenes"]
    if not dev or not test or set(dev) & set(test):
        raise ValueError("Development and evaluation scenes must be nonempty and disjoint.")
    if len(set(dev + test)) != len(dev + test): raise ValueError("Duplicate scene.")
    if p["primary_cutoff"] not in ("exclusive", "inclusive"):
        raise ValueError("Unknown cutoff.")
    for flag, evidence in (("checkpoint_semantics", "checkpoint_evidence"),
                           ("blank_label_semantics", "blank_label_evidence")):
        if p[flag] == "confirmed" and not p[evidence]:
            raise ValueError(f"{flag}: author/runner source and exact excerpt are required.")
    return p

def init_project(base: Path) -> dict:
    base.mkdir(parents=True, exist_ok=True)
    if not (base / "protocol.json").exists(): atomic_json(base / "protocol.json", DEFAULT_PROTOCOL)
    p = protocol(base)
    marker = base / "plan_lock.json"
    plan = {"protocol_sha256": digest(p), "harness_sha256": code_digest()}
    if marker.exists() and read_json(marker) != plan:
        raise ValueError("Plan or harness changed. Use a NEW project directory; do not overwrite a frozen study.")
    if not marker.exists(): atomic_json(marker, plan)
    return p

def scene_role(p: dict, scene: str) -> str:
    if scene in p["development_scenes"]: return "development"
    if scene in p["evaluation_scenes"]: return "evaluation"
    raise ValueError(f"Scene {scene!r} is not in the predeclared split.")

def eligible_count(checkpoint: int, mode: str, n: int) -> int:
    if type(checkpoint) is not int or checkpoint <= 0: raise ValueError("Invalid checkpoint")
    if mode not in ("exclusive", "inclusive"): raise ValueError("Unknown cutoff")
    count = checkpoint + (mode == "inclusive")
    if count > n: raise ValueError(f"Cutoff {mode} {checkpoint} exceeds {n} frames")
    return int(count)

def validate_prediction(r: dict) -> None:
    n = r["n_observations"]
    if r["eligible_frame_indices"] != list(range(n)): raise ValueError("Non-prefix input frames")
    if n != eligible_count(r["checkpoint"], r["cutoff"], r["scene_frame_count"]):
        raise ValueError("Inconsistent observation cutoff")
    if r["status"] == "error": return
    loc = xyz(r["prediction_xyz"])
    idx = r.get("evidence_frame_index")
    if idx is not None and not (0 <= idx < n): raise ValueError("Future/out-of-range evidence")
    if loc is not None and idx is None: raise ValueError("Returned location has no eligible evidence")
    if idx is not None and r.get("age_observation_steps") != n - 1 - idx:
        raise ValueError("Incorrect evidence age")

def parse_label(row: dict, blank_is_negative: bool = False) -> dict:
    vals = [str(row.get(k, "")).strip() for k in ("x", "y", "z", "tol")]
    if all(not s for s in vals):
        return {"label_status": "negative" if blank_is_negative else "unknown_blank",
                "target_xyz": None, "tolerance_m": None}
    if any(not s for s in vals): raise ValueError("Partially missing label; cannot assume negative")
    if not all(finite(s) for s in vals) or float(vals[3]) < 0:
        raise ValueError("Nonfinite or invalid label")
    return {"label_status": "positive", "target_xyz": list(map(float, vals[:3])),
            "tolerance_m": float(vals[3])}

def query_id(scene: str, checkpoint: int, row: int) -> str:
    return f"{scene}_q{checkpoint:04d}_{row:03d}"

def score_prediction(r: dict, label: dict) -> dict:
    out = {**r, **{k: label[k] for k in ("label_status", "target_xyz", "tolerance_m")}}
    if r["status"] == "error":
        return {**out, "correct": None, "distance_error_m": None, "outcome": "inference_error"}
    location = xyz(r["prediction_xyz"])
    d = None
    if label["label_status"] == "unknown_blank":
        correct = None; outcome = "unscored_location" if location is not None else "unscored_decline"
    elif label["label_status"] == "negative":
        correct = location is None; outcome = "correct_not_found" if correct else "wrong_location"
    else:
        d = math.dist(location, label["target_xyz"]) if location is not None else None
        correct = d is not None and d <= label["tolerance_m"]
        outcome = "decline" if location is None else ("correct_location" if correct else "wrong_location")
    return {**out, "distance_error_m": d, "correct": correct, "outcome": outcome}

def summary(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["status"] != "error"]
    known = [r for r in valid if r.get("correct") is not None]
    answered = [r for r in valid if r["prediction_xyz"] is not None]
    ka = [r for r in known if r["prediction_xyz"] is not None]
    pos = [r for r in known if r["label_status"] == "positive"]
    neg = [r for r in known if r["label_status"] == "negative"]
    rate = lambda a,b: a / b if b else None
    return {
        "queries": len(rows), "inference_errors": len(rows)-len(valid),
        "answers": len(answered), "declines": len(valid)-len(answered),
        "known_label_queries": len(known), "unscored_label_queries": len(valid)-len(known),
        "correct_locations": sum(r["correct"] for r in ka),
        "wrong_locations": sum(not r["correct"] for r in ka),
        "unscored_locations": len(answered)-len(ka),
        "coverage_all_queries": rate(len(answered),len(rows)),
        "coverage_scorable_queries": rate(len(ka),len(known)),
        "conditional_error_scorable": rate(sum(not r["correct"] for r in ka),len(ka)),
        "precision_scorable": rate(sum(r["correct"] for r in ka),len(ka)),
        "positive_localization_recall": rate(sum(r["correct"] for r in pos),len(pos)),
        "negative_not_found_accuracy": rate(sum(r["correct"] for r in neg),len(neg)),
        "negative_false_location_rate": rate(sum(r["prediction_xyz"] is not None for r in neg),len(neg)),
        "success_scorable": rate(sum(r["correct"] for r in known),len(known)),
    }

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text("", encoding="utf-8"); return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})

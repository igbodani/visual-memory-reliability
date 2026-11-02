"""Validate a non-destructive, four-stream scene view for the unchanged v3 harness.

This repair does not edit vmresearch, raw data, model settings, study locks,
preflight records, predictions, or frozen policies. It does not guess key aliases,
geometry, units, timing conventions, or the meaning of extra fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from vmresearch.core import (REVISION, atomic_json, code_digest, init_project,
                             read_json, scene_role, sha256)
from vmresearch.data import ArrayTensorUnpickler, download_scene, load_scene, prepare_scene
from vmresearch.policies import load_frozen

REQUIRED = ("rgb", "depth", "camera_poses", "camera_K")
REPAIR_ID = "env2-four-stream-view-v1"


def fingerprint_streams(env: dict) -> dict:
    """Hash each frame's type, shape, dtype and numeric bytes in original order."""
    result = {}
    for key in REQUIRED:
        values = env[key]
        try:
            count = len(values)
        except TypeError as exc:
            raise ValueError(f"{key}: expected a sequence of frames") from exc
        h = hashlib.sha256()
        h.update(f"{type(values).__module__}.{type(values).__name__}:{count}".encode())
        for i, value in enumerate(values):
            a = np.asarray(value)
            if a.dtype.hasobject:
                raise ValueError(f"{key}[{i}]: object arrays are not supported; no conversion attempted")
            header = json.dumps({"index": i, "type": f"{type(value).__module__}.{type(value).__name__}",
                                 "shape": list(a.shape), "dtype": a.dtype.str}, sort_keys=True).encode()
            body = a.tobytes(order="C")
            h.update(len(header).to_bytes(8, "little")); h.update(header)
            h.update(len(body).to_bytes(8, "little")); h.update(body)
        result[key] = {"frames": count, "sha256": h.hexdigest()}
    return result


def protected_hashes(base: Path, protocol: dict) -> dict:
    """Read-only snapshots of the frozen plan and existing development results."""
    paths = [base / p for p in ("protocol.json", "plan_lock.json", "preflight.json",
                               "model_lock.json", "analysis/frozen_policies.json")]
    for scene in protocol["development_scenes"]:
        paths += [base / folder / scene / name for folder, name in (
            ("runs", "run_manifest.json"), ("runs", "predictions.jsonl"),
            ("runs", "completion.json"), ("scoring", "scored.jsonl"), ("scoring", "metrics.json"))]
    return {str(p.relative_to(base)): sha256(p) for p in paths if p.is_file()}


def stage_and_prepare(base: Path, repo: Path, scene: str = "env2", *, allow_download: bool = True) -> dict:
    base = base.resolve(); repo = repo.resolve()
    p = init_project(base)  # Keeps the existing protocol AND harness-hash guard.
    if scene_role(p, scene) != "evaluation":
        raise ValueError("This repair is only for a predeclared held-out scene, not development data")
    load_frozen(base)  # Verify rules and development-score hashes; never refit.
    before = protected_hashes(base, p)
    raw = base / "data" / "DynaBench" / scene
    if not (raw / "env.pkl").is_file() or not (raw / "time.txt").is_file():
        if not allow_download:
            raise FileNotFoundError(f"Scene not downloaded: {raw}")
        raw = download_scene(base, scene)
    with (raw / "env.pkl").open("rb") as f:
        env = ArrayTensorUnpickler(f).load()
    actual = [str(k) for k in env] if isinstance(env, dict) else None
    print("Raw scene type:", type(env).__name__, flush=True)
    print("Raw scene keys:", actual, flush=True)
    if not isinstance(env, dict):
        raise ValueError(f"Unsupported scene container {type(env).__name__}; no layout was guessed. "
                         "Share the Raw scene type/keys output. Original data and study are unchanged.")
    missing = [k for k in REQUIRED if k not in env]
    if missing:
        raise ValueError(f"Missing required fields: {missing}; available keys: {actual}. "
                         "This is not an extra-fields-only case. No renaming or geometry conversion was attempted.")
    extra = sorted(str(k) for k in env if k not in REQUIRED)
    view = {k: env[k] for k in REQUIRED}  # Same objects, no slicing/transposing/scaling/reordering.
    original_fingerprints = fingerprint_streams(view)
    print("Fields excluded from the model-input copy:", extra, flush=True)
    print("Original frame counts:", {k: v["frames"] for k, v in original_fingerprints.items()}, flush=True)
    # CSV files are copied byte-for-byte; their coordinate labels are not parsed here.
    source_files = [raw / "env.pkl", raw / "time.txt"] + sorted(
        f for f in raw.glob("*.csv") if f.stem.isdecimal())
    if len(source_files) <= 2:
        raise ValueError("No checkpoint CSV files found")
    raw_hashes = {f.name: sha256(f) for f in source_files}
    root = base / "data" / "DynaBench_model_inputs"
    dest = root / scene
    prepared = base / "prepared" / scene
    if (prepared / "manifest.json").exists():
        old_root = Path(read_json(prepared / "manifest.json")["data_root"]).resolve()
        if old_root != dest.resolve():
            raise RuntimeError("A scene manifest already points elsewhere. It was not overwritten; "
                               "review that existing prepared scene before changing its input path.")
    adapter_source = Path(__file__).resolve()
    adapter_hash = sha256(adapter_source)
    audit_name = "schema_adapter_audit.json"
    root.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if not (dest / audit_name).is_file():
            raise RuntimeError(f"Unrecognized existing input-copy directory, left untouched: {dest}")
        audit = read_json(dest / audit_name)
        if (audit.get("repair_id") != REPAIR_ID or audit.get("source_files_sha256") != raw_hashes
                or audit.get("adapter_sha256") != adapter_hash):
            raise RuntimeError("Existing input copy has different source/adapter provenance; not overwritten")
        if audit.get("protected_before") != before:
            raise RuntimeError("Frozen plan or development result hashes changed since this copy was created")
        if sha256(dest / "env.pkl") != audit["model_input_pickle_sha256"]:
            raise RuntimeError("Model-input pickle changed after staging")
        normalized = load_scene(dest / "env.pkl")
        if fingerprint_streams(normalized) != original_fingerprints:
            raise RuntimeError("Input-copy stream values differ from the original scene")
        for f in source_files[1:]:
            if sha256(dest / f.name) != raw_hashes[f.name]:
                raise RuntimeError(f"Copied checkpoint file changed: {f.name}")
        if sha256(dest / "schema_adapter.py") != adapter_hash:
            raise RuntimeError("Saved adapter source changed")
        print("Reusing verified input copy:", dest, flush=True)
        del normalized
    else:
        import pickle
        # Commit the directory only after validation; a failed validation leaves no final copy.
        with tempfile.TemporaryDirectory(prefix=f".{scene}-staging-", dir=root) as temp:
            stage = Path(temp)
            with (stage / "env.pkl").open("wb") as f:
                pickle.dump(view, f, protocol=4)
            normalized = load_scene(stage / "env.pkl")  # Retain ALL original frame/camera checks.
            if fingerprint_streams(normalized) != original_fingerprints:
                raise RuntimeError("Serialization changed a required stream; aborting")
            del normalized
            for f in source_files[1:]:
                shutil.copyfile(f, stage / f.name)
                if sha256(stage / f.name) != raw_hashes[f.name]:
                    raise RuntimeError(f"Checkpoint-file copy verification failed: {f.name}")
            shutil.copyfile(adapter_source, stage / "schema_adapter.py")
            audit = {
                "repair_id": REPAIR_ID,
                "source_dataset_revision": REVISION,
                "original_scene_directory": str(raw.resolve()),
                "model_input_directory": str(dest.resolve()),
                "source_files_sha256": raw_hashes,
                "model_input_pickle_sha256": sha256(stage / "env.pkl"),
                "required_streams": list(REQUIRED), "original_top_level_keys": actual,
                "excluded_top_level_keys": extra, "stream_fingerprints": original_fingerprints,
                "adapter_sha256": adapter_hash, "unchanged_harness_sha256": code_digest(),
                "protected_before": before,
                "scope": "Four-stream projection only. No data values, frame order, units, geometry, "
                         "checkpoints, models, policies, labels, or scores were changed.",
                "limitations": "Excluded fields are not automatically meaningless. Timing/blank-label "
                               "conventions remain unresolved. No key aliases were guessed. "
                               "This is a derived input container, not an unmodified raw pickle.",
            }
            atomic_json(stage / audit_name, audit)
            stage.rename(dest)
    # All raw source files must still be unchanged.
    if {f.name: sha256(f) for f in source_files} != raw_hashes:
        raise RuntimeError("Raw source changed during staging")
    del view, env
    manifest = prepare_scene(base, scene, repo, data_root=root)
    # The existing exporter includes prepared/*, so the repair's provenance travels with results.
    shutil.copyfile(dest / audit_name, prepared / audit_name)
    shutil.copyfile(adapter_source, prepared / "schema_adapter.py")
    if protected_hashes(base, p) != before:
        raise RuntimeError("Protected study files changed; stop before inference")
    print("SCHEMA REPAIR PASSED: required streams, raw files, study locks and frozen rules preserved.", flush=True)
    print("Prepared queries:", manifest["query_count"], "checkpoints:", manifest["checkpoints"], flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/content/visual_memory"))
    parser.add_argument("--repo", type=Path, default=Path("/content/stretch_ai"))
    parser.add_argument("--scene", default="env2")
    args = parser.parse_args()
    stage_and_prepare(args.base, args.repo, args.scene)


if __name__ == "__main__":
    main()

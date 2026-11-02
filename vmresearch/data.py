"""Data adapters and convention audits. No model receives a label table."""
from __future__ import annotations
import csv
import hashlib
import io
import pickle
import re
from pathlib import Path
import numpy as np
from .core import (REVISION, atomic_json, digest, eligible_count, init_project,
                   protocol, query_id, read_json, read_jsonl, scene_role, sha256, write_jsonl)

class ArrayTensorUnpickler(pickle.Unpickler):
    """Allow numeric arrays/tensors in the pinned dataset, not arbitrary globals.

    Tensor storage is restored through torch.load(weights_only=True) on CPU.
    """
    def find_class(self, module, name):
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(np.core.multiarray, name)
        if module in ("numpy.core.numeric", "numpy._core.numeric") and name == "_frombuffer":
            return np.core.numeric._frombuffer
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
            import torch._utils
            return getattr(torch._utils, name)
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            return lambda value: torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)
        raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}")

def load_scene(path: Path) -> dict:
    with path.open("rb") as f: env = ArrayTensorUnpickler(f).load()
    keys = ("rgb", "depth", "camera_poses", "camera_K")
    if not isinstance(env, dict) or set(env) != set(keys): raise ValueError("Unexpected scene schema")
    counts = [len(env[k]) for k in keys]
    if len(set(counts)) != 1 or not counts[0]: raise ValueError("Unequal or empty observation arrays")
    for i in range(counts[0]):
        rgb,d,K,pose = (np.asarray(env[k][i]) for k in ("rgb","depth","camera_K","camera_poses"))
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.shape[:2] != d.shape:
            raise ValueError(f"Frame {i}: RGB/depth shape mismatch")
        if K.shape != (3,3) or pose.shape != (4,4): raise ValueError(f"Frame {i}: invalid camera shape")
        if not np.isfinite(K).all() or not np.isfinite(pose).all(): raise ValueError("Nonfinite calibration")
        if K[0,0] <= 0 or K[1,1] <= 0 or not np.allclose(pose[3], [0,0,0,1], atol=1e-5):
            raise ValueError("Invalid pinhole / homogeneous transform")
    return env

def download_scene(base: Path, scene: str) -> Path:
    from huggingface_hub import snapshot_download
    p = init_project(base)
    role = scene_role(p, scene)
    if role == "evaluation" and not (base / "analysis/frozen_policies.json").exists():
        raise RuntimeError("Freeze development policies BEFORE fetching evaluation data.")
    root = base / "data/DynaBench"
    snapshot_download(repo_id="hello-peiqi/DynaMem-DynaBench", repo_type="dataset", revision=REVISION,
                      allow_patterns=f"{scene}/*", local_dir=str(root))
    # A failed download must never be reported as an empty/negative scene.
    if not (root/scene/"env.pkl").is_file() or not (root/scene/"time.txt").is_file():
        raise FileNotFoundError(f"Incomplete scene download: {root/scene}")
    return root/scene

def _scan_source(repo: Path) -> list[dict]:
    """Collect possible author-runner evidence; string matches do NOT certify a cutoff."""
    matches = []
    if not repo.exists(): return matches
    for p in sorted(repo.rglob("*")):
        if not p.is_file() or p.suffix not in (".py", ".md", ".yaml", ".yml", ".txt"): continue
        if any(x in p.parts for x in (".git", "third_party", ".venv", "node_modules")): continue
        if p.stat().st_size > 1_000_000: continue
        text = p.read_text(errors="replace")
        lines = text.splitlines()
        for i,line in enumerate(lines):
            if any(s in line.lower() for s in ("dynabench", "time.txt", "env.pkl")):
                matches.append({"path": str(p.relative_to(repo)), "line": i+1,
                                "sha256": sha256(p), "excerpt": "\n".join(lines[max(0,i-3):i+6])})
    return matches[:100]

def prepare_scene(base: Path, scene: str, repo: Path, data_root: Path | None = None) -> dict:
    p = init_project(base); role = scene_role(p, scene)
    if role == "evaluation" and not (base / "analysis/frozen_policies.json").exists():
        raise RuntimeError("Do not inspect or prepare evaluation data before policies are frozen.")
    root = (data_root or base/"data/DynaBench") / scene
    env = load_scene(root/"env.pkl"); n = len(env["rgb"])
    checkpoints = [int(s) for s in (root/"time.txt").read_text().split()]
    if not checkpoints or checkpoints != sorted(set(checkpoints)):
        raise ValueError("Checkpoint file must be strictly increasing")
    rows = []; counts = {}
    modes = [p["primary_cutoff"]] + (["inclusive" if p["primary_cutoff"] == "exclusive" else "exclusive"]
                                     if p["cutoff_sensitivity"] else [])
    boundaries = []; cutoff_modes = {}
    for cp in checkpoints:
        eligible_count(cp, p["primary_cutoff"], n)
        cutoff_modes[str(cp)] = []
        for mode in modes:
            try: eligible_count(cp, mode, n)
            except ValueError: continue  # No boundary frame exists for this sensitivity branch.
            cutoff_modes[str(cp)].append(mode)
        with (root/f"{cp}.csv").open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["query","x","y","z","tol"]: raise ValueError("Unexpected query CSV schema")
            texts = [r["query"].strip() for r in reader]  # Coordinates not exported to inference.
        if any(not t or any(c in t for c in ("/","\\","\n","\x00")) for t in texts):
            raise ValueError("Empty or unsafe query text (upstream uses it in image filenames)")
        counts[str(cp)] = len(texts)
        rows += [{"query_id": query_id(scene,cp,i), "scene": scene, "checkpoint": cp,
                  "row_index": i, "query": text} for i,text in enumerate(texts)]
        boundaries.append({"checkpoint":cp, "exclusive_indices":[0,cp-1],
                           "inclusive_indices":[0,cp] if cp<n else None, "source_timestamps_present": False})
    if scene == "env1" and counts != {"14":18,"26":18,"41":18}:
        raise ValueError(f"Pinned env1 changed: {counts}")
    prepared = base/"prepared"/scene; prepared.mkdir(parents=True, exist_ok=True)
    hashes = {f.name: sha256(f) for f in sorted(root.iterdir()) if f.is_file()}
    manifest = {"scene":scene, "role":role, "frame_count":n, "checkpoints":checkpoints,
                "queries_per_checkpoint":counts, "query_count":len(rows), "file_hashes":hashes,
                "cutoff_modes":cutoff_modes,
                "dataset_revision":REVISION, "data_root":str(root.resolve()),
                "protocol_sha256":digest(p)}
    old = prepared/"manifest.json"
    if old.exists() and read_json(old) != manifest:
        raise ValueError("Prepared data changed. Use a new project directory.")
    atomic_json(old, manifest); write_jsonl(prepared/"queries.jsonl",rows)
    evidence = _scan_source(repo)
    audit = {
        "status":p["checkpoint_semantics"], "verified_automatically":False,
        "stated_evidence":p["checkpoint_evidence"], "frame_count":n,
        "observation_id_mapping":"Upstream increments obs_count before add: ID = array index + 1.",
        "dataset_checkpoint_mapping":"NOT established by upstream observation IDs or frame counts.",
        "boundary_cases":boundaries, "source_evidence_candidates":evidence,
        "primary_rule":p["primary_cutoff"],
        "sensitivity_warning":"Inclusive branch is a separate diagnostic; may include a forbidden frame under exclusive semantics. Never pool branches or select the better-scoring branch.",
        "blank_label_semantics":p["blank_label_semantics"],
        "resolution_needed":"Author/exporter documentation or official runner mapping CSV names to input slices. Boundary images and good scores cannot prove semantics.",
    }
    atomic_json(prepared/"convention_audit.json", audit)
    from PIL import Image, ImageDraw
    for cp in checkpoints:
        imgs=[]
        for idx in [j for j in [cp-1,cp] if j<n]:
            im=Image.fromarray(np.asarray(env["rgb"][idx]).clip(0,255).astype("uint8")).convert("RGB")
            panel=Image.new("RGB",(im.width,im.height+32),"white");panel.paste(im,(0,32))
            ImageDraw.Draw(panel).text((6,8),f"array frame {idx} / upstream ID {idx+1}",fill="black")
            imgs.append(panel)
        canvas=Image.new("RGB",(sum(x.width for x in imgs),max(x.height for x in imgs)),"white")
        offset=0
        for im in imgs: canvas.paste(im,(offset,0));offset+=im.width
        canvas.save(prepared/f"checkpoint_{cp}_boundary.png")
    return manifest

def verify_manifest(base: Path, scene: str) -> dict:
    p=protocol(base); m=read_json(base/"prepared"/scene/"manifest.json")
    if m["protocol_sha256"] != digest(p): raise ValueError("Protocol changed")
    root=Path(m["data_root"])
    for name,h in m["file_hashes"].items():
        if sha256(root/name) != h: raise ValueError(f"Dataset bytes changed: {name}")
    return m

"""Pinned upstream model execution; auditing and file I/O do not replace its rules."""
from __future__ import annotations
import csv
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from importlib.metadata import version
import numpy as np
from .core import (COMMIT, EXPERIMENT, REFERENCE, REVISION, atomic_json, code_digest, digest,
                   init_project, protocol, read_json, read_jsonl, scene_role, sha256,
                   validate_prediction, write_jsonl, xyz)
from .data import load_scene, verify_manifest
from .instrumentation import CallAudit, array, box_overlay, preserve_rng, scalar

PACKAGE_NAMES=("torch","torchvision","transformers","tokenizers","huggingface-hub",
               "setuptools","stretch-ai","sam-2","open3d","numpy","ultralytics")

def runtime_record(repo: Path) -> dict:
    import torch
    return {"python":sys.version,"executable":sys.executable,
            "packages":{k:version(k) for k in PACKAGE_NAMES},
            "code_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
            "source_modified":subprocess.check_output(["git","-C",str(repo),"status","--porcelain","--untracked-files=no"],text=True).strip(),
            "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "harness_sha256":code_digest()}

def preflight(base: Path, repo: Path) -> dict:
    init_project(base)
    if str(repo/"src") not in sys.path: sys.path.insert(0,str(repo/"src"))
    import torch
    from stretch.mapping.voxel.voxel_dynamem import SparseVoxelMap
    from stretch.perception.encoders.siglip_encoder import MaskSiglipEncoder
    from transformers import Qwen2_5_VLForConditionalGeneration
    import sam2
    r=runtime_record(repo)
    if sys.version_info[:2]!=(3,10) or r["code_commit"]!=COMMIT or r["source_modified"]:
        raise RuntimeError("Use Python 3.10 and the unchanged pinned source commit.")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; do not silently switch models/CPU modes")
    expected=read_json(REFERENCE/"compatibility_pass.json")["packages"]
    differences={k:{"recorded":v,"current":r["packages"][k]} for k,v in expected.items() if r["packages"][k]!=v}
    if differences:
        atomic_json(base/"preflight_version_differences.json",differences)
        raise RuntimeError("Versions differ from the working run. See preflight_version_differences.json; do not silently alter the baseline.")
    encoder=MaskSiglipEncoder(device="cuda")
    x=encoder.encode_text("toy owl")
    if not torch.is_tensor(x) or not torch.isfinite(x).all(): raise RuntimeError("SigLIP smoke test failed")
    r["siglip_text_shape"]=list(x.shape)
    r["known_dependency_overrides"]=["setuptools 69.5.1 below torch's declared >=77.0.3",
                                      "transformers 4.49.0 below stretch-ai's declared >=5.3.0"]
    atomic_json(base/"preflight.json",r)
    # Export the actual environment rather than pretending the metadata conflicts are clean.
    try:
        packages=subprocess.check_output(["uv","pip","freeze","--python",sys.executable],text=True)
        (base/"working_environment.txt").write_text(packages)
    except (OSError,subprocess.CalledProcessError) as e:
        (base/"working_environment.txt").write_text("Freeze unavailable: "+str(e))
    del encoder;gc.collect();torch.cuda.empty_cache()
    return r

def make_models(repo: Path, need_encoder: bool=True):
    if str(repo/"src") not in sys.path: sys.path.insert(0,str(repo/"src"))
    from stretch.perception import create_semantic_sensor
    from stretch.perception.encoders.siglip_encoder import MaskSiglipEncoder
    from stretch.mapping.voxel.voxel_dynamem import SparseVoxelMap
    encoder=MaskSiglipEncoder(device="cuda") if need_encoder else None
    if encoder is not None:
        import torch
        x=encoder.encode_text("toy owl")
        if not torch.is_tensor(x) or not torch.isfinite(x).all(): raise RuntimeError("SigLIP test failed")
    sensor=create_semantic_sensor(device_id=0,verbose=True)
    detector=sensor if callable(getattr(sensor,"compute_obj_coord",None)) else getattr(sensor,"_segmentation",None)
    backend=getattr(sensor,"_segmentation",detector)
    if detector is None or not callable(getattr(detector,"compute_obj_coord",None)):
        raise RuntimeError("Selected detector lacks compute_obj_coord")
    if "yoloe" not in type(backend).__module__.lower(): raise RuntimeError("Detector changed from recorded YOLOE")
    # The selected backend itself contains the auditable methods in the pinned working run.
    if detector is not backend: raise RuntimeError("Unexpected detector wrapper; validate before changing instrumentation")
    return encoder,detector,SparseVoxelMap

def model_metadata(repo: Path, encoder,detector) -> dict:
    out={"encoder":"google/siglip-base-patch16-224", "detector_backend":type(detector).__name__,
         "detector_confidence":scalar(getattr(detector,"confidence",None)),
         "encoder_revision":getattr(getattr(getattr(encoder,"model",None),"config",None),"_commit_hash",None),
         "default_config_sha256":sha256(repo/"src/stretch/config/default_planner.yaml"),
         "detector_source_sha256":sha256(Path(inspect.getfile(type(detector)))),
         "local_weights":{}}
    for name in ("yoloe-v8l-seg.pt","mobileclip_blt.ts"):
        p=repo/name
        if p.exists():out["local_weights"][name]=sha256(p)
    return out

def _seed(seed: int):
    import torch
    random.seed(seed);np.random.seed(seed % 2**32);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def _save_image(a,path):
    from PIL import Image
    Image.fromarray(array(a).clip(0,255).astype("uint8")).save(path)

def run_scene(base: Path, repo: Path, scene: str) -> dict:
    import torch
    p=init_project(base);m=verify_manifest(base,scene);role=scene_role(p,scene)
    if role=="evaluation" and not (base/"analysis/frozen_policies.json").exists():
        raise RuntimeError("Freeze development policies first")
    check=read_json(base/"preflight.json")
    now=runtime_record(repo)
    for k in ("code_commit","packages","harness_sha256"):
        if now[k]!=check[k]:raise RuntimeError(f"Environment changed after preflight: {k}")
    if now["source_modified"]:raise RuntimeError("Upstream source modified")
    os.chdir(repo)
    run=base/"runs"/scene;run.mkdir(parents=True,exist_ok=True)
    queries=read_jsonl(base/"prepared"/scene/"queries.jsonl")
    manifest={"protocol_sha256":digest(p),"prepared_manifest_sha256":digest(m),
              "runtime":{k:now[k] for k in ("packages","code_commit","harness_sha256")}}
    if (run/"run_manifest.json").exists() and read_json(run/"run_manifest.json")!=manifest:
        raise RuntimeError("Cannot resume a run with changed data, code, or environment")
    atomic_json(run/"run_manifest.json",manifest)
    modes=[p["primary_cutoff"]]
    if p["cutoff_sensitivity"]:modes+=["inclusive" if modes[0]=="exclusive" else "exclusive"]
    expected={(q["query_id"],mode) for q in queries for mode in m["cutoff_modes"][str(q["checkpoint"]) ]}
    complete=True
    for qid,mode in expected:
        fp=run/mode/qid/"prediction.json"
        if not fp.exists() or read_json(fp)["status"]=="error": complete=False;break
    if complete:
        print("All saved predictions exist; skipping GPU replay.");return finalize_run(base,scene,modes)
    env=load_scene(Path(m["data_root"])/"env.pkl")
    encoder,detector,Memory=make_models(repo)
    model=model_metadata(repo,encoder,detector)
    if model.get("encoder_revision") not in (None,"7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"):
        raise RuntimeError("SigLIP model revision changed from the recorded download; do not silently change weights.")
    model_lock=base/"model_lock.json"
    if model_lock.exists() and read_json(model_lock)!=model:
        raise RuntimeError("Model/configuration changed across scenes. Keep development and evaluation identical.")
    if not model_lock.exists():atomic_json(model_lock,model)
    model_file=run/"model_metadata.json"
    if model_file.exists() and read_json(model_file)!=model:raise RuntimeError("Model metadata/weight hash changed")
    atomic_json(model_file,model)
    logs=run/"memory_inputs";logs.mkdir(exist_ok=True)
    seed=int(p["seed"])+int(hashlib.sha256(scene.encode()).hexdigest()[:6],16)
    _seed(seed)
    memory=Memory(encoder=encoder,detection=detector,device="cuda",log=str(logs),mllm=False,run_eqa=False)
    atomic_json(run/"settings.json",{"seed":seed,"memory_image_shape":list(memory.image_shape),
                "min_depth":memory.min_depth,"max_depth":memory.max_depth,
                "query_default_signature":str(inspect.signature(memory.localize_with_feature_similarity)),
                "rng_note":"Fixed scene seed. Each query restores RNG states so queries/auditing cannot alter later memory subsampling.",
                "prior_run_seed":"Not recorded; newly seeded reconstruction may differ from the prior single query."})
    checkpoints=m["checkpoints"]
    by_count={}
    for cp in checkpoints:
        for mode in m["cutoff_modes"][str(cp)]:
            count=cp+(mode=="inclusive")
            by_count.setdefault(count,[]).append((cp,mode))
    max_count=max(by_count)
    for i in range(max_count):
        print(f"{scene}: building frame {i}/{max_count-1}",flush=True)
        memory.process_rgbd_images(rgb=np.asarray(env["rgb"][i]),depth=np.asarray(env["depth"][i]),
                                   intrinsics=np.asarray(env["camera_K"][i]),pose=np.asarray(env["camera_poses"][i]))
        if len(memory.observations)!=i+1:raise RuntimeError("Observation counter changed")
        for cp,mode in by_count.get(i+1,[]):
            for q in [q for q in queries if q["checkpoint"]==cp]:
                out=run/mode/q["query_id"];out.mkdir(parents=True,exist_ok=True)
                path=out/"prediction.json"
                if path.exists() and read_json(path)["status"]!="error":continue
                row={**q,"status":"ok","cutoff":mode,"experiment_type":EXPERIMENT,
                     "cutoff_status":p["checkpoint_semantics"],"scene_frame_count":len(env["rgb"]),
                     "n_observations":i+1,"eligible_frame_indices":list(range(i+1)),
                     "prediction_xyz":None,"evidence_frame_index":None,"age_observation_steps":None,
                     "seed":seed,"code_commit":COMMIT,"dataset_revision":REVISION}
                print(f"{scene}/{mode}/{cp}: {q['query']}",flush=True)
                start=time.perf_counter()
                try:
                    audit=CallAudit(detector,memory)
                    with preserve_rng(),audit:
                        result=memory.localize_with_feature_similarity(q["query"],debug=True,return_debug=True)
                    loc,debug,obs_id,raw=result
                    loc=xyz(loc); obs_id=scalar(obs_id)
                    idx=int(obs_id)-1 if obs_id is not None else None
                    if obs_id is not None and obs_id!=int(obs_id):raise ValueError("Noninteger evidence ID")
                    row.update(prediction_xyz=loc,raw_similarity_point=xyz(raw),debug_text=str(debug),
                               evidence_observation_id=int(obs_id) if obs_id is not None else None,
                               evidence_frame_index=idx,age_observation_steps=(i-idx) if idx is not None else None,
                               similarity_max=audit.record.get("similarity_max"),
                               detector_supported=audit.record.get("detector_supported"),
                               detector_confidence=audit.record.get("selected_confidence"),
                               return_route=audit.record.get("return_route"),
                               audit_status="ok" if not audit.record["capture_errors"] else "partial")
                    validate_prediction(row)
                    # Save the actual return before any labels are opened or any secondary analysis runs.
                    row["query_seconds"]=time.perf_counter()-start
                    atomic_json(path,row);atomic_json(out/"detector_trace.json",audit.record)
                    if idx is not None:
                        rgb=np.asarray(env["rgb"][idx])
                        _save_image(rgb,out/"evidence_rgb.png")
                        # Upstream supplies BGR as a PIL RGB array. Preserve it, do not fix the baseline here.
                        _save_image(rgb[...,::-1],out/"actual_detector_input.png")
                        box_overlay(rgb,audit.record,out/"selected_box.png")
                except Exception as e:
                    row.update(status="error",prediction_xyz=None,error_type=type(e).__name__,error=str(e),
                               query_seconds=time.perf_counter()-start)
                    atomic_json(path,row);(out/"error.txt").write_text(traceback.format_exc())
                    print(f"QUERY ERROR (not a decline): {e}",flush=True)
    result=finalize_run(base,scene,modes)
    del memory,encoder,detector,env;gc.collect();torch.cuda.empty_cache()
    if result["errors"]:raise RuntimeError(f"{result['errors']} inference errors; saved individually. Policies are blocked until repaired.")
    return result

def finalize_run(base: Path,scene: str,modes:list[str]) -> dict:
    queries=read_jsonl(base/"prepared"/scene/"queries.jsonl");rows=[]
    manifest=read_json(base/"prepared"/scene/"manifest.json")
    for mode in modes:
        for q in queries:
            if mode not in manifest["cutoff_modes"][str(q["checkpoint"])]:continue
            path=base/"runs"/scene/mode/q["query_id"]/"prediction.json"
            if not path.exists():raise RuntimeError("Incomplete batch; missing "+str(path))
            r=read_json(path);validate_prediction(r);rows.append(r)
    write_jsonl(base/"runs"/scene/"predictions.jsonl",rows)
    result={"scene":scene,"primary_queries":len(queries),"cutoff_branches":modes,
            "prediction_records":len(rows),"errors":sum(r["status"]=="error" for r in rows),
            "no_labels_used_by_model":True}
    atomic_json(base/"runs"/scene/"completion.json",result);return result

def audit_prior_detector(base: Path,repo: Path) -> dict:
    """Replay ONLY the stored supporting frame; do not replace the earlier prediction."""
    import torch,cv2
    p=init_project(base);m=verify_manifest(base,"env1");os.chdir(repo)
    env=load_scene(Path(m["data_root"])/"env.pkl")
    old=read_json(REFERENCE/"prediction.json");idx=old["evidence_frame_index"]
    out=base/"prior_audit";out.mkdir(parents=True,exist_ok=True)
    _,detector,_=make_models(repo,need_encoder=False)
    _seed(int(p["seed"]))
    rgb=np.asarray(env["rgb"][idx]);bgr=cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2BGR)
    audit=CallAudit(detector)
    with preserve_rng(),audit:
        returned=detector.compute_obj_coord(old["query"],bgr,torch.Tensor(env["depth"][idx]),
                     torch.Tensor(env["camera_K"][idx]),torch.Tensor(env["camera_poses"][idx]))
    current=xyz(returned)
    match=(current is not None and np.allclose(current,old["prediction_xyz"],rtol=0,atol=1e-6))
    result={"status":"detector_replayed_not_full_memory_reproduction", "query":old["query"],
            "prior_prediction_xyz":old["prediction_xyz"],"replay_detector_xyz":current,
            "matches_saved_location_at_1e_6m":bool(match),"evidence_frame_index":idx,
            "semantic_identity":"unreviewed", "detector_trace":audit.record,
            "caution":"Reproduced box/coordinate does not itself establish object identity. Use selected_box.png and projected annotation anchors for review."}
    atomic_json(out/"detector_replay.json",result)
    _save_image(rgb,out/"evidence_rgb.png");_save_image(bgr,out/"actual_detector_input.png")
    box_overlay(rgb,audit.record,out/"selected_box.png")
    # Labels are opened only now, after detector output is saved.
    from .analysis import prior_geometry_review
    prior_geometry_review(base,Path(m["data_root"]),audit.record)
    review={"semantic_identity":"unreviewed","reviewer":None,"rationale":None,
            "allowed_values":["unreviewed","requested_object","different_object","ambiguous"],
            "instruction":"Optional human review only. Never feed this file to model or policy selection."}
    if not (out/"manual_review.json").exists():atomic_json(out/"manual_review.json",review)
    return result

"""Post-inference scoring and geometry review. Never imported by model decisions."""
from __future__ import annotations
import csv
import html
import math
from pathlib import Path
import numpy as np
from .core import (REFERENCE, atomic_json, parse_label, protocol, query_id, read_json, read_jsonl,
                   score_prediction, summary, write_csv, write_jsonl)
from .data import load_scene, verify_manifest

def project_point(point, K, pose):
    cam=np.linalg.inv(np.asarray(pose,dtype=float))@np.r_[point,1.0]
    if not np.isfinite(cam).all() or cam[2]<=0:return None
    pixel=np.asarray(K,dtype=float)@cam[:3]
    return [float(pixel[0]/pixel[2]),float(pixel[1]/pixel[2]),float(cam[2])]

def prior_geometry_review(base: Path,scene_root: Path,trace:dict|None=None) -> dict:
    """Recompute actual saved geometry, with or without a new detector replay."""
    from PIL import Image,ImageDraw
    old=read_json(REFERENCE/"prediction.json"); saved=read_json(REFERENCE/"result.json")
    env=load_scene(scene_root/"env.pkl");idx=old["evidence_frame_index"]
    point=old["prediction_xyz"]
    if saved["prediction_xyz"]!=point:raise ValueError("Prior prediction/result mismatch")
    out=base/"prior_audit";out.mkdir(parents=True,exist_ok=True)
    trace=trace or {}
    with (scene_root/"14.csv").open(newline="",encoding="utf-8-sig") as f: raw=list(csv.DictReader(f))
    labels=[(row["query"],parse_label(row)) for row in raw]
    image=Image.fromarray(np.asarray(env["rgb"][idx]).clip(0,255).astype("uint8")).convert("RGB")
    draw=ImageDraw.Draw(image)
    anchors=[];b=trace.get("selected_box_pixels")
    for name,label in labels:
        if label["label_status"]!="positive":continue
        d=math.dist(point,label["target_xyz"])
        projection=project_point(label["target_xyz"],env["camera_K"][idx],env["camera_poses"][idx])
        inside=None
        if projection and b: inside=b[0]<=projection[0]<b[2] and b[1]<=projection[1]<b[3]
        anchor={"query_label":name,"distance_from_saved_prediction_m":d,
                "projected_query_time_anchor_uvz":projection,"anchor_inside_selected_box":inside,
                "passes_own_tolerance":d<=label["tolerance_m"]}
        anchors.append(anchor)
        if projection and 0<=projection[0]<image.width and 0<=projection[1]<image.height and name in ("toy owl","green toy"):
            u,v=projection[:2];draw.ellipse((u-4,v-4,u+4,v+4),fill="white",outline="black")
            draw.text((u+5,v),name,fill="white",stroke_width=1,stroke_fill="black")
    if b:draw.rectangle(b,outline="yellow",width=3)
    image.save(out/"projected_label_anchors.png")
    wanted=next(l for q,l in labels if q==old["query"])
    distance=math.dist(point,wanted["target_xyz"])
    nearest=min(anchors,key=lambda r:r["distance_from_saved_prediction_m"])
    result={"basis":"Uploaded saved real prediction, not a new model run",
            "query":old["query"],"distance_error_m":distance,"tolerance_m":wanted["tolerance_m"],
            "geometric_pass":distance<=wanted["tolerance_m"],"nearest_annotated_object":nearest,
            "possible_identity_mismatch":nearest["query_label"]!=old["query"],
            "anchors":sorted(anchors,key=lambda r:r["distance_from_saved_prediction_m"]),
            "semantic_identity":"not established by coordinate proximity or projected centroids",
            "time_convention":"unverified; saved run used exclusive 0..13",
            "anchor_caution":"Targets describe query time, not guaranteed exact supporting-frame pixels. An anchor inside a box is supporting evidence, not semantic ground truth.",
            "selected_box_available":bool(b)}
    atomic_json(out/"geometry_review.json",result)
    return result

def score_scene(base: Path,scene: str) -> dict:
    p=protocol(base);m=verify_manifest(base,scene)
    predictions=read_jsonl(base/"runs"/scene/"predictions.jsonl")
    completion=read_json(base/"runs"/scene/"completion.json")
    # Do not hide errors. They remain records and block policy fitting/evaluation.
    labels={};by_checkpoint={}
    for cp in m["checkpoints"]:
        with (Path(m["data_root"])/f"{cp}.csv").open(newline="",encoding="utf-8-sig") as f: raw=list(csv.DictReader(f))
        positives=[]
        for i,row in enumerate(raw):
            label=parse_label(row,p["blank_label_semantics"]=="confirmed")
            labels[query_id(scene,cp,i)]={**label,"query":row["query"].strip()}
            if label["label_status"]=="positive":positives.append((row["query"].strip(),label))
        by_checkpoint[cp]=positives
    scored=[]
    for pred in predictions:
        label=labels[pred["query_id"]]
        if label["query"]!=pred["query"]:raise ValueError("Query/label row mismatch")
        r=score_prediction(pred,label)
        # Posthoc diagnosis only: never turn nearest annotation into a model feature.
        if r["prediction_xyz"] is not None and r["status"]!="error":
            neighbors=[(math.dist(r["prediction_xyz"],l["target_xyz"]),name)
                       for name,l in by_checkpoint[r["checkpoint"]]]
            if neighbors:
                d,name=min(neighbors);r["nearest_annotation_query"]=name;r["nearest_annotation_distance_m"]=d
                r["possible_wrong_object_match"]=(r["label_status"]=="positive" and name!=r["query"]
                                                    and d+1e-9<r["distance_error_m"])
        scored.append(r)
    out=base/"scoring"/scene;write_jsonl(out/"scored.jsonl",scored);write_csv(out/"scored.csv",scored)
    modes=sorted({r["cutoff"] for r in scored})
    metrics={mode:summary([r for r in scored if r["cutoff"]==mode]) for mode in modes}
    metrics["label_semantics"]=p["blank_label_semantics"]
    # This is explicitly a sensitivity scenario, not the primary truth labels.
    alt=[]
    for r in scored:
        label={k:r[k] for k in ("label_status","target_xyz","tolerance_m")}
        if label["label_status"]=="unknown_blank":label["label_status"]="negative"
        alt.append(score_prediction(r,label))
    metrics["assumed_blank_negative_sensitivity"]={mode:summary([r for r in alt if r["cutoff"]==mode]) for mode in modes}
    metrics["warning"]="Blank-negative sensitivity is not verified benchmark scoring. No policy is trained on these assumed labels."
    atomic_json(out/"metrics.json",metrics)
    differences=[]
    if len(modes)==2:
        for qid in labels:
            present={r["cutoff"] for r in scored if r["query_id"]==qid}
            if present!={"exclusive","inclusive"}:continue
            a=next(r for r in scored if r["query_id"]==qid and r["cutoff"]=="exclusive")
            b=next(r for r in scored if r["query_id"]==qid and r["cutoff"]=="inclusive")
            changed=(a["prediction_xyz"] is None)!=(b["prediction_xyz"] is None)
            delta=None
            if a["prediction_xyz"] is not None and b["prediction_xyz"] is not None:
                delta=math.dist(a["prediction_xyz"],b["prediction_xyz"]);changed=delta>1e-6
            differences.append({"query_id":qid,"query":a["query"],"checkpoint":a["checkpoint"],
                                "output_changed":changed,"location_delta_m":delta,
                                "exclusive_outcome":a["outcome"],"inclusive_outcome":b["outcome"]})
        write_csv(out/"cutoff_sensitivity.csv",differences)
    gallery(out,scored,base)
    return {"scene":scene,"metrics":metrics,"inference_errors":completion["errors"]}

def gallery(out: Path,rows:list[dict],base:Path):
    chunks=["<!doctype html><meta charset='utf-8'><title>DynaBench error review</title>",
            "<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}article{border-top:1px solid #bbb;padding:20px 0}img{max-width:300px;vertical-align:top}pre{white-space:pre-wrap}</style>",
            "<h1>Saved detections and geometric scores</h1><p>No semantic identity is certified automatically. Unknown-label questions remain unscored. Selected boxes show the actual detector call, not a second detection.</p>"]
    for r in rows:
        folder=base/"runs"/r["scene"]/r["cutoff"]/r["query_id"]
        import os
        rel=os.path.relpath(folder,out)
        chunks.append(f"<article><h2>{html.escape(r['query_id'])}: {html.escape(r['query'])} / {r['cutoff']}</h2>")
        for img in ("evidence_rgb.png","selected_box.png","actual_detector_input.png"):
            if (folder/img).exists():chunks.append(f"<img src='{html.escape(rel+'/'+img)}' title='{img}'>")
        keys=("outcome","distance_error_m","tolerance_m","age_observation_steps","similarity_max",
              "detector_confidence","return_route","nearest_annotation_query","possible_wrong_object_match")
        chunks.append("<pre>"+html.escape("\n".join(f"{k}: {r.get(k)}" for k in keys))+"</pre></article>")
    (out/"review.html").write_text("\n".join(chunks),encoding="utf-8")

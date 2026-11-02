"""CLI used by the notebook. Every project command runs in the same Python 3.10 venv."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import zipfile
from .core import REFERENCE, init_project, protocol, read_json, atomic_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["init","preflight","prepare","audit-prior","batch","score","freeze","evaluate","report","export"])
    parser.add_argument("--base",type=Path,default=Path("/content/visual_memory"))
    parser.add_argument("--repo",type=Path,default=Path("/content/stretch_ai"))
    parser.add_argument("--scene",default="env1")
    parser.add_argument("--local-data-root",type=Path,help="Use existing pinned scene files instead of downloading")
    args=parser.parse_args();base=args.base.resolve();p=init_project(base)
    if args.command=="init":result=p
    elif args.command=="preflight":
        from .inference import preflight
        result=preflight(base,args.repo)
    elif args.command=="prepare":
        from .data import download_scene,prepare_scene
        if args.local_data_root is None:download_scene(base,args.scene)
        result=prepare_scene(base,args.scene,args.repo,args.local_data_root)
    elif args.command=="audit-prior":
        from .inference import audit_prior_detector
        result=audit_prior_detector(base,args.repo)
    elif args.command=="batch":
        from .inference import run_scene
        result=run_scene(base,args.repo,args.scene)
    elif args.command=="score":
        from .analysis import score_scene
        result=score_scene(base,args.scene)
    elif args.command=="freeze":
        from .policies import freeze
        result=freeze(base)
    elif args.command=="evaluate":
        from .policies import evaluate_heldout
        result=evaluate_heldout(base)
    elif args.command=="report":
        from .report import write_report
        result={"report":str(write_report(base))}
    else:
        # Only result/provenance artifacts: no full data, memory arrays, model weights or tokens.
        from .report import write_report
        write_report(base)
        out=base.parent/"DynaBench_Research_v3_results.zip"
        prefixes=("prepared","prior_audit","runs","scoring","analysis")
        root_files=("protocol.json","plan_lock.json","preflight.json","model_lock.json","working_environment.txt","report.md")
        with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
            for name in root_files:
                f=base/name
                if f.exists():z.write(f,name)
            for prefix in prefixes:
                folder=base/prefix
                if not folder.exists():continue
                for f in folder.rglob("*"):
                    if not f.is_file() or "memory_inputs" in f.parts or f.suffix in (".pkl",".pt",".npy",".safetensors"):continue
                    z.write(f,f.relative_to(base))
            for f in REFERENCE.iterdir():
                if f.is_file() and f.suffix in (".json",".txt"):z.write(f,"reference/"+f.name)
        result={"archive":str(out),"bytes":out.stat().st_size,
                "includes":"predictions, traces, evidence images, scores, frozen rules, report and provenance",
                "excludes":"raw scenes, full memory arrays, model weights, API credentials"}
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=="__main__":
    try:main()
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}",file=sys.stderr)
        raise

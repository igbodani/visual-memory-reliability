"""Read-only call observation: retain original return values and model decisions.

The profile callback observes the selected compute_obj_coord bounding box and the
query's existing alignment tensor at function return. It does not call the model
again, change thresholds, patch source, or choose a different candidate.
"""
from __future__ import annotations
from contextlib import contextmanager
import inspect
import random
import sys
import numpy as np


def array(x):
    if hasattr(x,"detach"): x=x.detach().cpu().numpy()
    return np.asarray(x)

def scalar(x):
    if x is None: return None
    a=array(x).reshape(-1)
    if a.size!=1 or not np.isfinite(a[0]): return None
    return float(a[0])

class CallAudit:
    def __init__(self, detector, memory=None):
        self.detector=detector; self.memory=memory; self.record={"capture_errors":[]}
        self.codes={inspect.unwrap(detector.compute_obj_coord).__code__:"detector"}
        if memory is not None:
            self.codes[inspect.unwrap(memory.localize_with_feature_similarity).__code__]="query"
        self.previous=None
    def _profile(self, frame, event, value):
        if event!="return" or frame.f_code not in self.codes: return
        try:
            kind=self.codes[frame.f_code]; loc=frame.f_locals
            if kind=="detector":
                self.record["detector_called"]=True
                self.record["detector_returned_location"]=value is not None
                scores=array(loc.get("scores",[])).reshape(-1)
                boxes=array(loc.get("boxes",[]))
                self.record["candidates"]=[{"confidence":float(s),"box_xyxy":b.tolist()}
                                           for s,b in zip(scores,boxes)]
                self.record["detector_return_xyz"]=None if value is None else array(value).reshape(-1).tolist()
                self.record["selected_confidence"]=scalar(loc.get("score")) if value is not None else None
                self.record["selected_box_xyxy"]=array(loc["bbox"]).tolist() if value is not None and "bbox" in loc else None
                names=("tl_x","tl_y","br_x","br_y")
                self.record["selected_box_pixels"]=([int(loc[k]) for k in names]
                    if value is not None and all(k in loc for k in names) else None)
                self.record["selected_rank"]=int(loc["idx"]) if value is not None and "idx" in loc else None
                self.record["depth_threshold"]=float(loc.get("depth_threshold",3.0))
                self.record["selection_region"]="bounding_box_not_segmentation_mask"
            else:
                a=array(loc.get("alignments",[])).reshape(-1)
                self.record["similarity_max"]=float(a.max()) if a.size else None
                self.record["alignment_count"]=int(a.size)
                self.record["detector_supported"]=loc.get("res") is not None
                self.record["return_route"]=("detector" if loc.get("res") is not None else
                                              "similarity_fallback" if loc.get("target_point") is not None else "decline")
        except Exception as exc:
            # Audit errors must never change the model's own returned prediction.
            self.record["capture_errors"].append(f"{type(exc).__name__}: {exc}")
    def __enter__(self):
        self.previous=sys.getprofile()
        if self.previous is not None:
            raise RuntimeError("Another profiler is active; disable it before auditable inference.")
        sys.setprofile(self._profile); return self
    def __exit__(self,*args): sys.setprofile(self.previous)

@contextmanager
def preserve_rng():
    """Query logging/inference must not change the RNG stream for later map frames."""
    import torch
    py_state=random.getstate();np_state=np.random.get_state()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        try: yield
        finally: random.setstate(py_state);np.random.set_state(np_state)


def box_overlay(rgb, trace: dict, out):
    """Visual audit of an existing image. No class identity is inferred here."""
    from PIL import Image,ImageDraw
    im=Image.fromarray(array(rgb).clip(0,255).astype("uint8")).convert("RGB")
    draw=ImageDraw.Draw(im)
    b=trace.get("selected_box_pixels")
    if b:
        draw.rectangle(b,outline="yellow",width=3)
        text="selected detector box"
        if trace.get("selected_confidence") is not None: text+=f" {trace['selected_confidence']:.3f}"
        draw.text((max(0,b[0]),max(0,b[1]-13)),text,fill="yellow",stroke_width=1,stroke_fill="black")
    im.save(out)
    if b and b[2]>b[0] and b[3]>b[1]: im.crop(b).save(str(out).replace(".png","_crop.png"))

"""Synthetic tests of the harness. These are NOT DynaBench/model performance results."""
import copy
import csv
import io
import json
import math
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from vmresearch import core
from vmresearch.data import ArrayTensorUnpickler,load_scene,prepare_scene
from vmresearch.instrumentation import CallAudit,scalar
from vmresearch.policies import (cdf,choose_threshold,feature_value,gate_prediction,reliability,
                                 freeze,evaluate_heldout,load_frozen)
from vmresearch.analysis import project_point,score_scene
from vmresearch.report import write_report


def record(qid="q",location=None,age=1,similarity=.2):
    return {"query_id":qid,"scene":"devscene","checkpoint":2,"cutoff":"exclusive","query":"item",
            "row_index":0,"status":"ok","n_observations":2,"scene_frame_count":5,
            "eligible_frame_indices":[0,1],"prediction_xyz":location,
            "evidence_frame_index":0,"age_observation_steps":age,
            "similarity_max":similarity,"detector_supported":True}

def positive():return {"label_status":"positive","target_xyz":[1.,2.,3.],"tolerance_m":.3}

class FakeEncoder: pass
class FakeDetector:
    def compute_obj_coord(self,text,rgb,depth,camera_K,camera_pose):
        scores=np.array([.8,.4]);boxes=np.array([[0,0,2,2],[1,1,3,3]])
        idx=0;score=scores[0];bbox=boxes[0];tl_x,tl_y,br_x,br_y=0,0,2,2
        depth_threshold=3.0
        return np.array([1.,2.,3.]) if text!="miss" else None

class FakeMemory:
    def __init__(self,encoder,detection,device,log,mllm=False,run_eqa=False):
        self.detection_model=detection;self.observations=[];self.image_shape=(4,4)
        self.min_depth=.25;self.max_depth=2.5
    def process_rgbd_images(self,rgb,depth,intrinsics,pose):
        self.observations.append((rgb,depth,intrinsics,pose))
    def localize_with_feature_similarity(self,text,debug=True,return_debug=True):
        if text=="crash":raise ValueError("synthetic inference failure")
        alignments=np.array([[.3 if text=="item" else .1]])
        obs_id=np.array([1]);rgb,depth,K,pose=self.observations[0]
        res=self.detection_model.compute_obj_coord(text,rgb,depth,K,pose)
        target_point=res
        return target_point,"synthetic test",obs_id,np.array([1.,2.,3.])

def synthetic_data(root,scene,crash=False):
    d=root/scene;d.mkdir(parents=True)
    env={"rgb":[np.zeros((4,4,3),dtype="uint8") for _ in range(5)],
         "depth":[np.ones((4,4),dtype="float32") for _ in range(5)],
         "camera_K":[np.eye(3,dtype="float32") for _ in range(5)],
         "camera_poses":[np.eye(4,dtype="float32") for _ in range(5)]}
    with (d/"env.pkl").open("wb") as f:pickle.dump(env,f)
    (d/"time.txt").write_text("2\n4\n")
    for cp in [2,4]:
        with (d/f"{cp}.csv").open("w",newline="") as f:
            w=csv.writer(f);w.writerow(["query","x","y","z","tol"])
            w.writerow(["item",1,2,3,.3]);w.writerow(["crash" if crash else "other",8,9,10,.3]);w.writerow(["miss","","","",""])

class CoreTests(unittest.TestCase):
    def test_cutoff_exclusive(self):self.assertEqual(core.eligible_count(14,"exclusive",42),14)
    def test_cutoff_inclusive(self):self.assertEqual(core.eligible_count(14,"inclusive",42),15)
    def test_invalid_cutoff(self):
        with self.assertRaises(ValueError):core.eligible_count(5,"inclusive",5)
    def test_future_evidence_rejected(self):
        r=record(location=[1,2,3]);r["evidence_frame_index"]=2
        with self.assertRaises(ValueError):core.validate_prediction(r)
    def test_nonprefix_rejected(self):
        r=record(location=[1,2,3]);r["eligible_frame_indices"]=[0,2]
        with self.assertRaises(ValueError):core.validate_prediction(r)
    def test_bad_age_rejected(self):
        r=record(location=[1,2,3],age=0)
        with self.assertRaises(ValueError):core.validate_prediction(r)
    def test_valid_evidence(self):core.validate_prediction(record(location=[1,2,3]))
    def test_blank_unknown(self):self.assertEqual(core.parse_label(dict(x="",y="",z="",tol=""))["label_status"],"unknown_blank")
    def test_blank_negative_only_explicit(self):self.assertEqual(core.parse_label(dict(x="",y="",z="",tol=""),True)["label_status"],"negative")
    def test_partial_label_rejected(self):
        with self.assertRaises(ValueError):core.parse_label(dict(x="1",y="",z="",tol=""))
    def test_nonfinite_label_rejected(self):
        with self.assertRaises(ValueError):core.parse_label(dict(x="nan",y="1",z="1",tol=".3"))
    def test_positive_pass(self):self.assertTrue(core.score_prediction(record(location=[1,2,3]),positive())["correct"])
    def test_wrong_location(self):self.assertFalse(core.score_prediction(record(location=[5,2,3]),positive())["correct"])
    def test_zero_tolerance(self):
        label=positive();label["tolerance_m"]=0
        self.assertTrue(core.score_prediction(record(location=[1,2,3]),label)["correct"])
    def test_always_decline_not_success(self):
        row=core.score_prediction(record(),positive());s=core.summary([row])
        self.assertEqual(s["positive_localization_recall"],0);self.assertIsNone(s["conditional_error_scorable"])
    def test_unknown_not_success(self):
        r=core.score_prediction(record(),{"label_status":"unknown_blank","target_xyz":None,"tolerance_m":None})
        self.assertIsNone(r["correct"]);self.assertEqual(core.summary([r])["unscored_label_queries"],1)
    def test_error_not_decline(self):
        r=record();r["status"]="error";r=core.score_prediction(r,positive());s=core.summary([r])
        self.assertEqual(s["declines"],0);self.assertEqual(s["inference_errors"],1)
    def test_join_cannot_restore_rejected_prediction(self):
        scored=core.score_prediction(record(location=[1,2,3]),positive())
        again=core.score_prediction(gate_prediction(scored,False),scored)
        self.assertIsNone(again["prediction_xyz"]);self.assertFalse(again["correct"])
    def test_no_truth_in_policy_prediction(self):
        r=core.score_prediction(record(location=[1,2,3]),positive());out=gate_prediction(r,True)
        self.assertNotIn("target_xyz",out);self.assertNotIn("correct",out)
    def test_unique_duplicate_text_ids(self):self.assertNotEqual(core.query_id("env1",14,0),core.query_id("env1",14,1))
    def test_known_prior_numeric_score(self):
        r=core.read_json(core.REFERENCE/"result.json")
        self.assertAlmostEqual(math.dist(r["prediction_xyz"],r["target_xyz"]),.2750690905825581,places=10)
    def test_project_point(self):self.assertEqual(project_point([1,2,2],np.eye(3),np.eye(4)),[.5,1.,2.])
    def test_behind_camera(self):self.assertIsNone(project_point([1,2,-1],np.eye(3),np.eye(4)))
    def test_pickle_disallows_globals(self):
        with self.assertRaises(pickle.UnpicklingError):ArrayTensorUnpickler(io.BytesIO(pickle.dumps(Path("x")))).load()
    def test_pytorch_tensor_pickle_roundtrip(self):
        import torch
        value={"rgb":[torch.arange(12).reshape(2,2,3)]}
        restored=ArrayTensorUnpickler(io.BytesIO(pickle.dumps(value))).load()
        self.assertTrue(torch.equal(restored["rgb"][0],value["rgb"][0]))
    def test_query_rng_does_not_affect_next_frame(self):
        import torch,random
        from vmresearch.instrumentation import preserve_rng
        random.seed(17);np.random.seed(17);torch.manual_seed(17)
        expected=(random.random(),float(np.random.rand()),float(torch.rand(1)))
        random.seed(17);np.random.seed(17);torch.manual_seed(17)
        with preserve_rng():
            random.random();np.random.rand();torch.rand(10)
        actual=(random.random(),float(np.random.rand()),float(torch.rand(1)))
        self.assertEqual(expected,actual)
    def test_cdf_ties(self):self.assertEqual(cdf(1,[1,1,2,3]),.25)
    def test_retention_ties_included(self):self.assertEqual(choose_threshold([.5,.5,.2],.5),.5)
    def test_full_retention_no_threshold(self):self.assertIsNone(choose_threshold([.5],1))
    def test_policy_does_not_consult_truth(self):
        m={"combined":{"status":"available","transforms":{"recency":[-2,-1],"similarity":[.1,.3]}}}
        a=record(location=[1,2,3]);b={**a,"target_xyz":[900,0,0],"correct":False}
        self.assertEqual(reliability(a,"combined",m),reliability(b,"combined",m))
    def test_combined_missing_signal(self):
        m={"combined":{"status":"available","transforms":{"recency":[-2,-1],"similarity":[.1,.3]}}}
        r=record(location=[1,2,3]);r["similarity_max"]=None
        self.assertIsNone(reliability(r,"combined",m))
    def test_readonly_profile_records_selected_box(self):
        d=FakeDetector();m=FakeMemory(None,d,None,None);m.process_rgbd_images(np.zeros((4,4,3)),np.ones((4,4)),np.eye(3),np.eye(4))
        with CallAudit(d,m) as audit:result=m.localize_with_feature_similarity("item")
        self.assertEqual(audit.record["selected_box_pixels"],[0,0,2,2])
        self.assertEqual(audit.record["selected_confidence"],.8)
        self.assertEqual(audit.record["return_route"],"detector")
        self.assertTrue(np.allclose(result[0],[1,2,3]));self.assertEqual(audit.record["capture_errors"],[])

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.base=self.root/"study";self.base.mkdir();self.data=self.root/"data";self.repo=self.root/"repo";self.repo.mkdir()
        p=copy.deepcopy(core.DEFAULT_PROTOCOL);p.update(development_scenes=["devscene"],evaluation_scenes=["testscene"],minimum_development_answers=2)
        core.atomic_json(self.base/"protocol.json",p);core.init_project(self.base)
        self.runtime={"packages":{"fake":"1"},"code_commit":core.COMMIT,"harness_sha256":core.code_digest(),"source_modified":""}
        core.atomic_json(self.base/"preflight.json",self.runtime)
    def tearDown(self):self.temp.cleanup()
    def do_batch(self,scene):
        from vmresearch.inference import run_scene
        with patch("vmresearch.inference.runtime_record",return_value=self.runtime),\
             patch("vmresearch.inference.make_models",return_value=(FakeEncoder(),FakeDetector(),FakeMemory)),\
             patch("vmresearch.inference.model_metadata",return_value={"fake_model":True}):
            return run_scene(self.base,self.repo,scene)
    def prepare_dev(self):
        synthetic_data(self.data,"devscene");prepare_scene(self.base,"devscene",self.repo,self.data)
    def test_protocol_change_rejected(self):
        p=core.read_json(self.base/"protocol.json");p["seed"]+=1;core.atomic_json(self.base/"protocol.json",p)
        with self.assertRaises(ValueError):core.init_project(self.base)
    def test_overlap_rejected(self):
        p=core.read_json(self.base/"protocol.json");p["evaluation_scenes"]=["devscene"];core.atomic_json(self.base/"protocol.json",p)
        with self.assertRaises(ValueError):core.protocol(self.base)
    def test_confirmation_requires_evidence(self):
        p=core.read_json(self.base/"protocol.json");p["checkpoint_semantics"]="confirmed";core.atomic_json(self.base/"protocol.json",p)
        with self.assertRaises(ValueError):core.protocol(self.base)
    def test_test_scene_blocked_before_freeze(self):
        synthetic_data(self.data,"testscene")
        with self.assertRaises(RuntimeError):prepare_scene(self.base,"testscene",self.repo,self.data)
    def test_queries_are_label_free(self):
        self.prepare_dev();rows=core.read_jsonl(self.base/"prepared/devscene/queries.jsonl")
        self.assertEqual(len(rows),6);self.assertNotIn("target_xyz",rows[0]);self.assertNotIn("x",rows[0])
    def test_missing_boundary_frame_skips_only_sensitivity(self):
        synthetic_data(self.data,"devscene")
        d=self.data/"devscene";(d/"time.txt").write_text("2\n5\n")
        (d/"4.csv").rename(d/"5.csv")
        m=prepare_scene(self.base,"devscene",self.repo,self.data)
        self.assertEqual(m["cutoff_modes"]["5"],["exclusive"])
        c=self.do_batch("devscene")
        self.assertEqual(c["prediction_records"],9)
    def test_unverified_convention_not_certified_by_shape(self):
        self.prepare_dev();a=core.read_json(self.base/"prepared/devscene/convention_audit.json")
        self.assertEqual(a["status"],"unverified");self.assertFalse(a["verified_automatically"])
    def test_complete_mocked_workflow(self):
        self.prepare_dev();result=self.do_batch("devscene")
        self.assertEqual(result["primary_queries"],6);self.assertEqual(result["prediction_records"],12)
        raw=core.read_jsonl(self.base/"runs/devscene/predictions.jsonl")
        self.assertTrue(all("target_xyz" not in r for r in raw))
        for r in raw:core.validate_prediction(r)
        score_scene(self.base,"devscene")
        frozen=freeze(self.base);frozen_before=(self.base/"analysis/frozen_policies.json").read_bytes()
        self.assertEqual(frozen["combined"]["status"],"available")
        synthetic_data(self.data,"testscene");prepare_scene(self.base,"testscene",self.repo,self.data)
        self.do_batch("testscene");score_scene(self.base,"testscene")
        result=evaluate_heldout(self.base)
        self.assertEqual(frozen_before,(self.base/"analysis/frozen_policies.json").read_bytes())
        grouped={}
        for r in result["matched_coverage_metrics"]:
            grouped.setdefault(r["retained_fraction_of_baseline_answers"],[]).append(r["answers"])
        self.assertTrue(all(len(set(v))==1 for v in grouped.values()))
        preds=core.read_jsonl(self.base/"analysis/heldout_policy_predictions.jsonl")
        self.assertTrue(all("target_xyz" not in r and "correct" not in r for r in preds))
        decl=next(r for r in result["fixed_threshold_metrics"] if r["policy"]=="always_decline" and r["scene"]=="pooled")
        self.assertEqual(decl["positive_localization_recall"],0);self.assertIsNone(decl["conditional_error_scorable"])
        # Changing held-out targets/correctness must not change rank-based selected IDs.
        before=(self.base/"analysis/matched_selected_ids.json").read_bytes()
        sf=self.base/"scoring/testscene/scored.jsonl";altered=core.read_jsonl(sf)
        for row in altered:
            if row["label_status"]=="positive":
                row["target_xyz"]=[1000,1000,1000];row["correct"]=False
        core.write_jsonl(sf,altered);evaluate_heldout(self.base)
        self.assertEqual(before,(self.base/"analysis/matched_selected_ids.json").read_bytes())
        path=write_report(self.base);self.assertTrue(path.exists());self.assertIn("Exploratory".lower(),path.read_text().lower())
        # Resume the completed batch without calling any model constructors.
        from vmresearch.inference import run_scene
        with patch("vmresearch.inference.runtime_record",return_value=self.runtime),\
             patch("vmresearch.inference.make_models",side_effect=AssertionError("Should not rerun")):
            self.assertEqual(run_scene(self.base,self.repo,"testscene")["errors"],0)
    def test_errors_are_logged_and_block_fitting(self):
        synthetic_data(self.data,"devscene",crash=True);prepare_scene(self.base,"devscene",self.repo,self.data)
        with self.assertRaises(RuntimeError):self.do_batch("devscene")
        c=core.read_json(self.base/"runs/devscene/completion.json");self.assertEqual(c["errors"],4)
        score_scene(self.base,"devscene")
        with self.assertRaises(RuntimeError):freeze(self.base)
    def test_tampered_frozen_file_rejected(self):
        self.prepare_dev();self.do_batch("devscene");score_scene(self.base,"devscene");freeze(self.base)
        f=self.base/"analysis/frozen_policies.json";m=core.read_json(f);m["thresholds"]["age_only"]["0.5"]=999;core.atomic_json(f,m)
        with self.assertRaises(ValueError):load_frozen(self.base)
    def test_partial_report_does_not_invent_results(self):
        text=write_report(self.base).read_text()
        self.assertIn("comparison pending",text);self.assertIn("devscene: pending",text)

if __name__=="__main__":unittest.main(verbosity=2)

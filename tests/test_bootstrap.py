"""Regression tests for v3.1 setup control flow, using fake commands (no installs/GPU)."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bootstrap_guard", ROOT / "bootstrap_guard.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)
COMMIT = "6272b28e3ff7be482af3b427a4551d1772ebf7d7"

FAKE = r'''#!/usr/bin/python3 -S
import json, os, sys
from pathlib import Path
state_path=Path(os.environ["MOCK_STATE"])
s=json.loads(state_path.read_text())
a=sys.argv[1:]; name=Path(sys.argv[0]).name
with open(os.environ["MOCK_CALLS"],"a") as f:f.write(json.dumps([name,*a])+"\n")
def save():state_path.write_text(json.dumps(s))
def core():return s.get("gpu",False) and s.get("compat",False) and s.get("stretch",False) and s.get("sam",False)
if name=="fakepython":
 if a[0]=="-c":sys.exit(0)
 mode=a[1]
 if mode=="core":sys.exit(0 if core() else 1)
 if mode=="gpu":sys.exit(0 if s.get("gpu",False) else 1)
 if mode=="installed":sys.exit(0 if s.get("stretch",False) and s.get("sam",False) else 1)
 if mode=="snapshot":print(json.dumps(s));sys.exit(0)
 if mode=="verify":
  if not core() or not s.get("cuda",True) or s.get("verify_fail",False):sys.exit(9)
  print("CORE VERSIONS AND GPU CHECK PASSED");sys.exit(0)
 sys.exit(90)
if name=="uv":
 if "freeze" in a:print("mock==1.0");sys.exit(0)
 if "check" in a:print("Known metadata exceptions (mock)");sys.exit(1)
 if "install" not in a:sys.exit(91)
 if "--no-deps" in a:
  assert "setuptools==69.5.1" in a and "transformers==4.49.0" in a
  assert not any(x.startswith("torch==") for x in a)
  s["compat"]=True;save();sys.exit(0)
 assert "--torch-backend" in a and a[a.index("--torch-backend")+1]=="cu126"
 assert "-c" in a
 constraints=Path(a[a.index("-c")+1]).read_text()
 assert "torch==2.14.1+cu126" in constraints and "torchvision==0.29.1+cu126" in constraints
 if "torch==2.14.1+cu126" in a:
  if s.get("fail_install",False):sys.exit(17)
  s["gpu"]=True
 if "-e" in a:
  target=a[a.index("-e")+1]
  s["sam" if target.endswith("segment-anything-2") else "stretch"]=True
  s["compat"]=False
 save();sys.exit(0)
if name=="git":
 if "clone" in a:Path(a[-1],".git").mkdir(parents=True);sys.exit(0)
 if "checkout" in a:s["commit"]=a[-1];save();sys.exit(0)
 if "status" in a:print(" M tracked.py" if s.get("dirty") else "");sys.exit(0)
 if "rev-parse" in a:print(s.get("commit","6272b28e3ff7be482af3b427a4551d1772ebf7d7"));sys.exit(0)
 if "submodule" in a:sys.exit(0)
 sys.exit(92)
if name=="nvidia-smi":sys.exit(0 if s.get("driver",True) else 7)
if name=="apt-get":sys.exit(0)
sys.exit(93)
'''

class BootstrapTests(unittest.TestCase):
    def run_case(self, overrides=None, repo_exists=True):
        state={"gpu":False,"compat":True,"stretch":True,"sam":True,"cuda":True,"commit":COMMIT}
        state.update(overrides or {})
        with tempfile.TemporaryDirectory() as temp:
            t=Path(temp); bindir=t/"bin";bindir.mkdir()
            for name in ("fakepython","uv","git","apt-get","nvidia-smi"):
                p=bindir/name;p.write_text(FAKE);p.chmod(0o755)
            repo=t/"repo"
            if repo_exists:(repo/".git").mkdir(parents=True)
            (t/"state.json").write_text(json.dumps(state))
            env={**os.environ,"PATH":str(bindir)+os.pathsep+os.environ["PATH"],
                 "MOCK_STATE":str(t/"state.json"),"MOCK_CALLS":str(t/"calls.jsonl"),
                 "DYNABENCH_PYTHON":str(bindir/"fakepython"),"DYNABENCH_REPO":str(repo),
                 "DYNABENCH_SETUP_LOG_DIR":str(t/"audits")}
            p=subprocess.run(["bash",str(ROOT/"bootstrap_colab.sh")],env=env,text=True,capture_output=True,timeout=20)
            calls=[json.loads(x) for x in (t/"calls.jsonl").read_text().splitlines()]
            files={str(x.relative_to(t)):x.read_text() for x in (t/"audits").rglob("*") if x.is_file()}
            return p,calls,files

    def installs(self,calls):return [c for c in calls if c[0]=="uv" and "install" in c]
    def test_exact_environment_not_modified(self):
        p,c,f=self.run_case({"gpu":True});self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(self.installs(c),[]);self.assertEqual(f,{})
    def test_cuda_failure_is_not_package_reinstall(self):
        p,c,_=self.run_case({"gpu":True,"cuda":False});self.assertNotEqual(p.returncode,0)
        self.assertEqual(self.installs(c),[])
    def test_drift_restores_pair_then_no_deps_overrides(self):
        p,c,f=self.run_case();self.assertEqual(p.returncode,0,p.stderr)
        installs=self.installs(c);self.assertEqual(len(installs),2)
        self.assertIn("torch==2.14.1+cu126",installs[0]);self.assertIn("--no-deps",installs[1])
        self.assertTrue(any(k.endswith("before_freeze.txt") for k in f))
        self.assertTrue(any(k.endswith("after_core.json") for k in f))
        self.assertTrue(any(k.endswith("dependency_check.txt") for k in f))
    def test_only_compatibility_drift_does_not_reinstall_gpu(self):
        p,c,_=self.run_case({"gpu":True,"compat":False});self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(len(self.installs(c)),1);self.assertIn("--no-deps",self.installs(c)[0])
    def test_fresh_stack_constrains_every_resolving_install(self):
        p,c,_=self.run_case({"stretch":False,"sam":False,"compat":False})
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(len(self.installs(c)),4)
        for call in self.installs(c)[:-1]:self.assertIn("-c",call);self.assertIn("cu126",call)
        self.assertIn("--no-deps",self.installs(c)[-1])
    def test_install_error_stops_before_overrides(self):
        p,c,_=self.run_case({"fail_install":True});self.assertEqual(p.returncode,17)
        self.assertEqual(len(self.installs(c)),1)
        self.assertNotIn("Section 2 core checks passed",p.stdout)
    def test_source_edits_not_overwritten(self):
        p,c,_=self.run_case({"dirty":True});self.assertNotEqual(p.returncode,0)
        self.assertEqual(self.installs(c),[]);self.assertFalse(any("checkout" in x for x in c))
    def test_existing_different_commit_not_replaced(self):
        p,c,_=self.run_case({"commit":"different"});self.assertNotEqual(p.returncode,0)
        self.assertEqual(self.installs(c),[]);self.assertFalse(any("checkout" in x for x in c))
    def test_new_clone_pinned_before_installing(self):
        p,c,_=self.run_case({"commit":"new-main","stretch":False,"sam":False},repo_exists=False)
        self.assertEqual(p.returncode,0,p.stderr)
        checkouts=[x for x in c if x[0]=="git" and "checkout" in x]
        self.assertEqual(checkouts[0][-1],COMMIT)
    def test_post_install_check_is_mandatory(self):
        p,c,_=self.run_case({"verify_fail":True});self.assertNotEqual(p.returncode,0)
        self.assertNotIn("Section 2 core checks passed",p.stdout)
    def test_compare_detects_local_cuda_build_change(self):
        diff=GUARD.compare_versions({"torch":"2.14.1+cu126"},{"torch":"2.14.1"})
        self.assertIn("torch",diff)
    def test_compare_detects_missing_and_wrong_package(self):
        d=GUARD.compare_versions({"a":"1","b":"2"},{"a":"3"})
        self.assertEqual(set(d),{"a","b"});self.assertIsNone(d["b"]["installed"])
    def test_expected_versions_from_supplied_reference(self):
        expected=GUARD.expected_versions();self.assertEqual(expected["torch"],"2.14.1+cu126")
        self.assertEqual(expected["setuptools"],"69.5.1")
    def test_shell_syntax(self):
        p=subprocess.run(["bash","-n",str(ROOT/"bootstrap_colab.sh")],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

if __name__ == "__main__":unittest.main()

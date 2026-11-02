# Setup and reproduction

This document reproduces the recorded **current-code YOLOE + SigLIP diagnostic**, not the original paper configuration. The most reliable entrypoint is the [notebook](../notebooks/DynaBench_research.ipynb), because it includes code export, the held-out schema adapter, setup checks and export steps.

## Requirements

- **Hosted Linux GPU runtime** (recorded: Colab Tesla T4 with about 15 GiB GPU memory). A local NVIDIA GPU setup is possible but was not validated for this release.
- **Python 3.10** isolated in `/content/stretch310` using `uv`; notebook kernel may have a different Python version.
- `git`, working network access for weights, and `apt` permissions for system audio headers used by upstream Stretch dependencies.
- Recorded upstream commit `6272b28e3ff7be482af3b427a4551d1772ebf7d7` and SAM2 submodule `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- Public dataset `hello-peiqi/DynaMem-DynaBench` revision `1d561e2de8718addf398e5e967f2c44dc015c366`.

The recorded working environment used PyTorch `2.14.1+cu126`, torchvision `0.29.1+cu126`, Transformers `4.49.0`, setuptools `69.5.1`, Open3D `0.20.0`, NumPy `1.26.4`, and Ultralytics `8.3.161`. The notebook validates these versions; these are compatibility observations, **not** a clean full-stack dependency lock. Torch/setuptools and Stretch/Transformers package metadata have documented conflicts; a newer unconstrained install can break the run.

## Recommended Colab workflow

1. Use a new [Google Colab](https://colab.research.google.com/) **GPU** notebook; open [`DynaBench_research.ipynb`](../notebooks/DynaBench_research.ipynb).
2. Run **Section 1** to unpack the embedded, hash-checked research source at `/content/DynaBench_Research_v3` (the export is part of the notebook).
3. Run **Sections 2–3** for guarded setup, unit tests, protocol freeze, and preflight. Do not install additional dependencies outside the recorded environment to make warnings disappear.
4. **Sections 4–6:** download/pin `env1`, inspect frame alignment and evidence, run 54 primary queries plus a separate cutoff sensitivity branch, save predictions before scoring.
5. **Sections 7–8:** inspect development evidence and freeze the age-only and age-plus-similarity rules using `env1` only.
6. **Section 9:** stage the held-out `env2` four-stream view using the non-destructive schema adapter, then batch and score. The notebook's adapter is also available separately at `scripts/prepare_env2_schema.py`.
7. **Sections 10–11:** produce the matched-answer and fixed-threshold comparisons, report and evidence ZIP. Download the private ZIP before ending the Colab session.

**Do not use held-out results to change thresholds, tie-break rules, or checkpoint interpretations.** The paper's official checkpoint convention remains unconfirmed; retain the recorded exclusive-cutoff branch and the sensitivity branch separately.

## Using the standalone source from a Colab terminal

After Section 2 has configured dependencies, you can inspect/help the CLI without running the entire notebook:

```bash
export PYTHONPATH=/content/DynaBench_Research_v3:/content/stretch_ai/src
PY=/content/stretch310/bin/python
BASE=/content/visual_memory
"$PY" -m vmresearch.cli --help
"$PY" -m unittest discover -s /content/DynaBench_Research_v3/tests -v
```

For an already frozen study, the held-out schema adaptation and baseline processing is:

```bash
"$PY" /content/repair_env2_schema.py --base "$BASE" --repo /content/stretch_ai --scene env2
"$PY" -m vmresearch.cli batch --base "$BASE" --scene env2
"$PY" -m vmresearch.cli score --base "$BASE" --scene env2
"$PY" -m vmresearch.cli evaluate --base "$BASE"
"$PY" -m vmresearch.cli report --base "$BASE"
```

Do not run the held-out commands before Section 8 has frozen the development rules. If cloning this repo instead of opening the notebook, use the repository root in `PYTHONPATH` and `bash bootstrap_colab.sh` from that root; initialize the study protocol with the CLI. The `env2` adapter should use the exact standalone script above when processing the raw `env2` pickle.

## Running unit tests without a GPU

The `tests` folder uses synthetic fixtures and mocked components; it is **not** a benchmark run. In a Python environment that already has NumPy, Pillow and PyTorch:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

`tests/test_bootstrap.py` also checks that the bootstrap script uses pinned GPU constraints. The results report separately documents the independent checks of saved outputs.

## Dataset and sharing

We use public DynaBench data downloaded by the notebook. This GitHub repository intentionally **does not redistribute** dataset `env.pkl` files, camera arrays, original photographs, segmentation crops, model weights, or the full 335 MB result bundle. Upstream data/image redistribution permissions were not verified. Keep the original results ZIP and independent audit privately if you need the full chain of evidence.

Repository exclusions are listed in [`.gitignore`](../.gitignore).

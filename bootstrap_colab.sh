#!/usr/bin/env bash
# v3.1 bootstrap-only correction; experiment code/configuration is unchanged.
# Keep an EXACTLY matching working environment. Repair recorded-core version drift.
# The two documented metadata exceptions are intentional, not a clean lockfile.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY=${DYNABENCH_PYTHON:-/content/stretch310/bin/python}
REPO=${DYNABENCH_REPO:-/content/stretch_ai}
COMMIT=6272b28e3ff7be482af3b427a4551d1772ebf7d7
GUARD="$HERE/bootstrap_guard.py"

# Never quietly modify a matching environment just because its GPU is unavailable.
if [ -x "$PY" ] && "$PY" "$GUARD" core >/dev/null 2>&1; then
    "$PY" "$GUARD" verify
    echo 'Exact recorded core found. No packages installed or upgraded. Continue to Section 3.'
    exit 0
fi

nvidia-smi
if ! command -v uv >/dev/null 2>&1; then python -m pip install -q uv; fi
if [ ! -x "$PY" ]; then
    uv --no-config venv --python 3.10 "$(dirname "$(dirname "$PY")")"
fi
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 is required"'

# Save what was present before making any package changes. Preserve all data/results.
AUDIT_ROOT=${DYNABENCH_SETUP_LOG_DIR:-/content/visual_memory/setup_audits}
mkdir -p "$AUDIT_ROOT"
AUDIT=$(mktemp -d "$AUDIT_ROOT/bootstrap_v3_1_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
"$PY" "$GUARD" snapshot > "$AUDIT/before_core.json"
uv --no-config pip freeze --python "$PY" > "$AUDIT/before_freeze.txt"

# Preserve the pinned source and refuse to overwrite tracked edits.
if [ ! -d "$REPO/.git" ]; then
    git clone https://github.com/hello-robot/stretch_ai.git "$REPO"
    git -C "$REPO" checkout --detach "$COMMIT"
fi
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    echo 'STOP: tracked source has edits; preserve them instead of overwriting.' >&2
    exit 1
fi
if [ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]; then
    echo 'STOP: source is not at the recorded commit. No package changes made.' >&2
    echo "Expected: $COMMIT" >&2
    exit 1
fi

# Every dependency-resolving installation is constrained to the recorded GPU pair.
CONSTRAINTS="$AUDIT/gpu_constraints.txt"
cat > "$CONSTRAINTS" <<'EOF'
torch==2.14.1+cu126
torchvision==0.29.1+cu126
EOF
NEEDS_STACK=0
if ! "$PY" "$GUARD" installed >/dev/null 2>&1; then NEEDS_STACK=1; fi

if ! "$PY" "$GUARD" gpu >/dev/null 2>&1; then
    echo 'Restoring the recorded CUDA 12.6 PyTorch/torchvision pair.'
    uv --no-config pip install --python "$PY" --torch-backend cu126 \
        -c "$CONSTRAINTS" 'torch==2.14.1+cu126' 'torchvision==0.29.1+cu126'
fi

if [ "$NEEDS_STACK" -eq 1 ]; then
    git -C "$REPO" submodule update --init third_party/segment-anything-2
    apt-get update -qq
    apt-get install -y libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0
    uv --no-config pip install --python "$PY" --torch-backend cu126 \
        -c "$CONSTRAINTS" -e "$REPO/third_party/segment-anything-2"
    uv --no-config pip install --python "$PY" --torch-backend cu126 \
        -c "$CONSTRAINTS" -e "$REPO/src"
else
    echo 'Stretch AI and SAM2 are installed; keeping them and repairing only recorded core versions.'
fi

# IMPORTANT: these are explicit compatibility overrides AFTER the full dependency install.
# --no-deps prevents their known metadata conflicts from replacing the GPU packages again.
# It is NOT a recommendation to bypass dependencies for the full Stretch/SAM2 installation.
uv --no-config pip install --python "$PY" --no-deps \
    'transformers==4.49.0' 'tokenizers==0.21.4' 'huggingface-hub==0.36.2' \
    'setuptools==69.5.1' 'open3d==0.20.0' 'numpy==1.26.4' 'ultralytics==8.3.161'

"$PY" "$GUARD" snapshot > "$AUDIT/after_core.json"
uv --no-config pip freeze --python "$PY" > "$AUDIT/after_freeze.txt"
# Preserve diagnostic output, including the known metadata exceptions; do not call it clean.
CHECK_RC=0
uv --no-config pip check --python "$PY" > "$AUDIT/dependency_check.txt" 2>&1 || CHECK_RC=$?
printf '%s\n' "$CHECK_RC" > "$AUDIT/dependency_check_exit_status.txt"
"$PY" "$GUARD" verify
printf 'Setup audit saved: %s\n' "$AUDIT"
echo 'Section 2 core checks passed. Run Section 3 before any data replay.'

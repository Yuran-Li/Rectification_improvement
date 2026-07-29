#!/bin/bash
# Alliance/Narval PAG env bootstrap (venv). Adapts docs/ENV.md to CC wheel tags.
set -euo pipefail

VENV=/scratch/yuranli/virtualenvs/PAG
REPO=/home/yuranli/scratch/verify_then_rectify/PAG-reproduction
WHL_DIR="$REPO/.tmp_whl"
CC_GENERIC=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic
CC_GENTOO=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic

source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$WHL_DIR"
cd "$WHL_DIR"

retag_manylinux() {
  # Usage: retag_manylinux input.whl
  # Rewrites *manylinux*_x86_64.whl / *manylinux1_x86_64.whl -> *linux_x86_64.whl
  local src="$1"
  local dst
  dst=$(echo "$src" | sed -E 's/manylinux[0-9_]*x86_64/linux_x86_64/; s/manylinux1_x86_64/linux_x86_64/')
  if [[ "$src" == "$dst" ]]; then
    echo "No retag needed: $src"
    echo "$src"
    return
  fi
  cp -f "$src" "$dst"
  echo "$dst"
}

download_pypi_wheel() {
  # Usage: download_pypi_wheel package version substr
  local pkg="$1" ver="$2" substr="$3"
  python - <<PY
import urllib.request, json, sys
pkg, ver, substr = "$pkg", "$ver", "$substr"
data = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/{ver}/json", timeout=60))
cands = [f for f in data["urls"] if f["packagetype"] == "bdist_wheel" and substr in f["filename"]]
if not cands:
    sys.stderr.write(f"No wheel for {pkg}=={ver} matching {substr}\n")
    sys.exit(1)
f = cands[0]
out = f["filename"]
print(f"Downloading {out}", flush=True)
urllib.request.urlretrieve(f["url"], out)
print(out)
PY
}

echo "=== [1/6] ray 2.56.0 (PyPI manylinux -> linux_x86_64) ==="
if [[ ! -f ray-2.56.0-cp310-cp310-manylinux2014_x86_64.whl ]]; then
  download_pypi_wheel ray 2.56.0 'cp310-cp310-manylinux2014_x86_64'
fi
RAY_WHL=$(retag_manylinux ray-2.56.0-cp310-cp310-manylinux2014_x86_64.whl)
pip install --no-deps "$RAY_WHL"

echo "=== [2/6] vllm 0.8.4+computecanada (CC wheel; 0.8.2 unavailable) ==="
VLLM_WHL="$CC_GENTOO/vllm-0.8.4+computecanada-cp38-abi3-linux_x86_64.whl"
pip install --no-deps "$VLLM_WHL"

echo "=== [3/6] Critical pins (numpy/transformers/accelerate/...) ==="
pip install --no-deps \
  "$CC_GENTOO/numpy-1.26.4+computecanada-cp310-cp310-linux_x86_64.whl" \
  "$CC_GENERIC/transformers-4.51.3+computecanada-py3-none-any.whl"

# accelerate / math-verify: pure py3 wheels from PyPI
if [[ ! -f accelerate-1.4.0-py3-none-any.whl ]]; then
  download_pypi_wheel accelerate 1.4.0 'py3-none-any'
fi
if [[ ! -f math_verify-0.9.0-py3-none-any.whl ]]; then
  download_pypi_wheel math-verify 0.9.0 'py3-none-any'
fi
pip install --no-deps accelerate-1.4.0-py3-none-any.whl math_verify-0.9.0-py3-none-any.whl

# tensordict needs retag
if [[ ! -f tensordict-0.6.2-cp310-cp310-manylinux1_x86_64.whl ]]; then
  download_pypi_wheel tensordict 0.6.2 'cp310-cp310-manylinux1_x86_64'
fi
TD_WHL=$(retag_manylinux tensordict-0.6.2-cp310-cp310-manylinux1_x86_64.whl)
pip install --no-deps "$TD_WHL"

# wandb retag
if [[ ! -f wandb-0.28.0-py3-none-manylinux_2_28_x86_64.whl ]]; then
  download_pypi_wheel wandb 0.28.0 'py3-none-manylinux_2_28_x86_64'
fi
WANDB_WHL=$(retag_manylinux wandb-0.28.0-py3-none-manylinux_2_28_x86_64.whl)
pip install --no-deps "$WANDB_WHL"

echo "=== [4/6] flash-attn (CC cp310 max is 2.5.7; torch cxx11_abi=True) ==="
# Dao FALSE wheel is incompatible with CC torch (abi True). Use CC 2.5.7.
pip install --no-deps "$CC_GENTOO/flash_attn-2.5.7+computecanada-cp310-cp310-linux_x86_64.whl"

echo "=== [5/6] vllm runtime deps + requirements.txt (best-effort) ==="
# Install remaining declared deps; allow CC resolver. Then re-pin critical packages.
pip install \
  'huggingface-hub>=0.30.0' \
  'tokenizers>=0.19.1' \
  'safetensors' \
  'sentencepiece' \
  'protobuf' \
  'fastapi' \
  'uvicorn' \
  'openai' \
  'pydantic' \
  'prometheus-client' \
  'prometheus-fastapi-instrumentator' \
  'pillow' \
  'tiktoken' \
  'lm-format-enforcer' \
  'outlines==0.1.11' \
  'lark' \
  'msgspec' \
  'gguf' \
  'importlib_metadata' \
  'einops' \
  'cloudpickle' \
  'watchfiles' \
  'python-json-logger' \
  'scipy' \
  'ninja' \
  'psutil' \
  'py-cpuinfo' \
  'blake3' \
  'cachetools' \
  'requests' \
  'tqdm' \
  'aiohttp' \
  'partial-json-parser' \
  'pyzmq' \
  'opencv-python-headless' \
  'compressed-tensors' \
  'depyf' \
  'mistral-common' \
  'opentelemetry-sdk' \
  'opentelemetry-api' \
  'opentelemetry-exporter-otlp' \
  'opentelemetry-semantic-conventions-ai>=0.4.1,<0.5.0' \
  'filelock' \
  'typing-extensions' \
  'PyYAML' \
  || true

# Repo declared deps
pip install \
  codetiming \
  datasets \
  dill \
  hydra-core \
  'liger-kernel' \
  pandas \
  peft \
  'pyarrow>=15.0.0' \
  pybind11 \
  pylatexenc \
  'pylint==3.3.6' \
  torchdata \
  || true

# Force critical pins again (resolver may have upgraded them)
pip install --no-deps --force-reinstall \
  "$CC_GENTOO/numpy-1.26.4+computecanada-cp310-cp310-linux_x86_64.whl" \
  "$CC_GENERIC/transformers-4.51.3+computecanada-py3-none-any.whl" \
  accelerate-1.4.0-py3-none-any.whl \
  math_verify-0.9.0-py3-none-any.whl \
  "$TD_WHL" \
  "$RAY_WHL" \
  "$VLLM_WHL" \
  "$CC_GENTOO/flash_attn-2.5.7+computecanada-cp310-cp310-linux_x86_64.whl"

echo "=== [6/6] editable install of this repo ==="
cd "$REPO"
pip install -e . --no-deps || pip install -e .

echo "=== DONE: versions ==="
python - <<'PY'
mods = [
  'torch','vllm','flash_attn','transformers','accelerate','numpy',
  'ray','math_verify','tensordict','wandb'
]
for m in mods:
  try:
    mod = __import__(m)
    ver = getattr(mod, '__version__', '?')
    print(f'{m}: {ver}')
  except Exception as e:
    print(f'{m}: IMPORT FAIL ({e})')
import torch
print('torch.cuda:', torch.version.cuda, 'cxx11_abi:', torch._C._GLIBCXX_USE_CXX11_ABI)
PY

#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["PYTHONPATH"] = ""
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import importlib
import traceback

OUT = "/home/yuranli/scratch/verify_then_rectify/PAG-reproduction/.tmp_whl/verify_out.txt"

def log(msg: str) -> None:
    with open(OUT, "a") as f:
        f.write(msg + "\n")
        f.flush()
    print(msg, flush=True)

open(OUT, "w").write("")
log("BEGIN")

checks = [
    "numpy", "orjson", "torch", "transformers", "accelerate", "ray",
    "flash_attn", "tensordict", "wandb", "math_verify", "pyarrow", "pandas",
    "datasets", "hydra", "omegaconf", "peft", "codetiming", "verl", "pydantic",
]
for m in checks:
    try:
        mod = importlib.import_module(m)
        log(f"OK  {m}: {getattr(mod, '__version__', 'OK')}")
    except Exception as e:
        log(f"FAIL {m}: {type(e).__name__}: {e}")

log("NON_VLLM_DONE")
try:
    import vllm
    log(f"OK  vllm: {vllm.__version__}")
except Exception as e:
    log(f"FAIL vllm: {type(e).__name__}: {e}")
    log(traceback.format_exc())
log("ALL_DONE")

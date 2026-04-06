from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT_INFERENCE = Path(__file__).resolve().parent.parent / "inference.py"
spec = importlib.util.spec_from_file_location("adctm_root_inference_pkg", str(ROOT_INFERENCE))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load root inference module from {ROOT_INFERENCE}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Note: The root inference.py (D:/ADCTM/ADCTMEnv/inference.py) does not actually 
# export `predict_action` or `execute_simulation` right now. 
# It has `parse_action`, `call_llm`, `run_task`, and `main`.
# However, the previous `inference/inference.py` had an execute_simulation that got deleted.
# For now we will try to make this import block not crash.

predict_action = getattr(module, "predict_action", None)
execute_simulation = getattr(module, "execute_simulation", None)

__all__ = ["predict_action", "execute_simulation"]

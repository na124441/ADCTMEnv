import importlib.util
from pathlib import Path


ROOT_INFERENCE = Path(__file__).resolve().parent.parent / "inference.py"
spec = importlib.util.spec_from_file_location("adctm_root_inference", str(ROOT_INFERENCE))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load root inference module from {ROOT_INFERENCE}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
execute_simulation = module.execute_simulation

__all__ = ["execute_simulation"]

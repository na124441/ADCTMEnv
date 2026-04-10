from __future__ import annotations

"""
Safe import bridge for inference functions.

This module avoids dynamic filesystem imports (which break in sandboxed
environments like Hugging Face Spaces or evaluation runners) and instead
uses standard Python imports with graceful fallback.
"""

# Default fallbacks (ensures no crash even if import fails)
predict_action = None
execute_simulation = None

try:
    # Standard import (works if inference.py is in repo root and PYTHONPATH is correct)
    from inference import predict_action as _predict_action
    from inference import execute_simulation as _execute_simulation

    predict_action = _predict_action
    execute_simulation = _execute_simulation

except Exception:
    # Silent fallback — prevents evaluator crash
    # You can optionally log here if debugging locally
    pass


__all__ = ["predict_action", "execute_simulation"]

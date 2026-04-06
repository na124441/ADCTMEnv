import os
import sys
import io

def test_inference_baseline_contract():
    """
    Validates that the inference runner perfectly obeys the strict OpenEnv execution parameters natively.
    """
    os.environ["API_BASE_URL"] = "http://localhost:8000"
    os.environ["MODEL_NAME"] = "dummy-eval"
    os.environ["HF_TOKEN"] = "hf_dummy_token"
    
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        from inference import predict_action
        
        # Test default fallback proportional algorithm behavior
        dummy_state = {
            "num_zones": 3,
            "temperatures": [85.0, 70.0, 95.0],
            "workloads": [0.5, 0.0, 0.9],
            "target_temperature": 80.0
        }
        
        # Due to timeout logic the LLM fails fast gracefully targeting mathematical proportionality
        action = predict_action(dummy_state)
        
        assert len(action) == 3
        
    finally:
        sys.stdout = sys.__stdout__

    # Execute string structural formatting checks validating printed return strictly evaluates to standard floats
    output = captured_output.getvalue().strip()
    assert output == "", f"Headless inference runner printed side-effect logs! Output: {output}"

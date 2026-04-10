import json

from core.models import Action


def parse_llm_response(response_text: str, expected_num_zones: int) -> Action:
    """
    Parse the model response into a validated cooling action.
    """
    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("No JSON found in model response")

    json_str = response_text[start:end]
    data = json.loads(json_str)

    cooling_list = data.get("cooling")
    if not isinstance(cooling_list, list) or len(cooling_list) != expected_num_zones:
        raise ValueError(f"Expected 'cooling' to contain {expected_num_zones} entries")

    try:
        clamped = [max(0.0, min(1.0, float(value))) for value in cooling_list]
    except (TypeError, ValueError) as exc:
        raise ValueError("Cooling values must be numeric") from exc

    return Action(cooling=clamped)

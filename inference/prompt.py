# inference/prompt.py

PROMPT_TEMPLATE = """You are controlling the cooling system of a multi-zone data-center.

Goal:
Keep every zone temperature â‰¤ {safe_temp:.1f}Â°C while using as little cooling as possible and avoiding rapid changes.

Constraints:
- Cooling for each zone must be a float between 0 (off) and 1 (full power).
- Output exactly a JSON object with a single key "cooling" holding a list of {n_zones} numbers.
- The order of numbers matches the order of temperatures in the observation.

Current step: {step}
Number of zones: {n_zones}
Ambient temperature: {ambient:.1f}Â°C

Zone states:
{zone_table}

Give your answer now:
"""


def build_zone_table(observation):
    """Build a formatted markdown-style table of zone data."""
    rows = []
    headers = "| Zone | Temp (Â°C) | Workload | Prev Cooling |"
    separator = "|------|-----------|----------|--------------|"

    for i, (temp, wl, cool) in enumerate(zip(
        observation.temperatures,
        observation.workloads,
        observation.cooling
    )):
        row = f"| {i} | {temp:.1f} | {wl:.2f} | {cool:.2f} |"
        rows.append(row)

    return "\n".join([headers, separator] + rows)


def build_prompt(observation, task_config):
    """Construct full prompt string for the LLM."""
    zone_table = build_zone_table(observation)
    return PROMPT_TEMPLATE.format(
        safe_temp=task_config["safe_temperature"],
        step=observation.time_step,
        n_zones=len(observation.temperatures),
        ambient=observation.ambient_temp,
        zone_table=zone_table,
    )

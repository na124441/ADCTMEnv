from inference.prompt import build_prompt, build_zone_table


def test_build_zone_table_contains_one_row_per_zone(observation):
    table = build_zone_table(observation)
    assert table.count("| 0 |") == 1
    assert table.count("| 1 |") == 1
    assert table.count("| 2 |") == 1


def test_build_prompt_includes_core_fields(observation, task_config):
    prompt = build_prompt(observation, task_config.model_dump())
    assert 'single key "cooling"' in prompt
    assert f"Current step: {observation.time_step}" in prompt
    assert f"Number of zones: {task_config.num_zones}" in prompt

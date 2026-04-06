import pytest

from grader.metrics import compute_metrics


def test_compute_metrics_returns_zeroes_for_empty_actions(task_config):
    assert compute_metrics([], [], task_config) == (0.0, 0.0, 0.0, 0.0)


def test_compute_metrics_computes_safety_energy_and_jitter(task_config):
    observations = [
        {"temperatures": [45.0, 48.0, 50.0]},
        {"temperatures": [60.0, 61.0, 62.0]},
        {"temperatures": [90.0, 61.0, 62.0]},
    ]
    actions = [
        {"cooling": [0.0, 0.5, 1.0]},
        {"cooling": [0.5, 0.5, 0.5]},
    ]
    safety, energy, jitter, target_error = compute_metrics(observations, actions, task_config)
    assert safety == pytest.approx(0.5)
    assert energy == pytest.approx(0.5)
    assert jitter == pytest.approx((0.5 + 0.0 + 0.5) / 3)

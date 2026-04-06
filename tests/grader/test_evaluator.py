import pytest

from grader.evaluator import evaluate_trajectory


def test_evaluate_trajectory_combines_metrics(monkeypatch, task_config):
    monkeypatch.setattr("grader.evaluator.compute_metrics", lambda observations, actions, config: (0.8, 0.25, 0.1, 0.05))
    score = evaluate_trajectory([], [], task_config)
    assert score == pytest.approx(0.4 * 0.8 + 0.3 * 0.95 + 0.2 * 0.75 + 0.1 * 0.9)


def test_evaluate_trajectory_clamps_score(monkeypatch, task_config):
    monkeypatch.setattr("grader.evaluator.compute_metrics", lambda observations, actions, config: (0.0, 5.0, 5.0, 2.0))
    assert evaluate_trajectory([], [], task_config) == 0.0

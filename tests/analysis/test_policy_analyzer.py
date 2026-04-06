from analysis.policy_analyzer import assess_policy_type, evaluate_action_quality


def test_evaluate_action_quality_covers_urgent_cases():
    labels = evaluate_action_quality([84.0, 69.0, 76.0], [0.9, 0.8, 0.1], [0.2, 0.0, 0.0], 85.0)
    assert labels[0] == "Optimal (Reactive)"
    assert labels[1] == "Over-reacting (Wasteful)"
    assert labels[2] == "Idle (Dangerous)"


def test_assess_policy_type_variants():
    assert assess_policy_type(0.5, 0.1) == "AGGRESSIVE-REACTIVE"
    assert assess_policy_type(0.5, 0.001) == "STEADY-STABLE"
    assert assess_policy_type(0.8, 0.01) == "STEADY-AGGRESSIVE"
    assert assess_policy_type(0.1, 0.01) == "PASSIVE-CONSERVATIVE"

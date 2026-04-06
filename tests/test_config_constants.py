from config.constants import ALPHA, BETA, GAMMA


def test_constants_are_positive_numeric():
    assert isinstance(ALPHA, (int, float))
    assert isinstance(BETA, (int, float))
    assert isinstance(GAMMA, (int, float))
    assert ALPHA > 0
    assert BETA > 0
    assert GAMMA > 0


def test_constants_regression_values():
    assert (ALPHA, BETA, GAMMA) == (7.5, 8.0, 0.1)

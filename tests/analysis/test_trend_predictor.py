from analysis.trend_predictor import predict_thermal_future


def test_predict_thermal_future_initializing():
    assert predict_thermal_future([70.0, 71.0], [[69.0, 70.0]], 85.0) == ["Initializing...", "Initializing..."]


def test_predict_thermal_future_stable_cooling():
    history = [
        [75.0, 76.0],
        [74.5, 75.5],
        [74.0, 75.0],
        [73.5, 74.5],
        [73.0, 74.0],
    ]
    assert predict_thermal_future([73.0, 74.0], history, 85.0) == ["Stable/Cooling", "Stable/Cooling"]


def test_predict_thermal_future_critical_and_projected():
    history = [
        [82.0, 70.0],
        [83.0, 71.0],
        [84.0, 72.0],
        [85.0, 73.0],
        [86.0, 74.0],
    ]
    output = predict_thermal_future([86.0, 74.0], history, 85.0)
    assert output[0] == "CRITICAL"
    assert output[1].startswith("~")

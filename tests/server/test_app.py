from fastapi.testclient import TestClient

import server.app as server_app


def test_server_root_endpoint():
    client = TestClient(server_app.app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"].startswith("Welcome")


def test_server_step_requires_reset():
    client = TestClient(server_app.app)
    response = client.post("/step", json={"cooling": [0.1, 0.2, 0.3]})
    assert response.status_code == 400


def test_server_reset_and_state(base_task_dict):
    client = TestClient(server_app.app)
    response = client.post("/reset", json=base_task_dict)
    assert response.status_code == 200
    assert response.json()["observation"]["time_step"] == 0
    state_response = client.get("/state")
    assert state_response.status_code == 200
    assert state_response.json()["step_counter"] == 0


def test_server_rejects_invalid_reset_payload():
    client = TestClient(server_app.app)
    response = client.post("/reset", json={"num_zones": 3})
    assert response.status_code == 422

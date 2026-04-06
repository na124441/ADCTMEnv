import app as app_entry
import server.app


def test_app_entry_re_exports_server_app():
    assert app_entry.app is server.app.app


def test_app_entry_main_delegates(monkeypatch):
    called = {"count": 0}

    def fake_main():
        called["count"] += 1

    monkeypatch.setattr(app_entry, "main", fake_main)
    app_entry.main()
    assert called["count"] == 1

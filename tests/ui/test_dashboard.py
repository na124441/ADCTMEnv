from rich.layout import Layout

from ui.dashboard import make_dashboard


def test_make_dashboard_returns_layout(dashboard_state):
    layout = make_dashboard(dashboard_state, "utf-8")
    assert isinstance(layout, Layout)
    assert layout["header"].name == "header"
    assert layout["footer"].name == "footer"


def test_make_dashboard_renders_default_log_when_empty(dashboard_state, render_rich_text):
    dashboard_state.log.clear()
    text = render_rich_text(make_dashboard(dashboard_state, "utf-8"))
    assert "System standby" in text

from rich.panel import Panel

from ui.panels.analysis import (
    render_action_quality_panel,
    render_behavior_summary,
    render_delta_graph,
    render_forecast_panel,
    render_reward_chart,
)


def test_analysis_panels_return_panels(dashboard_state):
    assert isinstance(render_delta_graph(dashboard_state), Panel)
    assert isinstance(render_reward_chart(dashboard_state), Panel)
    assert isinstance(render_forecast_panel(dashboard_state), Panel)
    assert isinstance(render_action_quality_panel(dashboard_state), Panel)
    assert isinstance(render_behavior_summary(dashboard_state, "BALANCED-OPTIMAL"), Panel)

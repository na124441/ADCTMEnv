from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.console import Group
from rich.align import Align
from rich import box
from core.dashboard_state import DashboardState
from ui.panels.header import render_header
from ui.panels.metrics import render_metrics_bar
from ui.panels.thermal import render_normalized_heatbars, render_action_overlay
from ui.panels.analysis import (
    render_delta_graph, render_reward_chart,
    render_forecast_panel, render_action_quality_panel, render_behavior_summary
)
from ui.panels.llm import render_llm_trace
from analysis.policy_analyzer import assess_policy_type

def make_dashboard(state: DashboardState, console_encoding: str) -> Layout:
    layout = Layout()
    
    # 1. Compact Dashboard Structure (Fits ~25-30 lines)
    # --------------------------------------------------------------
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="visuals", ratio=45),   # Thermal + Overlay
        Layout(name="intelligence", ratio=30), # Analytical block
        Layout(name="agent", ratio=25),      # LLM Trace
        Layout(name="footer", size=5)
    )

    # 2. Merged Top Bar (Header + Metrics)
    # --------------------------------------------------------------
    top_grid = Table.grid(expand=True)
    top_grid.add_column()
    top_grid.add_column(justify="right")
    top_grid.add_row(render_header(state), render_metrics_bar(state))
    
    layout["header"].update(Panel(top_grid, title="ADCTMS Command Center", border_style="cyan", box=box.SQUARE))

    # 3. Visuals Zone
    # --------------------------------------------------------------
    layout["visuals"].split_row(
        Layout(render_normalized_heatbars(state, console_encoding), ratio=2),
        Layout(render_action_overlay(state), ratio=3)
    )

    # 4. Intelligence Block (Combined Analysis)
    # --------------------------------------------------------------
    personality = assess_policy_type(state.policy_signature["avg"], state.policy_signature["var"])
    
    layout["intelligence"].split_row(
        Layout(render_delta_graph(state), ratio=1),
        Layout(render_forecast_panel(state), ratio=1),
        Layout(render_action_quality_panel(state), ratio=1),
        Layout(render_reward_chart(state), ratio=2),
        Layout(render_behavior_summary(state, personality), ratio=2)
    )

    # 5. Agent Logic (LLM Trace)
    # --------------------------------------------------------------
    layout["agent"].update(render_llm_trace(state))

    # 6. Active Alert System & Zone Status Strip (Footer)
    # --------------------------------------------------------------
    log_msg = state.log[-1] if state.log else "System standby"
    
    # Create the Zone Telemetry Strip
    zone_strip = Table.grid(padding=(0, 2))
    for i, (t, a) in enumerate(zip(state.temperatures, state.cooling)):
        t_style = "bold red" if t >= state.safe_temp else "yellow" if t > state.target_temp + 5.0 else "green"
        zone_strip.add_column()
        zone_strip.add_row(Text.assemble(
            (f"Z{i+1:02d}: ", "dim"),
            (f"{t:4.1f}Â°C", t_style),
            (f" [{a:.2f}]", "magenta")
        ))

    # Compile Footer Content
    max_t = max(state.temperatures)
    avg_a = state.policy_signature["avg"]
    alert_style = "cyan"
    
    if max_t > state.safe_temp:
        alert_style = "bold white on red"
        log_msg = f"ðŸš¨ CRITICAL THERMAL EVENT: Z{state.temperatures.index(max_t)+1:02d} AT {max_t:.1f}Â°C!"
    elif max_t > state.target_temp + 5.0 and avg_a < 0.1:
        alert_style = "yellow"
        log_msg = f"âš  OPTIMIZATION GAP: Rising heat ({max_t:.1f}Â°C), agent is idling."

    footer_content = Group(
        Align.center(zone_strip),
        Text(f"â–¶ {log_msg}", style=alert_style)
    )

    layout["footer"].update(Panel(footer_content, title="ADCTMS Master Console : Active Status & Log", box=box.SQUARE, border_style="magenta"))

    return layout

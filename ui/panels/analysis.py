from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group
from rich import box
from core.dashboard_state import DashboardState

def render_delta_graph(state: DashboardState) -> Panel:
    """Visualizes the rate of change per zone."""
    lines = []
    for i, d in enumerate(state.deltas):
        label = f"Z{i+1:02d}"
        if d > 1.0: icon, style = "â†‘â†‘", "red"
        elif d > 0.1: icon, style = "â†‘ ", "yellow"
        elif d < -1.0: icon, style = "â†“â†“", "green"
        elif d < -0.1: icon, style = "â†“ ", "cyan"
        else: icon, style = "â†’ ", "dim"
        
        lines.append(f"{label}: [{style}]{icon} {d:+.2f}Â°C/step[/]")
    
    return Panel("\n".join(lines), title="[bold]Thermal Delta (Velocity)[/]", box=box.SQUARE, border_style="cyan")

def render_reward_chart(state: DashboardState) -> Panel:
    """Graphic bar chart of reward penalties."""
    o_len = int(abs(state.overshoot_term) * 5)
    e_len = int(abs(state.energy_term) * 5)
    s_len = int(abs(state.smoothness_term) * 5)
    
    chart = Group(
        Text.assemble(("Overshoot:  ", "dim"), ("â–ˆ" * o_len, "red"), (f" {state.overshoot_term:+.3f}", "red")),
        Text.assemble(("Energy:     ", "dim"), ("â–ˆ" * e_len, "yellow"), (f" {state.energy_term:+.2f}", "yellow")),
        Text.assemble(("Smoothness: ", "dim"), ("â–ˆ" * s_len, "cyan"), (f" {state.smoothness_term:+.2f}", "cyan")),
        Text.assemble(("\nTotal Step Reward: ", "bold white"), (f"{state.reward:+.3f}", "bold magenta"))
    )
    return Panel(chart, title="[bold]Reward Decomposition (Penalty Breakdown)[/]", border_style="cyan", box=box.SQUARE)

def render_forecast_panel(state: DashboardState) -> Panel:
    """Display time-to-violation predictions."""
    lines = []
    for i, forecast in enumerate(state.forecasts):
        label = f"Z{i+1:02d}"
        # Handle forecast strings like "~5 steps" or "Stable"
        steps_val = 999
        if "steps" in forecast:
            try:
                # Remove '~' and split to get the number
                raw_num = forecast.replace("~", "").split()[0]
                steps_val = int(raw_num)
            except (ValueError, IndexError):
                pass
        
        style = "red" if steps_val < 3 else "yellow" if steps_val < 10 else "green"
        lines.append(f"{label}: [{style}]{forecast}[/]")
        
    return Panel("\n".join(lines), title="[bold]Thermal Forecast (Predict_85Â°C)[/]", border_style="yellow", box=box.SQUARE)

def render_action_quality_panel(state: DashboardState) -> Panel:
    """Control system validation table."""
    table = Table(box=box.SIMPLE, header_style="bold cyan")
    table.add_column("Zone")
    table.add_column("Assessment", justify="center")
    
    for i, quality in enumerate(state.action_quality):
        table.add_row(f"Z{i+1:02d}", quality)
        
    return Panel(table, title="[bold]Action Quality (Control Validation)[/]", border_style="magenta", box=box.SQUARE)

def render_behavior_summary(state: DashboardState, personality: str) -> Panel:
    """Synthesized behavior intel panel."""
    issues = []
    if "OSCILLATING" in personality: issues.append("High Thermal Jitter")
    if any("Under-reacting" in q for q in state.action_quality): issues.append("Response Lag (Risk)")
    if any("Over-reacting" in q for q in state.action_quality): issues.append("Power waste (Overcooling)")
    
    if not issues: issues = ["None - Stable performance"]
    
    content = Table.grid(padding=(0,1))
    content.add_row("Policy Type:", f"[bold cyan]{personality}[/]")
    content.add_row("Stability:  ", f"[bold {'green' if state.trajectory_status == 'STABLE' else 'yellow'}]{state.trajectory_status}[/]")
    content.add_row("Critical Issues:", f"[red]{', '.join(issues)}[/]")
    
    return Panel(content, title="[bold]Agent Behavioral Summary (Intelligence)[/]", border_style="magenta", box=box.SQUARE)

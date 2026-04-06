from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich import box
from core.dashboard_state import DashboardState

def render_metrics_bar(state: DashboardState) -> Panel:
    avg_temp = sum(state.temperatures)/len(state.temperatures) if state.temperatures else 0
    max_temp = max(state.temperatures) if state.temperatures else 0
    status = "COMPLETED" if state.done else state.trajectory_status
    status_style = "bold green" if status == "STABLE" else "bold yellow" if "TREND" in status else "bold red"

    metrics = Text.assemble(
        ("âš¡ Avg: ", "dim"), (f"{avg_temp:.1f}Â°C  ", "cyan"),
        ("ðŸ”¥ Max: ", "dim"), (f"{max_temp:.1f}Â°C  ", "red" if max_temp >= state.safe_temp else "yellow"),
        ("ðŸ“ˆ Status: ", "dim"), (f"{status}  ", status_style),
        ("ðŸ§  Conf: ", "dim"), (f"{state.confidence:.0f}%  ", "magenta"),
        ("ðŸ“Š Total: ", "dim"), (f"{state.total_reward:.1f}", "bold blue")
    )
    return Align.center(metrics)

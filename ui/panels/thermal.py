from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.dashboard_state import DashboardState


def render_normalized_heatbars(state: DashboardState, console_encoding: str) -> Panel:
    """Render thermal load normalized from target (0.0) to safe (1.0)."""
    lines = []
    is_utf = console_encoding.lower().startswith("utf")
    char = "â–ˆ" if is_utf else "#"

    for index, temperature in enumerate(state.temperatures):
        label = f"Z{index + 1:02d}"
        norm = (temperature - state.target_temp) / (state.safe_temp - state.target_temp)
        length = max(0, min(20, int(abs(norm) * 20)))

        if norm >= 1.0:
            color, bar_char = "bold white on red", "X"
        elif norm >= 0.8:
            color, bar_char = "red", char
        elif norm >= 0.0:
            color, bar_char = "yellow", char
        else:
            color, bar_char = "cyan", char

        bar = (bar_char * length).ljust(20)
        lines.append(f"{label} [{color}]{bar}[/] {temperature:5.1f} C (util: {norm:+.2f})")

    return Panel(
        "\n".join(lines),
        title="[bold]Normalized Thermal Load (Target -> Safe)[/]",
        box=box.SQUARE,
        border_style="cyan",
    )


def render_action_overlay(state: DashboardState) -> Panel:
    """Render a paired view comparing normalized temperature to agent action."""
    table = Table.grid(padding=(0, 1))
    table.add_column("Zone", style="dim")
    table.add_column("Load vs Action insight", width=40)

    for index, (temperature, action) in enumerate(zip(state.temperatures, state.cooling)):
        norm_t = (temperature - state.target_temp) / (state.safe_temp - state.target_temp)
        temp_len = max(0, min(15, int(norm_t * 15)))
        action_len = max(0, min(15, int(action * 15)))

        bar = Text.assemble(
            ("â–ˆ" * temp_len, "yellow"),
            (" " * max(0, 15 - temp_len)),
            (" | ", "dim"),
            ("â–®" * action_len, "magenta"),
        )
        table.add_row(f"Z{index + 1:02d}", bar)

    return Panel(table, title="[bold]Control Theory Overlay (Load | Action)[/]", box=box.SQUARE, border_style="magenta")

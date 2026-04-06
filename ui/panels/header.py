from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.align import Align
from rich import box
from core.dashboard_state import DashboardState

def render_title() -> Panel:
    """Renders the main project title in a bold, block-like style."""
    title_text = Text.assemble(
        (" ADCTMS ", "bold #ffffff on #bd93f9"),
        (" : Autonomous Data Centre Thermal Management System ", "bold white"),
        ("(RL based)", "dim italic cyan")
    )
    return Panel(
        Align.center(title_text),
        box=box.DOUBLE_EDGE,
        border_style="magenta",
        padding=(0, 2)
    )

def render_header(state: DashboardState) -> Panel:
    # Left: Basic Info
    info_text = Text.assemble(
        (f"Task: ", "dim"), (f"{state.task_name}  ", "green"),
        (f"Model: ", "dim"), (f"{state.model}  ", "cyan"),
        (f"Step: ", "dim"), (f"{state.step}", "yellow")
    )
    
    # Right: Hyper-params strip
    hp_table = Table.grid(padding=(0, 1))
    hp_table.add_row(
        Text.assemble(("Î±:", "dim"), (f"{state.alpha:.1f} ", "bold")),
        Text.assemble(("Î²:", "dim"), (f"{state.beta:.1f} ", "bold")),
        Text.assemble(("Î³:", "dim"), (f"{state.gamma:.1f} ", "bold")),
        Text.assemble(("Safe:", "dim"), (f"{state.safe_temp}Â°C", "bold yellow"))
    )
    
    content = Table.grid(expand=True)
    content.add_column(justify="left")
    content.add_column(justify="right")
    content.add_row(info_text, hp_table)
    return content

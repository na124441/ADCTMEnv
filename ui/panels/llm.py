from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.syntax import Syntax
from rich.console import Group
from rich import box
from core.dashboard_state import DashboardState

def render_llm_trace(state: DashboardState) -> Panel:
    """Renders the decision-making trace of the LLM agent."""
    # 1. Prompt (Condensed)
    prompt_p = Panel(state.prompt if state.prompt else "...", title="Prompt", border_style="dim", box=box.SQUARE)
    
    # 2. Raw Output
    json_syntax = Syntax(state.llm_raw_output if state.llm_raw_output else "{}", "json", theme="monokai", line_numbers=False)
    raw_p = Panel(json_syntax, title="Raw JSON Output", border_style="dim", box=box.SQUARE)
    
    # 3. Parsed Result
    table = Table(box=box.SIMPLE, header_style="bold magenta")
    table.add_column("Zone")
    table.add_column("Intensity", justify="right")
    
    if state.cooling:
        for i, v in enumerate(state.cooling):
            table.add_row(f"Z{i+1:02d}", f"{v:.3f}")
            
    res_group = Group(table)
    if state.parse_error:
        res_group = Group(Panel(f"[bold red]PARSE ERROR:[/] {state.parse_error}", border_style="red"), table)
        
    parsed_p = Panel(res_group, title="Parsed Decision", border_style="magenta", box=box.SQUARE)
    
    # Layout Assembly
    layout = Layout()
    layout.split_row(
        Layout(prompt_p, ratio=2),
        Layout(raw_p, ratio=1),
        Layout(parsed_p, ratio=1)
    )
    return Panel(layout, title="[bold magenta]LLM Decision Trace[/]", border_style="magenta", box=box.SQUARE)

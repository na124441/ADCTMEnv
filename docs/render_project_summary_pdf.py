from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "ADCTM_Project_Summary.md"
OUTPUT = ROOT / "ADCTM_Project_Summary.pdf"


@dataclass
class Block:
    kind: str
    text: str = ""
    level: int = 0


def parse_markdown(text: str) -> list[Block]:
    blocks: list[Block] = []
    paragraph: list[str] = []
    table: list[str] = []
    code: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            joined = " ".join(part.strip() for part in paragraph if part.strip())
            if joined:
                blocks.append(Block(kind="paragraph", text=joined))
            paragraph = []

    def flush_table() -> None:
        nonlocal table
        if table:
            blocks.append(Block(kind="table", text="\n".join(table)))
            table = []

    def flush_code() -> None:
        nonlocal code
        if code:
            blocks.append(Block(kind="code", text="\n".join(code)))
            code = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            flush_paragraph()
            flush_table()
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code.append(line)
            continue

        if not line.strip():
            flush_paragraph()
            flush_table()
            blocks.append(Block(kind="spacer"))
            continue

        if line.startswith("|"):
            flush_paragraph()
            table.append(line)
            continue

        flush_table()

        if line.startswith("#"):
            flush_paragraph()
            level = len(line) - len(line.lstrip("#"))
            blocks.append(Block(kind="heading", text=line[level:].strip(), level=level))
            continue

        if re.match(r"^\d+\.\s+", line):
            flush_paragraph()
            blocks.append(Block(kind="numbered", text=line))
            continue

        if line.startswith("- "):
            flush_paragraph()
            blocks.append(Block(kind="bullet", text=line[2:].strip()))
            continue

        paragraph.append(line)

    flush_paragraph()
    flush_table()
    flush_code()
    return blocks


def render_lines(block: Block) -> list[tuple[str, str]]:
    if block.kind == "heading":
        return [("heading", block.text)]

    if block.kind == "spacer":
        return [("spacer", "")]

    if block.kind == "paragraph":
        return [("body", line) for line in textwrap.wrap(block.text, width=100)]

    if block.kind == "bullet":
        wrapped = textwrap.wrap(block.text, width=96)
        lines = []
        for index, line in enumerate(wrapped):
            prefix = "- " if index == 0 else "  "
            lines.append(("body", f"{prefix}{line}"))
        return lines

    if block.kind == "numbered":
        match = re.match(r"^(\d+\.)\s+(.*)$", block.text)
        if not match:
            return [("body", block.text)]
        prefix, rest = match.groups()
        wrapped = textwrap.wrap(rest, width=94)
        lines = []
        for index, line in enumerate(wrapped):
            label = f"{prefix} " if index == 0 else "   "
            lines.append(("body", f"{label}{line}"))
        return lines

    if block.kind in {"table", "code"}:
        rendered: list[tuple[str, str]] = []
        for line in block.text.splitlines():
            wrapped = textwrap.wrap(line, width=110, break_long_words=False, break_on_hyphens=False) or [""]
            rendered.extend(("mono", item) for item in wrapped)
        return rendered

    return [("body", block.text)]


def ensure_page(pdf: PdfPages, page_number: int):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    return fig


def add_footer(fig, page_number: int) -> None:
    fig.text(0.5, 0.03, f"ADCTM Project Summary | Page {page_number}", ha="center", va="center", fontsize=9, color="#666666")


def write_pdf(blocks: list[Block]) -> None:
    with PdfPages(OUTPUT) as pdf:
        page_number = 1
        fig = ensure_page(pdf, page_number)
        y = 0.95

        def new_page():
            nonlocal fig, y, page_number
            add_footer(fig, page_number)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            page_number += 1
            fig = ensure_page(pdf, page_number)
            y = 0.95

        title_rendered = False
        for block in blocks:
            if block.kind == "heading":
                level = block.level
                if level == 1 and not title_rendered:
                    needed = 0.10
                    if y - needed < 0.08:
                        new_page()
                    fig.text(0.5, 0.88, block.text, ha="center", va="center", fontsize=22, fontweight="bold", color="#12304A")
                    fig.text(0.5, 0.84, "Detailed repository walkthrough generated from local source analysis", ha="center", va="center", fontsize=11, color="#4B5D6B")
                    y = 0.79
                    title_rendered = True
                    continue

            lines = render_lines(block)
            if block.kind == "heading":
                needed = 0.04 if block.level == 2 else 0.032
            elif block.kind == "spacer":
                needed = 0.01
            else:
                line_height = 0.017 if all(kind == "body" for kind, _ in lines) else 0.015
                needed = max(0.02, len(lines) * line_height + 0.006)

            if y - needed < 0.07:
                new_page()

            if block.kind == "heading":
                if block.level == 2:
                    fig.text(0.08, y, block.text, ha="left", va="top", fontsize=15, fontweight="bold", color="#12304A")
                    y -= 0.03
                elif block.level == 3:
                    fig.text(0.08, y, block.text, ha="left", va="top", fontsize=12.5, fontweight="bold", color="#2E5266")
                    y -= 0.024
                else:
                    fig.text(0.08, y, block.text, ha="left", va="top", fontsize=18, fontweight="bold", color="#12304A")
                    y -= 0.032
                continue

            if block.kind == "spacer":
                y -= 0.010
                continue

            for kind, line in lines:
                if kind == "mono":
                    fig.text(0.10, y, line, ha="left", va="top", fontsize=9.2, family="monospace", color="#333333")
                    y -= 0.0145
                else:
                    fig.text(0.08, y, line, ha="left", va="top", fontsize=10.2, color="#222222")
                    y -= 0.0168
            y -= 0.004

        add_footer(fig, page_number)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source markdown not found: {SOURCE}")

    text = SOURCE.read_text(encoding="utf-8")
    blocks = parse_markdown(text)
    write_pdf(blocks)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

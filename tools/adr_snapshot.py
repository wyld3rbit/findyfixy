"""Draw the architecture as it stood at the close of each phase, from the live page.

Why: an ADR must show the system as it was when the decision was made. The
page code/docs/architecture.html shows only the latest state, and GitHub shows
an .html file as source. This script reads the page's own SVG, keeps the nodes
that existed at a phase, hides the ones that did not, and writes a standalone
SVG that GitHub renders inside a Markdown file. One source, so the picture
cannot drift from the page.

The drift guard is a test (tools/test_adr_snapshot.py): the nodes the page marks
built must equal the nodes of the latest phase here. Flip a node on the page
without adding the phase here, and `make check` fails.

Run from the training root: `make adr-snapshot`. Standard library only.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "code" / "docs" / "architecture.html"
OUT = ROOT / "docs" / "adr" / "img"

# The phase in which each node first existed. A node absent here is planned.
FIRST_BUILT = {
    "Question": 1, "Small model": 1, "Route": 1, "Large model": 1, "Triage record": 1,
    "Investigate": 2, "Deploy tool": 2, "Validate": 2,
    "Recall": 4,
    "Evals": 5,
}
PHASES = range(1, 6)

# Nodes that sit in the main flow. When they do not exist yet they are not drawn
# at all, because the edges around them would point at nothing. The planned lane
# (Tracing, Policy, Gateway, Evals, Human review) is always drawn.
FLOW = {"Investigate", "Deploy tool", "Validate", "Recall"}

# Each edge in the page: (from node, to node).
EDGES = {
    "e_q_inv": ("Question", "Investigate"), "e_inv_small": ("Investigate", "Small model"),
    "e_small_val": ("Small model", "Validate"), "e_val_route": ("Validate", "Route"),
    "e_retry": ("Validate", "Small model"), "e_route_rec": ("Route", "Triage record"),
    "e_route_large": ("Route", "Large model"), "e_large_rec": ("Large model", "Triage record"),
    "e_q_recall": ("Question", "Recall"), "e_inv_tool": ("Investigate", "Deploy tool"),
}
# Straight edges that stand in for the page's two-edge route while a flow node is
# missing: (from, to) -> (path, the flow node whose presence makes the page's edges apply).
BYPASS = {
    ("Question", "Small model"): ("M120,260 L294,260", "Investigate"),
    ("Small model", "Route"): ("M424,260 L598,260", "Validate"),
}
LABELS = {  # label text -> the nodes that must exist for it to make sense
    "retry with the error text (once)": {"Validate"},
    "similar incidents": {"Recall"},
}

DEFS = (
    '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
    '<path d="M0,1 L10,5 L0,9 z" fill="#5b6f99"/></marker>'
    '<marker id="arrow-tool" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
    '<path d="M0,1 L10,5 L0,9 z" fill="#fbbf24"/></marker></defs>\n'
)

STYLE = """\
<style>
.node rect, .node polygon { fill: #0b1220; stroke: #5cc8ff; stroke-width: 1.5; }
.node .t { fill: #dbe6ff; font-size: 14px; font-weight: 600; text-anchor: middle; }
.node .s, .node .badge { fill: #8da2c8; font-size: 11px; text-anchor: middle; }
.node[data-status=planned] rect { stroke: #7f8fb0; stroke-dasharray: 5 4; }
.node[data-status=planned] .t, .node[data-status=planned] .s { fill: #7f8fb0; }
.edge { fill: none; stroke: #5b6f99; stroke-width: 2; marker-end: url(#arrow); }
.edge.tool { stroke-dasharray: 5 5; stroke: #fbbf24; marker-end: url(#arrow-tool); }
.edge-label { fill: #8da2c8; font-size: 11px; text-anchor: middle; }
.caption { fill: #dbe6ff; font-size: 14px; font-weight: 600; }
text { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
</style>
"""


def page_parts(page: str) -> tuple[list[str], dict[str, str], list[tuple[str, str]]]:
    """The page's node blocks, its edge paths by id, and its edge labels as (x y attrs, text)."""
    svg = re.search(r"<svg.*?</svg>", page, re.S).group(0)
    nodes = re.findall(r'<g class="node"[^>]*>.*?</g>', svg, re.S)
    edges = {m.group(1): m.group(0) for m in re.finditer(r'<path id="(e_\w+)"[^>]*/>', svg)}
    labels = [(m.group(1), m.group(2)) for m in re.finditer(r'<text class="edge-label"([^>]*)>([^<]+)</text>', svg)]
    return nodes, edges, labels


def node_name(block: str) -> str:
    return re.search(r'<text class="t"[^>]*>([^<]+)</text>', block).group(1)


def built_on_page(page: str) -> set[str]:
    """Names of the nodes the live page marks built."""
    nodes, _, _ = page_parts(page)
    return {node_name(b) for b in nodes if 'data-status="built"' in b.split(">", 1)[0]}


def built_in_phase(phase: int) -> set[str]:
    return {name for name, first in FIRST_BUILT.items() if first <= phase}


def snapshot(phase: int, page: str) -> str:
    """A standalone SVG of the architecture at the close of `phase`."""
    nodes, edges, labels = page_parts(page)
    built = built_in_phase(phase)
    out = []
    for block in nodes:
        name = node_name(block)
        if name in FLOW and name not in built:
            continue
        block = re.sub(r"<title>.*?</title>\s*", "", block, flags=re.S)
        status = "built" if name in built else "planned"
        block = re.sub(r'data-status="\w+"', f'data-status="{status}"', block, count=1)
        if status == "planned" and name in FIRST_BUILT:
            block = re.sub(r'(<text class="s"[^>]*>)[^<]*(</text>)', rf"\g<1>Day {FIRST_BUILT[name]}\g<2>", block)
        out.append(block)
    drawn = []
    for edge_id, (a, b) in EDGES.items():
        if a in built and b in built:
            path = re.sub(r'\s*style="[^"]*"', "", edges[edge_id])
            if edge_id == "e_large_rec":
                path = re.sub(r' d="[^"]*"', ' d="M723,405 C780,405 845,350 845,292"', path)
            drawn.append(path)
    for (a, b), (path, via) in BYPASS.items():
        if a in built and b in built and via not in built:
            drawn.append(f'<path class="edge" d="{path}"/>')
    text = [f'<text class="edge-label"{attrs}>{t}</text>' for attrs, t in labels
            if all(n in built for n in LABELS.get(t, set())) and t != "low confidence or invalid"]
    low = next((attrs for attrs, t in labels if t == "low confidence or invalid"), "")
    text.append(f'<text class="edge-label"{low}>{"low confidence or invalid" if "Validate" in built else "below 0.7"}</text>')
    text.extend(f'<text class="edge-label"{attrs}>{t}</text>' for attrs, t in labels if t == "confident")
    caption = f"Architecture at the close of Day {phase}. Solid is built. Dashed is planned."
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 20 940 450" width="940" height="450" role="img" '
        f'aria-label="{caption}">\n{STYLE}{DEFS}<rect x="0" y="20" width="940" height="450" fill="#0b1220"/>\n'
        f'<text class="caption" x="14" y="44">{caption}</text>\n'
        + "\n".join(drawn) + "\n" + "\n".join(text) + "\n" + "\n".join(out) + "\n</svg>\n"
    )


def main() -> int:
    page = PAGE.read_text()
    OUT.mkdir(parents=True, exist_ok=True)
    for phase in PHASES:
        path = OUT / f"{phase:04d}-architecture.svg"
        path.write_text(snapshot(phase, page))
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

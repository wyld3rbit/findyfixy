"""Draw the architecture as it stood at the close of each phase, from the live page.

Why: an ADR must show the system as it was when the decision was made. The
page code/docs/architecture.html shows only the latest state, and GitHub shows
an .html file as source. This script reads the page's own SVG, keeps the nodes
that existed at a phase, marks the ones built later as planned, and writes a
standalone SVG that GitHub renders inside a Markdown file. One source, so the
picture cannot drift from the page.

Frozen phases. Phases 1 to 4 were drawn from a one-lane page. On Day 5 the page
became two lanes (the harness on top, the app it tests below), and every node
moved. Redrawing an old phase from the new page would change a picture an
accepted ADR already shows, which is the drift this script exists to prevent.
Teaching the script both layouts would keep a second copy of the old geometry
in code, for pictures that must never change again. So those pictures are
frozen instead: the committed files are the source, `FROZEN` holds the SHA-256
of each, and `make adr-snapshot` checks them and leaves them alone. Phase 5 on
is drawn from the page. The root .gitattributes marks the pictures `-text`, so
no checkout converts line endings: the hash and a byte compare see the same
bytes on every machine. Hashing normalised text instead would accept a copy
that is no longer byte-identical to what the ADR was accepted with. If the page's layout changes again, freeze the phases
drawn from the old layout the same way.

The drift guard is a test (tools/test_adr_snapshot.py): the nodes the page marks
built must equal the nodes of the latest phase here. Flip a node on the page
without adding the phase here, and `make check` fails.

Run from the training root: `make adr-snapshot`. Standard library only.
"""

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "code" / "docs" / "architecture.html"
OUT = ROOT / "code" / "docs" / "adr" / "img"

# The phase in which each node first existed. A node absent here is planned.
FIRST_BUILT = {
    "Question": 1, "Small model": 1, "Route": 1, "Large model": 1, "Triage record": 1,
    "Investigate": 2, "Deploy tool": 2, "Validate": 2,
    "Recall": 4,
    "Eval gate": 5, "CI gate": 5,
}
PHASES = range(1, 6)

# Pictures drawn from the one-lane page, kept byte for byte (see the docstring).
FROZEN = {
    1: "87c495615b5f14a3bc67bd4e33907c444fa58057731744e89eb3813d680cfcfb",
    2: "244bb1d66ecb89649da913d5178a8554d80b5a0312fcb5112a2dfe24ecdd1f53",
    3: "105d9c2b0a306330e1f832ea1d158b2999f7881ed4b3061f26b215acb9c2c6bb",
    4: "a387d05a9fbd9564d59fd59e0b69c195ecf0aba4fd7cf7fa940b1f94523d2912",
}

# Planned nodes added to the page after a phase closed. They are drawn only from
# the phase they were added in, so a picture keeps the plan it was drawn with.
ADDED_IN: dict[str, int] = {}

# Each edge in the page: (from node, to node). Edges the page uses only for
# animation (the tool step's return path, the bypass shown when it is off) are
# left out, so they are never drawn.
EDGES = {
    "e_eval_q": ("Eval gate", "Question"), "e_rec_eval": ("Triage record", "Eval gate"),
    "e_q_inv": ("Question", "Investigate"), "e_inv_small": ("Investigate", "Small model"),
    "e_small_val": ("Small model", "Validate"), "e_val_route": ("Validate", "Route"),
    "e_retry": ("Validate", "Small model"), "e_route_rec": ("Route", "Triage record"),
    "e_route_large": ("Route", "Large model"), "e_large_rec": ("Large model", "Triage record"),
    "e_q_recall": ("Question", "Recall"), "e_inv_tool": ("Investigate", "Deploy tool"),
}
LABELS = {  # label text -> the nodes that must exist for it to make sense
    "cases in": {"Eval gate", "Question"},
    "answers scored": {"Eval gate", "Triage record"},
    "retry with the error text (once)": {"Validate"},
    "similar incidents": {"Recall"},
}
# Room above the page's drawing for the caption.
CAPTION_SPACE = 36

DEFS = (
    '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
    '<path d="M0,1 L10,5 L0,9 z" fill="#5b6f99"/></marker>'
    '<marker id="arrow-tool" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
    '<path d="M0,1 L10,5 L0,9 z" fill="#fbbf24"/></marker></defs>\n'
)

STYLE = """\
<style>
.lane { fill: #0d1729; stroke: #2a3a5c; stroke-width: 1; }
.lane.harness { stroke: #c084fc; stroke-opacity: .45; }
.lane-label { fill: #8da2c8; font-size: 12px; font-weight: 600; letter-spacing: .04em; }
.lane-label.harness { fill: #c084fc; }
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


class FrozenImageChanged(Exception):
    """A frozen ADR picture no longer has its recorded hash. Restore it from git."""


def _svg(page: str) -> str:
    """The page's schematic: its one <svg> element."""
    return re.search(r"<svg.*?</svg>", page, re.S).group(0)


def page_parts(page: str) -> tuple[list[str], dict[str, str], list[tuple[str, str]], list[str]]:
    """The page's node blocks, its edge paths by id, its visible edge labels, and its lanes.

    Labels come as (attributes, text). A label hidden by default on the page
    (`opacity:0`, shown only during a run) is left out. Lanes are the lane
    rectangles and their titles, in page order.
    """
    svg = _svg(page)
    nodes = re.findall(r'<g class="node"[^>]*>.*?</g>', svg, re.S)
    edges = {m.group(1): m.group(0) for m in re.finditer(r'<path id="(e_\w+)"[^>]*/>', svg)}
    labels = [(m.group(1), m.group(2)) for m in re.finditer(r'<text class="edge-label"([^>]*)>([^<]+)</text>', svg)
              if "opacity:0" not in m.group(1)]
    lanes = re.findall(r'<rect class="lane[^"]*"[^>]*/>|<text class="lane-label[^"]*"[^>]*>[^<]*</text>', svg)
    return nodes, edges, labels, lanes


def view_box(page: str) -> tuple[float, float, float, float]:
    """The page schematic's viewBox as (x, y, width, height)."""
    x, y, w, h = re.search(r'<svg viewBox="([^"]+)"', _svg(page)).group(1).split()
    return float(x), float(y), float(w), float(h)


def node_name(block: str) -> str:
    """A node block's display name: the text of its title line."""
    return re.search(r'<text class="t"[^>]*>([^<]+)</text>', block).group(1)


def built_on_page(page: str) -> set[str]:
    """Names of the nodes the live page marks built."""
    nodes, _, _, _ = page_parts(page)
    return {node_name(b) for b in nodes if 'data-status="built"' in b.split(">", 1)[0]}


def built_in_phase(phase: int) -> set[str]:
    """Names of the nodes that existed at the close of `phase`."""
    return {name for name, first in FIRST_BUILT.items() if first <= phase}


def _number(value: float) -> str:
    """A coordinate as the page writes it: no trailing `.0`."""
    return f"{value:g}"


def snapshot(phase: int, page: str) -> str:
    """A standalone SVG of the architecture at the close of `phase`, drawn from the page.

    Only for phases after the frozen ones: earlier pictures were drawn from a
    layout the page no longer has.
    """
    if phase in FROZEN:
        raise ValueError(f"phase {phase} is frozen: its picture is the committed file")
    nodes, edges, labels, lanes = page_parts(page)
    built = built_in_phase(phase)
    out = []
    for block in nodes:
        name = node_name(block)
        if phase < ADDED_IN.get(name, 0):
            continue
        block = re.sub(r"<title>.*?</title>\s*", "", block, flags=re.S)
        # The harness colour and the code pointer are page concerns. The lane
        # already shows which nodes are the harness.
        block = re.sub(r' data-(layer|code)="[^"]*"', "", block)
        status = "built" if name in built else "planned"
        block = re.sub(r'data-status="\w+"', f'data-status="{status}"', block, count=1)
        if status == "planned" and name in FIRST_BUILT:
            block = re.sub(r'(<text class="s"[^>]*>)[^<]*(</text>)', rf"\g<1>Day {FIRST_BUILT[name]}\g<2>", block)
        out.append(block)
    drawn = [
        re.sub(r'\s*style="[^"]*"', "", edges[edge_id])
        for edge_id, (a, b) in EDGES.items()
        if a in built and b in built
    ]
    text = [f'<text class="edge-label"{attrs}>{t}</text>' for attrs, t in labels
            if all(n in built for n in LABELS.get(t, set()))]
    x, y, w, h = view_box(page)
    top, height = y - CAPTION_SPACE, h + CAPTION_SPACE
    box = f"{_number(x)} {_number(top)} {_number(w)} {_number(height)}"
    caption = f"Architecture at the close of Day {phase}. Solid is built. Dashed is planned."
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{box}" width="{_number(w)}" height="{_number(height)}" '
        f'role="img" aria-label="{caption}">\n{STYLE}{DEFS}'
        f'<rect x="{_number(x)}" y="{_number(top)}" width="{_number(w)}" height="{_number(height)}" fill="#0b1220"/>\n'
        f'<text class="caption" x="{_number(x + 14)}" y="{_number(top + 24)}">{caption}</text>\n'
        + "\n".join(lanes) + "\n" + "\n".join(drawn) + "\n" + "\n".join(text) + "\n" + "\n".join(out) + "\n</svg>\n"
    )


def image_path(phase: int) -> Path:
    """Where the picture for `phase` lives."""
    return OUT / f"{phase:04d}-architecture.svg"


def check_frozen(phase: int) -> None:
    """Raise `FrozenImageChanged` unless the frozen picture still has its recorded hash."""
    digest = hashlib.sha256(image_path(phase).read_bytes()).hexdigest()
    if digest != FROZEN[phase]:
        raise FrozenImageChanged(f"{image_path(phase).name} changed; restore it with git, it is frozen")


def main() -> int:
    """Check the frozen pictures, then redraw every later phase from the page."""
    page = PAGE.read_text()
    OUT.mkdir(parents=True, exist_ok=True)
    for phase in PHASES:
        path = image_path(phase)
        if phase in FROZEN:
            try:
                check_frozen(phase)
            except FrozenImageChanged as err:
                print(f"error: {err}", file=sys.stderr)
                return 1
            print(f"kept {path.relative_to(ROOT)} (frozen)")
            continue
        path.write_text(snapshot(phase, page))
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

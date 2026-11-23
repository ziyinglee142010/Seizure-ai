"""Render a static SVG preview of a docs/diagrams graph spec (standard library only).

The editable source of truth is the ``.drawio`` file built from the same JSON
spec with ``.agents/skills/drawio-mxgraph/scripts/drawio_tool.py``. This preview
draws the spec's explicit node positions, colours, and edge waypoints so the
documentation has a readable image without the Draw.io desktop app.

Usage::

    python3 docs/diagrams/render_preview.py docs/diagrams/system-overview.json
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

LINE_HEIGHT = 15
FONT = "Helvetica, Arial, sans-serif"


def parse_style(style: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in style.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
        elif part:
            out[part] = "1"
    return out


def label_lines(label: str) -> list[list[tuple[str, bool, bool, str | None]]]:
    """Split a Draw.io HTML label into lines of (text, bold, italic, colour) runs."""

    lines = []
    for raw in re.split(r"<br\s*/?>", label):
        runs: list[tuple[str, bool, bool, str | None]] = []
        bold = italic = False
        colours: list[str | None] = []
        for token in re.split(r"(<[^>]+>)", raw):
            if not token:
                continue
            tag = token.lower()
            if tag == "<b>":
                bold = True
            elif tag == "</b>":
                bold = False
            elif tag == "<i>":
                italic = True
            elif tag == "</i>":
                italic = False
            elif tag.startswith("<font"):
                match = re.search(r"color=['\"]([^'\"]+)['\"]", token)
                colours.append(match.group(1) if match else None)
            elif tag == "</font>":
                if colours:
                    colours.pop()
            elif token.startswith("<"):
                continue
            else:
                runs.append((html.unescape(token), bold, italic, colours[-1] if colours else None))
        lines.append(runs)
    return lines


def text_block(label: str, cx: float, top: float, height: float, left: float | None, size: int) -> str:
    lines = label_lines(label)
    total = len(lines) * LINE_HEIGHT
    y = top + (height - total) / 2 + LINE_HEIGHT - 3 if height else top + LINE_HEIGHT
    anchor = "start" if left is not None else "middle"
    x = left if left is not None else cx
    parts = []
    for runs in lines:
        spans = "".join(
            f'<tspan font-weight="{"bold" if b else "normal"}" font-style="{"italic" if i else "normal"}"'
            f'{f" fill={chr(34)}{c}{chr(34)}" if c else ""}>{html.escape(t)}</tspan>'
            for t, b, i, c in runs
        ) or "<tspan> </tspan>"
        parts.append(
            f'<text x="{x}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="#1f2328">{spans}</text>'
        )
        y += LINE_HEIGHT
    return "\n".join(parts)


def node_svg(node: dict) -> str:
    s = parse_style(node.get("style", ""))
    x, y, w, h = node["x"], node["y"], node["width"], node["height"]
    fill = s.get("fillColor", "#ffffff")
    stroke = s.get("strokeColor", "#000000")
    width = s.get("strokeWidth", "1")
    dash = ' stroke-dasharray="6 4"' if s.get("dashed") == "1" else ""
    size = int(s.get("fontSize", "12"))
    left = x + 12 if s.get("align") == "left" else None
    shape = ""
    if s.get("text") == "1":
        return text_block(node["label"], x + w / 2, y, 0, x, size)
    if s.get("shape") == "cylinder3":
        r = 12
        shape = (
            f'<path d="M{x},{y + r} a{w / 2},{r} 0 0,1 {w},0 v{h - 2 * r} a{w / 2},{r} 0 0,1 {-w},0 z" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash}/>'
            f'<path d="M{x},{y + r} a{w / 2},{r} 0 0,0 {w},0" fill="none" stroke="{stroke}" stroke-width="{width}"/>'
        )
        top, height = y + 2 * r, h - 2 * r
    elif s.get("rhombus") == "1":
        shape = (
            f'<polygon points="{x + w / 2},{y} {x + w},{y + h / 2} {x + w / 2},{y + h} {x},{y + h / 2}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash}/>'
        )
        top, height = y, h
    else:
        rx = 10 if s.get("rounded") == "1" else 0
        shape = (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}"{dash}/>'
        )
        top, height = y, h
    return shape + "\n" + text_block(node["label"], x + w / 2, top, height, left, size)


def _exit(node: dict, toward: tuple[float, float]) -> tuple[float, float]:
    """Point on the node's border on an orthogonal line towards a waypoint."""

    x, y, w, h = node["x"], node["y"], node["width"], node["height"]
    px, py = toward
    if y <= py <= y + h:
        return (x + w if px > x + w / 2 else x, py)
    if x <= px <= x + w:
        return (px, y + h if py > y + h / 2 else y)
    cx, cy = x + w / 2, y + h / 2
    return (x + w if px > cx else x, cy)


def edge_path(edge: dict, nodes: dict) -> list[tuple[float, float]]:
    src, dst = nodes[edge["source"]], nodes[edge["target"]]
    points = [(p["x"], p["y"]) for p in edge.get("points", [])]
    if not points:
        sy0, sy1 = src["y"], src["y"] + src["height"]
        dy0, dy1 = dst["y"], dst["y"] + dst["height"]
        sx0, sx1 = src["x"], src["x"] + src["width"]
        dx0, dx1 = dst["x"], dst["x"] + dst["width"]
        if max(sy0, dy0) < min(sy1, dy1):
            yy = (max(sy0, dy0) + min(sy1, dy1)) / 2
            points = []
            start = _exit(src, (dx0 if dx0 > sx0 else dx1, yy))
            end = _exit(dst, (sx1 if dx0 > sx0 else sx0, yy))
            return [start, end]
        if max(sx0, dx0) < min(sx1, dx1):
            xx = (max(sx0, dx0) + min(sx1, dx1)) / 2
            start = _exit(src, (xx, dy0 if dy0 > sy0 else dy1))
            end = _exit(dst, (xx, sy1 if dy0 > sy0 else sy0))
            return [start, end]
        scx, scy = sx0 + src["width"] / 2, sy0 + src["height"] / 2
        dcx = dx0 + dst["width"] / 2
        points = [(dcx, scy)]
    start = _exit(src, points[0])
    end = _exit(dst, points[-1])
    return [start, *points, end]


def edge_svg(edge: dict, nodes: dict) -> str:
    s = parse_style(edge.get("style", ""))
    stroke = s.get("strokeColor", "#555555")
    width = s.get("strokeWidth", "1.3")
    dash = ' stroke-dasharray="6 4"' if s.get("dashed") == "1" else ""
    path = edge_path(edge, nodes)
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in path)
    marker = {"#82b366": "arrow-green", "#d79b00": "arrow-orange"}.get(stroke, "arrow")
    out = (
        f'<polyline points="{d}" fill="none" stroke="{stroke}" stroke-width="{width}"{dash} '
        f'marker-end="url(#{marker})"/>'
    )
    if edge.get("label"):
        segments = list(zip(path, path[1:]))
        (ax, ay), (bx, by) = max(segments, key=lambda seg: abs(seg[0][0] - seg[1][0]) + abs(seg[0][1] - seg[1][1]))
        if "label_at" in edge:
            mx, my = edge["label_at"]["x"], edge["label_at"]["y"]
        else:
            mx, my = (ax + bx) / 2, (ay + by) / 2
        lines = label_lines(edge["label"])
        longest = max(sum(len(t) for t, *_ in runs) for runs in lines)
        bw, bh = longest * 6.2 + 12, len(lines) * 14 + 6
        out += (
            f'<rect x="{mx - bw / 2:.1f}" y="{my - bh / 2:.1f}" width="{bw:.1f}" height="{bh}" rx="3" '
            f'fill="#ffffff" stroke="#d0d7de"/>'
        )
        y = my - bh / 2 + 14
        for runs in lines:
            text = "".join(html.escape(t) for t, *_ in runs)
            out += (
                f'<text x="{mx:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="11" '
                f'text-anchor="middle" fill="#3d444d">{text}</text>'
            )
            y += 14
    return out


def render(spec_path: Path) -> Path:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in spec["nodes"]}
    width, height = spec.get("page_width", 1600), spec.get("page_height", 900)
    defs = "".join(
        f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{colour}"/></marker>'
        for name, colour in (("arrow", "#555555"), ("arrow-green", "#82b366"), ("arrow-orange", "#d79b00"))
    )
    title = html.escape(spec.get("title", ""))
    body = [
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        *(group_svg(group) for group in spec.get("zones", [])),
        *(edge_svg(edge, nodes) for edge in spec["edges"]),
        *(node_svg(node) for node in spec["nodes"]),
    ]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}">'
        f"<title>{title}</title><defs>{defs}</defs>\n" + "\n".join(body) + "\n</svg>\n"
    )
    output = spec_path.with_suffix(".svg")
    output.write_text(svg, encoding="utf-8")
    return output


def group_svg(zone: dict) -> str:
    x, y, w, h = zone["x"], zone["y"], zone["width"], zone["height"]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{zone.get("fill", "#f6f8fa")}" '
        f'stroke="#d0d7de"/>'
        f'<text x="{x + 14}" y="{y + 22}" font-family="{FONT}" font-size="13" font-weight="bold" '
        f'fill="#57606a">{html.escape(zone["label"])}</text>'
    )


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        print(render(Path(argument)))

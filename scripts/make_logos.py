"""Generate the CaseGraph wordmark as a dot-matrix SVG,
in the same visual language as the 0neHackers logo (blue + green dot matrix).

    python scripts/make_logos.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"

# 5x7 dot-matrix glyphs
G = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "C": [".####", "#....", "#....", "#....", "#....", "#....", ".####"],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "G": [".####", "#....", "#....", "#.###", "#...#", "#...#", ".###."],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["####.", "....#", "....#", ".###.", "....#", "....#", "####."],
    "6": [".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."],
    "·": [".....", ".....", ".....", "..#..", ".....", ".....", "....."],
    " ": [".....", ".....", ".....", ".....", ".....", ".....", "....."],
}
STEP, R = 18, 7.2          # dot pitch and radius (the 0neHackers logo uses ~the same ratio)


def dots(text: str, x0: float, y0: float, fill: str) -> tuple[str, float]:
    out, x = [], x0
    for ch in text:
        for r, row in enumerate(G[ch]):
            for c, v in enumerate(row):
                if v == "#":
                    out.append(f'<circle cx="{x + c * STEP:.1f}" cy="{y0 + r * STEP:.1f}" r="{R}" fill="{fill}"/>')
        x += (3 if ch == " " else 6) * STEP
    return "\n".join(out), x


def casegraph() -> str:
    # icon: a small case graph, a flagged node (red) linked to its neighbourhood
    nodes = [(60, 64, "#0077FF"), (22, 22, "#2ECC40"), (100, 18, "#2ECC40"), (112, 96, "#2ECC40"),
             (18, 110, "#2ECC40"), (64, 124, "#6B6B75")]
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (3, 5)]
    icon = "".join(f'<line x1="{nodes[a][0]}" y1="{nodes[a][1]}" x2="{nodes[b][0]}" y2="{nodes[b][1]}" '
                   f'stroke="#6B6B75" stroke-width="5" stroke-linecap="round"/>' for a, b in edges)
    icon += "".join(f'<circle cx="{x}" cy="{y}" r="{16 if i == 0 else 11}" fill="{f}"/>' for i, (x, y, f) in enumerate(nodes))
    icon += '<circle cx="60" cy="64" r="26" fill="none" stroke="#FF4136" stroke-width="5"/>'
    case, x = dots("CASE", 170, 15, "#0077FF")
    graph, x = dots("GRAPH", x + 8, 15, "url(#g)")
    w = x - STEP + 10
    return (f'<svg width="{w:.0f}" height="142" viewBox="0 0 {w:.0f} 142" fill="none" xmlns="http://www.w3.org/2000/svg">\n'
            '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="0" gradientUnits="objectBoundingBox">'
            '<stop offset="0" stop-color="#2ECC40"/><stop offset="1" stop-color="#0B9A2E"/></linearGradient></defs>\n'
            f'<g>{icon}</g>\n{case}\n{graph}\n</svg>\n')


if __name__ == "__main__":
    (OUT / "casegraph.svg").write_text(casegraph())
    print("wrote", OUT / "casegraph.svg")

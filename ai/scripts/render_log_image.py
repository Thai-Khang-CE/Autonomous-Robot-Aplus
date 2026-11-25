"""Render a captured text log (real command output saved to a file) into a PNG.

The image is NOT a desktop screenshot: a banner on every image names the source log file it was
rendered from, so the raw evidence can always be checked.

    python scripts/render_log_image.py SOURCE.txt OUT.png --title "pytest -v" [--head N] [--tail N]
        [--grep REGEX] [--max-width 120]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from robot_aplus_rl.config import PROJECT_ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--title", required=True)
    ap.add_argument("--head", type=int)
    ap.add_argument("--tail", type=int)
    ap.add_argument("--grep")
    ap.add_argument("--max-width", type=int, default=118)
    args = ap.parse_args()

    src = Path(args.source).resolve()
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    if args.grep:
        lines = [ln for ln in lines if re.search(args.grep, ln)]
    if args.head:
        lines = lines[: args.head]
    if args.tail:
        lines = lines[-args.tail:]
    lines = [ln if len(ln) <= args.max_width else ln[: args.max_width - 3] + "..." for ln in lines]
    try:
        rel = src.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel = str(src)
    sel = []
    if args.grep:
        sel.append(f"lines matching /{args.grep}/")
    if args.head:
        sel.append(f"first {args.head}")
    if args.tail:
        sel.append(f"last {args.tail}")
    banner = (f"Rendered from captured log: {rel} ({total} lines"
              + (f"; showing {', '.join(sel)}" if sel else "") + ") - NOT a desktop screenshot")

    h = 0.75 + 0.128 * (len(lines) + 2)
    fig = plt.figure(figsize=(9.6, h))
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.012, 1 - 0.28 / h, args.title, color="#9cdcfe", family="monospace", fontsize=9.5,
             weight="bold", va="top")
    fig.text(0.012, 1 - 0.50 / h, banner, color="#ce9178", family="monospace", fontsize=6.6, va="top")
    body = "\n".join(lines)
    fig.text(0.012, 1 - 0.80 / h, body, color="#d4d4d4", family="monospace", fontsize=7.2,
             va="top", linespacing=1.32)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", args.output)


if __name__ == "__main__":
    main()

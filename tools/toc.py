"""Generate or refresh the table of contents in a Markdown file.

Every document in this project carries a navigable index at the top. Keeping
those by hand goes stale the first time a section is renamed, so they are
generated: run ``make toc`` after editing any document.

The index is written between ``<!-- toc -->`` and ``<!-- /toc -->`` markers. If
they are absent the block is inserted after the file's front matter — the title
plus any bold metadata lines directly beneath it — so a document only has to be
written, never wired up.

Anchors follow the GitHub/VS Code slug rule: lowercase, drop anything that is
not a letter, digit, space, hyphen or underscore, then spaces to hyphens. That
is why a heading carrying a status mark still links correctly, and why renaming
one silently breaks the link unless this is re-run.

Two things are deliberately not indexed. **Headings inside fenced code blocks**
— a document that quotes another one wholesale would otherwise have the quoted
headings appear as though they were its own sections. And **the index's own
heading**, which would otherwise be picked up on the next run and appended to
itself, once per run, for ever.

A single leading ``#`` is treated as the document title and left out of its own
index. A file that uses ``#`` for every section instead — ``CLAUDE.md`` does —
has them all indexed, because there no one heading is the title.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BEGIN = "<!-- toc -->"
END = "<!-- /toc -->"

HEADING = re.compile(r"^(?P<hashes>#{1,4})\s+(?P<title>.+?)\s*$")
FENCE = re.compile(r"^\s*(?P<ticks>```+|~~~+)")


def slug(title: str) -> str:
    """The anchor GitHub and the VS Code preview both derive from a heading."""
    kept = [c for c in title.strip().lower() if c.isalnum() or c in " -_"]
    return "".join(kept).replace(" ", "-")


def headings(lines: list[str]) -> list[tuple[int, str, str]]:
    """``(level, title, anchor)`` for every heading that belongs in an index."""
    found: list[tuple[int, str, str]] = []
    seen: dict[str, int] = {}
    fence: str | None = None
    in_toc = False

    for line in lines:
        stripped = line.strip()
        if stripped == BEGIN:
            in_toc = True
            continue
        if stripped == END:
            in_toc = False
            continue
        if in_toc:
            continue

        opener = FENCE.match(line)
        if opener is not None:
            ticks = opener.group("ticks")
            if fence is None:
                fence = ticks[0] * 3
            elif stripped.startswith(fence):
                fence = None
            continue
        if fence is not None:
            continue

        match = HEADING.match(line)
        if match is None:
            continue
        title = match.group("title")
        anchor = slug(title)
        # GitHub disambiguates repeats by suffixing -1, -2, ...
        count = seen.get(anchor, 0)
        seen[anchor] = count + 1
        found.append(
            (
                len(match.group("hashes")),
                title,
                anchor if not count else f"{anchor}-{count}",
            )
        )
    return found


def drop_title(found: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    """Leave the document's own title out of its index.

    Only when there is exactly one top-level heading, which is the ordinary
    shape. A file whose every section is an ``h1`` has no title to drop, and
    dropping the first would silently hide a section.
    """
    top = min(level for level, _, _ in found)
    if found[0][0] == top and sum(1 for level, _, _ in found if level == top) == 1:
        return found[1:]
    return found


def render(found: list[tuple[int, str, str]]) -> str:
    if found:
        found = drop_title(found)
    if not found:
        return f"{BEGIN}\n{END}"
    top = min(level for level, _, _ in found)
    rows = [
        f"{'  ' * (level - top)}- [{title}](#{anchor})"
        for level, title, anchor in found
    ]
    return "\n".join([BEGIN, "**Contents**", "", *rows, "", END])


def front_matter_end(lines: list[str]) -> int:
    """Where the index goes when the file has no markers yet.

    Immediately before the first horizontal rule or section heading after the
    title — which is where a reader looks, and which is the only boundary that
    survives front matter written as wrapped prose. Scanning forward over
    "lines that look like metadata" instead would stop at the first
    continuation line and wedge the index into the middle of a sentence.
    """
    start = 1 if lines and lines[0].startswith("# ") else 0
    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        if stripped == "---" or HEADING.match(lines[index]):
            return index
    return len(lines)


def rewrite(text: str) -> str:
    lines = text.splitlines()
    block = render(headings(lines))

    if BEGIN in text and END in text:
        start = text.index(BEGIN)
        stop = text.index(END) + len(END)
        return text[:start] + block + text[stop:]

    cut = front_matter_end(lines)
    return "\n".join([*lines[:cut], block, "", *lines[cut:]]) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh Markdown tables of contents.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any index is stale, without writing",
    )
    args = parser.parse_args(argv)

    stale: list[Path] = []
    for path in args.paths:
        if not path.is_file():
            # Private notes are gitignored; a clone will not have them.
            print(f"skip   {path} (not present)")
            continue

        before = path.read_text(encoding="utf-8")
        after = rewrite(before)
        if before == after:
            print(f"ok     {path}")
            continue
        if args.check:
            stale.append(path)
            print(f"STALE  {path}")
        else:
            path.write_text(after, encoding="utf-8")
            print(f"wrote  {path}")

    if stale:
        print(f"\n{len(stale)} index(es) are stale; run `make toc`", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

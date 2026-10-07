"""Engraving MusicXML into SVG pages with Verovio, in a separate process.

Verovio runs in its own short-lived Python process, never inside the app. Once the window has
started its media player, Qt's FFmpeg looks for hardware video decoding, and on some Linux
systems that loads the graphics driver and the system's LLVM library. LLVM brings its own copy
of parts of the C++ standard library, Verovio ends up calling those instead of its own, and
the whole app was killed ("free(): invalid pointer") the moment it engraved anything. In its
own process Verovio only sees its own libraries, and if it ever does crash, the Sheet music
window shows an error and the app keeps running.

This module does not import Qt, so the engraving process starts quickly.
Run as: python -m audioscribe.engrave   (a JSON job on stdin, the JSON result on stdout)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

TIMEOUT = 180   # seconds; a long song with every part takes a few seconds


def _first_family(families: str) -> str:
    first = families.split(",")[0].strip().strip("'\"")
    return first or "Times"


def svg_for_qt(svg: str) -> str:
    """Make Verovio's SVG drawable by QtSvg (SVG Tiny): replace the nested <svg> with a scaled
    group, and flatten each <text> to one plain run with its real font size."""
    m = re.search(r'<svg[^>]*width="([\d.]+)px"[^>]*height="([\d.]+)px"', svg)
    inner = re.search(r'<svg class="definition-scale"([^>]*)viewBox="0 0 ([\d.]+) ([\d.]+)"([^>]*)>', svg)
    if m and inner:
        w, h = float(m.group(1)), float(m.group(2))
        vw, vh = float(inner.group(2)), float(inner.group(3))
        attrs = (inner.group(1) + inner.group(4)).strip()
        g = f'<g transform="scale({w / vw:.6f},{h / vh:.6f})" {attrs}>'
        svg = svg[:inner.start()] + g + svg[inner.end():]
        last = svg.rfind("</svg>")
        before = svg.rfind("</svg>", 0, last)
        svg = svg[:before] + "</g>" + svg[before + 6:]
    # QtSvg reads a font list like "Times, serif" as one unknown name and falls back to the
    # window's sans font, which is wider, so lyrics ran into each other. Name one font.
    svg = re.sub(r'font-family="([^"]*)"', lambda m_: f'font-family="{_first_family(m_.group(1))}"', svg)
    if "<!DOCTYPE" in svg or "<!ENTITY" in svg:
        # Verovio never writes these. Refuse them so no entity tricks can reach the XML parser.
        return svg
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return svg
    tag = f"{{{SVG_NS}}}"
    parents = {c: p for p in root.iter() for c in p}
    for text in list(root.iter(tag + "text")):
        size = None
        x, y, anchor = text.get("x"), text.get("y"), text.get("text-anchor")
        style_bits = {}
        for el in text.iter():
            fs = el.get("font-size")
            if fs and fs not in ("0px", "0"):
                size = fs
            if x is None and el.get("x"):
                x = el.get("x")
            if y is None and el.get("y"):
                y = el.get("y")
            if anchor is None and el.get("text-anchor"):
                anchor = el.get("text-anchor")
            for k in ("font-style", "font-weight", "font-family"):
                if el.get(k) and k not in style_bits:
                    style_bits[k] = el.get(k)
        parts = []
        for el in text.iter():
            if el.tag == tag + "title":
                continue
            if el.text:
                parts.append(el.text)
            if el is not text and el.tail and parents.get(el) is not None and parents[el].tag != tag + "title":
                parts.append(el.tail)
        content = "".join(parts).strip()
        new = ET.Element(tag + "text")
        if x is not None or y is not None:
            new.set("x", x or "0")
            new.set("y", y or "0")
        if anchor:
            new.set("text-anchor", anchor)
        if size:
            new.set("font-size", size)
        for k, v in style_bits.items():
            new.set(k, _first_family(v) if k == "font-family" else v)
        cls = text.get("class")
        if cls:
            new.set("class", cls)
        new.text = content
        parent = parents.get(text)
        if parent is not None and content:
            idx = list(parent).index(text)
            parent.remove(text)
            parent.insert(idx, new)
        elif parent is not None:
            parent.remove(text)
    return ET.tostring(root, encoding="unicode")


def render_pages_here(xml: str, width: int, height: int, scale: int = 40) -> list[str]:
    """Engrave in this process. Only the engraving process calls this."""
    import verovio

    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": width, "pageHeight": height, "scale": scale, "adjustPageHeight": False,
                   "footer": "none", "header": "auto", "breaks": "auto", "pageMarginLeft": 60,
                   "pageMarginRight": 60, "pageMarginTop": 60, "pageMarginBottom": 60,
                   "lyricSize": 4.0, "spacingSystem": 12})
    if not tk.loadData(xml):
        raise ValueError("Verovio could not read the notation.")
    return [svg_for_qt(tk.renderToSVG(i)) for i in range(1, tk.getPageCount() + 1)]


def render_pages(xml: str, width: int, height: int, scale: int = 40) -> list[str]:
    """Engrave MusicXML into SVG pages (already made Qt friendly) in a separate process."""
    job = json.dumps({"xml": xml, "width": int(width), "height": int(height), "scale": int(scale)})
    package_parent = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = package_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    extra = {"creationflags": 0x08000000} if os.name == "nt" else {}      # CREATE_NO_WINDOW
    try:
        done = subprocess.run([sys.executable, "-m", "audioscribe.engrave"], input=job.encode("utf-8"),  # noqa: S603
                              capture_output=True, timeout=TIMEOUT, env=env, cwd=package_parent, **extra)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Engraving took too long. Try fewer parts or a shorter stretch.") from None
    try:
        answer = json.loads(done.stdout.decode("utf-8"))
    except ValueError:
        detail = done.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"The engraver stopped unexpectedly (code {done.returncode})."
                           + (f" {detail[-1]}" if detail else "")) from None
    if answer.get("error"):
        raise RuntimeError(answer["error"])
    return list(answer.get("pages") or [])


def main() -> int:
    # Verovio's own messages go to the C stdout. Send them to stderr so only the JSON answer
    # reaches the app.
    sys.stdout.flush()
    answer_fd = os.dup(1)
    os.dup2(2, 1)
    try:
        job = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        pages = render_pages_here(job["xml"], job["width"], job["height"], job.get("scale", 40))
        answer = {"pages": pages}
    except Exception as exc:  # reported back to the window
        answer = {"error": str(exc) or type(exc).__name__}
    with os.fdopen(answer_fd, "wb") as out:
        out.write(json.dumps(answer).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

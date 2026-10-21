"""
Convert a single-page PDF logo to SVG

ESA only provides its logo as EPS, PDF and PNG
(https://brand.esa.int/assets/logotype/).
We use SVG for our logos, so we convert the PDF with this script.
The logo itself is not changed, we only:

- convert the vector paths from PDF to SVG
- remove any full-page background fill
  (e.g. the white version of the ESA logo sits on a deep-space blue page)
  so the logo works on our light and dark themes
- crop the viewBox to the logo's content plus a small margin
  (the PDFs include a lot of empty space around the logo,
  which makes the logo tiny when shown next to our other logos)

The ESA logos in `docs/assets/logos` were created with

```sh
uv run --group dev python scripts/convert-pdf-to-svg.py \
    ESA_Logo_DS.pdf docs/assets/logos/esa-light.svg
uv run --group dev python scripts/convert-pdf-to-svg.py \
    ESA_Logo_WT.pdf docs/assets/logos/esa-dark.svg
```
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pymupdf


def is_full_page(rect: pymupdf.Rect, page_rect: pymupdf.Rect) -> bool:
    """
    Check whether a rectangle covers (essentially) the whole page
    """
    return (
        rect.width >= page_rect.width * 0.99 and rect.height >= page_rect.height * 0.99
    )


def main() -> None:
    """
    Convert the PDF to SVG
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("pdf", type=Path, help="PDF to convert")
    parser.add_argument("svg", type=Path, help="Path in which to write the SVG")
    parser.add_argument(
        "--margin",
        type=float,
        default=15.0,
        help="Margin to leave around the logo's content (in PDF points)",
    )
    args = parser.parse_args()

    page = pymupdf.open(args.pdf)[0]
    page_rect = page.rect

    # Bounding box of the logo, ignoring full-page shapes (backgrounds/clip paths)
    content = pymupdf.Rect()
    for drawing in page.get_drawings():
        if not is_full_page(drawing["rect"], page_rect):
            content |= drawing["rect"]

    svg = page.get_svg_image(text_as_path=True)

    # Remove full-page background fills.
    # Clip paths over the full page have no fill so they are left alone.
    width = f"{page_rect.width:g}"
    svg = re.sub(
        rf'<path [^>]*d="M0 0H{re.escape(width)}V[0-9.]+H0Z" fill="[^"]+"/>\n',
        "",
        svg,
    )

    # Crop to the content plus margin
    x0 = round(content.x0 - args.margin)
    y0 = round(content.y0 - args.margin)
    crop_width = round(content.x1 + args.margin) - x0
    crop_height = round(content.y1 + args.margin) - y0
    svg, n_subs = re.subn(
        r'width="[0-9.]+" height="[0-9.]+" viewBox="[0-9. ]+"',
        f'width="{crop_width}" height="{crop_height}" '
        f'viewBox="{x0} {y0} {crop_width} {crop_height}"',
        svg,
        count=1,
    )
    if n_subs != 1:
        msg = "Could not find the SVG's size attributes to crop"
        raise ValueError(msg)

    args.svg.write_text(svg)
    print(f"Wrote {args.svg}")


if __name__ == "__main__":
    main()

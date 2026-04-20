#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def render_pdf_first_page_to_image(pdf_path: Path, dpi: int = 200) -> Image.Image:
    """Render first page of a PDF to a PIL Image (RGB) using pdftoppm."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "page"
        _run(
            [
                "pdftoppm",
                "-png",
                "-singlefile",
                "-r",
                str(dpi),
                str(pdf_path),
                str(prefix),
            ]
        )
        png_path = prefix.with_suffix(".png")
        return Image.open(png_path).convert("RGB")


def stack_vertical(
    images: list[Image.Image], padding: int = 20, bg=(255, 255, 255)
) -> Image.Image:
    widths = [im.width for im in images]
    max_w = max(widths)
    total_h = sum(im.height for im in images) + padding * (len(images) - 1)

    canvas = Image.new("RGB", (max_w, total_h), color=bg)
    y = 0
    for im in images:
        x = (max_w - im.width) // 2
        canvas.paste(im, (x, y))
        y += im.height + padding
    return canvas


def stack_horizontal(
    images: list[Image.Image], padding: int = 20, bg=(255, 255, 255)
) -> Image.Image:
    heights = [im.height for im in images]
    max_h = max(heights)
    total_w = sum(im.width for im in images) + padding * (len(images) - 1)

    canvas = Image.new("RGB", (total_w, max_h), color=bg)
    x = 0
    for im in images:
        y = (max_h - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + padding
    return canvas


def save_as_pdf(image: Image.Image, out_pdf: Path, dpi: int = 200) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_pdf, "PDF", resolution=dpi)


def build_from_panels(
    panel_paths: list[Path], out_pdf: Path, dpi: int = 200, padding: int = 20
) -> None:
    images = [render_pdf_first_page_to_image(p, dpi=dpi) for p in panel_paths]
    combined = stack_vertical(images, padding=padding)
    save_as_pdf(combined, out_pdf, dpi=dpi)


def maybe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 1-4 PDFs (f1.pdf..f4.pdf)."
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument(
        "--prefer_existing",
        action="store_true",
        help="If plot/fX.pdf exists, copy it directly instead of re-assembling.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]

    # Outputs requested by the user (project root)
    out_f1 = root / "f1.pdf"
    out_f2 = root / "f2.pdf"
    out_f3 = root / "f3.pdf"
    out_f4 = root / "f4.pdf"

    plot_dir = root / "plot"

    if args.prefer_existing:
        # Fast path: copy existing PDFs if present.
        for i, out in [(1, out_f1), (2, out_f2), (3, out_f3), (4, out_f4)]:
            src = plot_dir / f"f{i}.pdf"
            if not maybe_copy(src, out):
                raise FileNotFoundError(f"Missing {src}; cannot copy.")
        return

    # Figure 2-4: assemble from panel PDFs in plot/pdfs (single-page PDFs stacked vertically)
    build_from_panels(
        [
            root / "plot/pdfs/f2_1.pdf",
            root / "plot/pdfs/f2_2.pdf",
            root / "plot/pdfs/f2_3.pdf",
        ],
        out_f2,
        dpi=args.dpi,
        padding=args.padding,
    )
    build_from_panels(
        [root / "plot/pdfs/f3_1.pdf", root / "plot/pdfs/f3_2.pdf"],
        out_f3,
        dpi=args.dpi,
        padding=args.padding,
    )
    build_from_panels(
        [root / "plot/pdfs/f4_1.pdf", root / "plot/pdfs/f4_2.pdf"],
        out_f4,
        dpi=args.dpi,
        padding=args.padding,
    )

    # Figure 1: prefer copying the existing final PDF (it was created on macOS, likely via PPT).
    # If it's missing, we fall back to composing available assets.
    if maybe_copy(plot_dir / "f1.pdf", out_f1):
        return

    # Fallback composition for figure 1 (best-effort): stack SCDRL drawio panels and the two PNGs.
    assets_pdf = [
        root / "draw/SCDRL_a.drawio.pdf",
        root / "draw/SCDRL_b.drawio.pdf",
        root / "draw/SCDRL_c.drawio.pdf",
    ]
    draw_imgs = [
        render_pdf_first_page_to_image(p, dpi=args.dpi)
        for p in assets_pdf
        if p.exists()
    ]

    png_paths = [root / "plot/pdfs/f1_2_1.png", root / "plot/pdfs/f1_2_2.png"]
    png_imgs = [Image.open(p).convert("RGB") for p in png_paths if p.exists()]

    if png_imgs:
        top = stack_horizontal(png_imgs, padding=args.padding)
        combined = stack_vertical([top] + draw_imgs, padding=args.padding)
    else:
        combined = stack_vertical(draw_imgs, padding=args.padding)

    save_as_pdf(combined, out_f1, dpi=args.dpi)


if __name__ == "__main__":
    main()

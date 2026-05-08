import fitz
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# pdf filename -> 1-indexed page numbers (from README)
PAGE_SELECTIONS: dict[str, list[int]] = {
    "1-s2.0-S0378775320310466-main.pdf": [9],
    "20120016539.pdf":                    [13],
    "am2c09841.pdf":                      [5],
    "chang2011.pdf":                      [6],
    "LiFePO4_zhao2017.pdf":               [10],
    "LixCoO2_mizushima1980.pdf":          [3, 6],
    "SEI_model_peled1979.pdf":            [2],
    "yazami1983.pdf":                     [6],
    "An_additive_for_lithium_ion_rechargeable_battery_cells_EP2430686B1.pdf": [22, 38],
    "Cathode_materials_for_secondary_(rechargeable)_lithium_batteriesUS6514640.pdf": [7],
    "fast_ion_conductors_US4357215.pdf":  [1, 3],
    "US7993780.pdf":                      [3],
    "US8791449.pdf":                      [6],
    "US9970711.pdf":                      [22],
    "US20210167417A1.pdf":                [16],
    "US20220109187A1.pdf":                [10, 11],
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _find_pdf(filename: str) -> Path | None:
    for subdir in ("papers", "patents"):
        p = DATA_DIR / subdir / filename
        if p.exists():
            return p
    return None


def extract_pages_as_images(dpi: int = 300) -> list[Path]:
    pages_root = DATA_DIR / "pages"
    saved: list[Path] = []

    for filename, page_numbers in PAGE_SELECTIONS.items():
        pdf_path = _find_pdf(filename)
        if pdf_path is None:
            print(f"[WARN] not found: {filename}")
            continue

        doc = fitz.open(pdf_path)
        stem = pdf_path.stem

        for page_num in page_numbers:
            idx = page_num - 1
            if idx < 0 or idx >= len(doc):
                print(f"[WARN] page {page_num} out of range for {filename}")
                continue

            page_dir = pages_root / stem / str(page_num)
            page_dir.mkdir(parents=True, exist_ok=True)

            # single-page pdf
            single = fitz.open()
            single.insert_pdf(doc, from_page=idx, to_page=idx)
            single.save(str(page_dir / f"original_p{page_num}.pdf"))
            single.close()

            # rendered png
            page = doc[idx]
            zoom = dpi / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_path = page_dir / f"page_{page_num}.png"
            pix.save(str(img_path))

            saved.append(img_path)
            print(f"  [{stem}] page {page_num} -> {page_dir}")

        doc.close()

    print(f"\nDone — {len(saved)} page(s)")
    return saved


# stubs — fill in as needed

def extract_tables(pdf_path: str, output_dir: str) -> list[Path]:
    raise NotImplementedError

def extract_figures(pdf_path: str, output_dir: str) -> list[Path]:
    raise NotImplementedError

def segment_regions(image: np.ndarray) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    extract_pages_as_images()

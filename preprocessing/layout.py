import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PAGES_ROOT = DATA_DIR / "pages"


@dataclass
class Region:
    label: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    method: str


TEXT_LABELS = {"text", "plain text", "title", "figure_caption",
               "table_caption", "table_footnote", "formula_caption"}


def _is_text(label: str) -> bool:
    return label in TEXT_LABELS


def _save_regions(img: np.ndarray, regions: list[Region], out_dir: Path):
    fig_dir = out_dir / "figures"
    txt_dir = out_dir / "text"
    fig_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    fig_counts: dict[str, int] = {}
    txt_idx = 0
    meta = []

    for r in regions:
        x1, y1, x2, y2 = r.bbox
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if _is_text(r.label):
            fname = f"text_{txt_idx}.png"
            cv2.imwrite(str(txt_dir / fname), crop)
            txt_idx += 1
        else:
            prefix = r.label.replace(" ", "_")
            idx = fig_counts.get(prefix, 0)
            fig_counts[prefix] = idx + 1
            fname = f"{prefix}_{idx}.png"
            cv2.imwrite(str(fig_dir / fname), crop)

        meta.append({**asdict(r), "file": fname})

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# -- DocLayout-YOLO --

_YOLO_MODEL = None
_YOLO_WEIGHTS = ("juliozhao/DocLayout-YOLO-DocStructBench",
                 "doclayout_yolo_docstructbench_imgsz1024.pt")


def _load_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download(_YOLO_WEIGHTS[0], _YOLO_WEIGHTS[1])
        _YOLO_MODEL = YOLOv10(weights)
    return _YOLO_MODEL


def doclayout_yolo(img: np.ndarray) -> list[Region]:
    model = _load_yolo()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb, imgsz=1024, conf=0.2, verbose=False)

    regions = []
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            if label == "abandon":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            regions.append(Region(
                label=label,
                bbox=(x1, y1, x2, y2),
                confidence=float(box.conf[0]),
                method="doclayout_yolo",
            ))

    return regions


METHODS = {"doclayout_yolo": doclayout_yolo}


def run_all(methods: list[str] | None = None):
    methods = methods or list(METHODS.keys())
    pages = sorted(PAGES_ROOT.glob("*/*/page_*.png"))

    if not pages:
        print("No page PNGs found — run extract first.")
        return

    print(f"Running {methods} on {len(pages)} page(s)\n")

    for png in pages:
        img = cv2.imread(str(png))
        if img is None:
            print(f"[WARN] couldn't read {png}")
            continue

        print(f"  {png.relative_to(PAGES_ROOT)}")
        for name in methods:
            regions = METHODS[name](img)
            _save_regions(img, regions, png.parent / name)
            n_fig = sum(1 for r in regions if not _is_text(r.label))
            n_txt = sum(1 for r in regions if _is_text(r.label))
            print(f"    {name}: {n_fig} non-text, {n_txt} text")


if __name__ == "__main__":
    run_all()

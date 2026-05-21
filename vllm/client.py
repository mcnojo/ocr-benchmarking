import argparse
import base64
import sys
import time
from pathlib import Path

from openai import OpenAI

VLLM_DIR = Path(__file__).resolve().parent
RESULTS_DIR = VLLM_DIR / "results"
INSTANCES_DIR = VLLM_DIR / "aws" / "instances"

CHANDRA_ALLOWED_TAGS = [
    "math", "br", "i", "b", "u", "del", "sup", "sub", "table", "tr", "td",
    "p", "th", "div", "pre", "h1", "h2", "h3", "h4", "h5", "ul", "ol", "li",
    "input", "a", "span", "img", "hr", "tbody", "small", "caption", "strong",
    "thead", "big", "code", "chem",
]
CHANDRA_ALLOWED_ATTRS = [
    "class", "colspan", "rowspan", "display", "checked", "type", "border",
    "value", "style", "href", "alt", "align", "data-bbox", "data-label",
]
CHANDRA_PROMPT_ENDING = f"""
Only use these tags {CHANDRA_ALLOWED_TAGS}, and these attributes {CHANDRA_ALLOWED_ATTRS}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()

CHANDRA_OCR_LAYOUT_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
- Chemical-Block
- Diagram
- Bibliography
- Blank-Page

{CHANDRA_PROMPT_ENDING}
""".strip()

CHANDRA_OCR_PROMPT = f"""
OCR this image to HTML.

{CHANDRA_PROMPT_ENDING}
""".strip()


PROMPTS = {
    "deepseek": {
        "general": "Convert this contents of this image into text logically, including a complete parse of any charts, tables, or figures, along with their data.",
        "chart": (
            "Parse this chart or plot. Extract all axis labels, units, scales, "
            "legend entries, and data series. Output the data as a markdown table "
            "with columns for each variable. Preserve all numerical values and units."
        ),
        "table": "Parse all tables. Extract data as markdown tables preserving headers, units, and all cell values.",
        "chemical": "Extract all chemical formulas, compound names, and SMILES notations.",
    },
    "chandra": {
        "general": CHANDRA_OCR_LAYOUT_PROMPT,
        "chart": CHANDRA_OCR_LAYOUT_PROMPT,
        "table": CHANDRA_OCR_LAYOUT_PROMPT,
        "chemical": CHANDRA_OCR_LAYOUT_PROMPT,
        "ocr": CHANDRA_OCR_PROMPT,
    },
    "dots": {
        "general": "Convert this image to markdown, including all representations of charts, tables, etc along with their data.",
        "chart": (
            "Extract all data from this chart. Include axis labels, units, legend "
            "entries, and tabulate the data series as a markdown table."
        ),
        "table": "Extract all tables as markdown tables preserving structure, headers, and units.",
        "chemical": "Extract all chemical formulas and compound names.",
    },
    "olmocr": {
        "general": "Convert this image to markdown, including all representations of charts, tables, etc along with their data.",
        "chart": (
            "Extract all data from this chart. Include axis labels, units, legend "
            "entries, and tabulate the data series as a markdown table."
        ),
        "table": "Extract all tables as markdown tables preserving structure, headers, and units.",
        "chemical": "Extract all chemical formulas and compound names.",
    },
}

MODELS = {
    "deepseek": {
        "port": 8001,
        "name": "deepseek-ai/DeepSeek-OCR-2",
    },
    "chandra": {
        "port": 8004,
        "name": "datalab-to/chandra-ocr-2",
    },
    "dots": {
        "port": 8002,
        "name": "rednote-hilab/dots.mocr",
    },
    "olmocr": {
        "port": 8003,
        "name": "allenai/olmOCR-2-7B-1025-FP8",
    },
}


def _resolve_host(model_key: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    ip_file = INSTANCES_DIR / f"{model_key}.ip"
    if ip_file.exists():
        return ip_file.read_text().strip()
    print(f"No --host and no tracked instance for '{model_key}'.", file=sys.stderr)
    print(f"  Launch one: ./vllm/aws/launch.sh {model_key}", file=sys.stderr)
    sys.exit(1)


def _encode(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def resolve_prompt(model_key: str, task: str = "general", prompt: str | None = None) -> str:
    if prompt:
        return prompt
    model_prompts = PROMPTS[model_key]
    if task not in model_prompts:
        raise ValueError(f"Unknown task '{task}' for model '{model_key}'. Available: {list(model_prompts)}")
    return model_prompts[task]


def ocr_image(image_path: str, model_key: str, host: str = "localhost",
              task: str = "general", prompt: str | None = None) -> dict:
    cfg = MODELS[model_key]
    client = OpenAI(
        base_url=f"http://{host}:{cfg['port']}/v1",
        api_key="unused",
    )

    b64 = _encode(image_path)
    ext = Path(image_path).suffix.lstrip(".")
    mime = "image/jpeg" if ext == "jpg" else f"image/{ext}"

    resolved_prompt = resolve_prompt(model_key, task, prompt)

    t0 = time.time()
    resp = client.chat.completions.create(
        model=cfg["name"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": resolved_prompt},
            ],
        }],
        max_tokens=2048,
    )

    return {
        "text": resp.choices[0].message.content,
        "model": model_key,
        "task": task,
        "elapsed_s": round(time.time() - t0, 2),
    }


def main():
    p = argparse.ArgumentParser(description="Send an image to a vLLM OCR model")
    p.add_argument("image")
    p.add_argument("--model", choices=list(MODELS), default="deepseek")
    p.add_argument("--task", choices=["general", "chart", "table", "chemical", "ocr"], default="general")
    p.add_argument("--host", default=None)
    p.add_argument("--prompt", default=None, help="Override the task prompt entirely")
    p.add_argument("--all", action="store_true", help="Send to all models")
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args()

    if not Path(args.image).exists():
        print(f"Error: {args.image} not found", file=sys.stderr)
        sys.exit(1)

    targets = list(MODELS) if args.all else [args.model]
    stem = Path(args.image).stem

    for key in targets:
        print(f"\n{'='*60}")
        print(f"Model: {key} ({MODELS[key]['name']})")
        print(f"{'='*60}")

        host = _resolve_host(key, args.host)
        result = ocr_image(args.image, key, host, task=args.task, prompt=args.prompt)
        print(f"Time: {result['elapsed_s']}s\n")
        print(result["text"])

        if args.output:
            out = Path(args.output)
            if args.all:
                out = out.with_stem(f"{out.stem}_{key}")
        else:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out = RESULTS_DIR / f"{stem}_{key}.md"

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result["text"])
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

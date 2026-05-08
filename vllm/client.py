import argparse
import base64
import sys
import time
from pathlib import Path

from openai import OpenAI

VLLM_DIR = Path(__file__).resolve().parent
RESULTS_DIR = VLLM_DIR / "results"
INSTANCES_DIR = VLLM_DIR / "aws" / "instances"

MODELS = {
    "deepseek": {
        "port": 8001,
        "name": "deepseek-ai/DeepSeek-OCR-2",
        "prompt": "Convert this image to markdown.",
    },
    "dots": {
        "port": 8002,
        "name": "rednote-hilab/dots.mocr",
        "prompt": "Convert this image to markdown.",
    },
    "olmocr": {
        "port": 8003,
        "name": "allenai/olmOCR-2-7B-1025-FP8",
        "prompt": "Convert this image to markdown.",
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


def ocr_image(image_path: str, model_key: str, host: str = "localhost",
              prompt: str | None = None) -> dict:
    cfg = MODELS[model_key]
    client = OpenAI(
        base_url=f"http://{host}:{cfg['port']}/v1",
        api_key="unused",
    )

    b64 = _encode(image_path)
    ext = Path(image_path).suffix.lstrip(".")
    mime = "image/jpeg" if ext == "jpg" else f"image/{ext}"

    t0 = time.time()
    resp = client.chat.completions.create(
        model=cfg["name"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt or cfg["prompt"]},
            ],
        }],
        max_tokens=2048,
    )

    return {
        "text": resp.choices[0].message.content,
        "model": model_key,
        "elapsed_s": round(time.time() - t0, 2),
    }


def main():
    p = argparse.ArgumentParser(description="Send an image to a vLLM OCR model")
    p.add_argument("image")
    p.add_argument("--model", choices=list(MODELS), default="deepseek")
    p.add_argument("--host", default=None)
    p.add_argument("--prompt", default=None)
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
        result = ocr_image(args.image, key, host, args.prompt)
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

"""Generate an HTML comparison page: original images vs OCR results side-by-side."""

import argparse
import html
import json
import re
from pathlib import Path

VLLM_DIR = Path(__file__).resolve().parent
RESULTS_DIR = VLLM_DIR / "results"
DATA_DIR = VLLM_DIR.parent / "data" / "pages"

RESULT_PATTERN = re.compile(
    r"^(?P<pdf_stem>.+)_p(?P<page>\d+)_(?P<img>.+)_(?P<model>[^_]+)_(?P<task>[^_]+)\.md$"
)


def parse_result_filename(name: str) -> dict | None:
    m = RESULT_PATTERN.match(name)
    if not m:
        return None
    return m.groupdict()


def find_source_image(pdf_stem: str, page: str, img: str) -> Path | None:
    candidates = [
        DATA_DIR / pdf_stem / page / "doclayout_yolo" / "figures" / f"{img}.png",
        DATA_DIR / pdf_stem / page / "doclayout_yolo" / "figures" / f"{img}.jpg",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def group_results() -> dict:
    """Group result files by (pdf_stem, page, img) so multiple models/tasks appear together."""
    groups = {}
    for f in sorted(RESULTS_DIR.glob("*.md")):
        parsed = parse_result_filename(f.name)
        if not parsed:
            continue
        key = (parsed["pdf_stem"], parsed["page"], parsed["img"])
        groups.setdefault(key, []).append({
            "path": f,
            "model": parsed["model"],
            "task": parsed["task"],
        })
    return groups


def _render_analysis(analysis: list[dict]) -> str:
    """Render <analyze> JSON as description list."""
    parts = []
    for i, subplot in enumerate(analysis):
        title = subplot.get("titles", f"Subplot {i + 1}")
        parts.append(f'<div class="analysis-block"><h4>{html.escape(title)}</h4><dl>')
        for key in ("x_labels", "y_labels", "x_ticks", "y_ticks", "legends", "series"):
            if key in subplot:
                label = key.replace("_", " ").title()
                parts.append(f"<dt>{label}</dt><dd>{html.escape(subplot[key])}</dd>")
        parts.append("</dl></div>")
    return "\n".join(parts)


def _render_data_tables(data: list) -> str:
    """Render JSON data point arrays as HTML tables, grouped by series."""
    if not data:
        return ""
    # flatten nested lists
    points = []
    for item in data:
        if isinstance(item, list):
            points.extend(item)
        elif isinstance(item, dict):
            points.append(item)

    if not points or not isinstance(points[0], dict):
        return ""

    # group by series name
    series_map: dict[str, list[dict]] = {}
    for pt in points:
        name = pt.get("series", "data")
        series_map.setdefault(name, []).append(pt)

    # determine value columns (exclude series/panel metadata)
    skip = {"series", "panel"}
    all_keys = []
    for pt in points:
        for k in pt:
            if k not in skip and k not in all_keys:
                all_keys.append(k)

    parts = []
    for series_name, pts in series_map.items():
        header = "".join(f"<th>{html.escape(k)}</th>" for k in all_keys)
        rows = []
        for pt in pts:
            cells = "".join(f"<td>{html.escape(str(pt.get(k, '')))}</td>" for k in all_keys)
            rows.append(f"<tr>{cells}</tr>")
        parts.append(
            f'<div class="data-series">'
            f'<h4>{html.escape(series_name)}</h4>'
            f'<table><thead><tr>{header}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
            f'</div>'
        )
    return "\n".join(parts)


def render_content(content: str) -> str:
    """Detect output format and render as structured HTML."""
    # check for <analyze> tag + JSON data pattern
    analyze_match = re.search(r"<analyze>\s*(.*?)\s*</analyze>", content, re.DOTALL)
    if analyze_match:
        rendered = ""
        try:
            analysis = json.loads(analyze_match.group(1))
            rendered += _render_analysis(analysis)
        except (json.JSONDecodeError, TypeError):
            rendered += f"<pre>{html.escape(analyze_match.group(1))}</pre>"

        # extract JSON data after </analyze>
        remainder = content[analyze_match.end():].strip()
        if remainder:
            try:
                data = json.loads(remainder)
                rendered += _render_data_tables(data)
            except json.JSONDecodeError:
                rendered += f"<pre>{html.escape(remainder)}</pre>"
        return rendered

    # check if it's chandra-style HTML (has data-bbox or data-label attributes)
    if "data-bbox=" in content or "data-label=" in content:
        return content

    # check for DePlot-style linearized tables (pipe-separated with <0x0A> newlines)
    if " | " in content and ("<0x0A>" in content or "\n" in content):
        rows = content.replace("<0x0A>", "\n").strip().split("\n")
        if len(rows) >= 2:
            table_rows = []
            for i, row in enumerate(rows):
                cells = [c.strip() for c in row.split(" | ")]
                tag = "th" if i == 0 else "td"
                table_rows.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
            return f'<table>{"".join(table_rows)}</table>'

    # fallback: plain text / markdown — show as preformatted
    return f"<pre>{html.escape(content)}</pre>"


def build_html(groups: dict, output_path: Path) -> str:
    output_dir = output_path.parent
    cards = []
    for (pdf_stem, page, img), results in groups.items():
        source = find_source_image(pdf_stem, page, img)
        if not source:
            continue

        img_rel = Path(*source.relative_to(VLLM_DIR.parent).parts)
        # compute relative path from HTML output dir to project root
        try:
            prefix = Path(*[".."] * len(output_dir.relative_to(VLLM_DIR.parent).parts))
        except ValueError:
            prefix = Path("../..")
        img_src = prefix / img_rel

        group_id = f"{pdf_stem}-{page}-{img}".replace(" ", "-")

        tabs_html = []
        panels_html = []
        for i, r in enumerate(results):
            tab_id = f"{group_id}-{r['model']}-{r['task']}"
            label = f"{r['model']} / {r['task']}"
            content = r["path"].read_text()
            checked = "checked" if i == 0 else ""

            tabs_html.append(
                f'<input type="radio" name="tab-{group_id}" '
                f'id="{tab_id}" {checked}>'
                f'<label for="{tab_id}">{html.escape(label)}</label>'
            )
            rendered = render_content(content)
            panels_html.append(
                f'<div class="panel">'
                f'<div class="rendered">{rendered}</div>'
                f'<details><summary>Raw output</summary>'
                f'<pre>{html.escape(content)}</pre>'
                f'</details>'
                f'</div>'
            )

        cards.append(f"""
        <div class="card">
            <h2>{html.escape(pdf_stem)} &mdash; page {html.escape(page)}, {html.escape(img)}</h2>
            <div class="comparison">
                <div class="original">
                    <h3>Original</h3>
                    <img src="{img_src}" alt="original">
                </div>
                <div class="ocr-output">
                    <h3>OCR Output</h3>
                    <div class="tabs">
                        {''.join(tabs_html)}
                        {''.join(panels_html)}
                    </div>
                </div>
            </div>
        </div>
        """)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OCR Comparison</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 2rem; }}
h1 {{ text-align: center; margin-bottom: 2rem; color: #fff; }}
.card {{ background: #16213e; border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem; }}
.card h2 {{ font-size: 1rem; color: #a0a0c0; margin-bottom: 1rem; font-weight: 500; }}
.comparison {{ display: flex; gap: 1.5rem; }}
.original {{ flex: 1; min-width: 0; }}
.original img {{ width: 100%; height: auto; border-radius: 4px; border: 1px solid #333; }}
.ocr-output {{ flex: 1; min-width: 0; }}
h3 {{ font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }}

.tabs {{ position: relative; }}
.tabs input[type="radio"] {{ display: none; }}
.tabs label {{
    display: inline-block; padding: 0.4rem 0.8rem; cursor: pointer;
    background: #0f3460; border-radius: 4px 4px 0 0; margin-right: 2px;
    font-size: 0.8rem; color: #aaa; transition: background 0.2s;
}}
.tabs input[type="radio"]:checked + label {{ background: #1a1a2e; color: #fff; }}
.panel {{ display: none; background: #1a1a2e; border-radius: 0 4px 4px 4px; padding: 1rem; max-height: 600px; overflow: auto; }}

/* rendered OCR content */
.rendered {{ font-size: 0.85rem; line-height: 1.6; color: #d0d0d0; }}
.rendered div[data-label] {{ margin: 0.5rem 0; padding: 0.5rem 0.5rem 0.5rem 0.8rem; border-left: 3px solid #0f3460; display: block; }}
.rendered div[data-label]::before {{
    content: attr(data-label);
    display: inline-block; font-size: 0.6rem; text-transform: uppercase;
    background: #0f3460; color: #aaa; padding: 0.1rem 0.4rem; border-radius: 3px;
    margin-bottom: 0.4rem; letter-spacing: 0.05em;
}}
/* label colors */
.rendered div[data-label="Figure"],
.rendered div[data-label="Image"] {{ border-left-color: #e94560; }}
.rendered div[data-label="Figure"]::before,
.rendered div[data-label="Image"]::before {{ background: #3a1525; color: #e94560; }}
.rendered div[data-label="Diagram"] {{ border-left-color: #f5a623; }}
.rendered div[data-label="Diagram"]::before {{ background: #3a2a10; color: #f5a623; }}
.rendered div[data-label="Text"],
.rendered div[data-label="Caption"],
.rendered div[data-label="Footnote"] {{ border-left-color: #4ecca3; }}
.rendered div[data-label="Text"]::before,
.rendered div[data-label="Caption"]::before,
.rendered div[data-label="Footnote"]::before {{ background: #1a3a2e; color: #4ecca3; }}
.rendered div[data-label="Table"] {{ border-left-color: #7b68ee; }}
.rendered div[data-label="Table"]::before {{ background: #1f1a3e; color: #7b68ee; }}
.rendered div[data-label="Equation-Block"],
.rendered div[data-label="Chemical-Block"] {{ border-left-color: #88bbff; }}
.rendered div[data-label="Equation-Block"]::before,
.rendered div[data-label="Chemical-Block"]::before {{ background: #0d1b36; color: #88bbff; }}
.rendered div[data-label="Section-Header"] {{ border-left-color: #fff; }}
.rendered div[data-label="Section-Header"]::before {{ background: #333; color: #fff; }}
.rendered img[alt] {{ display: block; font-size: 0.75rem; color: #888; font-style: italic;
    padding: 0.3rem; background: #111; border-radius: 3px; margin: 0.3rem 0; }}
.rendered img {{ max-width: 100%; }}
.rendered p {{ margin: 0.3rem 0; }}
.rendered b, .rendered strong {{ color: #fff; }}
.rendered table {{ border-collapse: collapse; margin: 0.5rem 0; width: 100%; }}
.rendered th, .rendered td {{ border: 1px solid #333; padding: 0.3rem 0.6rem; font-size: 0.8rem; text-align: left; }}
.rendered th {{ background: #0f3460; }}
.rendered chem {{ font-family: monospace; color: #f5a623; background: #2a1f0a; padding: 0.1rem 0.3rem; border-radius: 2px; }}
.rendered h1, .rendered h2, .rendered h3, .rendered h4, .rendered h5 {{ color: #fff; margin: 0.5rem 0 0.3rem; }}
.rendered ul, .rendered ol {{ margin: 0.3rem 0 0.3rem 1.5rem; }}
.rendered pre {{ background: #111; padding: 0.5rem; border-radius: 4px; }}
.rendered code {{ background: #111; padding: 0.1rem 0.3rem; border-radius: 2px; font-size: 0.8rem; }}

/* analysis blocks */
.analysis-block {{ margin-bottom: 1rem; padding: 0.8rem; background: #0d1b36; border-radius: 4px; }}
.analysis-block h4 {{ font-size: 0.85rem; color: #88bbff; margin-bottom: 0.5rem; }}
dl {{ display: grid; grid-template-columns: auto 1fr; gap: 0.2rem 0.8rem; font-size: 0.8rem; }}
dt {{ color: #888; font-weight: 500; }}
dd {{ color: #d0d0d0; }}

/* data series tables */
.data-series {{ margin: 0.8rem 0; }}
.data-series h4 {{ font-size: 0.8rem; color: #4ecca3; margin-bottom: 0.3rem; }}

/* raw output toggle */
details {{ margin-top: 0.8rem; }}
summary {{ cursor: pointer; font-size: 0.75rem; color: #666; }}
summary:hover {{ color: #aaa; }}
pre {{ white-space: pre-wrap; word-break: break-word; font-size: 0.75rem; line-height: 1.4;
       color: #888; margin-top: 0.5rem; background: #111; padding: 0.8rem; border-radius: 4px; }}

/* tab visibility */
.tabs input:nth-of-type(1):checked ~ .panel:nth-of-type(1),
.tabs input:nth-of-type(2):checked ~ .panel:nth-of-type(2),
.tabs input:nth-of-type(3):checked ~ .panel:nth-of-type(3),
.tabs input:nth-of-type(4):checked ~ .panel:nth-of-type(4),
.tabs input:nth-of-type(5):checked ~ .panel:nth-of-type(5),
.tabs input:nth-of-type(6):checked ~ .panel:nth-of-type(6),
.tabs input:nth-of-type(7):checked ~ .panel:nth-of-type(7),
.tabs input:nth-of-type(8):checked ~ .panel:nth-of-type(8),
.tabs input:nth-of-type(9):checked ~ .panel:nth-of-type(9),
.tabs input:nth-of-type(10):checked ~ .panel:nth-of-type(10) {{ display: block; }}

@media (max-width: 900px) {{ .comparison {{ flex-direction: column; }} }}
</style>
</head>
<body>
<h1>OCR Model Comparison</h1>
{''.join(cards) if cards else '<p style="text-align:center">No results with matching source images found.</p>'}
<script>
document.addEventListener("DOMContentLoaded", function() {{
    document.querySelectorAll("math").forEach(function(el) {{
        var tex = el.textContent;
        var display = el.getAttribute("display") === "block";
        try {{
            katex.render(tex, el, {{ throwOnError: false, displayMode: display }});
        }} catch(e) {{}}
    }});
}});
</script>
</body>
</html>"""


def main():
    p = argparse.ArgumentParser(description="Generate OCR comparison HTML page")
    p.add_argument("-o", "--output", default=str(RESULTS_DIR / "compare.html"))
    args = p.parse_args()

    output_path = Path(args.output).resolve()
    groups = group_results()
    page = build_html(groups, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)
    print(f"Wrote {output_path} ({len(groups)} image(s), {sum(len(v) for v in groups.values())} result(s))")


if __name__ == "__main__":
    main()

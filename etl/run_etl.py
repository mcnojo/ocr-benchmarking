#!/usr/bin/env python3
"""
CLI entry point for the sodium-ion electrolyte ETL pipeline.

Usage:
    python run_etl.py --pdf /path/to/paper.pdf
    python run_etl.py --pdf-dir /path/to/papers/ --workers 2
    python run_etl.py --pdf paper.pdf --skip-enrichment
"""

import asyncio
import json
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console

console = Console()


def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Stash the config directory so modules can find sibling files
    cfg["_config_dir"] = str(Path(config_path).resolve().parent)
    return cfg


def _flatten(nodes):
    for n in nodes:
        yield n
        yield from _flatten(n.nodes)


def _save_tree(tree, path: Path, config: dict):
    indent = 2 if config["output"]["pretty_print_json"] else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree.model_dump(), f, indent=indent, ensure_ascii=False)


async def process_single_pdf(
    pdf_path: str,
    config: dict,
    skip_enrichment: bool = False,
) -> Path:
    from pipeline.tree_builder import build_tree_async
    from pipeline.asset_extractor import AssetExtractor
    from pipeline.enricher import Enricher, assign_elements_to_tree

    paper_id = Path(pdf_path).stem
    output_dir = Path(config["output"]["kb_root"]) / paper_id
    output_dir.mkdir(parents=True, exist_ok=True)
    tree_path = output_dir / "tree.json"

    console.print(f"[bold cyan]Processing:[/bold cyan] {paper_id}")

    # Step 1: Build structural tree
    console.print("  [yellow]Building document tree...[/yellow]")
    tree = await build_tree_async(pdf_path, config)
    node_count = sum(1 for _ in _flatten(tree.root_nodes))
    console.print(f"  [green]✓[/green] Tree built: {node_count} nodes")

    if skip_enrichment:
        _save_tree(tree, tree_path, config)
        console.print(f"  [green]✓[/green] Saved (no enrichment): {tree_path}")
        return tree_path

    # Step 2: Extract visual assets
    console.print("  [yellow]Extracting visual elements...[/yellow]")
    extractor = AssetExtractor(pdf_path, paper_id, config)
    all_pages = set()
    for node in _flatten(tree.root_nodes):
        for p in range(node.start_index, node.end_index + 1):
            all_pages.add(p)

    page_elements = extractor.extract_all_pages(all_pages)
    total_elements = sum(len(v) for v in page_elements.values())
    console.print(
        f"  [green]✓[/green] Detected {total_elements} visual elements "
        f"across {len(page_elements)} pages"
    )
    extractor.close()

    # Step 3: Enrich with VLM + OCR
    console.print("  [yellow]Running VLM enrichment...[/yellow]")
    enricher = Enricher(config)
    page_elements = await enricher.enrich_all(page_elements, config)
    console.print("  [green]✓[/green] Enrichment complete")

    # Step 4: Assign elements to tree + chemistry extraction
    console.print("  [yellow]Assigning elements to tree nodes...[/yellow]")
    pages_dir = Path(config["output"]["kb_root"]) / paper_id / "assets" / "pages"
    tree = assign_elements_to_tree(tree, page_elements, pdf_path, pages_dir, config)

    # Step 5: Save
    _save_tree(tree, tree_path, config)
    console.print(f"  [bold green]✓ Done:[/bold green] {tree_path}")
    return tree_path


@click.command()
@click.option("--pdf", default=None, help="Path to a single PDF to process.")
@click.option("--pdf-dir", default=None, help="Directory of PDFs to process.")
@click.option(
    "--config", "config_path",
    default="config/pipeline_config.yaml",
    help="Path to pipeline config YAML.",
)
@click.option(
    "--skip-enrichment", is_flag=True, default=False,
    help="Skip VLM/OCR enrichment; build tree structure only.",
)
@click.option(
    "--workers", default=1, show_default=True,
    help="Number of PDFs to process concurrently.",
)
@click.option(
    "-v", "--verbose", is_flag=True, default=False,
    help="Enable debug logging (shows LLM prompts/responses).",
)
def main(pdf, pdf_dir, config_path, skip_enrichment, workers, verbose):
    import logging
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Only show our own debug logs, not every HTTP library's internals
    for noisy in ("httpcore", "httpx", "LiteLLM", "matplotlib", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = load_config(config_path)

    if pdf and pdf_dir:
        console.print("[red]Provide --pdf or --pdf-dir, not both.[/red]")
        sys.exit(1)

    if pdf:
        pdfs = [Path(pdf)]
    elif pdf_dir:
        pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
        console.print(f"Found {len(pdfs)} PDFs in {pdf_dir}")
    else:
        console.print("[red]Provide --pdf or --pdf-dir.[/red]")
        sys.exit(1)

    if not pdfs:
        console.print("[red]No PDFs found.[/red]")
        sys.exit(1)

    async def run_all():
        sem = asyncio.Semaphore(workers)
        async def bounded(p):
            async with sem:
                return await process_single_pdf(str(p), config, skip_enrichment)
        await asyncio.gather(*[bounded(p) for p in pdfs])

    asyncio.run(run_all())


if __name__ == "__main__":
    main()

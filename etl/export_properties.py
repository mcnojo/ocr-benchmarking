#!/usr/bin/env python3
"""Export the typed property table from processed tree.json files to a single CSV.

This is the artifact the science loop consumes: one row = one molecule's property under stated
conditions, with structure (SMILES/InChIKey) and document provenance. Feed it straight into the
HOMO↔experiment calibration (data/prob2.md).

Usage:
    python export_properties.py --kb ./kb --out properties.csv
    python export_properties.py --tree ./kb/<paper_id>/tree.json
    python export_properties.py --kb ./kb --resolved-only      # drop unresolved-structure rows
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console

# allow running from etl/ with `pipeline` importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.property_schema import PropertyRecord, PropertyTable  # noqa: E402

console = Console()


def _load_records(tree_path: Path) -> list[PropertyRecord]:
    data = json.loads(tree_path.read_text(encoding="utf-8"))
    recs = data.get("property_records") or []
    return [PropertyRecord(**r) for r in recs]


@click.command()
@click.option("--kb", default="./kb", help="KB root holding <paper_id>/tree.json dirs.")
@click.option("--tree", "tree_path", default=None, help="A single tree.json to export.")
@click.option("--out", default="properties.csv", help="Output CSV path.")
@click.option("--resolved-only", is_flag=True, default=False,
              help="Keep only rows whose molecule resolved to a structure.")
def main(kb, tree_path, out, resolved_only):
    trees = [Path(tree_path)] if tree_path else sorted(Path(kb).glob("*/tree.json"))
    if not trees:
        console.print(f"[red]No tree.json found under {tree_path or kb}[/red]")
        sys.exit(1)

    all_recs: list[PropertyRecord] = []
    for t in trees:
        recs = _load_records(t)
        all_recs.extend(recs)
        console.print(f"  {t.parent.name}: {len(recs)} records")

    if resolved_only:
        all_recs = [r for r in all_recs if r.molecule.resolved_by != "unresolved"]

    table = PropertyTable(records=all_recs)
    Path(out).write_text(table.to_csv(), encoding="utf-8")

    resolved = sum(1 for r in all_recs if r.molecule.smiles)
    by_prop: dict[str, int] = {}
    for r in all_recs:
        by_prop[r.property] = by_prop.get(r.property, 0) + 1
    console.print(f"\n[bold green]✓[/bold green] {len(all_recs)} records → {out}")
    console.print(f"  structure-resolved: {resolved}/{len(all_recs)}")
    console.print(f"  by property: {by_prop}")


if __name__ == "__main__":
    main()

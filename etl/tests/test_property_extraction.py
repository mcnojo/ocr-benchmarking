#!/usr/bin/env python3
"""Self-checking tests for the typed property extractor — runnable with no GPU and no network.

Run:  python tests/test_property_extraction.py    (from etl/)
Mirrors the PASS/FAIL style of the recipher_* scripts: deterministic, prints a verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import property_extractor as PX
from pipeline import smiles_resolver as SR
from pipeline.property_schema import PropertyTable

results = []
def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


print("=" * 80)
print("TYPED PROPERTY EXTRACTION — molecule↔property↔conditions")
print("=" * 80)
print(f"\nbackends: {SR.backends_available()}")

# A realistic ether oxidative-stability table as chandra would emit it (HTML), plus a caption
# carrying the conditions (reference electrode, technique, temperature).
TABLE_HTML = """
<table>
  <tr><th>Solvent</th><th>Oxidation potential (V vs Na+/Na)</th><th>HOMO (eV)</th></tr>
  <tr><td>1,2-dimethoxyethane</td><td>4.5</td><td>-6.8</td></tr>
  <tr><td>diglyme</td><td>4.3</td><td>-6.6</td></tr>
  <tr><td>tetraethylene glycol dimethyl ether</td><td>4.1</td><td>-6.5</td></tr>
  <tr><td>NaPF6</td><td>—</td><td>—</td></tr>
</table>
"""
CAPTION = ("Table 2. Computed oxidative stability of glyme solvents by LSV, "
           "1 M NaPF6 in DME, measured vs Na+/Na at 25 °C.")

elem_table = {
    "element_type": "table",
    "element_id": "t2_p4",
    "page_index": 4,
    "caption": CAPTION,
    "ocr_text": "",
    "structured_data": TABLE_HTML,
}

recs = PX.extract_from_element(elem_table, paper_id="ethers_demo")
print(f"\n[table] extracted {len(recs)} records")
for r in recs:
    print(f"    {r.molecule.name or r.molecule.formula:42s} {r.property:20s} "
          f"{str(r.value):6s} {str(r.unit):8s} smiles={r.molecule.smiles} "
          f"({r.molecule.resolved_by})")

# ─ assertions ─
props = {(r.molecule.name, r.property): r for r in recs}
check("DME oxidation_potential present", ("1,2-dimethoxyethane", "oxidation_potential") in props)
check("DME HOMO present", ("1,2-dimethoxyethane", "homo") in props)
check("two property columns × three valued rows = 6 records", len(recs) == 6,
      f"got {len(recs)}")

dme_ox = props.get(("1,2-dimethoxyethane", "oxidation_potential"))
if dme_ox:
    check("DME oxidation value parsed = 4.5 V", dme_ox.value == 4.5 and dme_ox.unit == "V",
          f"{dme_ox.value} {dme_ox.unit}")
    check("DME resolved to SMILES COCCOC via OPSIN",
          dme_ox.molecule.smiles == "COCCOC" if SR.backends_available()["opsin"] else True,
          dme_ox.molecule.smiles or "(opsin absent — skipped)")
    check("conditions: reference electrode = Na+/Na",
          dme_ox.conditions.reference_electrode == "Na+/Na", str(dme_ox.conditions.reference_electrode))
    check("conditions: technique = LSV", dme_ox.conditions.technique == "LSV",
          str(dme_ox.conditions.technique))
    check("conditions: temperature = 25 °C", dme_ox.conditions.temperature_c == 25.0,
          str(dme_ox.conditions.temperature_c))
    check("conditions: solvent = DME", dme_ox.conditions.solvent == "DME",
          str(dme_ox.conditions.solvent))
    check("provenance: page 4, element t2_p4",
          dme_ox.provenance.page_index == 4 and dme_ox.provenance.element_id == "t2_p4")

# the salt row has no numeric values → no record; ensure we didn't fabricate one
check("NaPF6 (no value) produced no record",
      not any(r.molecule.formula == "NaPF6" for r in recs))

# unit normalization: an S/cm conductivity should canonicalize to mS/cm with ×1000
COND_TABLE = """<table><tr><th>Electrolyte</th><th>Ionic conductivity (S/cm)</th></tr>
<tr><td>diglyme</td><td>0.012</td></tr></table>"""
crecs = PX.extract_from_element(
    {"element_type": "table", "element_id": "t3", "page_index": 5,
     "caption": "", "ocr_text": "", "structured_data": COND_TABLE}, "ethers_demo")
cond = crecs[0] if crecs else None
check("S/cm → mS/cm normalization (0.012 S/cm = 12 mS/cm)",
      cond is not None and abs(cond.value - 12.0) < 1e-9 and cond.unit == "mS/cm",
      f"{getattr(cond,'value',None)} {getattr(cond,'unit',None)}")

# figure-panel digitization (chandra figure_analysis path)
FIG = {
    "format": "figure_analysis",
    "panels": [{
        "title": "Oxidative stability window",
        "y_label": "Oxidation potential (V vs Na+/Na)",
        "x_label": "Number of -O- units",
        "legend": ["diglyme"],
        "series": [{"points": [{"x": 1, "y": 4.1}, {"x": 2, "y": 4.3}, {"x": 3, "y": 4.5}]}],
    }],
}
frecs = PX.extract_from_element(
    {"element_type": "figure", "element_id": "f1", "page_index": 6,
     "caption": "Figure 1. vs Na+/Na.", "ocr_parsed": FIG}, "ethers_demo")
check("figure panel yields a record (plot digitization)", len(frecs) >= 1,
      f"{len(frecs)} records")
if frecs:
    check("figure record property = oxidation_potential, peak value 4.5",
          frecs[0].property == "oxidation_potential" and frecs[0].value == 4.5,
          f"{frecs[0].property}={frecs[0].value}")

# CSV emit
table = PropertyTable(records=recs + crecs + frecs)
csv_text = table.to_csv()
check("CSV emits header + one row per record",
      csv_text.count("\n") == len(recs + crecs + frecs) + 1)

p = sum(results); n = len(results)
print("\n" + "=" * 80)
print(f"VERDICT: {p}/{n}. Tables/figures/captions → typed molecule↔property↔conditions rows,")
print("  structures resolved offline (OPSIN+RDKit), units normalized, provenance attached.")
print("  This is the calibration-set CSV the HOMO↔experiment loop consumes.")
print("=" * 80)
sys.exit(0 if p == n else 1)

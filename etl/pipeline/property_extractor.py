"""Turn OCR output into typed PropertyRecords — the load-bearing fix.

Three sources, in order of reliability:
  A. HTML tables (chandra layout_html / table structured_data): the richest source. We parse
     the grid, find the molecule column and the property columns (by header keyword + unit),
     and emit one record per (molecule row × property cell).
  B. Figure panels (chandra figure_analysis): axis labels + series → records, the
     plot-digitization path. Reuses chandra's own panel extraction (already in ocr_parsed).
  C. Free text (caption + body OCR): regex fallback "<molecule> … <value><unit>" near a
     property keyword, for values stated in prose rather than tabulated.

Conditions (solvent, reference electrode, temperature, technique) are parsed from the caption
and nearby text and attached to every record from that element. Everything is deterministic;
an LLM verifier can be layered on top later but is not required for the table path.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Optional

from .property_schema import (
    Molecule, Conditions, Provenance, PropertyRecord,
    classify_property, normalize_unit, PROPERTY_DEFAULT_UNIT,
)
from . import smiles_resolver


# ─── value / unit tokens ─────────────────────────────────────────────────────────
_UNIT_TOKEN = (
    r"(?:V\s*vs\.?\s*[A-Za-z+/]+|mV|V|eV|kcal/mol|kJ/mol|mS/cm|μS/cm|uS/cm|S/cm|"
    r"mAh/g|cm²?/s|mPa·s|cP)"
)
_VALUE = r"[-+]?\d+(?:\.\d+)?"
_VALUE_UNIT_RE = re.compile(rf"({_VALUE})\s*({_UNIT_TOKEN})")
_BARE_VALUE_RE = re.compile(rf"^\s*({_VALUE})\s*$")

# molecule-ish header names
_MOLECULE_HEADERS = ("molecule", "solvent", "compound", "electrolyte", "name",
                     "structure", "species", "abbrev", "additive")

# condition cues
_REF_RE = re.compile(r"vs\.?\s*([A-Za-z]+\+?/[A-Za-z]+|Li\+?/Li|Na\+?/Na|SHE|Ag/AgCl)", re.I)
_TEMP_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*([CK])\b")
_TECH_RE = re.compile(r"\b(LSV|CV|EIS|DFT|MD|xTB|ORCA|cyclic voltammetry|"
                      r"linear sweep|impedance)\b", re.I)
_SOLVENT_LEXICON = {
    "dme": "DME", "diglyme": "diglyme", "triglyme": "triglyme", "tetraglyme": "tetraglyme",
    "ec": "EC", "pc": "PC", "dmc": "DMC", "emc": "EMC", "dec": "DEC",
    "thf": "THF", "acetonitrile": "MeCN", "fec": "FEC",
}


# ─── HTML table → matrix ─────────────────────────────────────────────────────────
class _TableMatrix(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: Optional[list[str]] = None
        self._cell: Optional[list[str]] = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(c for c in self._row):
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_html_table(html: str) -> list[list[str]]:
    p = _TableMatrix()
    p.feed(html or "")
    p.close()
    return p.rows


# ─── conditions ──────────────────────────────────────────────────────────────────
def parse_conditions(text: str) -> Conditions:
    c = Conditions()
    if not text:
        return c
    low = text.lower()
    m = _REF_RE.search(text)
    if m:
        c.reference_electrode = m.group(1)
    m = _TEMP_RE.search(text)
    if m:
        val = float(m.group(1))
        c.temperature_c = val - 273.15 if m.group(2).upper() == "K" else val
    m = _TECH_RE.search(text)
    if m:
        c.technique = m.group(1).upper()
    for key, name in _SOLVENT_LEXICON.items():
        if re.search(rf"\b{re.escape(key)}\b", low):
            c.solvent = name
            break
    return c


# ─── helpers ───────────────────────────────────────────────────────────────────
def _split_value_unit(cell: str) -> tuple[Optional[str], Optional[str]]:
    """('4.7 V vs Na') → ('4.7', 'V vs Na'); ('4.7') → ('4.7', None); else (None, None)."""
    if not cell:
        return None, None
    m = _VALUE_UNIT_RE.search(cell)
    if m:
        return m.group(1), m.group(2)
    m = _BARE_VALUE_RE.match(cell)
    if m:
        return m.group(1), None
    return None, None


def _header_property(header: str) -> tuple[Optional[str], Optional[str]]:
    """Header cell → (canonical property, unit token in header if any)."""
    prop = classify_property(header)
    unit_m = re.search(_UNIT_TOKEN, header)
    return prop, (unit_m.group(0) if unit_m else None)


def _make_record(mol: Molecule, prop: str, value_raw: Optional[str], unit_raw: Optional[str],
                 conditions: Conditions, prov: Provenance, confidence: float) -> PropertyRecord:
    unit_canon, factor = normalize_unit(unit_raw or PROPERTY_DEFAULT_UNIT.get(prop))
    value = None
    flags: list[str] = []
    if value_raw is not None:
        try:
            value = float(value_raw) * factor
        except ValueError:
            flags.append("unparsable_value")
    if mol.resolved_by == "unresolved":
        flags.append("unresolved_structure")
    if unit_raw is None:
        flags.append("unit_assumed_from_header")
    return PropertyRecord(
        molecule=mol, property=prop, value=value, unit=unit_canon,
        value_raw=value_raw, unit_raw=unit_raw, conditions=conditions,
        provenance=prov, confidence=confidence, flags=flags,
    )


# ─── A. table extraction ─────────────────────────────────────────────────────────
def extract_from_table(rows: list[list[str]], conditions: Conditions, prov: Provenance,
                       seed_smiles: Optional[dict[str, str]] = None) -> list[PropertyRecord]:
    if len(rows) < 2:
        return []
    header = rows[0]
    ncol = len(header)

    # molecule column: explicit header name, else first column
    mol_col = 0
    for i, h in enumerate(header):
        if any(k in h.lower() for k in _MOLECULE_HEADERS):
            mol_col = i
            break

    # property columns: header maps to a canonical property
    prop_cols: dict[int, tuple[str, Optional[str]]] = {}
    for i, h in enumerate(header):
        if i == mol_col:
            continue
        prop, unit = _header_property(h)
        if prop:
            prop_cols[i] = (prop, unit)

    records: list[PropertyRecord] = []
    for row in rows[1:]:
        if mol_col >= len(row):
            continue
        mol_name = row[mol_col].strip()
        if not mol_name:
            continue
        mol = smiles_resolver.resolve(mol_name, seed_smiles)
        for ci, (prop, header_unit) in prop_cols.items():
            if ci >= len(row):
                continue
            value_raw, cell_unit = _split_value_unit(row[ci])
            if value_raw is None:
                continue
            cell_prov = prov.model_copy(update={"source_text": f"{mol_name} | {header[ci]} = {row[ci]}"})
            records.append(_make_record(
                mol, prop, value_raw, cell_unit or header_unit,
                conditions, cell_prov, confidence=0.8))
    return records


# ─── B. figure panel digitization ────────────────────────────────────────────────
def extract_from_figure(ocr_parsed: dict, conditions: Conditions, prov: Provenance,
                        seed_smiles: Optional[dict[str, str]] = None) -> list[PropertyRecord]:
    """chandra figure_analysis panels → records. The y-axis label gives the property/unit;
    each legend series names a molecule; series points carry the values."""
    if not ocr_parsed or ocr_parsed.get("format") != "figure_analysis":
        return []
    records: list[PropertyRecord] = []
    for panel in ocr_parsed.get("panels", []):
        y_label = _first_str(panel.get("y_label"))
        prop = classify_property(y_label or "") or classify_property(
            _first_str(panel.get("title")) or "")
        if not prop:
            continue
        unit_m = re.search(_UNIT_TOKEN, y_label or "")
        unit_raw = unit_m.group(0) if unit_m else None
        legends = panel.get("legend") or []
        series = panel.get("series") or []
        for idx, s in enumerate(series):
            label = legends[idx] if idx < len(legends) else _first_str(panel.get("title"))
            mol = smiles_resolver.resolve(label or "", seed_smiles)
            pts = _series_points(s)
            if not pts:
                continue
            # report the extremum most relevant to stability screening: max y (e.g. onset)
            top = max(pts, key=lambda xy: xy[1])
            cell_prov = prov.model_copy(update={
                "source_text": f"figure series '{label}' peak y={top[1]} at x={top[0]}"})
            records.append(_make_record(
                mol, prop, str(top[1]), unit_raw, conditions, cell_prov, confidence=0.4))
    return records


def _first_str(v) -> Optional[str]:
    if isinstance(v, list):
        return str(v[0]) if v else None
    return str(v) if v not in (None, "") else None


def _series_points(s) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    seq = s.get("points") if isinstance(s, dict) else s
    if not isinstance(seq, list):
        return pts
    for p in seq:
        try:
            if isinstance(p, dict):
                pts.append((float(p["x"]), float(p["y"])))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
        except (ValueError, KeyError, TypeError):
            continue
    return pts


# ─── C. free-text fallback ──────────────────────────────────────────────────────
def extract_from_text(text: str, conditions: Conditions, prov: Provenance,
                      seed_smiles: Optional[dict[str, str]] = None) -> list[PropertyRecord]:
    if not text:
        return []
    records: list[PropertyRecord] = []
    for m in _VALUE_UNIT_RE.finditer(text):
        window = text[max(0, m.start() - 80):m.start()]
        prop = classify_property(window)
        if not prop:
            continue
        # nearest preceding capitalized/multiword token as the molecule guess
        name_m = re.search(r"([A-Z][A-Za-z0-9,\-']+(?:\s+[A-Za-z0-9,\-']+){0,3})\s*$",
                           window.strip())
        mol = smiles_resolver.resolve(name_m.group(1), seed_smiles) if name_m else Molecule(
            resolved_by="unresolved")
        cell_prov = prov.model_copy(update={
            "source_text": text[max(0, m.start() - 60):m.end() + 5].strip()})
        records.append(_make_record(
            mol, prop, m.group(1), m.group(2), conditions, cell_prov, confidence=0.3))
    return records


# ─── element dispatcher (called from enricher) ───────────────────────────────────
def extract_from_element(elem: dict, paper_id: str, seed_smiles: Optional[dict[str, str]] = None
                         ) -> list[PropertyRecord]:
    """Single entry point: dispatch by element_type, build provenance + conditions, extract."""
    etype = elem.get("element_type")
    caption = elem.get("caption") or ""
    ocr_text = elem.get("ocr_text") or ""
    conditions = parse_conditions(caption + " " + ocr_text)
    prov = Provenance(
        paper_id=paper_id, page_index=elem.get("page_index"),
        element_id=elem.get("element_id"), element_type=etype, caption=caption or None,
    )
    out: list[PropertyRecord] = []
    if etype == "table":
        html = elem.get("structured_data") or ""
        if not html:
            m = re.search(r"<table[\s\S]*?</table>", ocr_text, re.IGNORECASE)
            html = m.group(0) if m else ""
        out += extract_from_table(parse_html_table(html), conditions, prov, seed_smiles)
    elif etype == "figure":
        out += extract_from_figure(elem.get("ocr_parsed") or {}, conditions, prov, seed_smiles)
    # always sweep the caption text as a cheap fallback
    out += extract_from_text(caption, conditions, prov, seed_smiles)
    return out

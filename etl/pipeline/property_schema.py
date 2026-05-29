"""Typed molecule↔property↔conditions records — the structured output that feeds the
HOMO/oxidation-potential calibration loop (see data/prob2.md).

The pipeline already OCRs tables and figures; what the science loop actually consumes is a
*typed* table: each row is one measured/computed property of one molecule, with units
normalized and the experimental conditions and document provenance attached. This module
defines that schema. Extraction lives in property_extractor.py; structure resolution in
smiles_resolver.py.

Pure pydantic + stdlib — no chemistry dependency required to *hold* a record. SMILES/InChIKey
are optional and stay None when no resolver is installed (resolved_by="unresolved").
"""
from __future__ import annotations

import csv
import io
import re
from typing import Optional

from pydantic import BaseModel, Field


# ─── Canonical properties + unit normalization ────────────────────────────────
# Map a raw unit string → (canonical_unit, multiplicative factor to canonical).
UNIT_NORMALIZE: dict[str, tuple[str, float]] = {
    # potentials / energies
    "v": ("V", 1.0),
    "mv": ("V", 1e-3),
    "ev": ("eV", 1.0),
    "kcal/mol": ("kJ/mol", 4.184),
    "kj/mol": ("kJ/mol", 1.0),
    # ionic conductivity → mS/cm
    "s/cm": ("mS/cm", 1e3),
    "ms/cm": ("mS/cm", 1.0),
    "μs/cm": ("mS/cm", 1e-3),
    "us/cm": ("mS/cm", 1e-3),
    # gravimetric capacity
    "mah/g": ("mAh/g", 1.0),
    # transport
    "cm2/s": ("cm2/s", 1.0),
    "cm²/s": ("cm2/s", 1.0),
    "mpa·s": ("mPa·s", 1.0),
    "mpa.s": ("mPa·s", 1.0),
    "cp": ("mPa·s", 1.0),
}

# Header/keyword → canonical property name. Checked as substrings, longest first.
PROPERTY_KEYWORDS: list[tuple[str, str]] = [
    ("oxidative stability", "oxidation_potential"),
    ("oxidation potential", "oxidation_potential"),
    ("anodic stability", "oxidation_potential"),
    ("oxidation onset", "oxidation_potential"),
    ("reduction potential", "reduction_potential"),
    ("cathodic stability", "reduction_potential"),
    ("homo", "homo"),
    ("lumo", "lumo"),
    ("ionization potential", "ionization_potential"),
    ("ionic conductivity", "ionic_conductivity"),
    ("conductivity", "ionic_conductivity"),
    ("solvation energy", "solvation_energy"),
    ("binding energy", "solvation_energy"),
    ("diffusion", "diffusion_coefficient"),
    ("viscosity", "viscosity"),
    ("capacity", "capacity"),
    ("voltage", "voltage"),
    ("potential", "voltage"),
]

# Default canonical unit per property (used when a header gives a property but no unit token).
PROPERTY_DEFAULT_UNIT: dict[str, str] = {
    "oxidation_potential": "V",
    "reduction_potential": "V",
    "voltage": "V",
    "homo": "eV",
    "lumo": "eV",
    "ionization_potential": "eV",
    "ionic_conductivity": "mS/cm",
    "solvation_energy": "kJ/mol",
    "diffusion_coefficient": "cm2/s",
    "viscosity": "mPa·s",
    "capacity": "mAh/g",
}


def classify_property(text: str) -> Optional[str]:
    """Map a free string (column header, caption fragment) to a canonical property name."""
    if not text:
        return None
    low = text.lower()
    for kw, prop in PROPERTY_KEYWORDS:
        if kw in low:
            return prop
    return None


def normalize_unit(raw: Optional[str]) -> tuple[Optional[str], float]:
    """Return (canonical_unit, factor) for a raw unit token; (raw, 1.0) if unknown."""
    if not raw:
        return None, 1.0
    key = raw.strip().lower().replace(" ", "")
    # collapse "v vs na+/na" → "v"
    key = re.split(r"vs", key)[0].strip("/. ") or key
    if key in UNIT_NORMALIZE:
        return UNIT_NORMALIZE[key]
    return raw.strip(), 1.0


# ─── Record model ──────────────────────────────────────────────────────────────

class Molecule(BaseModel):
    name: Optional[str] = None
    formula: Optional[str] = None
    smiles: Optional[str] = None
    inchikey: Optional[str] = None
    resolved_by: str = "unresolved"   # opsin | seed_smiles | rdkit | unresolved

    def key(self) -> str:
        return self.inchikey or self.smiles or self.name or self.formula or "?"


class Conditions(BaseModel):
    solvent: Optional[str] = None
    salt: Optional[str] = None
    concentration: Optional[str] = None
    reference_electrode: Optional[str] = None    # e.g. "Na+/Na"
    temperature_c: Optional[float] = None
    technique: Optional[str] = None              # LSV | CV | DFT | MD | EIS ...
    method: Optional[str] = None                 # e.g. "B3LYP/6-31G* SMD"
    notes: Optional[str] = None


class Provenance(BaseModel):
    paper_id: Optional[str] = None
    page_index: Optional[int] = None
    element_id: Optional[str] = None
    element_type: Optional[str] = None           # table | figure | text
    source_text: Optional[str] = None            # the cell / snippet the value came from
    caption: Optional[str] = None


class PropertyRecord(BaseModel):
    molecule: Molecule
    property: str
    value: Optional[float] = None                # normalized numeric value
    unit: Optional[str] = None                   # canonical unit
    value_raw: Optional[str] = None              # exactly as OCR'd
    unit_raw: Optional[str] = None
    conditions: Conditions = Field(default_factory=Conditions)
    provenance: Provenance = Field(default_factory=Provenance)
    confidence: float = 0.5
    flags: list[str] = Field(default_factory=list)


# ─── Table container + CSV emit ─────────────────────────────────────────────────

CSV_COLUMNS = [
    "paper_id", "molecule_name", "formula", "smiles", "inchikey", "resolved_by",
    "property", "value", "unit", "value_raw", "unit_raw",
    "solvent", "salt", "concentration", "reference_electrode", "temperature_c",
    "technique", "method",
    "page_index", "element_id", "element_type", "confidence", "flags", "source_text",
]


def record_to_row(r: PropertyRecord) -> dict:
    return {
        "paper_id": r.provenance.paper_id,
        "molecule_name": r.molecule.name,
        "formula": r.molecule.formula,
        "smiles": r.molecule.smiles,
        "inchikey": r.molecule.inchikey,
        "resolved_by": r.molecule.resolved_by,
        "property": r.property,
        "value": r.value,
        "unit": r.unit,
        "value_raw": r.value_raw,
        "unit_raw": r.unit_raw,
        "solvent": r.conditions.solvent,
        "salt": r.conditions.salt,
        "concentration": r.conditions.concentration,
        "reference_electrode": r.conditions.reference_electrode,
        "temperature_c": r.conditions.temperature_c,
        "technique": r.conditions.technique,
        "method": r.conditions.method,
        "page_index": r.provenance.page_index,
        "element_id": r.provenance.element_id,
        "element_type": r.provenance.element_type,
        "confidence": r.confidence,
        "flags": "|".join(r.flags) if r.flags else "",
        "source_text": (r.provenance.source_text or "")[:200],
    }


class PropertyTable(BaseModel):
    paper_id: Optional[str] = None
    records: list[PropertyRecord] = Field(default_factory=list)

    def to_rows(self) -> list[dict]:
        return [record_to_row(r) for r in self.records]

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for row in self.to_rows():
            w.writerow(row)
        return buf.getvalue()

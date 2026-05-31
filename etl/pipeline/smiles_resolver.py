"""Resolve a chemical name or formula to a structure (SMILES + InChIKey).

This is the seam the science loop needs: oxidation potentials are tied to *molecules*, and a
calibration set keyed on free-text names cannot be deduplicated or fed to RDKit/xTB. We resolve
names to canonical structure here.

Resolution ladder (each step optional; degrades gracefully):
  1. seed map      — a curated name→SMILES dict (config/chem_smiles.yaml), highest confidence.
  2. OPSIN         — IUPAC / systematic names → SMILES (py2opsin, needs Java). Handles
                     "1,2-dimethoxyethane", "tetraethylene glycol dimethyl ether", etc.
  3. RDKit canon   — canonicalize any SMILES we got + compute InChIKey for dedup.
If nothing resolves, returns a Molecule with smiles=None, resolved_by="unresolved" — the record
is still emitted (name/formula preserved) so coverage is visible, not silently dropped.

For *drawn-structure images* (no text name) you need an OSR model (MolScribe / DECIMER); that is a
separate vision endpoint — see resolve_structure_image() for the seam (not implemented here, by
design: it needs a model server like the OCR ones in vllm/).
"""
from __future__ import annotations

import functools
import re
from typing import Optional

from .property_schema import Molecule

# ─── optional backends ──────────────────────────────────────────────────────────
try:
    from py2opsin import py2opsin as _opsin
    _HAS_OPSIN = True
except Exception:
    _HAS_OPSIN = False

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False


def backends_available() -> dict[str, bool]:
    return {"opsin": _HAS_OPSIN, "rdkit": _HAS_RDKIT}


# A name that is really just an inorganic salt formula (OPSIN won't help, RDKit won't parse a
# bare formula). We keep these as formula-only molecules.
_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*){1,12}$")
_SALT_NAMES = {"napf6", "nabf4", "naclo4", "nafsi", "natfsi", "naotf"}


def _canonicalize(smiles: str) -> tuple[Optional[str], Optional[str]]:
    """SMILES → (canonical_smiles, inchikey) via RDKit; (smiles, None) if RDKit absent/invalid."""
    if not _HAS_RDKIT:
        return smiles, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol)


@functools.lru_cache(maxsize=4096)
def _opsin_smiles(name: str) -> Optional[str]:
    if not _HAS_OPSIN:
        return None
    try:
        out = _opsin(name)
    except Exception:
        return None
    out = (out or "").strip()
    return out or None


def resolve(name_or_formula: str, seed_smiles: Optional[dict[str, str]] = None) -> Molecule:
    """Resolve one identifier to a Molecule. Never raises; returns unresolved on failure."""
    raw = (name_or_formula or "").strip()
    if not raw:
        return Molecule(resolved_by="unresolved")

    low = raw.lower()

    # 1. seed map (curated, exact or case-insensitive)
    if seed_smiles:
        hit = seed_smiles.get(raw) or seed_smiles.get(low)
        if hit:
            smi, ikey = _canonicalize(hit)
            return Molecule(name=raw, smiles=smi or hit, inchikey=ikey, resolved_by="seed_smiles")

    # bare inorganic salt / formula → keep as formula, no structure attempt
    if low in _SALT_NAMES or _FORMULA_RE.match(raw):
        return Molecule(name=raw if not _FORMULA_RE.match(raw) else None,
                        formula=raw, resolved_by="formula_only")

    # 2. OPSIN (IUPAC/systematic name → SMILES)
    smi = _opsin_smiles(raw)
    if smi:
        canon, ikey = _canonicalize(smi)
        return Molecule(name=raw, smiles=canon or smi, inchikey=ikey, resolved_by="opsin")

    # 3. maybe the string already *is* a SMILES (e.g. from a <chem> tag)
    if _HAS_RDKIT and any(c in raw for c in "()=#[]") and " " not in raw:
        canon, ikey = _canonicalize(raw)
        if canon:
            return Molecule(smiles=canon, inchikey=ikey, resolved_by="rdkit")

    return Molecule(name=raw, resolved_by="unresolved")


def resolve_structure_image(image_path: str) -> Molecule:  # pragma: no cover - seam
    """Optical Structure Recognition for *drawn* molecules (no text name available).

    Intentionally a stub: this needs an OSR model server (MolScribe / DECIMER), mirroring the
    OCR vLLM endpoints in vllm/. Wire it the same way as Enricher's vision client and return a
    Molecule(resolved_by="molscribe"). Until then we surface the gap explicitly rather than
    silently producing no structure.
    """
    return Molecule(resolved_by="unresolved", name=None,
                    formula=None, smiles=None, inchikey=None)

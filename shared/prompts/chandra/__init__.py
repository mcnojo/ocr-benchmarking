"""Chandra-OCR-2 prompt package — public API tracks the latest version.

Default usage: `from shared.prompts.chandra import CHANDRA_OCR_LAYOUT_PROMPT`
Pinned usage:  `from shared.prompts.chandra.v1 import CHANDRA_OCR_LAYOUT_PROMPT`

To ship a new version: add `vN.py` alongside the existing ones, then update the
two lines below (the import and `LATEST`). Old version files stay frozen.
"""
from .v1 import (
    CHANDRA_OCR_LAYOUT_PROMPT,
    CHANDRA_OCR_PROMPT,
)

LATEST = "v1"

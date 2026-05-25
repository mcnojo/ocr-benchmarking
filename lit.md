The Critical Ingestion List for Your Knowledge Base
Here's how to structure this for your ETL pipeline. I'd organize your corpus into tiers — Tier 1 is foundational, must be in the knowledge base; Tier 2 is tracking literature, pull regularly; Tier 3 is the live frontier, set up automated ingestion.

Tier 1: Foundational Reviews (Ingest First, Index Densely)
These are the papers your agents will cite back constantly. Full-text ingestion, dense chunk overlap, high retrieval weight.
The Ponrouch / Palacin review lineage — Antoni Ponrouch and Rosa Palacin (ICMAB Barcelona) are the most important research group for electrolyte fundamentals in SIBs. Their papers from 2012–2024 on salt selection, SEI composition, and carbonate vs. ether comparison are the foundational experimental corpus. Search specifically: Ponrouch, Palacin, electrolyte, sodium-ion, Journal of Power Sources / Energy & Environmental Science, 2012–2022.
Eshetu et al. 2020 — "Electrolytes for Sodium-based Rechargeable Batteries: Challenges, Solutions, and the Path Forward," Advanced Energy Materials 10, 2000093. The most cited comprehensive review of the salt and solvent design space. This is your knowledge base's anchor document.
Darjazi et al. 2024 — "Electrolytes for Sodium Ion Batteries: The Current Transition from Liquid to Solid and Hybrid Systems," Advanced Materials, 2024 — covers the full transition from liquid to quasi-solid systems with mechanistic depth. Wiley Online Library
Li et al. 2025 — "Unveiling the Electrolyte and Solid Electrolyte Interphase in Sodium Ion Batteries," Advanced Materials, 2025 — the most current comprehensive SEI review. Ingest this in full. Wiley Online Library
"Reviving Ether-Based Electrolytes for SIBs" — Energy & Environmental Science, April 2025 — covers the full decade of ether electrolyte development from 2021 to 2024 and frames the current state of the art. Essential. RSC Publishing
Cui et al. 2025 — "Fundamentals, Status, and Prospects of Liquid Organic Electrolytes for High-Energy SIBs," Advanced Materials, December 2025. The most recent comprehensive liquid electrolyte review. This is your freshest foundational document.

Tier 2: Mechanistic / Experimental Papers (Structured Ingestion)
These require a schema beyond raw text — your ETL should extract structured fields: salt tested, solvent tested, additive if any, anode material, cathode material, key performance metric, SEI characterization method used, primary finding.
SEI characterization papers: The XPS (X-ray photoelectron spectroscopy) literature on SIB SEI composition is essential because it gives you ground truth on what SEI components form under what conditions. Fondard et al. 2020 (Journal of the Electrochemical Society 167, 070526) is the key reference for NaPF₆/NaTFSI + FEC/DMCF additive SEI composition. This is the paper your agents should check every time they're reasoning about SEI formation.
LHCE papers: The Zhang/Wang group papers on localized high concentration electrolytes from Pacific Northwest National Laboratory (2019–2022) established the paradigm for LHCEs generally. Search: Zhang, localized high concentration electrolyte, sodium, fluorinated ether, TTE. About six key papers define the space.
The hard carbon / additive matching papers: The PTFSI additive paper (Nano Research, 2022) showing that Li-ion film-forming additives like FEC/VC don't transfer directly to SIBs is important because it establishes why this is an original design problem rather than a technology transfer problem. Your agents should know this. Springer
Electrode/Electrolyte Interphases review (PSEI Community, 2023) — cited as the PSE Community paper — provides a systematic comparison of SEI composition across all major salt types in both carbonate and glyme solvents. Dense reference material.

Tier 3: Live Frontier Literature (Automated Pull)
Set up semantic scholar / arXiv monitoring on these query strings:

"sodium-ion" AND "electrolyte" AND ("SEI" OR "solid electrolyte interphase") — pull weekly, flag papers with computational components
"sodium" AND "LHCE" OR "localized high concentration" — monthly sweep
"NaFSI" OR "NaPF6" AND "hard carbon" AND "additive" — flag any new additive papers
"ether electrolyte" AND "sodium" AND "oxidative stability" — the key unsolved problem flag
"sodium ion" AND "DFT" OR "molecular dynamics" AND "electrolyte" — find computational papers in your exact domain to understand what's been simulated and what hasn't

The journals to prioritize in your ingestion pipeline in rough priority order: Nature Energy, Joule, Energy & Environmental Science (RSC), Advanced Materials, Advanced Energy Materials, Journal of the American Chemical Society, ACS Energy Letters, Journal of The Electrochemical Society, Batteries & Supercaps (Wiley).
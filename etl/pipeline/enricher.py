from __future__ import annotations
import asyncio
import base64
from pathlib import Path
from openai import AsyncOpenAI

from .node_schema import TreeNode, DocumentTree, VisualElement, NodeSource
from .chem_extractor import extract_chem_entities, load_seed_entities


# ─── Prompt templates ─────────────────────────────────────────────────────────
# These are intentionally module-level constants so they're easy to find and swap.

VLM_SYSTEM = (
    "You are an expert in electrochemistry and sodium-ion battery research. "
    "You are analyzing a visual element extracted from a scientific paper. "
    "Provide a detailed, precise description. Use correct electrochemical terminology. "
    "Include any chemical formulas exactly as shown."
)

VLM_PROMPTS = {
    "figure": (
        "Describe this figure from a sodium-ion electrolyte research paper. "
        "Include: (1) figure type (Nyquist plot, cycling curves, SEM, XRD, etc.), "
        "(2) axes labels and units, (3) labeled species/compounds/conditions, "
        "(4) key trends visible in the data, (5) chemical formulas or material names shown."
    ),
    "table": (
        "Describe this table from a sodium-ion electrolyte research paper. "
        "Include: (1) what property or comparison is presented, "
        "(2) all column and row headers verbatim, "
        "(3) key data values for ionic conductivity, electrochemical window, or stability, "
        "(4) chemical compounds or material names. "
        "Then reproduce the table in GitHub-flavored markdown."
    ),
    "isolate_formula": (
        "Describe this equation or chemical formula from a sodium-ion research paper. "
        "Provide: (1) plain-text rendering, (2) LaTeX representation if mathematical, "
        "(3) IUPAC name if a chemical structure, (4) contextual meaning if inferable."
    ),
}

OCR_PROMPT = (
    "Extract all text from this image with high fidelity. "
    "Pay special attention to: chemical formulas (e.g. Na3Zr2Si2PO12, NaFSI, NaPF6), "
    "mathematical expressions and subscripts/superscripts, "
    "units (S/cm, mAh/g, V vs Na+/Na), axis labels and tick values, table cell contents. "
    "Return the extracted text verbatim, preserving structure where possible."
)


def _image_to_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class Enricher:
    """Enriches visual elements via VLM descriptions and OCR."""

    def __init__(self, config: dict):
        vllm_cfg = config["vision_server"]
        self.client = AsyncOpenAI(
            base_url=vllm_cfg["base_url"],
            api_key=vllm_cfg["api_key"],
        )
        self.vlm_model = vllm_cfg["vlm_model"]
        self.ocr_model = vllm_cfg["ocr_model"]
        self.semaphore = asyncio.Semaphore(
            config["enrichment"]["max_concurrent_enrichments"]
        )

    async def _vlm_describe(self, image_path: str, element_type: str) -> str:
        b64 = _image_to_b64(image_path)
        prompt = VLM_PROMPTS.get(element_type, VLM_PROMPTS["figure"])
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {"role": "system", "content": VLM_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                max_tokens=1024,
                temperature=0.1,
            )
        return response.choices[0].message.content

    async def _ocr_extract(self, image_path: str) -> str:
        b64 = _image_to_b64(image_path)
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.ocr_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": OCR_PROMPT},
                        ],
                    },
                ],
                max_tokens=2048,
                temperature=0.0,
            )
        return response.choices[0].message.content

    async def enrich_element(self, element: dict, run_vlm: bool, run_ocr: bool) -> dict:
        tasks = []
        keys = []

        if run_vlm and element.get("asset_path"):
            tasks.append(self._vlm_describe(element["asset_path"], element["element_type"]))
            keys.append("vlm_description")

        if run_ocr and element.get("asset_path"):
            tasks.append(self._ocr_extract(element["asset_path"]))
            keys.append("ocr_text")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for key, result in zip(keys, results):
                if isinstance(result, Exception):
                    element[key] = f"ERROR: {result}"
                else:
                    element[key] = result

        return element

    async def enrich_all(
        self, page_elements: dict[int, list[dict]], config: dict
    ) -> dict[int, list[dict]]:
        run_vlm = config["enrichment"]["run_vlm_description"]
        run_ocr = config["enrichment"]["run_ocr"]

        all_tasks = []
        for elements in page_elements.values():
            for elem in elements:
                all_tasks.append(self.enrich_element(elem, run_vlm, run_ocr))

        await asyncio.gather(*all_tasks)
        return page_elements


# ─── Tree enrichment: assign elements to nodes ───────────────────────────────

def flatten_tree(nodes: list[TreeNode]) -> list[TreeNode]:
    result = []
    for node in nodes:
        result.append(node)
        if node.nodes:
            result.extend(flatten_tree(node.nodes))
    return result


def assign_elements_to_tree(
    tree: DocumentTree,
    page_elements: dict[int, list[dict]],
    pdf_path: str,
    pages_dir: Path,
    config: dict,
) -> DocumentTree:
    """Attach visual elements to the deepest tree node whose page range contains them."""
    seed_entities = load_seed_entities(
        str(Path(config.get("_config_dir", "config")) / "chem_entities.yaml")
    )
    flat_nodes = flatten_tree(tree.root_nodes)

    # Build page -> deepest node mapping (later = deeper in DFS order)
    page_to_node: dict[int, TreeNode] = {}
    for node in flat_nodes:
        for p in range(node.start_index, node.end_index + 1):
            page_to_node[p] = node

    # Populate NodeSource
    save_pages = config["output"]["save_page_images"]
    for node in flat_nodes:
        page_image_paths = []
        if save_pages:
            for p in range(node.start_index, node.end_index + 1):
                img_path = pages_dir / f"p{p:04d}.png"
                if img_path.exists():
                    page_image_paths.append(str(img_path.resolve()))
        node.source = NodeSource(
            pdf_path=str(Path(pdf_path).resolve()),
            paper_id=tree.paper_id,
            page_images=page_image_paths,
        )

    # Assign elements
    run_chem = config["enrichment"]["run_chem_entity_extraction"]
    for page_idx, elements in page_elements.items():
        target_node = page_to_node.get(page_idx)
        if target_node is None:
            continue
        for elem_dict in elements:
            if run_chem:
                combined_text = " ".join(filter(None, [
                    elem_dict.get("ocr_text", ""),
                    elem_dict.get("vlm_description", ""),
                    elem_dict.get("caption", ""),
                ]))
                elem_dict["chem_entities"] = extract_chem_entities(combined_text, seed_entities)

            # For tables: extract markdown table from VLM description
            if elem_dict["element_type"] == "table":
                desc = elem_dict.get("vlm_description") or ""
                table_lines = [l for l in desc.split("\n") if "|" in l]
                if table_lines:
                    elem_dict["structured_data"] = "\n".join(table_lines)

            target_node.visual_elements.append(VisualElement(**elem_dict))

    return tree

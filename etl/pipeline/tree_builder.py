"""
Tree builder — ported from VectifyAI/PageIndex (page_index.py + utils.py).

Builds a hierarchical document tree from a PDF by:
1. Extracting text per page via PyMuPDF
2. Detecting/extracting a table of contents (if present)
3. Mapping TOC entries to physical page indices
4. Falling back to LLM-generated structure when no TOC exists
5. Verifying and correcting page assignments via async LLM checks
6. Post-processing into a nested tree with start/end page ranges

LLM calls use the OpenAI client against any OpenAI-compatible endpoint (Ollama, vLLM, MLX, cloud).
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import os
import random
import re
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import tiktoken
import yaml
from openai import AsyncOpenAI, OpenAI

from .node_schema import DocumentTree, TreeNode

# ─── Module-level clients, set once by build_tree_async ───────────────────────
_sync_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None
_model_name: str = ""       # strong model for hard tasks
_model_name_fast: str = ""  # fast model for easy tasks (TOC detection, verification, summaries)
_max_tokens: int = 32768    # context window to request from the server
_ollama_base: str = ""      # Ollama native API base (without /v1), empty if non-Ollama
_active_model: str | None = None  # currently loaded model in VRAM

log = logging.getLogger("tree_builder")


def _init_clients(config: dict):
    """Initialize OpenAI clients from tree_llm config. Called once per run."""
    global _sync_client, _async_client, _model_name, _model_name_fast
    global _max_tokens, _ollama_base, _active_model

    cfg = config["tree_llm"]
    _model_name = cfg["model"]
    _model_name_fast = cfg.get("model_fast") or _model_name
    _max_tokens = cfg.get("max_tokens", 32768)
    base_url = cfg.get("base_url") or "http://localhost:11434/v1"
    api_key = "local"

    api_key_env = cfg.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env, "local")

    _sync_client = OpenAI(base_url=base_url, api_key=api_key)
    _async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    _active_model = None

    # Detect Ollama by its default port — enables model lifecycle management.
    # Non-Ollama backends (vLLM, MLX, cloud) ignore unload calls gracefully.
    if ":11434" in base_url and "/v1" in base_url:
        _ollama_base = base_url.rsplit("/v1", 1)[0]
    else:
        _ollama_base = ""

    log.info("Tree LLM: model=%s, model_fast=%s, base_url=%s, max_tokens=%d",
             _model_name, _model_name_fast, base_url, _max_tokens)


def _ensure_model_exclusive(model_name: str):
    """Unload the previously active model before loading a different one.

    On memory-constrained devices (e.g. M3 Pro 18GB), two models loaded
    simultaneously causes VRAM spill to CPU and inference timeouts.
    This ensures only one model occupies VRAM at a time.
    """
    global _active_model

    if not _ollama_base or model_name == _active_model:
        return

    if _active_model is not None:
        try:
            import httpx
            httpx.post(
                f"{_ollama_base}/api/generate",
                json={"model": _active_model, "keep_alive": 0},
                timeout=10,
            )
            log.info("Unloaded model %s from VRAM", _active_model)
        except Exception as e:
            log.debug("Failed to unload model %s: %s", _active_model, e)

    _active_model = model_name


# ─── LLM call wrappers ───────────────────────────────────────────────────────

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str | None = None) -> int:
    if not text:
        return 0
    return len(_enc.encode(text))


def llm_completion(
    model: str,
    prompt: str,
    chat_history: list | None = None,
    return_finish_reason: bool = False,
) -> str | tuple[str, str]:
    """model arg is kept for call-site compat but the actual model is resolved here."""
    use_model = model if model and model != _model_name else _model_name
    _ensure_model_exclusive(use_model)
    max_retries = 5
    messages = (list(chat_history) + [{"role": "user", "content": prompt}]
                if chat_history else [{"role": "user", "content": prompt}])

    log.debug("llm_completion [%s]: prompt length=%d chars", use_model, len(prompt))

    for i in range(max_retries):
        try:
            response = _sync_client.chat.completions.create(
                model=use_model, messages=messages, temperature=0,
                extra_body={"num_ctx": _max_tokens},
            )
            content = response.choices[0].message.content
            finish = response.choices[0].finish_reason

            log.debug(
                "llm_completion [%s]: response length=%d chars, finish_reason=%s, first 200: %.200s",
                use_model, len(content) if content else 0, finish, content or "(empty)",
            )

            if return_finish_reason:
                reason = ("max_output_reached" if finish == "length" else "finished")
                return content, reason
            return content
        except Exception as e:
            log.error("LLM call failed (attempt %d/%d): %s", i + 1, max_retries, e)
            if i < max_retries - 1:
                time.sleep(2)
            else:
                if return_finish_reason:
                    return "", "error"
                return ""


async def llm_acompletion(model: str, prompt: str) -> str:
    use_model = model if model and model != _model_name else _model_name
    _ensure_model_exclusive(use_model)
    max_retries = 5
    messages = [{"role": "user", "content": prompt}]

    log.debug("llm_acompletion [%s]: prompt length=%d chars", use_model, len(prompt))

    for i in range(max_retries):
        try:
            response = await _async_client.chat.completions.create(
                model=use_model, messages=messages, temperature=0,
                extra_body={"num_ctx": _max_tokens},
            )
            content = response.choices[0].message.content

            log.debug(
                "llm_acompletion [%s]: response length=%d chars, first 200: %.200s",
                use_model, len(content) if content else 0, content or "(empty)",
            )

            return content
        except Exception as e:
            log.error("Async LLM call failed (attempt %d/%d): %s", i + 1, max_retries, e)
            if i < max_retries - 1:
                await asyncio.sleep(2)
            else:
                return ""


# ─── JSON extraction ──────────────────────────────────────────────────────────

def _get_json_content(response: str) -> str:
    start_idx = response.find("```json")
    if start_idx != -1:
        response = response[start_idx + 7:]
    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]
    return response.strip()


def _strip_thinking(content: str) -> str:
    """Remove <think>...</think> blocks that reasoning models (Qwen3.5, DeepSeek) emit."""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def extract_json(content: str) -> dict | list:
    if not content or not content.strip():
        logging.error("extract_json: received empty response from LLM")
        return {}

    # Strip thinking blocks from reasoning models
    content = _strip_thinking(content)

    if not content.strip():
        logging.error("extract_json: response was only thinking tokens, no content")
        return {}

    try:
        # Strip markdown fences
        start_idx = content.find("```json")
        if start_idx != -1:
            json_content = content[start_idx + 7:content.rfind("```")].strip()
        else:
            json_content = content.strip()

        json_content = json_content.replace("None", "null")
        json_content = json_content.replace("\n", " ").replace("\r", " ")
        json_content = " ".join(json_content.split())
        return json.loads(json_content)
    except json.JSONDecodeError:
        # Try fixing trailing commas
        try:
            json_content = json_content.replace(",]", "]").replace(",}", "}")
            return json.loads(json_content)
        except Exception:
            pass

        # Last resort: find the outermost JSON object or array
        try:
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", content)
            if match:
                return json.loads(match.group(1))
        except Exception:
            pass

        logging.error(
            "extract_json: failed to parse. First 500 chars of response:\n%s",
            content[:500],
        )
        return {}
    except Exception as e:
        logging.error("extract_json: unexpected error: %s", e)
        return {}


# ─── PDF text extraction ──────────────────────────────────────────────────────

def get_page_tokens(pdf_path: str, model: str | None = None) -> list[tuple[str, int]]:
    doc = pymupdf.open(pdf_path)
    page_list = []
    for page in doc:
        text = page.get_text()
        tokens = count_tokens(text, model)
        page_list.append((text, tokens))
    doc.close()
    return page_list


def is_likely_scanned(page_list: list[tuple[str, int]], threshold: int = 30) -> bool:
    word_counts = [len(text.split()) for text, _ in page_list]
    if not word_counts:
        return True
    median_words = sorted(word_counts)[len(word_counts) // 2]
    return median_words < threshold


# ─── Utility functions (from PageIndex utils.py) ──────────────────────────────

def convert_physical_index_to_int(data):
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "physical_index" in item:
                val = item["physical_index"]
                if isinstance(val, str):
                    if val.startswith("<physical_index_"):
                        item["physical_index"] = int(val.split("_")[-1].rstrip(">").strip())
                    elif val.startswith("physical_index_"):
                        item["physical_index"] = int(val.split("_")[-1].strip())
    elif isinstance(data, str):
        if data.startswith("<physical_index_"):
            return int(data.split("_")[-1].rstrip(">").strip())
        elif data.startswith("physical_index_"):
            return int(data.split("_")[-1].strip())
        return None
    return data


def convert_page_to_int(data):
    for item in data:
        if "page" in item and isinstance(item["page"], str):
            try:
                item["page"] = int(item["page"])
            except ValueError:
                pass
    return data


def write_node_id(data, node_id=0):
    if isinstance(data, dict):
        data["node_id"] = str(node_id).zfill(4)
        node_id += 1
        if "nodes" in data:
            node_id = write_node_id(data["nodes"], node_id)
    elif isinstance(data, list):
        for item in data:
            node_id = write_node_id(item, node_id)
    return node_id


def list_to_tree(data):
    def get_parent_structure(structure):
        if not structure:
            return None
        parts = str(structure).split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else None

    nodes = {}
    root_nodes = []

    for item in data:
        structure = item.get("structure")
        node = {
            "title": item.get("title"),
            "start_index": item.get("start_index"),
            "end_index": item.get("end_index"),
            "nodes": [],
        }
        nodes[structure] = node
        parent = get_parent_structure(structure)
        if parent and parent in nodes:
            nodes[parent]["nodes"].append(node)
        else:
            root_nodes.append(node)

    def clean_node(node):
        if not node["nodes"]:
            del node["nodes"]
        else:
            for child in node["nodes"]:
                clean_node(child)
        return node

    return [clean_node(n) for n in root_nodes]


def add_preface_if_needed(data):
    if not isinstance(data, list) or not data:
        return data
    if data[0].get("physical_index") is not None and data[0]["physical_index"] > 1:
        data.insert(0, {"structure": "0", "title": "Preface", "physical_index": 1})
    return data


def validate_and_truncate_physical_indices(
    toc, page_list_length, start_index=1, logger=None
):
    if not toc:
        return toc
    max_allowed = page_list_length + start_index - 1
    for item in toc:
        if item.get("physical_index") is not None:
            if item["physical_index"] > max_allowed:
                if logger:
                    logger.info(
                        f"Removed physical_index for '{item.get('title')}' "
                        f"(was {item['physical_index']}, beyond document)"
                    )
                item["physical_index"] = None
    return toc


def page_list_to_group_text(
    page_contents, token_lengths, max_tokens=10000, overlap_page=1
):
    num_tokens = sum(token_lengths)
    if num_tokens <= max_tokens:
        return ["".join(page_contents)]

    expected_parts = math.ceil(num_tokens / max_tokens)
    avg_tokens = math.ceil(((num_tokens / expected_parts) + max_tokens) / 2)

    subsets = []
    current_subset = []
    current_count = 0

    for i, (content, tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_count + tokens > avg_tokens:
            subsets.append("".join(current_subset))
            overlap_start = max(i - overlap_page, 0)
            current_subset = list(page_contents[overlap_start:i])
            current_count = sum(token_lengths[overlap_start:i])
        current_subset.append(content)
        current_count += tokens

    if current_subset:
        subsets.append("".join(current_subset))

    return subsets


def post_processing(structure, end_physical_index):
    for i, item in enumerate(structure):
        item["start_index"] = item.get("physical_index")
        if i < len(structure) - 1:
            nxt = structure[i + 1]
            if nxt.get("appear_start") == "yes":
                item["end_index"] = nxt["physical_index"] - 1
            else:
                item["end_index"] = nxt["physical_index"]
        else:
            item["end_index"] = end_physical_index

    tree = list_to_tree(structure)
    if tree:
        return tree

    for node in structure:
        node.pop("appear_start", None)
        node.pop("physical_index", None)
    return structure


def structure_to_list(structure):
    if isinstance(structure, dict):
        nodes = [structure]
        if "nodes" in structure:
            nodes.extend(structure_to_list(structure["nodes"]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes
    return []


def get_text_of_pdf_pages(pdf_pages, start_page, end_page):
    return "".join(pdf_pages[i][0] for i in range(start_page - 1, end_page))


def add_node_text(node, pdf_pages):
    if isinstance(node, dict):
        node["text"] = get_text_of_pdf_pages(
            pdf_pages, node["start_index"], node["end_index"]
        )
        if "nodes" in node:
            add_node_text(node["nodes"], pdf_pages)
    elif isinstance(node, list):
        for item in node:
            add_node_text(item, pdf_pages)


def remove_structure_text(data):
    if isinstance(data, dict):
        data.pop("text", None)
        if "nodes" in data:
            remove_structure_text(data["nodes"])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)


def remove_page_number(data):
    if isinstance(data, dict):
        data.pop("page_number", None)
        for key in list(data.keys()):
            if "nodes" in key:
                remove_page_number(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_page_number(item)
    return data


def format_structure(structure, order=None):
    if not order:
        return structure
    if isinstance(structure, dict):
        if "nodes" in structure:
            structure["nodes"] = format_structure(structure["nodes"], order)
        if not structure.get("nodes"):
            structure.pop("nodes", None)
        structure = {k: structure[k] for k in order if k in structure}
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


def create_clean_structure_for_description(structure):
    if isinstance(structure, dict):
        clean = {}
        for key in ("title", "node_id", "summary"):
            if key in structure:
                clean[key] = structure[key]
        if "nodes" in structure and structure["nodes"]:
            clean["nodes"] = create_clean_structure_for_description(structure["nodes"])
        return clean
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    return structure


# ─── JSON logger (from PageIndex) ─────────────────────────────────────────────

class JsonLogger:
    def __init__(self, log_dir: str = "./logs"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"tree_builder_{ts}.json"
        os.makedirs(log_dir, exist_ok=True)
        self._dir = log_dir
        self.log_data = []

    def info(self, message):
        self.log_data.append(
            message if isinstance(message, dict) else {"message": message}
        )
        with open(os.path.join(self._dir, self.filename), "w") as f:
            json.dump(self.log_data, f, indent=2)

    def error(self, message):
        self.info(message)


# ─── TOC detection & extraction ───────────────────────────────────────────────

def toc_detector_single_page(content, model=None):
    model = _model_name_fast  # simple yes/no — always use fast model
    prompt = f"""Does this page contain a TABLE OF CONTENTS — a structured listing of
section/chapter titles that serves as a navigation guide for the document?

A real table of contents:
- Lists multiple section or chapter titles in hierarchical or sequential order
- May include page numbers, but not always
- Is explicitly labeled "Contents", "Table of Contents", or similar
- Exists as a dedicated listing, not inline within running text

These are NOT tables of contents — answer "no" for all of these:
- Abstracts or summaries
- Lists of figures, tables, abbreviations, or notation
- Numbered section headings within the body text of a paper
- Reference lists or bibliographies
- Keyword lists
- Author affiliations or acknowledgments
- Journal article headers that list sections inline (e.g. "Introduction ... Methods ... Results")

Most scientific journal articles (typically 4-30 pages) do NOT have a table of contents.
Only answer "yes" if you see a clearly dedicated, structured listing of the document's sections.

Text: {content}

Reply with only this JSON, nothing else:
{{"toc_detected": "yes or no"}}"""

    response = llm_completion(model=model, prompt=prompt)
    return extract_json(response).get("toc_detected", "no")


def find_toc_pages(start_page_index, page_list, opt, logger=None):
    last_page_is_yes = False
    toc_page_list = []
    i = start_page_index

    while i < len(page_list):
        if i >= opt.toc_check_page_num and not last_page_is_yes:
            break
        result = toc_detector_single_page(page_list[i][0], model=opt.model)
        if result == "yes":
            toc_page_list.append(i)
            last_page_is_yes = True
        elif result == "no" and last_page_is_yes:
            break
        i += 1

    return toc_page_list


def detect_page_index(toc_content, model=None):
    model = _model_name_fast  # simple yes/no — always use fast model
    prompt = f"""Does this table of contents contain page numbers or page indices?

Text: {toc_content}

Reply with only this JSON, nothing else:
{{"page_index_given_in_toc": "yes or no"}}"""

    response = llm_completion(model=model, prompt=prompt)
    return extract_json(response).get("page_index_given_in_toc", "no")


def toc_extractor(page_list, toc_page_list, model):
    def transform_dots(text):
        text = re.sub(r"\.{5,}", ": ", text)
        text = re.sub(r"(?:\. ){5,}\.?", ": ", text)
        return text

    toc_content = ""
    for idx in toc_page_list:
        toc_content += page_list[idx][0]
    toc_content = transform_dots(toc_content)
    has_page_index = detect_page_index(toc_content, model=model)
    return {"toc_content": toc_content, "page_index_given_in_toc": has_page_index}


def check_if_toc_transformation_is_complete(content, toc, model=None):
    model = _model_name_fast  # simple yes/no — always use fast model
    prompt = f"""Does the cleaned table of contents contain all sections from the raw table of contents?

Raw Table of contents:
{content}

Cleaned Table of contents:
{toc}

Reply with only this JSON, nothing else:
{{"completed": "yes or no"}}"""

    response = llm_completion(model=model, prompt=prompt)
    return extract_json(response).get("completed", "no")


def extract_toc_content(content, model=None):
    prompt = f"""
    Your job is to extract the full table of contents from the given text, replace ... with :

    Given text: {content}

    Directly return the full table of contents content. Do not output anything else."""

    response, finish_reason = llm_completion(
        model=model, prompt=prompt, return_finish_reason=True
    )
    if_complete = check_if_toc_transformation_is_complete(content, response, model)
    if if_complete == "yes" and finish_reason == "finished":
        return response

    chat_history = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    cont_prompt = "please continue the generation of table of contents, directly output the remaining part of the structure"

    for _ in range(5):
        new_response, finish_reason = llm_completion(
            model=model, prompt=cont_prompt,
            chat_history=chat_history, return_finish_reason=True,
        )
        response += new_response
        if_complete = check_if_toc_transformation_is_complete(content, response, model)
        if if_complete == "yes" and finish_reason == "finished":
            return response
        chat_history = [
            {"role": "user", "content": cont_prompt},
            {"role": "assistant", "content": response},
        ]

    raise RuntimeError("Failed to complete table of contents after maximum retries")


def _extract_toc_list(parsed: dict | list) -> list | None:
    """Pull the TOC list from whatever shape the LLM returned.

    Expected: {"table_of_contents": [...]}, but models sometimes return
    just the list directly, or use variant keys.
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("table_of_contents", "toc", "contents"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        # Single-key dict wrapping a list
        if len(parsed) == 1:
            val = next(iter(parsed.values()))
            if isinstance(val, list):
                return val
    return None


def toc_transformer(toc_content, model=None):
    init_prompt = """
    You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

    structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format:
    {
    table_of_contents: [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "page": <page number or None>,
        },
        ...
        ],
    }
    You should transform the full table of contents in one go.
    Directly return the final JSON structure, do not output anything else. """

    prompt = init_prompt + "\n Given table of contents\n:" + toc_content
    last_complete, finish_reason = llm_completion(
        model=model, prompt=prompt, return_finish_reason=True
    )
    if_complete = check_if_toc_transformation_is_complete(toc_content, last_complete, model)
    if if_complete == "yes" and finish_reason == "finished":
        parsed = extract_json(last_complete)
        toc_list = _extract_toc_list(parsed)
        if toc_list is not None:
            return convert_page_to_int(toc_list)

    last_complete = _get_json_content(last_complete)

    for _ in range(5):
        position = last_complete.rfind("}")
        if position != -1:
            last_complete = last_complete[: position + 2]

        cont_prompt = f"""
        Your task is to continue the table of contents json structure, directly output the remaining part of the json structure.

        The raw table of contents json structure is:
        {toc_content}

        The incomplete transformed table of contents json structure is:
        {last_complete}

        Please continue the json structure, directly output the remaining part of the json structure."""

        new_complete, finish_reason = llm_completion(
            model=model, prompt=cont_prompt, return_finish_reason=True
        )
        if new_complete.startswith("```json"):
            new_complete = _get_json_content(new_complete)
            last_complete += new_complete

        if_complete = check_if_toc_transformation_is_complete(
            toc_content, last_complete, model
        )
        if if_complete == "yes" and finish_reason == "finished":
            break

    parsed = extract_json(last_complete)
    toc_list = _extract_toc_list(parsed)
    if toc_list is None:
        log.error("toc_transformer: could not extract TOC list from LLM response")
        return []
    return convert_page_to_int(toc_list)


def toc_index_extractor(toc, content, model=None):
    prompt = """
    You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents.

    The response should be in the following JSON format:
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "physical_index": "<physical_index_X>" (keep the format)
        },
        ...
    ]

    Only add the physical_index to the sections that are in the provided pages.
    If the section is not in the provided pages, do not add the physical_index to it.
    Directly return the final JSON structure. Do not output anything else."""

    prompt += "\nTable of contents:\n" + str(toc) + "\nDocument pages:\n" + content
    response = llm_completion(model=model, prompt=prompt)
    return extract_json(response)


# ─── Page number assignment ───────────────────────────────────────────────────

def add_page_number_to_toc(part, structure, model=None):
    prompt = """
    You are given an JSON structure of a document and a partial part of the document. Your task is to check if the title that is described in the structure is started in the partial given document.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    If the full target section starts in the partial given document, insert the given JSON structure with the "start": "yes", and "start_index": "<physical_index_X>".

    If the full target section does not start in the partial given document, insert "start": "no",  "start_index": None.

    The response should be in the following format.
        [
            {
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "start": "<yes or no>",
                "physical_index": "<physical_index_X> (keep the format)" or None
            },
            ...
        ]
    The given structure contains the result of the previous part, you need to fill the result of the current part, do not change the previous result.
    Directly return the final JSON structure. Do not output anything else."""

    prompt += f"\n\nCurrent Partial Document:\n{part}\n\nGiven Structure\n{json.dumps(structure, indent=2)}\n"
    response = llm_completion(model=model, prompt=prompt)
    result = extract_json(response)

    for item in result:
        item.pop("start", None)
    return result


# ─── No-TOC tree generation ──────────────────────────────────────────────────

def generate_toc_init(part, model=None):
    prompt = """
    You are an expert in extracting hierarchical tree structure, your task is to generate the tree structure of the document.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X.

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format.
        [
            {{
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            }},

        ],

    Directly return the final JSON structure. Do not output anything else."""

    prompt += "\nGiven text\n:" + part
    response, finish_reason = llm_completion(
        model=model, prompt=prompt, return_finish_reason=True
    )
    if finish_reason == "finished":
        return extract_json(response)
    raise RuntimeError(f"generate_toc_init: finish_reason={finish_reason}")


def generate_toc_continue(toc_content, part, model=None):
    prompt = """
    You are an expert in extracting hierarchical tree structure.
    You are given a tree structure of the previous part and the text of the current part.
    Your task is to continue the tree structure from the previous part to include the current part.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X.

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format.
        [
            {
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            },
            ...
        ]

    Directly return the additional part of the final JSON structure. Do not output anything else."""

    prompt += "\nGiven text\n:" + part + "\nPrevious tree structure\n:" + json.dumps(toc_content, indent=2)
    response, finish_reason = llm_completion(
        model=model, prompt=prompt, return_finish_reason=True
    )
    if finish_reason == "finished":
        return extract_json(response)
    raise RuntimeError(f"generate_toc_continue: finish_reason={finish_reason}")


# ─── TOC processing paths ────────────────────────────────────────────────────

def process_no_toc(page_list, start_index=1, model=None, logger=None):
    page_contents = []
    token_lengths = []
    for page_index in range(start_index, start_index + len(page_list)):
        text = (
            f"<physical_index_{page_index}>\n"
            f"{page_list[page_index - start_index][0]}\n"
            f"<physical_index_{page_index}>\n\n"
        )
        page_contents.append(text)
        token_lengths.append(count_tokens(text, model))

    group_texts = page_list_to_group_text(page_contents, token_lengths)
    if logger:
        logger.info(f"process_no_toc: {len(group_texts)} group(s)")

    toc = generate_toc_init(group_texts[0], model)
    for group_text in group_texts[1:]:
        additional = generate_toc_continue(toc, group_text, model)
        toc.extend(additional)

    if logger:
        logger.info(f"generate_toc result: {toc}")

    return convert_physical_index_to_int(toc)


def process_toc_no_page_numbers(
    toc_content, toc_page_list, page_list, start_index=1, model=None, logger=None
):
    toc_structured = toc_transformer(toc_content, model)
    if logger:
        logger.info(f"toc_transformer: {toc_structured}")

    page_contents = []
    token_lengths = []
    for page_index in range(start_index, start_index + len(page_list)):
        text = (
            f"<physical_index_{page_index}>\n"
            f"{page_list[page_index - start_index][0]}\n"
            f"<physical_index_{page_index}>\n\n"
        )
        page_contents.append(text)
        token_lengths.append(count_tokens(text, model))

    group_texts = page_list_to_group_text(page_contents, token_lengths)
    if logger:
        logger.info(f"process_toc_no_page_numbers: {len(group_texts)} group(s)")

    toc_with_pages = copy.deepcopy(toc_structured)
    for group_text in group_texts:
        toc_with_pages = add_page_number_to_toc(group_text, toc_with_pages, model)

    if logger:
        logger.info(f"add_page_number_to_toc: {toc_with_pages}")

    return convert_physical_index_to_int(toc_with_pages)


def process_none_page_numbers(toc_items, page_list, start_index=1, model=None):
    """Fill in missing physical_index values by searching between known anchors."""
    for i, item in enumerate(toc_items):
        if "physical_index" not in item:
            prev = 0
            for j in range(i - 1, -1, -1):
                if toc_items[j].get("physical_index") is not None:
                    prev = toc_items[j]["physical_index"]
                    break

            nxt = -1
            for j in range(i + 1, len(toc_items)):
                if toc_items[j].get("physical_index") is not None:
                    nxt = toc_items[j]["physical_index"]
                    break

            page_contents = []
            for page_index in range(prev, nxt + 1):
                list_index = page_index - start_index
                if 0 <= list_index < len(page_list):
                    text = (
                        f"<physical_index_{page_index}>\n"
                        f"{page_list[list_index][0]}\n"
                        f"<physical_index_{page_index}>\n\n"
                    )
                    page_contents.append(text)

            item_copy = copy.deepcopy(item)
            item_copy.pop("page", None)
            result = add_page_number_to_toc(page_contents, item_copy, model)
            if (isinstance(result, list) and result
                    and isinstance(result[0].get("physical_index"), str)
                    and result[0]["physical_index"].startswith("<physical_index")):
                item["physical_index"] = int(
                    result[0]["physical_index"].split("_")[-1].rstrip(">").strip()
                )
                item.pop("page", None)

    return toc_items


def extract_matching_page_pairs(toc_page, toc_physical_index, start_page_index):
    pairs = []
    for phy_item in toc_physical_index:
        for page_item in toc_page:
            if phy_item.get("title") == page_item.get("title"):
                pi = phy_item.get("physical_index")
                if pi is not None and int(pi) >= start_page_index:
                    pairs.append({
                        "title": phy_item["title"],
                        "page": page_item["page"],
                        "physical_index": pi,
                    })
    return pairs


def calculate_page_offset(pairs):
    diffs = []
    for p in pairs:
        try:
            diffs.append(p["physical_index"] - p["page"])
        except (KeyError, TypeError):
            continue
    if not diffs:
        return None
    counts = {}
    for d in diffs:
        counts[d] = counts.get(d, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]


def add_page_offset_to_toc_json(data, offset):
    for item in data:
        if item.get("page") is not None and isinstance(item["page"], int):
            item["physical_index"] = item["page"] + offset
            del item["page"]
    return data


def process_toc_with_page_numbers(
    toc_content, toc_page_list, page_list,
    toc_check_page_num=None, model=None, logger=None,
):
    toc_with_pages = toc_transformer(toc_content, model)
    if logger:
        logger.info(f"toc_with_page_number: {toc_with_pages}")

    toc_no_pages = remove_page_number(copy.deepcopy(toc_with_pages))

    start_page = toc_page_list[-1] + 1
    main_content = ""
    end = min(start_page + (toc_check_page_num or 25), len(page_list))
    for page_index in range(start_page, end):
        main_content += (
            f"<physical_index_{page_index + 1}>\n"
            f"{page_list[page_index][0]}\n"
            f"<physical_index_{page_index + 1}>\n\n"
        )

    toc_with_physical = toc_index_extractor(toc_no_pages, main_content, model)
    if logger:
        logger.info(f"toc_with_physical_index: {toc_with_physical}")

    toc_with_physical = convert_physical_index_to_int(toc_with_physical)

    pairs = extract_matching_page_pairs(toc_with_pages, toc_with_physical, start_page)
    if logger:
        logger.info(f"matching_pairs: {pairs}")

    offset = calculate_page_offset(pairs)
    if logger:
        logger.info(f"offset: {offset}")

    toc_with_pages = add_page_offset_to_toc_json(toc_with_pages, offset)
    toc_with_pages = process_none_page_numbers(toc_with_pages, page_list, model=model)

    if logger:
        logger.info(f"final toc_with_page_number: {toc_with_pages}")

    return toc_with_pages


def check_toc(page_list, opt):
    # Cap TOC search to the first third of the document — a TOC deeper than
    # that is effectively absent for structural purposes, and scanning further
    # just increases false-positive risk on short scientific papers.
    opt = copy.copy(opt)
    opt.toc_check_page_num = min(opt.toc_check_page_num, max(3, len(page_list) // 3))

    toc_page_list = find_toc_pages(start_page_index=0, page_list=page_list, opt=opt)
    if not toc_page_list:
        return {"toc_content": None, "toc_page_list": [], "page_index_given_in_toc": "no"}

    toc_json = toc_extractor(page_list, toc_page_list, opt.model)
    if toc_json["page_index_given_in_toc"] == "yes":
        return {
            "toc_content": toc_json["toc_content"],
            "toc_page_list": toc_page_list,
            "page_index_given_in_toc": "yes",
        }

    current_start = toc_page_list[-1] + 1
    while (toc_json["page_index_given_in_toc"] == "no"
           and current_start < len(page_list)
           and current_start < opt.toc_check_page_num):
        additional = find_toc_pages(
            start_page_index=current_start, page_list=page_list, opt=opt
        )
        if not additional:
            break
        additional_json = toc_extractor(page_list, additional, opt.model)
        if additional_json["page_index_given_in_toc"] == "yes":
            return {
                "toc_content": additional_json["toc_content"],
                "toc_page_list": additional,
                "page_index_given_in_toc": "yes",
            }
        current_start = additional[-1] + 1

    return {
        "toc_content": toc_json["toc_content"],
        "toc_page_list": toc_page_list,
        "page_index_given_in_toc": "no",
    }


# ─── Verification & correction ───────────────────────────────────────────────

async def check_title_appearance(item, page_list, start_index=1, model=None):
    model = _model_name_fast  # simple yes/no — always use fast model
    title = item["title"]
    if "physical_index" not in item or item["physical_index"] is None:
        return {"list_index": item.get("list_index"), "answer": "no",
                "title": title, "page_number": None}

    page_number = item["physical_index"]
    page_text = page_list[page_number - start_index][0]

    prompt = f"""Does the section titled "{title}" appear or start in this page text? Use fuzzy matching, ignore spacing differences.

Page text: {page_text}

Reply with only this JSON, nothing else:
{{"answer": "yes or no"}}"""

    response = await llm_acompletion(model=model, prompt=prompt)
    parsed = extract_json(response)
    answer = parsed.get("answer", "no")
    return {"list_index": item.get("list_index"), "answer": answer,
            "title": title, "page_number": page_number}


async def check_title_appearance_in_start(title, page_text, model=None):
    model = _model_name_fast  # simple yes/no — always use fast model
    prompt = f"""Is the section titled "{title}" the very first content on this page? Answer "no" if other content appears before it.

Page text: {page_text}

Reply with only this JSON, nothing else:
{{"start_begin": "yes or no"}}"""

    response = await llm_acompletion(model=model, prompt=prompt)
    return extract_json(response).get("start_begin", "no")


async def check_title_appearance_in_start_concurrent(structure, page_list, model=None, logger=None):
    for item in structure:
        if item.get("physical_index") is None:
            item["appear_start"] = "no"

    tasks = []
    valid_items = []
    for item in structure:
        if item.get("physical_index") is not None:
            page_text = page_list[item["physical_index"] - 1][0]
            tasks.append(
                check_title_appearance_in_start(item["title"], page_text, model=model)
            )
            valid_items.append(item)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(valid_items, results):
        if isinstance(result, Exception):
            item["appear_start"] = "no"
        else:
            item["appear_start"] = result

    return structure


async def verify_toc(page_list, list_result, start_index=1, N=None, model=None):
    last_pi = None
    for item in reversed(list_result):
        if item.get("physical_index") is not None:
            last_pi = item["physical_index"]
            break

    if last_pi is None or last_pi < len(page_list) / 2:
        return 0, []

    if N is None:
        sample_indices = range(len(list_result))
    else:
        N = min(N, len(list_result))
        sample_indices = random.sample(range(len(list_result)), N)

    indexed = []
    for idx in sample_indices:
        item = list_result[idx]
        if item.get("physical_index") is not None:
            item_copy = item.copy()
            item_copy["list_index"] = idx
            indexed.append(item_copy)

    results = await asyncio.gather(
        *[check_title_appearance(item, page_list, start_index, model) for item in indexed]
    )

    correct = sum(1 for r in results if r["answer"] == "yes")
    incorrect = [r for r in results if r["answer"] != "yes"]
    accuracy = correct / len(results) if results else 0
    return accuracy, incorrect


async def single_toc_item_index_fixer(section_title, content, model=None):
    prompt = """
    You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    Reply in a JSON format:
    {
        "thinking": <explain which page contains the start of this section>,
        "physical_index": "<physical_index_X>" (keep the format)
    }
    Directly return the final JSON structure. Do not output anything else."""

    prompt += "\nSection Title:\n" + str(section_title) + "\nDocument pages:\n" + content
    response = await llm_acompletion(model=model, prompt=prompt)
    parsed = extract_json(response)
    return convert_physical_index_to_int(parsed.get("physical_index", ""))


async def fix_incorrect_toc(
    toc, page_list, incorrect_results, start_index=1, model=None, logger=None
):
    incorrect_indices = {r["list_index"] for r in incorrect_results}
    end_index = len(page_list) + start_index - 1

    async def process_item(incorrect_item):
        list_index = incorrect_item["list_index"]
        if list_index < 0 or list_index >= len(toc):
            return {"list_index": list_index, "title": incorrect_item["title"],
                    "physical_index": None, "is_valid": False}

        prev_correct = start_index - 1
        for j in range(list_index - 1, -1, -1):
            if j not in incorrect_indices and 0 <= j < len(toc):
                pi = toc[j].get("physical_index")
                if pi is not None:
                    prev_correct = pi
                    break

        next_correct = end_index
        for j in range(list_index + 1, len(toc)):
            if j not in incorrect_indices and 0 <= j < len(toc):
                pi = toc[j].get("physical_index")
                if pi is not None:
                    next_correct = pi
                    break

        page_contents = []
        for page_index in range(prev_correct, next_correct + 1):
            li = page_index - start_index
            if 0 <= li < len(page_list):
                text = (
                    f"<physical_index_{page_index}>\n"
                    f"{page_list[li][0]}\n"
                    f"<physical_index_{page_index}>\n\n"
                )
                page_contents.append(text)

        content_range = "".join(page_contents)
        pi_int = await single_toc_item_index_fixer(
            incorrect_item["title"], content_range, model
        )

        check_item = incorrect_item.copy()
        check_item["physical_index"] = pi_int
        check_result = await check_title_appearance(check_item, page_list, start_index, model)

        return {
            "list_index": list_index,
            "title": incorrect_item["title"],
            "physical_index": pi_int,
            "is_valid": check_result["answer"] == "yes",
        }

    results = await asyncio.gather(
        *[process_item(item) for item in incorrect_results],
        return_exceptions=True,
    )
    results = [r for r in results if not isinstance(r, Exception)]

    invalid = []
    for result in results:
        if result["is_valid"]:
            idx = result["list_index"]
            if 0 <= idx < len(toc):
                toc[idx]["physical_index"] = result["physical_index"]
            else:
                invalid.append(result)
        else:
            invalid.append(result)

    return toc, invalid


async def fix_incorrect_toc_with_retries(
    toc, page_list, incorrect_results,
    start_index=1, max_attempts=3, model=None, logger=None,
):
    current_toc = toc
    current_incorrect = incorrect_results

    for attempt in range(max_attempts):
        if not current_incorrect:
            break
        current_toc, current_incorrect = await fix_incorrect_toc(
            current_toc, page_list, current_incorrect, start_index, model, logger
        )

    return current_toc, current_incorrect


# ─── Main orchestration ──────────────────────────────────────────────────────

async def meta_processor(
    page_list, mode=None, toc_content=None, toc_page_list=None,
    start_index=1, opt=None, logger=None,
):
    if mode == "process_toc_with_page_numbers":
        toc = process_toc_with_page_numbers(
            toc_content, toc_page_list, page_list,
            toc_check_page_num=opt.toc_check_page_num, model=opt.model, logger=logger,
        )
    elif mode == "process_toc_no_page_numbers":
        toc = process_toc_no_page_numbers(
            toc_content, toc_page_list, page_list,
            model=opt.model, logger=logger,
        )
    else:
        toc = process_no_toc(
            page_list, start_index=start_index, model=opt.model, logger=logger,
        )

    toc = [item for item in toc if item.get("physical_index") is not None]
    toc = validate_and_truncate_physical_indices(
        toc, len(page_list), start_index=start_index, logger=logger,
    )

    accuracy, incorrect = await verify_toc(
        page_list, toc, start_index=start_index, model=opt.model,
    )

    if logger:
        logger.info({"mode": mode, "accuracy": accuracy, "incorrect_count": len(incorrect)})

    if accuracy == 1.0 and not incorrect:
        return toc
    if accuracy > 0.6 and incorrect:
        toc, _ = await fix_incorrect_toc_with_retries(
            toc, page_list, incorrect,
            start_index=start_index, max_attempts=3, model=opt.model, logger=logger,
        )
        return toc

    # Fallback cascade
    if mode == "process_toc_with_page_numbers":
        return await meta_processor(
            page_list, mode="process_toc_no_page_numbers",
            toc_content=toc_content, toc_page_list=toc_page_list,
            start_index=start_index, opt=opt, logger=logger,
        )
    elif mode == "process_toc_no_page_numbers":
        return await meta_processor(
            page_list, mode="process_no_toc",
            start_index=start_index, opt=opt, logger=logger,
        )
    else:
        raise RuntimeError("Tree building failed: all processing modes exhausted")


async def process_large_node_recursively(node, page_list, opt=None, logger=None):
    node_pages = page_list[node["start_index"] - 1: node["end_index"]]
    token_num = sum(p[1] for p in node_pages)

    if (node["end_index"] - node["start_index"] > opt.max_page_num_each_node
            and token_num >= opt.max_token_num_each_node):

        sub_toc = await meta_processor(
            node_pages, mode="process_no_toc",
            start_index=node["start_index"], opt=opt, logger=logger,
        )
        sub_toc = await check_title_appearance_in_start_concurrent(
            sub_toc, page_list, model=opt.model, logger=logger,
        )

        valid = [item for item in sub_toc if item.get("physical_index") is not None]

        if valid and node["title"].strip() == valid[0]["title"].strip():
            node["nodes"] = post_processing(valid[1:], node["end_index"])
            node["end_index"] = valid[1]["start_index"] if len(valid) > 1 else node["end_index"]
        else:
            node["nodes"] = post_processing(valid, node["end_index"])
            node["end_index"] = valid[0]["start_index"] if valid else node["end_index"]

    if node.get("nodes"):
        await asyncio.gather(
            *[process_large_node_recursively(child, page_list, opt, logger) for child in node["nodes"]]
        )

    return node


async def generate_node_summary(node, model=None):
    model = _model_name_fast  # straightforward summarization — always use fast model
    prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

    Partial Document Text: {node['text']}

    Directly return the description, do not include any other text."""

    return await llm_acompletion(model, prompt)


async def generate_summaries_for_structure(structure, model=None):
    nodes = structure_to_list(structure)
    summaries = await asyncio.gather(
        *[generate_node_summary(node, model=model) for node in nodes]
    )
    for node, summary in zip(nodes, summaries):
        node["summary"] = summary
    return structure


def generate_doc_description(structure, model=None):
    prompt = f"""Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.

    Document Structure: {structure}

    Directly return the description, do not include any other text."""

    return llm_completion(model, prompt)


# ─── Public entry points ─────────────────────────────────────────────────────

async def build_tree_async(pdf_path: str, config: dict) -> DocumentTree:
    """Build a document tree from a PDF. Returns a DocumentTree (no enrichment)."""
    tree_cfg = config["tree"]
    model = config["tree_llm"]["model"]

    # Build an opt namespace matching what PageIndex functions expect
    opt = SimpleNamespace(
        model=model,
        toc_check_page_num=tree_cfg["toc_check_pages"],
        max_page_num_each_node=tree_cfg["max_pages_per_node"],
        max_token_num_each_node=tree_cfg["max_tokens_per_node"],
        if_add_node_id="yes" if tree_cfg.get("add_node_id", True) else "no",
        if_add_node_summary="yes" if tree_cfg.get("add_node_summary", True) else "no",
        if_add_doc_description="yes" if tree_cfg.get("add_doc_description", False) else "no",
        if_add_node_text="no",
    )

    _init_clients(config)

    logger = JsonLogger()
    page_list = get_page_tokens(pdf_path, model=model)

    if is_likely_scanned(page_list):
        logging.warning(
            f"PDF appears to be scanned (low extractable text). "
            f"Tree quality may be poor — consider OCR preprocessing."
        )

    logger.info({"total_pages": len(page_list), "total_tokens": sum(p[1] for p in page_list)})

    # Build tree structure
    check_toc_result = check_toc(page_list, opt)
    logger.info({"check_toc_result": str(check_toc_result)})

    has_toc = (check_toc_result.get("toc_content")
               and check_toc_result["toc_content"].strip())

    if has_toc and check_toc_result["page_index_given_in_toc"] == "yes":
        toc = await meta_processor(
            page_list, mode="process_toc_with_page_numbers",
            start_index=1,
            toc_content=check_toc_result["toc_content"],
            toc_page_list=check_toc_result["toc_page_list"],
            opt=opt, logger=logger,
        )
    elif has_toc:
        toc = await meta_processor(
            page_list, mode="process_toc_no_page_numbers",
            start_index=1,
            toc_content=check_toc_result["toc_content"],
            toc_page_list=check_toc_result["toc_page_list"],
            opt=opt, logger=logger,
        )
    else:
        toc = await meta_processor(
            page_list, mode="process_no_toc",
            start_index=1, opt=opt, logger=logger,
        )

    toc = add_preface_if_needed(toc)
    toc = await check_title_appearance_in_start_concurrent(
        toc, page_list, model=opt.model, logger=logger,
    )

    valid_toc = [item for item in toc if item.get("physical_index") is not None]
    tree_nodes = post_processing(valid_toc, len(page_list))

    # Subdivide large nodes
    await asyncio.gather(
        *[process_large_node_recursively(node, page_list, opt, logger) for node in tree_nodes]
    )

    # Add node IDs
    if opt.if_add_node_id == "yes":
        write_node_id(tree_nodes)

    # Add summaries
    if opt.if_add_node_summary == "yes":
        add_node_text(tree_nodes, page_list)
        await generate_summaries_for_structure(tree_nodes, model=model)
        remove_structure_text(tree_nodes)

    # Doc description
    doc_description = None
    if opt.if_add_doc_description == "yes":
        clean = create_clean_structure_for_description(tree_nodes)
        doc_description = generate_doc_description(clean, model=model)

    # Format and convert to Pydantic models
    tree_nodes = format_structure(
        tree_nodes,
        order=["title", "node_id", "start_index", "end_index", "summary", "nodes"],
    )

    paper_id = Path(pdf_path).stem
    root_nodes = _dicts_to_tree_nodes(tree_nodes)

    return DocumentTree(
        paper_id=paper_id,
        pdf_path=str(Path(pdf_path).resolve()),
        total_pages=len(page_list),
        doc_description=doc_description,
        root_nodes=root_nodes,
    )


def _dicts_to_tree_nodes(nodes: list[dict]) -> list[TreeNode]:
    result = []
    for n in nodes:
        children = _dicts_to_tree_nodes(n.get("nodes", []) or [])
        result.append(TreeNode(
            title=n.get("title", ""),
            node_id=n.get("node_id", "0000"),
            start_index=n.get("start_index", 1),
            end_index=n.get("end_index", 1),
            summary=n.get("summary"),
            nodes=children,
        ))
    return result


def build_tree(pdf_path: str, config: dict) -> DocumentTree:
    """Synchronous wrapper around build_tree_async."""
    return asyncio.run(build_tree_async(pdf_path, config))

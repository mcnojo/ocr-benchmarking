const $ = (s) => document.querySelector(s);

const setStatus = (m) => { $("#status").textContent = m; };

const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// asset_path in tree.json is absolute (/Users/.../etl/kb/<paper>/assets/...).
// Strip everything up to "/kb/" so /kb/* statics handle it.
function assetUrl(absPath) {
  if (!absPath) return "";
  const i = absPath.indexOf("/kb/");
  return i === -1 ? absPath : absPath.slice(i);
}

const RAW_CAP = 2000; // chandra often emits multi-KB repeating x/y blobs after </analyze>
function rawBlock(content) {
  const trimmed = content.length > RAW_CAP
    ? content.slice(0, RAW_CAP) + `\n…[truncated, ${content.length - RAW_CAP} more chars]`
    : content;
  return `<details class="raw-toggle"><summary>raw</summary><pre class="raw">${esc(trimmed)}</pre></details>`;
}

// Render an <analyze>[…]</analyze> block (chandra "figure analysis" schema) as a <dl>.
// Used for figure-OCR output.
function renderAnalyze(jsonStr) {
  let parsed;
  try { parsed = JSON.parse(jsonStr); }
  catch { return `<pre class="raw">${esc(jsonStr.slice(0, RAW_CAP))}</pre>`; }
  const items = Array.isArray(parsed) ? parsed : [parsed];
  return items.map((sub) => {
    const rows = ["x_labels", "y_labels", "x_ticks", "y_ticks", "legends", "series"]
      .filter((k) => sub[k]).map((k) => `<dt>${k.replace(/_/g, " ")}</dt><dd>${esc(sub[k])}</dd>`).join("");
    return `<div class="analysis"><h5>${esc(sub.titles || "subplot")}</h5><dl>${rows}</dl></div>`;
  }).join("");
}

// Pre-parsed chandra envelope from etl/pipeline/chandra_parser.py.
function renderParsed(p) {
  if (!p) return "";
  if (p.format === "layout_html") {
    return (p.blocks || []).map((b) => {
      const label = esc(b.label || "Block");
      return `<div data-label="${label}">${b.html || esc(b.text || "")}</div>`;
    }).join("");
  }
  if (p.format === "figure_analysis") {
    const note = p.truncated ? `<div class="truncated-note">⚠ truncated output</div>` : "";
    const panels = (p.panels || []).map((sub) => {
      const rows = ["x_label", "y_label", "x_tick", "y_tick", "legend", "series"]
        .filter((k) => sub[k]).map((k) => `<dt>${k.replace(/_/g, " ")}</dt><dd>${esc(sub[k])}</dd>`).join("");
      return `<div class="analysis"><h5>${esc(sub.title || "subplot")}</h5><dl>${rows}</dl></div>`;
    }).join("");
    return note + panels;
  }
  return `<pre class="raw">${esc(JSON.stringify(p).slice(0, RAW_CAP))}</pre>`;
}

// Chandra output for either field can be (a) labeled-bbox HTML, (b) <analyze>…</analyze>JSON,
// or (c) a truncated/malformed mix. Detect and dispatch.
function renderChandra(content) {
  if (!content) return "";
  const m = content.match(/<analyze>\s*([\s\S]*?)\s*<\/analyze>/);
  if (m) return renderAnalyze(m[1]) + rawBlock(content);
  if (/data-bbox=|data-label=/.test(content)) return content; // trusted local-pipeline HTML
  // truncated <analyze> with no closing tag, or unknown shape — try to salvage the JSON head
  const opener = content.indexOf("<analyze>");
  if (opener !== -1) {
    const head = content.slice(opener + "<analyze>".length);
    // attempt parse of progressively-shorter prefixes ending at the last "}"
    const lastBrace = head.lastIndexOf("}");
    if (lastBrace !== -1) {
      const candidate = head.slice(0, lastBrace + 1) + (head.trimStart().startsWith("[") ? "]" : "");
      try {
        JSON.parse(candidate);
        return `<div class="truncated-note">⚠ truncated output, showing salvaged head</div>`
          + renderAnalyze(candidate) + rawBlock(content);
      } catch {}
    }
  }
  return rawBlock(content);
}

function renderVisuals(els) {
  if (!els?.length) return "";
  return `<div class="visuals">` + els.map((e) => {
    const img = e.asset_path ? `<img src="${assetUrl(e.asset_path)}" alt="${esc(e.element_id)}">` : "";
    const cap = e.caption ? `<p class="caption">${esc(e.caption)}</p>` : "";
    const ocrBody = e.ocr_parsed ? renderParsed(e.ocr_parsed)
                  : e.ocr_text   ? renderChandra(e.ocr_text)
                  : "";
    const ocr = ocrBody ? `<div class="block"><h4>OCR</h4><div class="ocr">${ocrBody}</div></div>` : "";
    const chems = e.chem_entities?.length
      ? `<div class="block"><h4>Chem entities</h4><div class="chems">${e.chem_entities.map((c) => `<span>${esc(c)}</span>`).join("")}</div></div>` : "";
    return `
      <details class="visual" data-t="${esc(e.element_type)}">
        <summary><code>${esc(e.element_id)}</code><span class="type">${esc(e.element_type)}</span><span style="color:#556;font-size:0.7rem;">p${e.page_index}</span></summary>
        <div class="visual-body">${img}${cap}${ocr}${chems}</div>
      </details>`;
  }).join("") + `</div>`;
}

function renderTree(nodes, depth = 0) {
  return nodes.map((n) => `
    <details class="node depth-${depth}" ${depth === 0 ? "open" : ""}>
      <summary>
        <span class="title">${esc(n.title)}</span>
        <span class="meta">pp ${n.start_index}–${n.end_index}</span>
      </summary>
      ${n.summary ? `<div class="summary-md">${marked.parse(n.summary)}</div>` : ""}
      ${renderVisuals(n.visual_elements)}
      ${n.nodes?.length ? renderTree(n.nodes, depth + 1) : ""}
    </details>
  `).join("");
}

function renderMath(root) {
  root.querySelectorAll("math").forEach((el) => {
    const tex = el.textContent;
    const display = el.getAttribute("display") === "block";
    try { katex.render(tex, el, { throwOnError: false, displayMode: display }); } catch {}
  });
  // also pass over $...$ / $$...$$ in markdown summaries
  root.querySelectorAll(".summary-md").forEach((el) => {
    el.innerHTML = el.innerHTML.replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => {
      try { return katex.renderToString(t, { throwOnError: false, displayMode: true }); } catch { return _; }
    }).replace(/\$([^\$\n]+?)\$/g, (_, t) => {
      try { return katex.renderToString(t, { throwOnError: false, displayMode: false }); } catch { return _; }
    });
  });
}

async function renderPdf(paper) {
  const container = $("#pdf");
  container.innerHTML = "";
  setStatus("loading pdf…");
  try {
    const pdf = await pdfjsLib.getDocument(`/pdf/${paper}`).promise;
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const viewport = page.getViewport({ scale: 1.5 });
      const canvas = document.createElement("canvas");
      canvas.className = "page";
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      container.appendChild(canvas);
      await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
    }
    setStatus(`${pdf.numPages} pages`);
  } catch (e) {
    container.innerHTML = `<p style="color:#e94560;padding:1rem;">PDF unavailable: ${esc(e.message || e)}</p>`;
    setStatus("pdf error");
  }
}

async function loadPaper(paper) {
  setStatus("loading tree…");
  const tree = await (await fetch(`/kb/${paper}/tree.json`)).json();
  $("#tree").innerHTML = renderTree(tree.root_nodes);
  renderMath($("#tree"));
  await renderPdf(paper);
}

async function init() {
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js";

  const papers = await (await fetch("/api/papers")).json();
  const sel = $("#paper-select");
  if (!papers.length) {
    sel.innerHTML = `<option>no papers found</option>`;
    setStatus("etl/kb is empty");
    return;
  }
  papers.forEach((p) => sel.add(new Option(p, p)));
  sel.addEventListener("change", () => loadPaper(sel.value));
  const initial = new URLSearchParams(location.search).get("paper") || papers[0];
  sel.value = papers.includes(initial) ? initial : papers[0];
  await loadPaper(sel.value);
}

init();

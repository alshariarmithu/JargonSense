/* ==========================================================================
   SE Sentiment Audit — interface logic

   Vanilla JS on purpose: no build step, no node_modules, nothing to install
   beyond Flask. The whole interface is three files.
   ========================================================================== */

const LABELS = ["negative", "neutral", "positive"];
const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = { models: [], examples: [], results: null, stress: null, findings: null };

const api = {
  async get(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  async post(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    return data;
  },
};

/* ------------------------------------------------------------------ theme */
function initTheme() {
  const saved = (() => {
    try { return localStorage.getItem("theme"); } catch { return null; }
  })();
  if (saved) document.documentElement.dataset.theme = saved;

  $("#theme-toggle").addEventListener("click", () => {
    const root = document.documentElement;
    const isDark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = isDark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch { /* private mode */ }
  });
}

/* ------------------------------------------------------------------- tabs */
function initTabs() {
  $$(".tab").forEach(tab => tab.addEventListener("click", () => {
    $$(".tab").forEach(t => t.setAttribute("aria-selected", String(t === tab)));
    $$(".view").forEach(v => v.classList.remove("active"));
    $(`#view-${tab.dataset.view}`).classList.add("active");
    loadView(tab.dataset.view);
  }));
}

/* --------------------------------------------------------------- rendering */
/** Colour intensity is relative to the strongest word in the sentence, so a
 *  faint explanation is still readable rather than uniformly pale. */
function renderSentence(target, tokens) {
  const peak = Math.max(...tokens.map(t => Math.abs(t.weight)), 1e-9);
  target.innerHTML = "";
  tokens.forEach(({ token, weight }) => {
    const alpha = Math.min(Math.abs(weight) / peak, 1) * 0.6;
    const span = document.createElement("span");
    span.className = "word";
    span.textContent = token;
    if (Math.abs(weight) > 1e-6) {
      const hue = weight > 0 ? "var(--negative)" : "var(--positive)";
      span.style.background = `color-mix(in srgb, ${hue} ${(alpha * 100).toFixed(0)}%, transparent)`;
      span.dataset.tip = weight.toFixed(4);
    }
    target.append(span, document.createTextNode(" "));
  });
}

function renderBars(target, probabilities) {
  target.innerHTML = "";
  LABELS.forEach((label, i) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span class="bar-label">${label}</span>
      <span class="bar-track"><i class="bar-fill ${label}"></i></span>
      <span class="bar-value num">${(probabilities[i] * 100).toFixed(1)}%</span>`;
    target.append(row);
    requestAnimationFrame(() => {
      $(".bar-fill", row).style.width = `${probabilities[i] * 100}%`;
    });
  });
}

/** A sortable table from an array of objects. */
function renderTable(table, rows, columns, options = {}) {
  const { highlight = () => false, sortKey = null, descending = true } = options;
  const thead = $("thead", table), tbody = $("tbody", table);

  let data = [...rows];
  let key = sortKey, desc = descending;

  const draw = () => {
    if (key) {
      data.sort((a, b) => {
        const x = a[key], y = b[key];
        const cmp = (typeof x === "number" && typeof y === "number")
          ? x - y : String(x ?? "").localeCompare(String(y ?? ""));
        return desc ? -cmp : cmp;
      });
    }
    thead.innerHTML = `<tr>${columns.map(c =>
      `<th data-key="${c.key}">${c.label}${key === c.key ? (desc ? " ↓" : " ↑") : ""}</th>`
    ).join("")}</tr>`;

    tbody.innerHTML = data.map(row => {
      const cells = columns.map(c => {
        const value = c.render ? c.render(row[c.key], row) : (row[c.key] ?? "");
        return `<td class="${c.className || ""}">${value}</td>`;
      }).join("");
      return `<tr class="${highlight(row) ? "is-best" : ""}">${cells}</tr>`;
    }).join("");

    $$("th", thead).forEach(th => th.addEventListener("click", () => {
      const clicked = th.dataset.key;
      desc = clicked === key ? !desc : true;
      key = clicked;
      draw();
    }));
  };
  draw();
}

function renderBarChart(target, entries, { format = v => v.toFixed(3), colour } = {}) {
  const peak = Math.max(...entries.map(e => e[1]), 1e-9);
  target.innerHTML = entries.map(([label, value]) => `
    <div class="fa-row">
      <span>${label}</span>
      <span class="fa-track"><i class="fa-fill" style="width:0;background:${
        colour ? colour(label, value) : "var(--accent)"}"></i></span>
      <span class="num" style="text-align:right">${format(value)}</span>
    </div>`).join("");
  requestAnimationFrame(() => {
    $$(".fa-fill", target).forEach((bar, i) => {
      bar.style.width = `${(entries[i][1] / peak) * 100}%`;
    });
  });
}

const pct  = v => (v == null || v === "" ? "—" : `${(v * 100).toFixed(1)}%`);
const f4   = v => (v == null || v === "" ? "—" : Number(v).toFixed(4));
const verdictHTML = label => `<span class="verdict ${label}">${label}</span>`;

/* ---------------------------------------------------------------- predict */
function modelOption(model) {
  const score = model.macro_f1 != null ? ` — ${Number(model.macro_f1).toFixed(4)}` : "";
  const speed = model.exact ? "" : " (LIME)";
  return `<option value="${model.name}">${model.name}${score}${speed}</option>`;
}

async function initPredict() {
  const { models, default: fallback } = await api.get("/api/models");
  state.models = models;

  const options = models.map(modelOption).join("");
  ["#predict-model", "#cmp-a", "#cmp-b", "#cmp-c"].forEach(sel => {
    $(sel).innerHTML = options;
  });
  $("#predict-model").value = fallback;
  $("#cmp-a").value = "vader";
  $("#cmp-b").value = models.find(m => m.name.startsWith("glove"))?.name || fallback;
  $("#cmp-c").value = fallback;

  const { examples } = await api.get("/api/examples");
  state.examples = examples;
  const chips = examples.map((ex, i) =>
    `<button class="chip" data-i="${i}" title="${ex.text}">${ex.kind}</button>`).join("");

  $("#examples").innerHTML = chips;
  $("#compare-examples").innerHTML = chips;

  $$("#examples .chip").forEach(chip => chip.addEventListener("click", () => {
    $("#predict-text").value = examples[chip.dataset.i].text;
    runPredict();
  }));
  $$("#compare-examples .chip").forEach(chip => chip.addEventListener("click", () => {
    $("#compare-text").value = examples[chip.dataset.i].text;
    runCompare();
  }));

  $("#predict-model").addEventListener("change", () => {
    const model = models.find(m => m.name === $("#predict-model").value);
    const badge = $("#predict-method");
    badge.textContent = model?.exact ? "exact" : "LIME";
    badge.className = `badge ${model?.exact ? "exact" : "slow"}`;
    runPredict();
  });

  $("#predict-go").addEventListener("click", runPredict);
  $("#predict-text").addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runPredict();
  });
  $("#compare-go").addEventListener("click", runCompare);

  runPredict();
}

async function runPredict() {
  const button = $("#predict-go");
  const text = $("#predict-text").value.trim();
  if (!text) return;

  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span> Working';
  try {
    const result = await api.post("/api/predict", {
      text, model: $("#predict-model").value,
    });
    $("#predict-out").hidden = false;
    $("#predict-verdict").textContent = result.label;
    $("#predict-verdict").className = `verdict ${result.label}`;
    renderSentence($("#predict-sentence"), result.tokens);
    renderBars($("#predict-bars"), result.probabilities);
    $("#predict-note").textContent =
      `Explaining the predicted class (${result.explained_class || result.label}) · ${result.method}.` +
      (result.note ? ` ${result.note}` : "");
  } catch (err) {
    $("#predict-out").hidden = false;
    $("#predict-sentence").innerHTML = `<span class="empty">${err.message}</span>`;
  } finally {
    button.disabled = false;
    button.textContent = "Classify";
  }
}

/* ---------------------------------------------------------------- compare */
const TAGLINE = {
  vader: "general-purpose lexicon, never saw a StackOverflow post",
  glove: "pretrained on Wikipedia and news",
  w2v:   "vectors learned from this corpus",
  tfidf: "term weights learned from this corpus",
};
const taglineFor = name =>
  TAGLINE[name] || TAGLINE[Object.keys(TAGLINE).find(k => name.startsWith(k))] || "";

async function runCompare() {
  const button = $("#compare-go");
  const text = $("#compare-text").value.trim();
  if (!text) return;

  const models = [$("#cmp-a").value, $("#cmp-b").value, $("#cmp-c").value];
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span> Working';
  $("#compare-out").innerHTML = "";

  try {
    const { results } = await api.post("/api/compare", { text, models });
    const labels = new Set(results.map(r => r.label));

    const banner = labels.size > 1
      ? `<div class="disagree"><strong>They disagree.</strong> The same sentence is
         ${results.map(r => `<code>${r.model}</code> → <strong>${r.label}</strong>`).join(", ")}.
         That gap is the domain-shift problem, live.</div>`
      : `<div class="disagree">All three agree on <strong>${[...labels][0]}</strong>
         for this sentence. Try one of the jargon examples above.</div>`;

    const cards = results.map(r => `
      <div class="compare-card">
        <h3>${r.model} ${verdictHTML(r.label)}</h3>
        <p class="tagline">${taglineFor(r.model)}</p>
        <div class="sentence" data-sentence="${r.model}" style="font-size:.95rem;line-height:2.2"></div>
        <div class="bars" data-bars="${r.model}" style="margin-top:12px"></div>
        <p class="hint" style="margin-top:10px;font-size:.75rem">${r.method}</p>
      </div>`).join("");

    $("#compare-out").innerHTML =
      `<div class="card">${banner}<div class="compare-grid">${cards}</div></div>`;

    results.forEach(r => {
      renderSentence($(`[data-sentence="${r.model}"]`), r.tokens);
      renderBars($(`[data-bars="${r.model}"]`), r.probabilities);
    });
  } catch (err) {
    $("#compare-out").innerHTML = `<div class="card"><p class="empty">${err.message}</p></div>`;
  } finally {
    button.disabled = false;
    button.textContent = "Compare";
  }
}

/* ---------------------------------------------------------------- results */
async function loadResults() {
  if (state.results) return;
  const data = await api.get("/api/results");
  state.results = data;
  const rows = data.rows;
  if (!rows.length) return;

  const best = rows[0];
  const byKind = k => rows.filter(r => r.kind === k)
    .reduce((s, r) => s + r.macro_f1, 0) / rows.filter(r => r.kind === k).length;

  $("#results-stats").innerHTML = `
    <div class="card stat"><div class="k">Best model</div>
      <div class="v">${best.macro_f1.toFixed(4)}</div>
      <div class="s">${best.model} · test macro-F1</div></div>
    <div class="card stat"><div class="k">Over VADER</div>
      <div class="v">+${(best.macro_f1 - data.baselines.vader).toFixed(3)}</div>
      <div class="s">baseline ${data.baselines.vader.toFixed(4)}</div></div>
    <div class="card stat"><div class="k">Discriminative edge</div>
      <div class="v">+${(byKind("discriminative") - byKind("generative")).toFixed(3)}</div>
      <div class="s">mean macro-F1, McNemar p = 4.7e-9</div></div>`;

  renderTable($("#results-table"), rows, [
    { key: "model", label: "Model", render: v => `<code>${v}</code>` },
    { key: "representation", label: "Representation" },
    { key: "kind", label: "Kind" },
    { key: "macro_f1", label: "Test macro-F1", render: (v) =>
        `<span class="num">${f4(v)}</span>` },
    { key: "accuracy", label: "Accuracy", render: v => `<span class="num">${f4(v)}</span>` },
    { key: "neutral_to_negative", label: "Neutral → negative",
      render: v => `<span class="num">${pct(v)}</span>` },
    { key: "negative_recall", label: "Negative recall",
      render: v => `<span class="num">${pct(v)}</span>` },
  ], { sortKey: "macro_f1", highlight: r => r.model === best.model });

  const maxBy = (list, key) => {
    const out = {};
    list.forEach(r => { out[r[key]] = Math.max(out[r[key]] ?? 0, r.macro_f1); });
    return Object.entries(out).sort((a, b) => b[1] - a[1]);
  };
  renderBarChart($("#results-by-rep"), maxBy(rows, "representation"),
    { format: v => v.toFixed(4) });

  renderBarChart($("#results-by-kind"), [
    ["discriminative", byKind("discriminative")],
    ["generative", byKind("generative")],
  ], {
    format: v => v.toFixed(4),
    colour: label => label === "discriminative" ? "var(--positive)" : "var(--neutral)",
  });
}

/* ----------------------------------------------------------------- stress */
async function loadStress() {
  if (state.stress) return;
  const data = await api.get("/api/stress");
  state.stress = data;

  const families = Object.entries(data.false_alarm_by_family)
    .sort((a, b) => b[1] - a[1]);
  renderBarChart($("#stress-family"), families, {
    format: v => `${(v * 100).toFixed(0)}%`,
    colour: label => /general/i.test(label) ? "var(--negative)" : "var(--positive)",
  });

  const best = [...data.summary].sort(
    (a, b) => a.false_alarm_rate - b.false_alarm_rate)[0];
  renderTable($("#stress-table"), data.summary, [
    { key: "model", label: "Model", render: v => `<code>${v}</code>` },
    { key: "false_alarm_rate", label: "False alarm ↓",
      render: v => `<span class="num">${pct(v)}</span>` },
    { key: "negative_recall", label: "Neg recall ↑",
      render: v => `<span class="num">${pct(v)}</span>` },
    { key: "recall_with_jargon", label: "Jargon",
      render: v => `<span class="num">${pct(v)}</span>` },
    { key: "recall_without_jargon", label: "Plain",
      render: v => `<span class="num">${pct(v)}</span>` },
    { key: "minimal_pair_accuracy", label: "Pairs",
      render: v => `<span class="num">${pct(v)}</span>` },
    { key: "accuracy", label: "Accuracy",
      render: v => `<span class="num">${pct(v)}</span>` },
  ], { sortKey: "false_alarm_rate", descending: false,
       highlight: r => r.model === best.model });

  const FILTERS = {
    all:     () => true,
    neutral: s => s.intended_label === "neutral",
    complaint: s => s.intended_label === "negative",
    jargon:  s => s.lexicon_word !== "",
    pairs:   s => s.pair_id > 0,
  };
  $("#stress-filters").innerHTML = Object.keys(FILTERS).map((k, i) =>
    `<button class="chip" data-f="${k}" aria-pressed="${i === 0}">${k}</button>`).join("");

  const drawSentences = filter => {
    renderTable($("#stress-sentences"),
      data.sentences.filter(FILTERS[filter]), [
        { key: "id", label: "ID", render: v => `<code>${v}</code>` },
        { key: "text", label: "Sentence", className: "wrap" },
        { key: "intended_label", label: "Intended", render: verdictHTML },
        { key: "lexicon_word", label: "Jargon",
          render: v => v ? `<code>${v}</code>` : "—" },
        { key: "pair_id", label: "Pair", render: v => v > 0 ? `#${v}` : "—" },
      ]);
  };
  drawSentences("all");

  $$("#stress-filters .chip").forEach(chip => chip.addEventListener("click", () => {
    $$("#stress-filters .chip").forEach(c =>
      c.setAttribute("aria-pressed", String(c === chip)));
    drawSentences(chip.dataset.f);
  }));
}

/* --------------------------------------------------------------- findings */
async function loadFindings() {
  if (state.findings) return;
  const data = await api.get("/api/findings");
  state.findings = data;

  // Nearest neighbours, pivoted so both embeddings sit on one row per word.
  const words = [...new Set(data.neighbours.map(r => r.word))];
  const pivot = words.map(word => {
    const rows = data.neighbours.filter(r => r.word === word);
    const pick = name => rows.find(r => r.embedding === name)?.neighbours ?? "—";
    return {
      word,
      occurrences: rows[0]?.train_occurrences ?? "",
      glove: pick("glove"),
      w2v: pick("w2v"),
    };
  });
  renderTable($("#neighbours-table"), pivot, [
    { key: "word", label: "Word", render: v => `<code>${v}</code>` },
    { key: "occurrences", label: "In corpus",
      render: v => `<span class="num">${v === "" ? "—" : v + "×"}</span>` },
    { key: "glove", label: "GloVe (general English)", className: "wrap" },
    { key: "w2v", label: "Word2Vec (this corpus)", className: "wrap" },
  ]);

  const tick = v => (v === true || v === "True") ? "✓" : "—";
  renderTable($("#phrases-table"), data.compound_phrases, [
    { key: "phrase", label: "Phrase", render: v => `<code>${v}</code>` },
    { key: "in_tfidf11", label: "(1,1)", render: tick },
    { key: "in_tfidf12", label: "(1,2)", render: tick },
    { key: "in_tfidf13", label: "(1,3)", render: tick },
    { key: "n_negative", label: "neg", render: v => `<span class="num">${v || 0}</span>` },
    { key: "n_neutral",  label: "neu", render: v => `<span class="num">${v || 0}</span>` },
    { key: "n_positive", label: "pos", render: v => `<span class="num">${v || 0}</span>` },
  ]);

  renderTable($("#ablation-table"), data.ablation, [
    { key: "mode", label: "Mode", render: v => `<code>${v}</code>` },
    { key: "vocabulary_size", label: "Vocabulary",
      render: v => `<span class="num">${Number(v).toLocaleString()}</span>` },
    { key: "macro_f1", label: "Macro-F1",
      render: v => `<span class="num">${f4(v)}</span>` },
    { key: "neutral_to_negative_rate", label: "Neutral → negative",
      render: v => `<span class="num">${pct(v)}</span>` },
  ]);

  const tokens = data.shap_tokens.length ? data.shap_tokens : data.lime_tokens;
  renderTable($("#tokens-table"), tokens.slice(0, 15), [
    { key: "token", label: "Token", render: v => `<code>${v}</code>` },
    { key: "occurrences", label: "n", render: v => `<span class="num">${v}</span>` },
    { key: "mean_weight_toward_negative", label: "Mean weight",
      render: v => `<span class="num">${Number(v).toFixed(4)}</span>` },
  ], { sortKey: "mean_weight_toward_negative" });

  renderAgreement(data.agreement);
}

/** Inline SVG scatter — no chart library, so nothing to load or break. */
function renderAgreement(points) {
  const target = $("#agreement-plot");
  if (!points.length) { target.innerHTML = '<p class="empty">Not available.</p>'; return; }

  const xs = points.map(p => p.mean_weight_toward_negative_lime);
  const ys = points.map(p => p.mean_weight_toward_negative_shap);
  const pad = 34, size = 300;
  const span = (arr) => {
    const lo = Math.min(...arr), hi = Math.max(...arr);
    const margin = (hi - lo) * 0.08 || 0.01;
    return [lo - margin, hi + margin];
  };
  const [x0, x1] = span(xs), [y0, y1] = span(ys);
  const sx = v => pad + ((v - x0) / (x1 - x0)) * (size - pad * 1.4);
  const sy = v => size - pad - ((v - y0) / (y1 - y0)) * (size - pad * 1.4);

  const dots = points.map(p => `<circle cx="${sx(p.mean_weight_toward_negative_lime).toFixed(1)}"
      cy="${sy(p.mean_weight_toward_negative_shap).toFixed(1)}" r="3.4"
      fill="var(--accent)" fill-opacity=".55"><title>${p.token}</title></circle>`).join("");

  target.innerHTML = `
    <svg viewBox="0 0 ${size} ${size}" width="100%" style="max-width:340px"
         role="img" aria-label="LIME versus SHAP token weights, Spearman 0.872">
      <line x1="${pad}" y1="${size - pad}" x2="${size - pad * 0.4}" y2="${size - pad}"
            stroke="var(--border-strong)" stroke-width="1"/>
      <line x1="${pad}" y1="${pad * 0.4}" x2="${pad}" y2="${size - pad}"
            stroke="var(--border-strong)" stroke-width="1"/>
      ${dots}
      <text x="${size / 2}" y="${size - 6}" text-anchor="middle"
            font-size="11" fill="var(--text-muted)">LIME weight</text>
      <text x="12" y="${size / 2}" text-anchor="middle" font-size="11"
            fill="var(--text-muted)" transform="rotate(-90 12 ${size / 2})">SHAP weight</text>
    </svg>`;
}

/* ------------------------------------------------------------------- boot */
const LOADERS = { results: loadResults, stress: loadStress, findings: loadFindings };

async function loadView(name) {
  const loader = LOADERS[name];
  if (!loader) return;
  try { await loader(); }
  catch (err) { console.error(`${name} failed:`, err); }
}

(async function start() {
  initTheme();
  initTabs();
  try { await initPredict(); }
  catch (err) {
    $("#predict-out").hidden = false;
    $("#predict-sentence").innerHTML =
      `<span class="empty">Could not reach the server: ${err.message}</span>`;
  }
})();

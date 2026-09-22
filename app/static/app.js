/* JargonSense — interface for the retained SE sentiment pipeline.
 *
 * Predictions and explanations come from the API, which serves the saved
 * tfidf13_svm pipeline.  Nothing is computed in the browser except layout.
 */

const $ = selector => document.querySelector(selector);

const LABELS = ["negative", "neutral", "positive"];
const MODEL = "tfidf13_svm";

const api = {
  async get(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  },
  async post(path, body) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    return data;
  },
};

const pct = value => `${(Number(value) * 100).toFixed(1)}%`;
const escapeHtml = text => String(text).replace(/[&<>"']/g, character => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]
));

/* ------------------------------------------------------------------ theme */
function initTheme() {
  const stored = (() => {
    try { return localStorage.getItem("theme"); } catch { return null; }
  })();
  if (stored === "light" || stored === "dark") {
    document.documentElement.setAttribute("data-theme", stored);
  }

  const isDark = () => {
    const explicit = document.documentElement.getAttribute("data-theme");
    return explicit
      ? explicit === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
  };
  const paint = () => {
    $("#theme-icon").textContent = isDark() ? "☾" : "☀";
    $("#theme-label").textContent = isDark() ? "Dark" : "Light";
  };

  paint();
  $("#theme-toggle").addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch { /* private mode */ }
    paint();
  });
}

/* -------------------------------------------------------------- classifier */
function renderSentence(tokens, predicted) {
  const maximum = Math.max(...tokens.map(item => Math.abs(item.weight)), 0.001);
  // Evidence for the prediction is drawn in the accent, evidence against it in
  // slate.  Deliberately not red/green: those two already mean "negative" and
  // "positive" everywhere else on the page, and reusing them here would say
  // the opposite thing whenever the prediction itself is negative.
  //
  // The strength rides on a --w custom property so the actual colour resolves
  // from the theme's variables rather than a hardcoded light-mode RGB.
  $("#predict-sentence").innerHTML = tokens.map(item => {
    const strength = Math.min(Math.abs(item.weight) / maximum, 1);
    const weight = item.weight === 0 ? 0 : (0.16 + strength * 0.62).toFixed(3);
    const side = item.weight >= 0 ? "for" : "against";
    return `<span class="word ${side}" data-tip="${item.weight.toFixed(4)}" ` +
      `style="--w:${weight}">${escapeHtml(item.token)}</span>`;
  }).join(" ");

  $("#predict-legend").innerHTML = `
    <span><i class="swatch for" style="--w:.7"></i>
      supports <b>${escapeHtml(predicted)}</b></span>
    <span><i class="swatch against" style="--w:.7"></i>
      argues against it</span>`;
}

function renderBars(probabilities, predicted) {
  $("#predict-bars").innerHTML = LABELS.map((label, index) => `
    <div class="bar-row${label === predicted ? " is-predicted" : ""}">
      <span class="bar-label">${label}</span>
      <div class="bar-track">
        <div class="bar-fill ${label}" style="width:${probabilities[index] * 100}%"></div>
      </div>
      <span class="bar-value">${pct(probabilities[index])}</span>
    </div>`).join("");
}

let expectedLabel = null;

async function classify() {
  const button = $("#predict-go");
  const text = $("#predict-text").value.trim();
  if (!text) return;
  button.disabled = true;
  $("#predict-status").innerHTML = '<i class="spinner"></i> sampling LIME …';
  try {
    const result = await api.post("/api/predict", { text, model: MODEL });
    $("#predict-out").hidden = false;
    $("#predict-verdict").textContent = result.label;
    $("#predict-verdict").className = `verdict ${result.label}`;
    renderSentence(result.tokens, result.label);
    renderBars(result.probabilities, result.label);
    $("#predict-note").textContent =
      `explaining "${result.explained_class}" · ${result.method}`;

    if (expectedLabel) {
      const hit = expectedLabel === result.label;
      $("#predict-expect").innerHTML = hit
        ? `This example is labelled <b>${escapeHtml(expectedLabel)}</b> — the model agrees.`
        : `This example is labelled <b>${escapeHtml(expectedLabel)}</b>, but the
           model said <b>${escapeHtml(result.label)}</b>.`;
    } else {
      $("#predict-expect").textContent = "";
    }
  } catch (error) {
    $("#predict-out").hidden = false;
    $("#predict-sentence").textContent = error.message;
    $("#predict-legend").innerHTML = "";
  } finally {
    button.disabled = false;
    $("#predict-status").textContent = "";
  }
}

/* ---------------------------------------------------------------- examples */
async function initExamples() {
  const { examples } = await api.get("/api/examples");
  const groups = [...new Set(examples.map(example => example.kind || "Examples"))];

  $("#examples").innerHTML = groups.map(group => {
    const cards = examples
      .map((example, index) => ({ example, index }))
      .filter(item => (item.example.kind || "Examples") === group)
      .map(({ example, index }) => {
        // Guard every optional field: a server serving an older payload must
        // render a card with something missing, never the string "undefined".
        const name = example.label || example.kind || `example ${index + 1}`;
        const gold = LABELS.includes(example.expected) ? example.expected : null;
        const tag = gold
          ? `<span class="tag ${gold} ex-gold">${gold}</span>`
          : "";
        return `
        <button class="example-card" type="button" data-index="${index}"
                aria-pressed="false">
          <span class="ex-top">
            <span class="ex-name">${escapeHtml(name)}</span>
            ${tag}
          </span>
          <span class="ex-text">${escapeHtml(example.text || "")}</span>
        </button>`;
      }).join("");
    return `<div class="example-group">
      <div class="example-group-head">${escapeHtml(group)}</div>
      <div class="example-grid">${cards}</div>
    </div>`;
  }).join("");

  document.querySelectorAll(".example-card").forEach(card => {
    card.addEventListener("click", () => {
      const example = examples[card.dataset.index];
      document.querySelectorAll(".example-card").forEach(other =>
        other.setAttribute("aria-pressed", String(other === card)));
      $("#predict-text").value = example.text;
      expectedLabel = LABELS.includes(example.expected) ? example.expected : null;
      classify();
      $("#sec-result").scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });
}

/* ------------------------------------------------------------------- boot */
async function init() {
  initTheme();

  $("#predict-go").addEventListener("click", () => {
    expectedLabel = null;
    document.querySelectorAll(".example-card")
      .forEach(card => card.setAttribute("aria-pressed", "false"));
    classify();
  });
  $("#predict-text").addEventListener("keydown", event => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      $("#predict-go").click();
    }
  });

  await initExamples();
  await classify();
}

init();

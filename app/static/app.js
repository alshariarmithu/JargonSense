const $ = selector => document.querySelector(selector);

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

function renderSentence(tokens) {
  const maximum = Math.max(...tokens.map(item => Math.abs(item.weight)), 0.001);
  $("#predict-sentence").innerHTML = tokens.map(item => {
    const strength = Math.min(Math.abs(item.weight) / maximum, 1);
    const colour = item.weight >= 0 ? "209,57,74" : "18,133,95";
    return `<span class="word" data-tip="${item.weight.toFixed(4)}" ` +
      `style="background:rgba(${colour},${0.08 + strength * 0.34})">${item.token}</span>`;
  }).join(" ");
}

function renderBars(probabilities) {
  const labels = ["negative", "neutral", "positive"];
  $("#predict-bars").innerHTML = labels.map((label, index) => `
    <div class="bar-row">
      <span class="bar-label">${label}</span>
      <div class="bar-track"><div class="bar-fill ${label}" style="width:${probabilities[index] * 100}%"></div></div>
      <span class="bar-value">${(probabilities[index] * 100).toFixed(1)}%</span>
    </div>`).join("");
}

async function classify() {
  const button = $("#predict-go");
  const text = $("#predict-text").value.trim();
  if (!text) return;
  button.disabled = true;
  button.textContent = "Working...";
  try {
    const result = await api.post("/api/predict", { text, model: "tfidf13_svm" });
    $("#predict-out").hidden = false;
    $("#predict-verdict").textContent = result.label;
    $("#predict-verdict").className = `verdict ${result.label}`;
    renderSentence(result.tokens);
    renderBars(result.probabilities);
    $("#predict-note").textContent = result.method;
  } catch (error) {
    $("#predict-out").hidden = false;
    $("#predict-sentence").textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Classify";
  }
}

async function loadSummary() {
  const results = await api.get("/api/results");
  const row = results.rows.find(item => item.model === "tfidf13_svm") || results.rows[0];
  if (row) {
    const stats = [
      ["Test macro-F1", Number(row.macro_f1).toFixed(4)],
      ["Test accuracy", Number(row.accuracy).toFixed(4)],
      ["Neutral to negative", `${(Number(row.neutral_to_negative) * 100).toFixed(1)}%`],
    ];
    $("#metrics").innerHTML = stats.map(([key, value]) =>
      `<div class="stat"><div class="k">${key}</div><div class="v">${value}</div></div>`
    ).join("");
  }

  const stress = await api.get("/api/stress");
  const item = stress.summary[0];
  if (item) {
    const columns = ["model", "false_alarm_rate", "negative_recall", "minimal_pair_accuracy", "accuracy"];
    $("#stress-table thead").innerHTML = `<tr>${columns.map(c => `<th>${c.replaceAll("_", " ")}</th>`).join("")}</tr>`;
    $("#stress-table tbody").innerHTML = `<tr>${columns.map(c => `<td>${c === "model" ? item[c] : (Number(item[c]) * 100).toFixed(1) + "%"}</td>`).join("")}</tr>`;
  }
}

async function init() {
  const examples = await api.get("/api/examples");
  $("#examples").innerHTML = examples.examples.map((example, index) =>
    `<button class="chip" data-index="${index}">${example.kind}</button>`
  ).join("");
  document.querySelectorAll("#examples .chip").forEach(button => {
    button.addEventListener("click", () => {
      $("#predict-text").value = examples.examples[button.dataset.index].text;
      classify();
    });
  });
  $("#predict-go").addEventListener("click", classify);
  await Promise.all([classify(), loadSummary()]);
}

init();

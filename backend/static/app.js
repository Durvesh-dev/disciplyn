/* Disciplyn prediction demo - vanilla JS, no framework, no build step. */

const el = (id) => document.getElementById(id);
const DEFAULT_PRESET = "impulsive";
const RING_LENGTH = 2 * Math.PI * 34;

const PROFILE_INFO = {
  impulsive: "Overtrades, chases entries and rushes back in after losses.",
  disciplined: "Follows a plan, uses stop-losses and sizes positions consistently.",
  mixed: "Mostly controlled, with a few specific weaknesses.",
  median: "The middle value of every feature in the dataset.",
};
const MODEL_INFO = {
  stop_loss: "Does the trader place stop-losses and respect them?",
  frequency: "Is the trader taking too many trades?",
  discipline: "Does the trader stick to a plan and limits?",
  profit_exit: "Does the trader book profits or hold on too long?",
  entry_timing: "Does the trader chase moves or wait for a setup?",
  leverage: "Is the trader taking on too much leverage and risk?",
  revenge: "Does the trader rush back in, larger, after a loss?",
  panic: "Does the trader exit in fear during adverse moves?",
  loss_aversion: "Does the trader hold losers and average down?",
  confirmation_bias: "Does the trader keep trading against the evidence?",
  emotional_stability: "Does behaviour stay steady after wins and losses?",
  consistent_sizing: "Are position sizes kept consistent?",
  learning: "Does the trader journal and improve over time?",
  strategic_consistency: "Does the trader stay with one approach?",
};
const UNIT_LABEL = { ratio: "ratio", count: "count", x: "×", "%": "%", minutes: "min", R: "R", "z-score": "z", "1-5 scale": "/ 5" };
const FILTERS = [["all", "All"], ["negative", "Concerns"], ["positive", "Strengths"], ["uncertain", "Uncertain"]];

let schema = null;          // /api/feature-schema payload
let featureByName = {};
let modelByName = {};
let baseline = null;        // values of the loaded preset (for change highlighting)
let currentPreset = DEFAULT_PRESET;
let activeGroup = null;
let activeFilter = "all";
let lastData = null;
let lastScore = null;
let requestId = 0;          // only the newest response is drawn

/* ------------------------------------------------------------------ utils */

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function fmt(v) {
  if (v === null || v === undefined || v === "") return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 100) return n.toFixed(0);
  if (Math.abs(n) >= 10) return n.toFixed(1);
  return n.toFixed(2);
}

const scoreClass = (s) => (s < 40 ? "red" : s < 70 ? "amber" : "green");
const verdictText = (s) => (s < 40 ? "High behavioural risk" : s < 70 ? "Mixed behaviour, room to improve" : "Healthy trading behaviour");
const kindOf = (p) => (p.uncertain ? "uncertain" : p.polarity);

function modelledTag(name) {
  const m = modelByName[name];
  return m.type === "pair" ? m.negative_tag : m.tag;
}

function showError(msg) {
  el("error").textContent = msg;
  el("error").hidden = !msg;
}

/* --------------------------------------------------------------- profiles */

function buildProfiles() {
  const host = el("profiles");
  Object.keys(schema.presets).forEach((key) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "profile";
    b.dataset.preset = key;
    b.innerHTML = `<b>${esc(schema.preset_labels[key] || key)}</b><span>${esc(PROFILE_INFO[key] || "")}</span>`;
    b.addEventListener("click", () => applyPreset(key));
    host.appendChild(b);
  });
}

function applyPreset(key) {
  currentPreset = key;
  lastScore = null;
  baseline = schema.presets[key];
  resetCoach();
  schema.features.forEach((f) => setValue(f, baseline[f.name]));
  refreshInputs();
  document.querySelectorAll(".profile").forEach((b) => {
    const on = b.dataset.preset === key;
    b.classList.toggle("active", on);
    b.setAttribute("aria-pressed", on);
  });
  return runPredict();
}

/* ------------------------------------------------------------------- form */

function buildFields() {
  const host = el("fields");
  schema.features.forEach((f) => {
    const row = document.createElement("div");
    row.className = "field";
    row.dataset.name = f.name;
    const models = f.used_by.map((m) => m.replace(/_/g, " ")).join(", ");
    row.innerHTML = `
      <div>
        <label for="in-${f.name}">${esc(f.label)}</label>
        <div class="desc">${esc(f.description)}</div>
        <div class="desc">Used by: ${esc(models)}</div>
        <div class="edited-note"></div>
      </div>
      <div>
        <div class="control">
          <input type="range" min="${f.min}" max="${f.max}" step="${f.step}" aria-label="${esc(f.label)} slider">
          <input id="in-${f.name}" type="number" step="any" ${f.optional ? 'placeholder="Blank"' : ""}>
          <span class="unit">${esc(UNIT_LABEL[f.unit] || f.unit || "")}</span>
        </div>
        <div class="note"></div>
      </div>`;
    const slider = row.querySelector("input[type=range]");
    const number = row.querySelector("input[type=number]");
    slider.addEventListener("input", () => {
      number.value = Number(Number(slider.value).toFixed(4));
      onEdit(f);
    });
    number.addEventListener("input", () => {
      if (number.value !== "") slider.value = number.value;
      onEdit(f);
    });
    host.appendChild(row);
  });
}

function fieldParts(name) {
  const row = document.querySelector(`.field[data-name="${name}"]`);
  return { row, slider: row.querySelector("input[type=range]"), number: row.querySelector("input[type=number]") };
}

function setValue(f, v) {
  const { slider, number } = fieldParts(f.name);
  const blank = v === null || v === undefined;
  number.value = blank ? "" : v;
  slider.value = blank ? f.median : v;
}

function isChanged(f) {
  const raw = fieldParts(f.name).number.value.trim();
  const base = baseline ? baseline[f.name] : null;
  const baseBlank = base === null || base === undefined;
  if (raw === "") return !baseBlank;
  return baseBlank || Math.abs(Number(raw) - base) > 1e-9;
}

function markField(f) {
  const { row, number } = fieldParts(f.name);
  const raw = number.value.trim();
  const changed = isChanged(f);
  row.classList.toggle("changed", changed);
  if (changed) {
    const base = baseline[f.name];
    row.querySelector(".edited-note").textContent =
      `Edited. Preset value: ${base === null || base === undefined ? "blank" : fmt(base)}`;
  }

  const out = raw !== "" && (Number(raw) < f.min || Number(raw) > f.max);
  row.classList.toggle("out-of-range", out);
  const note = row.querySelector(".note");
  note.classList.toggle("warn", out);
  const exact = (v) => Number(v.toFixed(3));   // 3 d.p. so 0.3165 never reads as outside "0.32"
  const range = `${exact(f.min)} to ${exact(f.max)}`;
  if (raw === "") {
    note.textContent = f.optional ? "Blank means the trader did not answer." : "Blank: treated as a missing value.";
  } else {
    note.textContent = out ? `Outside the typical range (${range})` : `Typical range ${range}, median ${exact(f.median)}`;
  }
}

function onEdit(f) {
  markField(f);
  refreshCounts();
  setStale(schema.features.some(isChanged) || lastSentDiffers());
  if (el("only-edited").checked) applyFieldFilter();
}

let lastSent = null;
function lastSentDiffers() {
  return lastSent !== null && JSON.stringify(collectFeatures()) !== lastSent;
}

function setStale(on) {
  const stale = on && lastSentDiffers();
  el("stale").hidden = !stale;
  el("cards").classList.toggle("dim", stale);
}

function refreshCounts() {
  const edited = schema.features.filter(isChanged);
  const badge = el("edited-count");
  badge.textContent = `${edited.length} edited`;
  badge.hidden = edited.length === 0;
  document.querySelectorAll(".group-btn").forEach((b) => {
    const total = schema.features.filter((f) => f.group === b.dataset.group).length;
    const n = edited.filter((f) => f.group === b.dataset.group).length;
    const small = b.querySelector("small");
    small.textContent = n ? `${n} edited` : total;
    small.classList.toggle("edited", n > 0);
  });
}

function refreshInputs() {
  schema.features.forEach(markField);
  refreshCounts();
  applyFieldFilter();
}

function buildGroupNav() {
  const host = el("group-nav");
  schema.groups.forEach((g) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "group-btn";
    b.dataset.group = g;
    b.innerHTML = `<span>${esc(g)}</span><small></small>`;
    b.addEventListener("click", () => {
      activeGroup = g;
      el("search").value = "";
      el("only-edited").checked = false;
      applyFieldFilter();
    });
    host.appendChild(b);
  });
}

function applyFieldFilter() {
  const q = el("search").value.trim().toLowerCase();
  const onlyEdited = el("only-edited").checked;
  const acrossAll = q !== "" || onlyEdited;
  let shown = 0;
  schema.features.forEach((f) => {
    const text = `${f.label} ${f.description} ${f.name} ${f.group}`.toLowerCase();
    const show = acrossAll
      ? (!q || text.includes(q)) && (!onlyEdited || isChanged(f))
      : f.group === activeGroup;
    fieldParts(f.name).row.hidden = !show;
    if (show) shown += 1;
  });
  el("no-fields").hidden = shown > 0;
  el("group-title").textContent = acrossAll ? `${shown} matching feature${shown === 1 ? "" : "s"}` : activeGroup;
  document.querySelectorAll(".group-btn").forEach((b) => {
    b.classList.toggle("active", !acrossAll && b.dataset.group === activeGroup);
  });
}

function collectFeatures() {
  const out = {};
  schema.features.forEach((f) => {
    const raw = fieldParts(f.name).number.value.trim();
    out[f.name] = raw === "" ? null : Number(raw);
  });
  return out;
}

/* ---------------------------------------------------------------- predict */

async function runPredict() {
  const id = ++requestId;
  const btn = el("btn-predict");
  btn.disabled = true;
  btn.textContent = "Predicting…";
  showError("");
  const body = collectFeatures();

  try {
    const res = await fetch("api/predict-features", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: body }),
    });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try {
        const d = (await res.json()).detail;                 // string, or FastAPI's validation list
        if (typeof d === "string") msg = d;
        else if (Array.isArray(d)) msg = d.map((x) => `${(x.loc || []).slice(-1)}: ${x.msg}`).join("; ");
      } catch (e) { /* non-JSON body */ }
      throw new Error(msg);
    }
    const data = await res.json();
    if (id !== requestId) return;
    lastSent = JSON.stringify(body);
    render(data);
    setStale(false);
  } catch (err) {
    if (id !== requestId) return;
    const offline = err instanceof TypeError;
    showError(offline
      ? "Cannot reach the backend. Check that uvicorn is still running, then press Predict again."
      : `The prediction failed: ${err.message}`);
  } finally {
    if (id === requestId) {
      btn.disabled = false;
      btn.textContent = "Predict";
    }
  }
}

/* ----------------------------------------------------------------- render */

function render(data) {
  lastData = data;
  const s = data.behaviour_score;
  const summary = el("summary");
  summary.classList.remove("is-red", "is-amber", "is-green");
  summary.classList.add("is-" + scoreClass(s));
  el("score-num").textContent = s;
  el("ring-value").style.strokeDasharray = `${(s / 100) * RING_LENGTH} ${RING_LENGTH}`;
  el("verdict-title").textContent = verdictText(s);

  const delta = lastScore === null ? 0 : s - lastScore;
  el("score-delta").textContent = delta === 0 ? "" : `${delta > 0 ? "+" : "−"}${Math.abs(delta)} from your last run (${lastScore})`;
  lastScore = s;

  const u = data.summary;
  el("score-summary").textContent = `Detected: ${u.negative} negative · ${u.positive} positive · ${u.uncertain} uncertain`;
  renderFilters();
  renderCards();
}

function renderFilters() {
  const u = lastData.summary;
  const counts = { all: lastData.predictions.length, negative: u.negative, positive: u.positive, uncertain: u.uncertain };
  const host = el("filters");
  host.innerHTML = "";
  FILTERS.forEach(([key, label]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip" + (activeFilter === key ? " active" : "");
    b.textContent = `${label} (${counts[key]})`;
    b.addEventListener("click", () => {
      activeFilter = key;
      renderFilters();
      renderCards();
    });
    host.appendChild(b);
  });
}

function renderCards() {
  const rank = { negative: 0, uncertain: 1, positive: 2 };
  const preds = lastData.predictions
    .filter((p) => activeFilter === "all" || kindOf(p) === activeFilter)
    .sort((a, b) => rank[kindOf(a)] - rank[kindOf(b)] || b.confidence - a.confidence);
  const host = el("cards");
  host.innerHTML = "";
  preds.forEach((p) => host.appendChild(buildCard(p)));
}

function buildCard(p) {
  const kind = kindOf(p);
  const meta = modelByName[p.model];
  const pct = Math.round(p.probability * 100);
  const feats = Object.entries(p.features_used).map(([name, v]) => {
    const blank = v === null || v === undefined;
    const label = featureByName[name] ? featureByName[name].label : name;
    return `<li><span>${esc(label)}</span><b class="${blank ? "blank" : ""}">${blank ? "blank" : fmt(v)}</b></li>`;
  }).join("");

  const card = document.createElement("article");
  card.className = "card " + kind;
  card.innerHTML = `
    <div class="card-name">${esc(p.display_name)}</div>
    <div class="badge ${kind}">${p.uncertain ? "Uncertain" : esc(p.predicted_tag)}</div>
    <div class="card-info">${p.uncertain ? `Too close to call. Leaning towards ${esc(p.predicted_tag)}.` : esc(MODEL_INFO[p.model] || "")}</div>
    <div class="prob-row"><span>Chance of ${esc(modelledTag(p.model))}</span><b>${pct}%</b></div>
    <div class="bar" title="Decision threshold ${p.threshold.toFixed(2)}">
      <div class="bar-fill" style="width:${pct}%"></div>
      <div class="bar-mark" style="left:${(p.threshold * 100).toFixed(1)}%"></div>
    </div>
    <div class="card-meta">Threshold ${p.threshold.toFixed(2)} · ${esc(p.algorithm)} · test accuracy ${(meta.test_accuracy * 100).toFixed(1)}%</div>
    <details>
      <summary>Features used (${Object.keys(p.features_used).length})</summary>
      <ul class="feat-list">${feats}</ul>
    </details>`;
  return card;
}

/* ------------------------------------------------------------------ coach */

const SUGGESTIONS = ["Why is my score at this level?", "What should I fix first?", "How do I stop revenge trading?"];
let chat = [];
let coachBusy = false;

function resetCoach() {
  chat = [];
  el("coach-log").innerHTML = "";
}

function markdownLite(text) {
  const inline = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/(^|\s)[*_]([^*_]+?)[*_](?=\s|[.,]|$)/g, "$1<i>$2</i>");
  let html = "";
  let inList = false;
  text.split("\n").forEach((line) => {
    const t = line.trim();
    const bullet = /^[-*]\s+/.test(t);
    if (bullet && !inList) { html += "<ul>"; inList = true; }
    if (!bullet && inList) { html += "</ul>"; inList = false; }
    if (bullet) html += `<li>${inline(t.replace(/^[-*]\s+/, ""))}</li>`;
    else if (t) html += `<p>${inline(t)}</p>`;
  });
  return html + (inList ? "</ul>" : "");
}

function addMessage(kind, html) {
  const div = document.createElement("div");
  div.className = "msg " + kind;
  div.innerHTML = html;
  el("coach-log").appendChild(div);
  return div;
}

async function askCoach(question) {
  const q = question.trim();
  if (!q || coachBusy) return;
  coachBusy = true;
  el("coach-send").disabled = true;
  el("coach-input").value = "";
  addMessage("user", esc(q));
  chat.push({ role: "user", content: q });
  const box = addMessage("assistant", "Thinking…");
  let reply = "";
  try {
    const res = await fetch("api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: collectFeatures(), messages: chat }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      reply += decoder.decode(value, { stream: true });
      box.innerHTML = markdownLite(reply);
      box.scrollIntoView({ block: "nearest" });
    }
    if (!reply.trim()) throw new Error("empty reply");
    chat.push({ role: "assistant", content: reply });
  } catch (err) {
    chat.pop();
    box.className = "msg error";
    box.textContent = "The coach could not answer. Check that the backend is running, then try again.";
  } finally {
    coachBusy = false;
    el("coach-send").disabled = false;
  }
}

function buildCoach() {
  SUGGESTIONS.forEach((q) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = q;
    b.addEventListener("click", () => askCoach(q));
    el("coach-suggest").appendChild(b);
  });
  el("coach-form").addEventListener("submit", (e) => {
    e.preventDefault();
    askCoach(el("coach-input").value);
  });
}

/* ------------------------------------------------------------------- tabs */

function showTab(name) {
  ["results", "inputs", "coach"].forEach((t) => {
    const on = t === name;
    el("tab-" + t).classList.toggle("active", on);
    el("tab-" + t).setAttribute("aria-selected", on);
    el("panel-" + t).hidden = !on;
  });
}

/* ------------------------------------------------------------------- boot */

async function init() {
  try {
    const res = await fetch("api/feature-schema");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    schema = await res.json();
  } catch (err) {
    el("verdict-title").textContent = "Backend not available";
    showError("Could not load the models. Start the backend with “uvicorn main:app --port 8000”, then reload this page.");
    return;
  }

  schema.features.forEach((f) => { featureByName[f.name] = f; });
  schema.models.forEach((m) => { modelByName[m.model] = m; });

  const mean = (k) => schema.models.reduce((a, m) => a + m[k], 0) / schema.models.length;
  el("stats").innerHTML = [
    [`${(mean("test_accuracy") * 100).toFixed(1)}%`, "mean test accuracy"],
    [mean("test_roc_auc").toFixed(3), "mean ROC-AUC"],
    [schema.n_dataset_rows.toLocaleString("en-IN"), "training windows"],
  ].map(([v, k]) => `<div><dd>${v}</dd><dt>${k}</dt></div>`).join("");

  activeGroup = schema.groups[0];
  buildProfiles();
  buildGroupNav();
  buildFields();
  buildCoach();

  el("btn-predict").addEventListener("click", runPredict);
  el("btn-reset").addEventListener("click", () => applyPreset(currentPreset));
  el("tab-results").addEventListener("click", () => showTab("results"));
  el("tab-inputs").addEventListener("click", () => showTab("inputs"));
  el("tab-coach").addEventListener("click", () => showTab("coach"));
  el("search").addEventListener("input", applyFieldFilter);
  el("only-edited").addEventListener("change", applyFieldFilter);

  // deep links for the presentation: ?preset=disciplined&tab=inputs
  const params = new URLSearchParams(location.search);
  if (["inputs", "coach"].includes(params.get("tab"))) showTab(params.get("tab"));
  const asked = params.get("preset");
  applyPreset(asked && schema.presets[asked] ? asked : DEFAULT_PRESET);
}

init();

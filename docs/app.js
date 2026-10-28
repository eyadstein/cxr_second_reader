const checklistKey = "cxr-second-reader-checklist-v1";
const boxes = [...document.querySelectorAll("input[data-task]")];
const resetButton = document.querySelector("#reset-checklist");

function loadChecklist() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(checklistKey) || "{}"); } catch { saved = {}; }
  for (const box of boxes) box.checked = Boolean(saved[box.dataset.task]);
}
function saveChecklist() {
  const saved = Object.fromEntries(boxes.map((box) => [box.dataset.task, box.checked]));
  localStorage.setItem(checklistKey, JSON.stringify(saved));
}
for (const box of boxes) box.addEventListener("change", saveChecklist);
resetButton.addEventListener("click", () => {
  for (const box of boxes) box.checked = false;
  saveChecklist();
});
loadChecklist();

const workflowRoot = document.querySelector("#workflow-list");
const repo = "eyadstein/cxr_second_reader";
const stateText = (run) => {
  if (run.status !== "completed") return "Running";
  return run.conclusion === "success" ? "Passed" : (run.conclusion || "Finished");
};
const stateClass = (run) => run.status !== "completed" ? "" : (run.conclusion === "success" ? "success" : "failure");
async function showWorkflowRuns() {
  try {
    const response = await fetch(`https://api.github.com/repos/${repo}/actions/runs?per_page=6`, {
      headers: { Accept: "application/vnd.github+json" }, cache: "no-store",
    });
    if (!response.ok) throw new Error("GitHub workflow status is temporarily unavailable.");
    const data = await response.json();
    workflowRoot.replaceChildren();
    for (const run of (data.workflow_runs || []).slice(0, 6)) {
      const link = document.createElement("a");
      link.className = "workflow-row";
      link.href = run.html_url;
      link.target = "_blank";
      link.rel = "noopener";
      const left = document.createElement("span");
      const title = document.createElement("b");
      title.textContent = run.name;
      const details = document.createElement("small");
      details.textContent = `${run.head_branch || "main"} · ${new Date(run.updated_at).toLocaleString()}`;
      left.append(title, details);
      const state = document.createElement("span");
      state.className = `run-state ${stateClass(run)}`;
      state.textContent = stateText(run);
      link.append(left, state);
      workflowRoot.append(link);
    }
    if (!workflowRoot.childElementCount) workflowRoot.textContent = "No workflow runs are available yet.";
  } catch (error) {
    workflowRoot.textContent = error instanceof Error ? error.message : "Workflow status is unavailable.";
  }
}
showWorkflowRuns();

// Case review uses the local Docker API. Files are held in memory only.
const apiUrlInput = document.querySelector("#api-url");
const apiStatus = document.querySelector("#api-status");
const caseForm = document.querySelector("#case-form");
const imageInput = document.querySelector("#case-image");
const notesInput = document.querySelector("#doctor-notes");
const previewWrap = document.querySelector("#image-preview-wrap");
const previewImage = document.querySelector("#image-preview");
const resultsRoot = document.querySelector("#case-results");
const analysisStatus = document.querySelector("#analysis-status");
const analyzeButton = document.querySelector("#analyze-button");
let caseState = null;
let previewUrl = null;
let chatHistory = [];

function apiBase() { return apiUrlInput.value.trim().replace(/\/$/, ""); }
function status(message, kind = "") {
  apiStatus.textContent = message;
  apiStatus.className = kind;
}
async function checkApi() {
  status("Connecting…");
  try {
    const response = await fetch(`${apiBase()}/health`, { cache: "no-store" });
    const health = await response.json();
    if (!response.ok || health.status !== "ready") throw new Error(health.detail || `Model service: ${health.status || response.status}`);
    status("Model ready", "success");
  } catch (error) {
    status(error instanceof Error ? error.message : "Could not reach local API", "failure");
  }
}
document.querySelector("#check-api").addEventListener("click", checkApi);
apiUrlInput.addEventListener("change", () => status("Not checked"));
if (["localhost", "127.0.0.1"].includes(location.hostname)) {
  apiUrlInput.value = location.origin;
  checkApi();
}
imageInput.addEventListener("change", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const file = imageInput.files?.[0];
  if (!file) { previewWrap.hidden = true; return; }
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewWrap.hidden = false;
});
document.querySelector("#remove-image").addEventListener("click", () => {
  imageInput.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  previewWrap.hidden = true;
});

function appendChat(role, text) {
  const item = document.createElement("article");
  item.className = `chat-message ${role}`;
  const label = document.createElement("b");
  label.textContent = role === "assistant" ? "Case assistant" : "You";
  const body = document.createElement("p");
  body.textContent = text;
  item.append(label, body);
  document.querySelector("#chat-thread").append(item);
  item.scrollIntoView({ block: "nearest", behavior: "smooth" });
}
function renderComparison(comparison) {
  document.querySelector("#comparison-status").textContent = comparison.status;
  document.querySelector("#comparison-limit").textContent = comparison.limitation;
  const output = document.querySelector("#comparison-output");
  output.replaceChildren();
  const differences = comparison.differences || [];
  if (!differences.length) {
    const empty = document.createElement("p");
    empty.textContent = "No differing mentions were detected by the limited term check.";
    output.append(empty);
    return;
  }
  for (const item of differences) {
    const row = document.createElement("div");
    row.className = "difference-row";
    const title = document.createElement("b");
    title.textContent = item.concept;
    const desc = document.createElement("span");
    desc.textContent = `Doctor: ${item.doctor} · Model: ${item.model}`;
    row.append(title, desc);
    output.append(row);
  }
}
caseForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files?.[0];
  if (!file) return;
  analyzeButton.disabled = true;
  analysisStatus.textContent = "Generating an image-only draft… this may take a few minutes.";
  resultsRoot.hidden = true;
  try {
    const form = new FormData();
    form.append("image", file, file.name);
    form.append("doctor_notes", notesInput.value);
    form.append("max_new_tokens", document.querySelector("#max-tokens").value);
    const response = await fetch(`${apiBase()}/v1/analyze`, { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    caseState = { file, doctorNotes: notesInput.value, draft: data.draft_report, comparison: data.comparison };
    document.querySelector("#draft-output").textContent = data.draft_report || "The model returned an empty draft.";
    renderComparison(data.comparison);
    document.querySelector("#chat-thread").replaceChildren();
    chatHistory = [];
    resultsRoot.hidden = false;
    analysisStatus.textContent = "Case review is ready.";
  } catch (error) {
    analysisStatus.textContent = error instanceof Error ? error.message : "Analysis failed.";
  } finally {
    analyzeButton.disabled = false;
  }
});
document.querySelectorAll("[data-copy]").forEach((button) => button.addEventListener("click", async () => {
  const text = document.querySelector(`#${button.dataset.copy}`).textContent;
  try { await navigator.clipboard.writeText(text); button.textContent = "Copied"; setTimeout(() => { button.textContent = "Copy"; }, 1200); } catch { button.textContent = "Copy unavailable"; }
}));
document.querySelector("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!caseState) return;
  const questionInput = document.querySelector("#chat-question");
  const question = questionInput.value.trim();
  if (!question) return;
  appendChat("user", question);
  questionInput.value = "";
  const submit = event.currentTarget.querySelector("button");
  submit.disabled = true;
  try {
    const form = new FormData();
    form.append("image", caseState.file, caseState.file.name);
    form.append("doctor_notes", caseState.doctorNotes);
    form.append("draft_report", caseState.draft);
    form.append("comparison_json", JSON.stringify(caseState.comparison));
    form.append("question", question);
    form.append("chat_history", JSON.stringify(chatHistory.slice(-8)));
    const response = await fetch(`${apiBase()}/v1/chat`, { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    const answer = data.answer || "The model did not return an answer.";
    chatHistory.push({ role: "user", content: question }, { role: "assistant", content: answer });
    chatHistory = chatHistory.slice(-8);
    appendChat("assistant", answer);
  } catch (error) {
    appendChat("assistant", error instanceof Error ? `Could not answer: ${error.message}` : "Could not reach the local model.");
  } finally { submit.disabled = false; }
});

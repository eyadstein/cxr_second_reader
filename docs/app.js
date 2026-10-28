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

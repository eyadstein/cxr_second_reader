const input = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const uploadContent = document.querySelector("#upload-content");
const previewCard = document.querySelector("#preview-card");
const imagePreview = document.querySelector("#image-preview");
const fileName = document.querySelector("#file-name");
const fileSize = document.querySelector("#file-size");
const removeButton = document.querySelector("#remove-image");
const generateButton = document.querySelector("#generate-button");
const tokenSlider = document.querySelector("#token-count");
const tokenValue = document.querySelector("#token-value");
const formMessage = document.querySelector("#form-message");
const resultEmpty = document.querySelector("#result-empty");
const resultLoading = document.querySelector("#result-loading");
const result = document.querySelector("#report-result");
const resultError = document.querySelector("#result-error");
const errorDetail = document.querySelector("#error-detail");
const reportText = document.querySelector("#report-text");
const reportModel = document.querySelector("#report-model");
const resultActions = document.querySelector("#result-actions");
const statusPill = document.querySelector("#service-status");
const statusText = document.querySelector("#service-status-text");

let selectedFile = null;
let previewUrl = null;
let latestDraft = "";

function readableSize(bytes) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(0)} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showOnly(view) {
  resultEmpty.classList.toggle("hidden", view !== "empty");
  resultLoading.classList.toggle("hidden", view !== "loading");
  result.classList.toggle("hidden", view !== "result");
  resultError.classList.toggle("hidden", view !== "error");
  resultActions.classList.toggle("hidden", view !== "result");
}

function selectFile(file) {
  formMessage.textContent = "";
  if (!file) return;
  if (!["image/png", "image/jpeg"].includes(file.type)) {
    formMessage.textContent = "Choose a PNG or JPEG image.";
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    formMessage.textContent = "This image is larger than 20 MB. Choose a smaller image.";
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  fileName.textContent = file.name;
  fileSize.textContent = readableSize(file.size);
  dropZone.classList.add("hidden");
  uploadContent.classList.add("hidden");
  previewCard.classList.remove("hidden");
  generateButton.disabled = !statusPill.classList.contains("status-ready");
  showOnly("empty");
  latestDraft = "";
}

function clearFile() {
  selectedFile = null;
  input.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  imagePreview.removeAttribute("src");
  previewCard.classList.add("hidden");
  dropZone.classList.remove("hidden");
  uploadContent.classList.remove("hidden");
  generateButton.disabled = true;
  formMessage.textContent = "";
}

dropZone.addEventListener("click", () => input.click());
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    input.click();
  }
});
input.addEventListener("change", () => selectFile(input.files?.[0]));
removeButton.addEventListener("click", clearFile);
for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
  });
}
dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer?.files?.[0]));

tokenSlider.addEventListener("input", () => {
  tokenValue.textContent = `${tokenSlider.value} tokens`;
});

async function refreshServiceStatus() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    const health = await response.json();
    statusPill.classList.remove("status-loading", "status-ready", "status-error");
    if (health.status === "ready") {
      statusPill.classList.add("status-ready");
      statusText.textContent = `Ready · ${health.device || "GPU"}`;
      generateButton.disabled = !selectedFile;
    } else if (health.status === "error") {
      statusPill.classList.add("status-error");
      statusText.textContent = "Model unavailable";
      if (health.detail) formMessage.textContent = "The model could not start. Check the service logs.";
    } else {
      statusPill.classList.add("status-loading");
      statusText.textContent = "Loading model…";
      generateButton.disabled = true;
    }
  } catch {
    statusPill.classList.remove("status-ready", "status-error");
    statusPill.classList.add("status-loading");
    statusText.textContent = "Service reconnecting…";
    generateButton.disabled = true;
  }
}

generateButton.addEventListener("click", async () => {
  if (!selectedFile) {
    formMessage.textContent = "Choose an image first.";
    return;
  }
  generateButton.disabled = true;
  formMessage.textContent = "";
  showOnly("loading");
  const data = new FormData();
  data.append("image", selectedFile, selectedFile.name);
  data.append("max_new_tokens", tokenSlider.value);
  try {
    const response = await fetch("/v1/generate", { method: "POST", body: data });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof payload.detail === "string" ? payload.detail : "The service returned an unexpected error.";
      throw new Error(detail);
    }
    latestDraft = String(payload.draft_report || "").trim();
    if (!latestDraft) throw new Error("The model returned an empty draft. Try once more.");
    reportText.textContent = latestDraft;
    reportModel.textContent = payload.model_id || "Qwen3-VL";
    showOnly("result");
    reportText.focus();
  } catch (error) {
    errorDetail.textContent = error instanceof Error ? error.message : "Please try again.";
    showOnly("error");
  } finally {
    generateButton.disabled = !selectedFile || !statusPill.classList.contains("status-ready");
  }
});

document.querySelector("#copy-button").addEventListener("click", async (event) => {
  try {
    await navigator.clipboard.writeText(latestDraft);
    event.currentTarget.textContent = "Copied";
    window.setTimeout(() => { event.currentTarget.textContent = "Copy"; }, 1400);
  } catch {
    formMessage.textContent = "Clipboard access was blocked by the browser.";
  }
});

document.querySelector("#download-button").addEventListener("click", () => {
  const blob = new Blob([latestDraft], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "cxr-research-draft.txt";
  link.click();
  URL.revokeObjectURL(url);
});

refreshServiceStatus();
window.setInterval(refreshServiceStatus, 15000);

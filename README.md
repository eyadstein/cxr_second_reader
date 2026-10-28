# CXR Second Reader

An educational research prototype for comparing chest X-ray report-generation
and retrieval workflows. **Not for clinical use or patient-care decisions.**
Generated text is an experimental draft and requires qualified clinician review.

## Project status

| Area | Status | Location |
|---|---|---|
| Indiana metadata preparation and study-level splits | Implemented; 6,685 image rows, 3,440 studies, no study overlap across splits in the reported run | `notebooks/09_qwen3vl_cxr.ipynb` |
| Qwen3-VL 2B zero-shot baseline | An earlier exploratory run generated 24 reports; those study IDs are treated as exposed and excluded from the new final test set | Drive results copied to `qwen3vl_cxr/` |
| Qwen3-VL QLoRA | Latest supplied archive records step 920 (about 230 optimizer updates at accumulation 4) and includes adapter, optimizer, and loss history; this is a checkpoint count, not evidence of convergence or quality | `qwen3vl_cxr/`, `notebooks/09_qwen3vl_cxr.ipynb` |
| FastAPI inference service | Implemented with a research-only browser UI; expects a CUDA GPU and saved adapter | `api/` |
| Docker packaging and local API | Rebuilt and started; UI endpoints return HTTP 200, Qwen reached ready on RTX 3050, synthetic-image inference returned HTTP 200 | `Dockerfile`, `compose.yaml`, `api/` |
| Constant-normal/random and text retrieval controls | Implemented | `notebooks/00_baselines.ipynb`, `notebooks/05_approach4_retrieval.ipynb` |
| DenseNet-attention LSTM and ViT-GPT2 report baselines | Runnable code added; GPU training has not yet been executed in this environment | `notebooks/02`–`03` |
| Qwen3-VL QLoRA artifact review | Reads saved evaluation files; training remains centralized in notebook 09 | `notebooks/04`, `09_qwen3vl_cxr.ipynb` |
| Image/indication fusion and Grad-CAM review | Runnable exploratory workflows added; GPU runs and validation remain | `notebooks/06`–`07` |
| RAG-style research context | Retrieves training-report examples; external knowledge ingestion and generated answers remain future work | `notebooks/08_rag_assistant.ipynb` |

## Observed run and evaluation limits

An earlier saved exploratory evaluation recorded these results on 24 studies:

| Stage | ROUGE-L | BLEU | Studies |
|---|---:|---:|---:|
| Qwen zero-shot | 0.1879 | 0.0266 | 24 |
| Qwen QLoRA | 0.3312 | 0.0700 | 24 |

The later archive supplied on 2026-10-28 records 920 training micro-batches
(about 230 optimizer updates at gradient accumulation 4). The previous scores
were copied from the earlier 24-case run, not computed at step 920, so they
must not be presented as the newer checkpoint's result. The maintained Qwen
notebook now uses validation for development, checkpoints after each optimizer
update, and includes a paired final test stage that excludes previously exposed
test studies. Run that final stage once after model selection is complete.

ROUGE and BLEU measure text overlap, not clinical correctness. The sample
generated report included unsupported rib/sternum observations and missed a
small calcified granuloma. The model is not validated for diagnosis, triage, or
patient care. The test set is small and should remain untouched for final
reporting; use validation data for tuning and have qualified reviewers assess
hallucinations, omissions, and clinically meaningful errors. The repository's
keyword metrics are weak heuristics, not clinical F1 or a validated
hallucination rate. This project is a research prototype and is not
production-ready for patient care.

## Project history: problems encountered and resolutions

This section records the issues encountered during the work so they are not
mistaken for model findings or repeated unnecessarily.

1. **Gated MedGemma download returned HTTP 403.** A Hugging Face login and a
   token do not grant access to a gated repository unless that account has been
   approved and accepted the model terms. The notebook was moved to public
   `Qwen/Qwen3-VL-2B-Instruct`; the maintained workflow no longer depends on
   MedGemma access.
2. **A Colab secret was missing, followed by a second 403.** `HF_TOKEN` had not
   been created in Colab Secrets. After creating it, the account authenticated,
   but MedGemma file access still returned 403. Do not paste tokens into
   notebooks, source control, or chat.
3. **The VS Code local kernel had no GPU.** The notebook was first attached to
   local `Python 3 (ipykernel)`, so `torch.cuda.is_available()` was false. The
   model workflow needs a Colab hosted GPU or a local CUDA-capable machine; a
   local kernel does not inherit Colab's T4. The VS Code Colab extension also
   reported that too many servers were assigned, requiring an old server to be
   removed before another connection could be made.
4. **Colab GPU allocator/NVML failures and out-of-memory errors.** Restarting
   the runtime and setting allocator environment variables alone did not prove
   that the hosted driver issue was fixed. The original image token budget also
   caused an allocation request that exceeded available T4 memory. The
   maintained notebook limits image pixels and output length, checks the GPU
   before setup, and saves artifacts to Drive so a runtime restart does not
   discard completed results. A remote Colab driver failure still requires a
   healthy hosted runtime; source code cannot repair the service itself.
5. **A saved split manifest missed seven new study IDs.** Rebuilding the split
   would have changed existing assignments. The split logic now retains old
   study assignments and deterministically adds new IDs, with checks that no
   study appears across multiple splits.
6. **A generated draft contained unsupported anatomy and missed a finding.**
   Text-overlap metrics do not reveal this reliably. The prompt was tightened,
   image resolution and generated length were reduced to control memory, and
   the output remains explicitly research-only pending a structured error
   review.
7. **A Transformers image-processor deprecation warning appeared.** The
   processor configuration was changed to use the `size` mapping rather than
   the deprecated `min_pixels`/`max_pixels` call arguments. That warning was
   not itself a failed evaluation.
8. **A Colab kernel restarted during a later run.** The attached Jupyter logs
   contain a restart event but no traceback that identifies the underlying
   trigger. The debugger/frozen-module, websocket ping, and TensorFlow CPU
   feature messages are warnings, not the cause shown by the logs. The attached
   notebook records state 0 and a validation run before its training loop. The
   local copy now restores the supplied step-920 checkpoint if Drive is behind,
   trains before validation, uses a smaller image-token budget, and saves after
   every optimizer update. Neither restart has a preceding traceback in the
   supplied logs, so the trigger remains unconfirmed. These changes lower memory
   and lost work but cannot prevent a Colab quota or infrastructure reset.

## Recommended next stages

1. **Review saved examples.** Inspect `qwen3vl_cxr/eval_qwen_zero_shot.csv` and
   `qwen3vl_cxr/eval_qwen_lora_finetuned.csv`; label unsupported claims,
   omitted reference findings, and section/format problems. Do not use the test
   set to make repeated training choices.
2. **Use validation for model development.** Increase training in resumable
   chunks only after checking the data and loss, and compare checkpoints on the
   validation set. Record the exact configuration and retain the current 24
   case output as a small preliminary baseline, not the final study.
3. **Use the local API demo.** Follow `api/README.md`. The rebuilt local
   service reached ready state on an RTX 3050 and passed a synthetic-image
   inference smoke check. Review outputs as research drafts only.
4. **Complete research evaluation.** Increase held-out evaluation coverage,
   report confidence intervals where appropriate, perform blinded expert review,
   and document subgroup and failure analysis before making broader claims.
5. **Run the experimental notebooks in small stages.** Notebooks 02–03 have
   GPU-dependent report training; 06–07 have exploratory weak-label
   classification and review code. They have not yet been run here and have no
   verified results. Notebook 08 currently retrieves training-report examples
   only; external source ingestion and grounded answer generation remain future
   work. Compare validation results during model selection.

## Run locally in VS Code/Jupyter (no Colab required)

Use Python 3.11 in a dedicated environment. Install CUDA-enabled PyTorch for
Windows from the official PyTorch selector first, then install the project
notebook dependencies and register the kernel. The project does not install or
replace PyTorch automatically because the correct CUDA build depends on the
machine.

In a PowerShell terminal from the repository root:

```powershell
py -3.11 -m venv .venv-jupyter
.\.venv-jupyter\Scripts\python.exe -m pip install --upgrade pip
# Install the CUDA-enabled torch/torchvision build using https://pytorch.org/get-started/locally/
.\.venv-jupyter\Scripts\python.exe -m pip install -r requirements.txt jupyterlab ipykernel
.\.venv-jupyter\Scripts\python.exe -m ipykernel install --user --name cxr-second-reader --display-name "Python (CXR Second Reader)"
$env:CXR_SHARED_DIR = (Join-Path (Get-Location) "cxr_shared")
$env:CXR_DATASET_ROOT = "D:\path\to\existing\indiana_dataset"
$env:QWEN_SAVE_DIR = (Join-Path (Get-Location) "qwen3vl_cxr")
.\.venv-jupyter\Scripts\python.exe -m jupyter lab
```

Replace `D:\path\to\existing\indiana_dataset` with the folder that
already contains the extracted Indiana images and metadata. The notebook will
stop with a clear message if the dataset path is missing. Automatic dataset
downloads are disabled unless you explicitly set
`$env:CXR_ALLOW_DATASET_DOWNLOAD = "1"`. In VS Code, select the
**Python (CXR Second Reader)** kernel. Run each notebook by stage; avoid **Run
All** for GPU training workflows.

The Qwen notebook checks CUDA before model setup. This computer's RTX 3050 has
4 GB VRAM, which is enough for the local 4-bit inference service that was
verified, but may not fit Qwen vision-language LoRA training. Training also
needs substantial free disk space for dependencies, the base model, cached
weights, dataset, and checkpoints. Do not start a long run until CUDA is
visible, the data path resolves, and sufficient disk space is available.

Notebooks 02–03 now include a final held-out test cell to run once after
validation-based model selection. Notebook 05 queries retrieval by indication
rather than by the target report. Qwen's final paired evaluation excludes
studies already exposed in earlier preview runs and is only for final reporting.
It cannot undo information already observed in prior experiments.

## Run the FastAPI and Docker service

The API serves the saved adapter from the locally mounted
`qwen3vl_cxr/qwen3vl-cxr-lora/` folder and includes a responsive research UI
at `/`. The model adapter and dataset are ignored by Git; the Docker build
contains only application code. It downloads the public base model
on first start; the Hugging Face cache is kept in a Docker volume. A CUDA GPU
is required by default. From the project root run:

```powershell
docker compose up --build
```

The service binds to `127.0.0.1:8000`. Browse to
<http://127.0.0.1:8000> for the UI or <http://127.0.0.1:8000/docs> for API
docs; use `/health` for readiness and `POST
/v1/generate` for PNG/JPEG report drafts. See `api/README.md` for the request
example and configuration. Do not expose this prototype publicly or send
identifiable patient data to it. Public deployment requires a specified GPU
host plus HTTPS, authentication, and privacy controls; no public target has
been configured yet.

## Data and shared utilities

The maintained notebook retrieves the public Indiana dataset through KaggleHub
in its notebook runtime and stores split tables and metrics in Google Drive.
The dataset summary provided during this run was 6,685 image rows with columns
`uid`, `MeSH`, `Problems`, `indication`, `filename`, `projection`,
`processed_report`, `class`, and `split`. The reported split summary had 5,359
training, 659 validation, and 667 test image rows; no study crossed splits and
no reports were blank.

- `src/config.py`: configurable shared Drive path and dataset identifiers.
- `src/data.py`: normalized CSV loader, study split selection, and lazy image
  resolver.
- `src/evaluate.py`: text overlap metrics, exploratory keyword metrics,
  baselines, and result logging.
- `api/main.py`: GPU-backed FastAPI inference for the saved QLoRA adapter.
- `notebooks/archive/`: source notebook snapshots retained for reference.

## Reproducibility and GitHub

Run notebooks from this project checkout when possible. Colab can use GitHub as
a fallback if it cannot see the checkout or Drive copy. Colab supplies its own
CUDA-matched PyTorch build, so the notebook `requirements.txt` intentionally
does not replace Torch. The API uses the separate `api/requirements.txt` and
GPU Docker image.

The local GitHub remote is configured, but changes in this working copy are not
published automatically. `qwen3vl_cxr/` is ignored by Git because it contains
model weights, optimizer state, and evaluation outputs; keep a backup and share
those artifacts separately when needed.

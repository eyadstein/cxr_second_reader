"""FastAPI inference service for the experimental Qwen3-VL CXR adapter.

This service is for research demos only. It is not a diagnostic device and must
not be used to make patient-care decisions.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("cxr_api")

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", "qwen3vl_cxr/qwen3vl-cxr-lora"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "200704"))
MAX_NEW_TOKENS_DEFAULT = int(os.getenv("MAX_NEW_TOKENS_DEFAULT", "96"))
MAX_NEW_TOKENS_LIMIT = int(os.getenv("MAX_NEW_TOKENS_LIMIT", "160"))
PROMPT_TEXT = (
    "Describe only visible chest X-ray findings. Write a concise draft with "
    "Findings and Impression sections. Do not invent findings. If a finding "
    "cannot be determined from the image, say so rather than guessing. "
    "This is an experimental draft for qualified human review."
)

_model: Any | None = None
_processor: Any | None = None
_model_lock = threading.Lock()
_device_name = "not loaded"
_model_loading = False
_load_error: str | None = None


def _load_model() -> None:
    """Load the base VLM and the locally saved LoRA adapter once at startup."""
    global _model, _processor, _device_name

    import torch
    from peft import PeftModel
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
    )

    if not ADAPTER_DIR.is_dir():
        raise FileNotFoundError(
            f"Adapter directory not found: {ADAPTER_DIR}. Copy the saved "
            "qwen3vl-cxr-lora folder into the project or set ADAPTER_DIR."
        )
    if not torch.cuda.is_available() and os.getenv("ALLOW_CPU_INFERENCE", "0") != "1":
        raise RuntimeError(
            "No CUDA GPU is available. Start the container with NVIDIA GPU "
            "support, or explicitly set ALLOW_CPU_INFERENCE=1 (slow and memory intensive)."
        )

    if torch.cuda.is_available():
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quantization,
            torch_dtype=compute_dtype,
            device_map="auto",
            attn_implementation=os.getenv("ATTN_IMPLEMENTATION", "sdpa"),
        )
        # Keep host-specific GPU details out of the public UI/health response.
        _device_name = "GPU accelerated"
    else:
        compute_dtype = torch.float32
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=compute_dtype,
            device_map="cpu",
            attn_implementation="sdpa",
        )
        _device_name = "CPU (explicitly enabled)"

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        size={
            "shortest_edge": 64 * 28 * 28,
            "longest_edge": MAX_IMAGE_PIXELS,
        },
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model.eval()
    _processor = processor
    _model = model
    logger.info("Loaded %s with adapter %s on %s", MODEL_ID, ADAPTER_DIR, _device_name)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Serve the UI immediately while the large model loads in the background."""
    global _model_loading, _load_error
    _model_loading = True

    def load_model_safely() -> None:
        global _model_loading, _load_error
        try:
            _load_model()
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Model startup failed")
        finally:
            _model_loading = False

    loader = threading.Thread(
        target=load_model_safely, name="qwen-model-loader", daemon=True
    )
    loader.start()
    try:
        yield
    finally:
        loader.join(timeout=5)


app = FastAPI(
    title="CXR Second Reader Research API",
    description=(
        "Experimental chest X-ray report drafting. Not for diagnosis or "
        "patient-care decisions; qualified clinician review is required."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://eyadstein.github.io",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
STATIC_DIR = Path(__file__).resolve().parent / "static"
SITE_DIR = Path(__file__).resolve().parents[1] / "docs"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(SITE_DIR / "index.html")


@app.get("/styles.css")
def site_styles() -> FileResponse:
    return FileResponse(SITE_DIR / "styles.css", media_type="text/css")


@app.get("/app.js")
def site_script() -> FileResponse:
    return FileResponse(SITE_DIR / "app.js", media_type="text/javascript")


@app.middleware("http")
async def allow_private_network_preflight(request, call_next):
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network", "").lower() == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


@app.get("/api-info")
def api_info() -> dict[str, str]:
    return {
        "service": "CXR Second Reader Research API",
        "docs": "/docs",
        "warning": "Research prototype only; not for clinical use.",
    }


@app.get("/health")
def health(response: Response) -> dict[str, Any]:
    ready = _model is not None and _processor is not None
    if not ready:
        response.status_code = 503
    return {
        "status": "ready" if ready else ("error" if _load_error else "loading"),
        "model_id": MODEL_ID,
        "adapter_available": ADAPTER_DIR.is_dir(),
        "device": _device_name,
        "detail": "Model startup failed; check the container logs." if _load_error else None,
    }


def _ensure_model_ready() -> None:
    if _model is None or _processor is None:
        if _load_error:
            raise HTTPException(status_code=503, detail="The model failed to load. Check service logs and /health.")
        raise HTTPException(status_code=503, detail="The model is still loading. Wait for /health to report ready.")


def _load_rgb(image: UploadFile) -> Image.Image:
    payload = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MiB upload limit.")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format not in {"PNG", "JPEG"}:
                raise HTTPException(status_code=415, detail="Upload a PNG or JPEG image.")
            if source.width * source.height > 40_000_000:
                raise HTTPException(status_code=413, detail="Image dimensions are too large.")
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=415, detail="The uploaded file is not a valid image.") from exc


def _generate(rgb: Image.Image, prompt_text: str, max_new_tokens: int) -> str:
    _ensure_model_ready()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": rgb},
        {"type": "text", "text": prompt_text},
    ]}]
    import torch
    with _model_lock, torch.inference_mode():
        inputs = _processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = {key: value.to(_model.device) for key, value in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(next(_model.parameters()).dtype)
        input_length = inputs["input_ids"].shape[-1]
        output = _model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        )
    return _processor.decode(output[0, input_length:], skip_special_tokens=True).strip()


@app.post("/v1/generate")
def generate(
    image: UploadFile = File(..., description="PNG or JPEG chest X-ray image"),
    max_new_tokens: int = Form(
        MAX_NEW_TOKENS_DEFAULT,
        ge=16,
        le=MAX_NEW_TOKENS_LIMIT,
        description="Maximum generated tokens (bounded to limit GPU memory use).",
    ),
) -> dict[str, Any]:
    rgb = _load_rgb(image)
    report = _generate(rgb, PROMPT_TEXT, max_new_tokens)
    return {
        "draft_report": report,
        "model_id": MODEL_ID,
        "adapter": ADAPTER_DIR.name,
        "device": _device_name,
        "warning": "Experimental draft only. Review image and report; not for clinical use.",
    }


@app.post("/v1/analyze")
def analyze(
    image: UploadFile = File(..., description="PNG or JPEG chest X-ray image"),
    doctor_notes: str = Form(..., min_length=1, max_length=10000),
    max_new_tokens: int = Form(MAX_NEW_TOKENS_DEFAULT, ge=16, le=MAX_NEW_TOKENS_LIMIT),
) -> dict[str, Any]:
    """Draft from image alone, then show transparent term-level comparison."""
    from api.review import compare_reports
    rgb = _load_rgb(image)
    # Deliberately do not provide doctor notes to the image model: this avoids
    # anchoring its draft to the report it is meant to compare against.
    draft = _generate(rgb, PROMPT_TEXT, max_new_tokens)
    comparison = compare_reports(doctor_notes, draft)
    return {
        "draft_report": draft,
        "comparison": comparison,
        "model_id": MODEL_ID,
        "adapter": ADAPTER_DIR.name,
        "warning": "Research prototype. Text comparison is heuristic, not an independent medical model or correctness judgment.",
    }


@app.post("/v1/chat")
def chat(
    image: UploadFile = File(...),
    doctor_notes: str = Form(..., max_length=10000),
    draft_report: str = Form(..., max_length=10000),
    comparison_json: str = Form(..., max_length=10000),
    question: str = Form(..., min_length=1, max_length=2000),
    chat_history: str = Form("[]", max_length=12000),
    max_new_tokens: int = Form(128, ge=16, le=MAX_NEW_TOKENS_LIMIT),
) -> dict[str, Any]:
    """Answer using only the image and report-pair context for this case."""
    import json
    from api.review import retrieve_case_context
    rgb = _load_rgb(image)
    try:
        comparison = json.loads(comparison_json)
        if not isinstance(comparison, dict):
            raise ValueError("comparison must be an object")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid comparison data.") from exc
    context = retrieve_case_context(question, doctor_notes, draft_report, comparison)
    context_text = "\n".join(f"- {line}" for line in context) or "No matching report sentences were found."
    try:
        history = json.loads(chat_history)
        if not isinstance(history, list) or any(
            not isinstance(turn, dict)
            or turn.get("role") not in {"user", "assistant"}
            or not isinstance(turn.get("content"), str)
            for turn in history[-8:]
        ):
            raise ValueError("chat history must be a list of user/assistant turns")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid chat history.") from exc
    history_text = "\n".join(
        f"{turn['role'].title()}: {turn['content'][:1000]}" for turn in history[-8:]
    )
    prompt = (
        "You are a research-demo assistant discussing this one chest X-ray case. "
        "Use the image and retrieved report context below. Explain agreements or differences plainly. "
        "Do not declare a report correct, make a diagnosis, or recommend treatment. "
        "If evidence is insufficient, say so and defer to a qualified radiologist.\n\n"
        f"Doctor note:\n{doctor_notes}\n\nModel draft:\n{draft_report}\n\n"
        f"Retrieved case context:\n{context_text}\n\n"
        f"Recent chat:\n{history_text or 'No earlier turns.'}\n\nQuestion: {question}"
    )
    answer = _generate(rgb, prompt, max_new_tokens)
    return {
        "answer": answer,
        "retrieved_context": context,
        "warning": "Case-grounded research chat only; not clinical advice or an independent correctness check.",
    }

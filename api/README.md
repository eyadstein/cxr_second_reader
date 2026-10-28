# CXR Second Reader API

This API serves the saved Qwen3-VL QLoRA adapter for a local research demo. It
does not diagnose disease and must not be used for patient-care decisions.

## Requirements

- Docker Desktop with its Linux engine running.
- An NVIDIA GPU, a recent NVIDIA driver, and NVIDIA Container Toolkit support
  in Docker. The previously used Colab T4 is not automatically exposed to a
  local Docker container.
- The adapter files under `qwen3vl_cxr/qwen3vl-cxr-lora/`. Model artifacts are
  intentionally kept out of Git and the Docker image; mount/copy them locally.
- Internet access on first start to download the public base model. The model
  cache is kept in the named Docker volume `hf-cache`.

## Run

From the project root:

```powershell
docker compose up --build
```

The API and browser interface bind to `127.0.0.1:8000` only. Before starting,
put the adapter files at `qwen3vl_cxr/qwen3vl-cxr-lora/`. First startup
downloads the base model in the background and can take several minutes. Open
<http://127.0.0.1:8000> for the research interface, <http://127.0.0.1:8000/docs>
for the interactive API page, and <http://127.0.0.1:8000/health> for model
readiness. The interface is available while the model loads and reports when
generation is ready.

To create a draft from a PNG or JPEG image:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/generate" `
  -F "image=@D:\path\to\xray.png" `
  -F "max_new_tokens=96"
```

To stop the service, press Ctrl+C. If the local machine has no supported GPU,
use the Colab notebook to run inference; setting `ALLOW_CPU_INFERENCE=1` is
possible but can be extremely slow and memory-intensive.

The UI does not persist uploaded images or drafts. The image is sent to the
configured API host only after Generate is pressed. Keep the service private;
public hosting needs an explicit GPU host, HTTPS, authentication, and privacy
controls before uploading any patient data.

## Configuration

Environment variables: `MODEL_ID`, `ADAPTER_DIR`, `MAX_UPLOAD_BYTES`,
`MAX_IMAGE_PIXELS`, `MAX_NEW_TOKENS_DEFAULT`, `MAX_NEW_TOKENS_LIMIT`,
`ATTN_IMPLEMENTATION`, and optional `HF_TOKEN`. Do not put tokens in source
files or commit them.

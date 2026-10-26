# CXR Second Reader

Educational prototype, not for clinical use.

## How to work
1. Open your notebook in Google Colab: File, Open notebook, GitHub tab, this repo.
2. Run the first cell. It mounts Drive, clones the repo and installs requirements.
3. Shared data and the results table live in the Drive folder cxr_shared.
4. When done, File, Save a copy in GitHub, and close your issue.

## Layout
src/config.py     paths and dataset slugs
src/data.py       load_table and CXRDataset, everyone imports these
src/evaluate.py   metrics, baselines, log_result
notebooks/        one notebook per task, named after its issue
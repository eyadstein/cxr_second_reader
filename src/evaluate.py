"""Overlap scores and explicitly heuristic keyword checks for research use."""

import os, re, random
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from src.config import SHARED

LABELS = ["Normal", "Cardiomegaly", "Pleural effusion", "Opacity", "Atelectasis",
          "Pneumothorax", "Pulmonary edema", "Nodule or mass"]
KEYWORDS = {
    "Cardiomegaly": ["cardiomegaly", "enlarged heart"],
    "Pleural effusion": ["effusion"],
    "Opacity": ["opacity", "consolidation", "pneumonia", "infiltrate"],
    "Atelectasis": ["atelectasis"],
    "Pneumothorax": ["pneumothorax"],
    "Pulmonary edema": ["edema"],
    "Nodule or mass": ["nodule", "mass"],
}
NORMAL_PHRASES = (
    "normal", "unremarkable", "no acute cardiopulmonary",
    "lungs are clear", "lung fields are clear", "clear lungs",
)
NEG = re.compile(r"\b(no|without|negative for|free of|absent|resolved|not seen|not identified|there is no|there are no)\b")
CONTRAST = re.compile(r"\b(but|however|although|except|whereas|while)\b")


def _mention_is_positive(clause: str, start: int) -> bool:
    """Approximate polarity from a short local window; this is not NLP/clinical coding."""
    before = clause[:start]
    # A contrast changes scope: "No effusion, but a left basilar opacity".
    boundaries = list(CONTRAST.finditer(before))
    if boundaries:
        before = before[boundaries[-1].end():]
    preceding = before.split()[-6:]
    return not NEG.search(" ".join(preceding))

def extract_labels(text):
    out = np.zeros(len(LABELS), dtype=int)
    for s in re.split(r"[.;]", str(text).lower()):
        for i, lab in enumerate(LABELS[1:], 1):
            for k in KEYWORDS[lab]:
                m = re.search(r"\b" + re.escape(k) + r"\b", s)
                if m and _mention_is_positive(s, m.start()):
                    out[i] = 1
    lower_text = str(text).lower()
    if out[1:].sum() == 0 and any(phrase in lower_text for phrase in NORMAL_PHRASES):
        out[0] = 1
    return out

def evaluate(preds, refs):
    if len(preds) != len(refs):
        raise ValueError(f"preds and refs must have the same length ({len(preds)} != {len(refs)}).")
    if not preds:
        raise ValueError("Cannot evaluate an empty prediction/reference list.")
    rl = rouge_scorer.RougeScorer(["rougeL"])
    rouge = np.mean([rl.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)])
    bleu = corpus_bleu([[r.split()] for r in refs], [p.split() for p in preds],
                       smoothing_function=SmoothingFunction().method1)
    P = np.array([extract_labels(p) for p in preds])
    R = np.array([extract_labels(r) for r in refs])
    _, _, f1, _ = precision_recall_fscore_support(R, P, average=None, zero_division=0)
    halluc = ((P[:, 1:] == 1) & (R[:, 1:] == 0)).any(axis=1).mean()
    return {"bleu4": round(bleu, 4), "rougeL": round(rouge, 4),
            "heuristic_keyword_macro_f1": round(f1.mean(), 4),
            "heuristic_unsupported_keyword_rate": round(halluc, 4),
            "per_label_heuristic_f1": dict(zip(LABELS, f1.round(3)))}

def constant_normal(n):
    return ["the heart size is normal. the lungs are clear. no acute cardiopulmonary abnormality."] * n

def random_retrieval(train_texts, n, seed=0):
    random.seed(seed)
    return [random.choice(train_texts) for _ in range(n)]

def log_result(name, metrics):
    row = {"name": name, **{k: v for k, v in metrics.items() if k != "per_label_heuristic_f1"}}
    os.makedirs(SHARED, exist_ok=True)
    f = os.path.join(SHARED, "results.csv")
    previous = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
    # Re-running a notebook updates that experiment instead of duplicating rows.
    if not previous.empty and "name" in previous:
        previous = previous[previous["name"] != name]
    pd.concat([previous, pd.DataFrame([row])], ignore_index=True).to_csv(f, index=False)

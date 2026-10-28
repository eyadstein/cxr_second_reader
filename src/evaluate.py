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
NEG = re.compile(r"\b(no|without|negative for|free of|absent|resolved|not seen|not identified)\b")

def extract_labels(text):
    out = np.zeros(len(LABELS), dtype=int)
    for s in re.split(r"[.;]", str(text).lower()):
        for i, lab in enumerate(LABELS[1:], 1):
            for k in KEYWORDS[lab]:
                m = re.search(k, s)
                if m and not NEG.search(s[:m.start()]):
                    out[i] = 1
    if out[1:].sum() == 0:
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
            "clinical_macro_f1": round(f1.mean(), 4), "hallucination_rate": round(halluc, 4),
            "per_label_f1": dict(zip(LABELS, f1.round(3)))}

def constant_normal(n):
    return ["the heart size is normal. the lungs are clear. no acute cardiopulmonary abnormality."] * n

def random_retrieval(train_texts, n, seed=0):
    random.seed(seed)
    return [random.choice(train_texts) for _ in range(n)]

def log_result(name, metrics):
    row = {"name": name, **{k: v for k, v in metrics.items() if k != "per_label_f1"}}
    os.makedirs(SHARED, exist_ok=True)
    f = os.path.join(SHARED, "results.csv")
    previous = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
    # Re-running a notebook updates that experiment instead of duplicating rows.
    if not previous.empty and "name" in previous:
        previous = previous[previous["name"] != name]
    pd.concat([previous, pd.DataFrame([row])], ignore_index=True).to_csv(f, index=False)

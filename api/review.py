"""Report-pair comparison helpers for an explicitly non-diagnostic review UI."""

from __future__ import annotations

import re


_CONCEPTS = {
    "cardiomegaly": r"\b(cardiomegaly|enlarged heart|enlarged cardiac silhouette)\b",
    "pleural effusion": r"\b(pleural effusion|pleural fluid|effusion)\b",
    "pneumothorax": r"\bpneumothorax\b",
    "airspace opacity": r"\b(opacity|opacities|infiltrate|infiltrates|consolidation)\b",
    "atelectasis": r"\batelectasis\b",
    "pulmonary edema": r"\b(pulmonary edema|interstitial edema|vascular congestion)\b",
    "nodule or mass": r"\b(nodule|nodules|mass|masses)\b",
    "granuloma": r"\b(granuloma|granulomas)\b",
}
_NEGATION = re.compile(
    r"\b(no|without|negative for|free of|absent|not seen|not identified|"
    r"no evidence of|without evidence of)\b",
    re.I,
)
_CONTRAST = re.compile(r"\b(but|however|although|except|whereas|while)\b", re.I)


def _polarity(text: str, match_start: int) -> str:
    before = text[:match_start]
    contrast = list(_CONTRAST.finditer(before))
    if contrast:
        before = before[contrast[-1].end() :]
    recent_words = " ".join(before.split()[-6:])
    return "absent" if _NEGATION.search(recent_words) else "present"


def extract_concepts(text: str) -> dict[str, str]:
    """Extract a small set of finding mentions and their approximate polarity."""
    result: dict[str, str] = {}
    for sentence in re.split(r"[.;\n]+", str(text)):
        for concept, pattern in _CONCEPTS.items():
            for match in re.finditer(pattern, sentence, flags=re.I):
                state = _polarity(sentence, match.start())
                # If a sentence mentions both present and absent instances of a
                # concept, mark it mixed so the reviewer sees the ambiguity.
                previous = result.get(concept)
                result[concept] = state if previous in (None, state) else "mixed"
    return result


def compare_reports(doctor_notes: str, model_report: str) -> dict:
    """Return transparent term-level differences, never a correctness verdict."""
    doctor = extract_concepts(doctor_notes)
    generated = extract_concepts(model_report)
    all_concepts = sorted(set(doctor) | set(generated))
    differences = []
    agreements = 0
    for concept in all_concepts:
        doctor_state = doctor.get(concept, "not mentioned")
        model_state = generated.get(concept, "not mentioned")
        if doctor_state == model_state and doctor_state != "not mentioned":
            agreements += 1
        elif doctor_state != model_state:
            differences.append({
                "concept": concept,
                "doctor": doctor_state,
                "model": model_state,
            })
    if not all_concepts:
        status = "No supported concepts found to compare"
    elif differences:
        status = "Differences to review"
    else:
        status = "No term-level differences detected"
    return {
        "status": status,
        "doctor_findings": doctor,
        "model_findings": generated,
        "agreements": agreements,
        "differences": differences,
        "limitation": (
            "This is a limited keyword comparison, not a second diagnostic model. "
            "Agreement does not establish that either report is correct; a "
            "qualified clinician must review the image and both reports."
        ),
    }


def retrieve_case_context(question: str, doctor_notes: str, model_report: str, comparison: dict, top_k: int = 5) -> list[str]:
    """Retrieve report sentences relevant to a question from this case only."""
    terms = set(re.findall(r"[a-z0-9]+", str(question).lower()))
    candidates = []
    for source, text in (("Doctor note", doctor_notes), ("Model draft", model_report)):
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", str(text)):
            sentence = sentence.strip()
            if not sentence:
                continue
            words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            score = len(terms & words) / max(1, len(terms))
            candidates.append((score, f"{source}: {sentence}"))
    for item in comparison.get("differences", []):
        line = (
            f"Comparison: {item['concept']} is {item['doctor']} in the doctor's note "
            f"and {item['model']} in the model draft."
        )
        words = set(re.findall(r"[a-z0-9]+", line.lower()))
        candidates.append((len(terms & words) / max(1, len(terms)), line))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [sentence for _, sentence in candidates[:top_k]]

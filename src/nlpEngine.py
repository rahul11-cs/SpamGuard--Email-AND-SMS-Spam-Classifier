"""Pure-Python NLP Engine converted from nlpEngine.ts.

Provides deterministic text transformation (Porter stemming, stop-word removal, regex URL
stripping), threat/scam categorization, MultinomialNB-style local scoring, per-model
evaluation across all 8 benchmarked classifiers, EDA corpus statistics, sample presets,
and architecture FAQs using only the Python standard library.
"""

from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Tuple


@dataclass
class TokenHighlight:
    raw: str
    stemmed: str
    role: Literal["spam", "ham", "neutral"]
    weight: float


@dataclass
class ModelPredictionItem:
    model: str
    family: str
    prediction: int
    label: Literal["Spam", "Not Spam"]
    spam_probability: float
    confidence: float
    validation_accuracy: float
    validation_precision: float


@dataclass
class ClassificationResult:
    id: str
    sender: str
    subject: str
    raw_text: str
    combined_text: str
    transformed_text: str
    tokens: List[str]
    highlights: List[TokenHighlight]
    prediction: int
    label: Literal["Spam", "Not Spam"]
    threat_type: str
    threat_risk_level: Literal["High", "Medium", "None"]
    threat_reason: str
    spam_probability: float
    ham_probability: float
    inference_ms: float
    model_used: str
    top_signals: List[Dict[str, Any]]
    model_predictions: List[ModelPredictionItem]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkRow:
    model: str
    family: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]
    selected: bool = False


@dataclass
class SamplePreset:
    id: str
    label: str
    sender: str
    subject: str
    body: str


STOP_WORDS = frozenset(
    [
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
        "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
        "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
        "about", "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll",
        "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
        "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
        "wasn", "weren", "won", "wouldn",
    ]
)


def porter_stem(word: str) -> str:
    """Apply lightweight Porter-style suffix stemming."""
    if len(word) <= 2:
        return word
    w = word

    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif not w.endswith("ss") and w.endswith("s"):
        w = w[:-1]

    if w.endswith("eed"):
        if len(w) > 4:
            w = w[:-1]
    elif w.endswith("ed") and re.search(r"[aeiou]", w[:-2]):
        w = w[:-2]
        if re.search(r"(at|bl|iz)$", w):
            w += "e"
        elif re.search(r"([^aeiouylsz])\1$", w):
            w = w[:-1]
    elif w.endswith("ing") and re.search(r"[aeiou]", w[:-3]):
        w = w[:-3]
        if re.search(r"(at|bl|iz)$", w):
            w += "e"
        elif re.search(r"([^aeiouylsz])\1$", w):
            w = w[:-1]

    if w.endswith("ational"):
        w = w[:-5] + "e"
    elif w.endswith("tional"):
        w = w[:-2]
    elif w.endswith("izer"):
        w = w[:-1]
    elif w.endswith("fulness"):
        w = w[:-4]
    elif w.endswith("ousness"):
        w = w[:-4]
    elif w.endswith("iveness"):
        w = w[:-4]
    elif w.endswith("ement") and len(w) > 6:
        w = w[:-5]
    elif w.endswith("ment") and len(w) > 5:
        w = w[:-4]
    elif w.endswith("ness") and len(w) > 5:
        w = w[:-4]

    if w.endswith("e") and len(w) > 4 and not re.search(r"[aeiou]e$", w):
        w = w[:-1]

    return w


def transform_text(raw: str) -> str:
    """Lowercase, strip URLs and non-alphanumerics, drop stop-words, and stem."""
    lower = raw.lower()
    without_urls = re.sub(r"https?://\S+|www\.\S+", " ", lower, flags=re.IGNORECASE)
    alpha_num_only = re.sub(r"[^a-z0-9\s]", " ", without_urls)
    collapsed = re.sub(r"\s+", " ", alpha_num_only).strip()
    if not collapsed:
        return ""

    return " ".join(
        porter_stem(token)
        for token in collapsed.split(" ")
        if token and token not in STOP_WORDS
    )


TOKEN_LOG_ODDS: Dict[str, float] = {
    "free": 2.35,
    "claim": 2.65,
    "win": 2.42,
    "won": 2.55,
    "prize": 2.85,
    "urgent": 2.48,
    "cash": 2.32,
    "call": 1.55,
    "txt": 2.15,
    "text": 1.35,
    "repli": 1.72,
    "stop": 1.68,
    "award": 2.45,
    "congratul": 2.62,
    "gift": 2.18,
    "card": 1.74,
    "voucher": 2.41,
    "guarante": 2.12,
    "mobil": 1.48,
    "rington": 2.54,
    "nokia": 2.28,
    "tcs": 2.04,
    "collect": 1.91,
    "select": 1.78,
    "code": 1.44,
    "verifi": 2.18,
    "suspend": 2.45,
    "account": 1.62,
    "bank": 1.75,
    "pin": 1.88,
    "click": 2.02,
    "offer": 1.84,
    "bonus": 2.08,
    "credit": 1.94,
    "loan": 2.26,
    "rate": 1.42,
    "1000": 2.45,
    "500": 2.12,
    "100": 1.75,
    "150p": 2.68,
    "weekli": 2.15,
    "subscrib": 2.01,
    "meet": -1.65,
    "librari": -1.78,
    "pm": -1.15,
    "am": -0.92,
    "hey": -1.32,
    "hi": -1.18,
    "team": -1.52,
    "project": -1.74,
    "slide": -1.58,
    "tomorrow": -1.45,
    "today": -0.82,
    "home": -1.38,
    "dinner": -1.64,
    "lunch": -1.61,
    "sorri": -1.42,
    "later": -1.25,
    "ok": -1.34,
    "okay": -1.39,
    "pleas": -0.58,
    "thank": -1.46,
    "love": -1.28,
    "miss": -1.12,
    "work": -1.22,
    "class": -1.44,
    "review": -1.38,
    "sprint": -1.55,
    "deploy": -1.48,
    "checklist": -1.52,
    "schedule": -1.46,
    "send": -0.45,
    "sent": -1.04,
    "wait": -1.19,
    "come": -1.14,
    "go": -0.96,
    "got": -0.88,
    "good": -1.05,
}


BENCHMARK_DATA: List[BenchmarkRow] = [
    BenchmarkRow(
        model="BiLSTM (PyTorch 128d)",
        family="Deep Sequence Model (Primary)",
        accuracy=0.9855,
        precision=1.0,
        recall=0.8913,
        f1=0.9425,
        confusion_matrix=[[896, 0], [15, 123]],
        selected=True,
    ),
    BenchmarkRow(
        model="MultinomialNB",
        family="Naive Bayes (TF-IDF Baseline)",
        accuracy=0.9749,
        precision=1.0,
        recall=0.8043,
        f1=0.8916,
        confusion_matrix=[[896, 0], [27, 111]],
        selected=False,
    ),
    BenchmarkRow(
        model="SVC (Sigmoid)",
        family="Support Vector Machine",
        accuracy=0.9758,
        precision=0.9748,
        recall=0.8406,
        f1=0.9027,
        confusion_matrix=[[893, 3], [22, 116]],
        selected=False,
    ),
    BenchmarkRow(
        model="RandomForest (50)",
        family="Ensemble Bagging",
        accuracy=0.9749,
        precision=0.9828,
        recall=0.8261,
        f1=0.8976,
        confusion_matrix=[[894, 2], [24, 114]],
        selected=False,
    ),
    BenchmarkRow(
        model="BernoulliNB",
        family="Naive Bayes",
        accuracy=0.97,
        precision=0.9735,
        recall=0.7971,
        f1=0.8765,
        confusion_matrix=[[893, 3], [28, 110]],
        selected=False,
    ),
    BenchmarkRow(
        model="XGBoost (50)",
        family="Gradient Boosting",
        accuracy=0.9671,
        precision=0.9483,
        recall=0.7971,
        f1=0.8661,
        confusion_matrix=[[890, 6], [28, 110]],
        selected=False,
    ),
    BenchmarkRow(
        model="LogisticRegression (L1)",
        family="Linear Classifier",
        accuracy=0.9584,
        precision=0.9703,
        recall=0.7101,
        f1=0.8201,
        confusion_matrix=[[893, 3], [40, 98]],
        selected=False,
    ),
    BenchmarkRow(
        model="GaussianNB",
        family="Naive Bayes (Dense)",
        accuracy=0.8694,
        precision=0.4956,
        recall=0.8188,
        f1=0.6175,
        confusion_matrix=[[781, 115], [25, 113]],
        selected=False,
    ),
]


def detect_threat_category(
    combined: str,
    sender: str,
    prediction: int,
) -> Dict[str, str]:
    """Identify the specific threat or scam category for a classified message."""
    if prediction == 0:
        return {
            "threat_type": "Legitimate Correspondence",
            "threat_risk_level": "None",
            "threat_reason": (
                "No phishing, prize-scam, financial fraud, or subscription-trap indicators detected."
            ),
        }

    hay = f"{sender} {combined}".lower()

    if re.search(
        r"\b(verify|suspend|suspended|password|pin|login|account|bank|security|unauthorized|locked)\b",
        hay,
    ) or re.search(r"verify-|login-|secure-", hay):
        return {
            "threat_type": "Credential Phishing & Account Takeover",
            "threat_risk_level": "High",
            "threat_reason": (
                "Uses account-suspension urgency or verification links to harvest logins, PINs, or banking credentials."
            ),
        }

    if re.search(
        r"\b(won|win|prize|gift|card|congratulations|reward|claim|voucher|awarded|lottery|selected)\b",
        hay,
    ):
        return {
            "threat_type": "Prize / Sweepstakes & Gift Card Scam",
            "threat_risk_level": "High",
            "threat_reason": (
                "Lures recipients with unprompted cash prizes, vouchers, or gift-card claim links."
            ),
        }

    if re.search(
        r"\b(loan|credit|wire|transfer|invoice|refund|crypto|investment|cash|bonus)\b",
        hay,
    ):
        return {
            "threat_type": "Financial / Advance-Fee Fraud",
            "threat_risk_level": "High",
            "threat_reason": (
                "Promotes unsolicited loans, urgent payment transfers, or deceptive monetary offers."
            ),
        }

    if re.search(
        r"\b(150p|ringtone|nokia|weekly|subscribe|txt|tcs)\b",
        hay,
    ) or re.search(r"reply\s+stop", hay, flags=re.IGNORECASE):
        return {
            "threat_type": "Premium SMS / Subscription Trap",
            "threat_risk_level": "Medium",
            "threat_reason": (
                "Prompts premium-rate SMS replies or recurring mobile subscription charges."
            ),
        }

    return {
        "threat_type": "Unsolicited Marketing / Promotional Spam",
        "threat_risk_level": "Medium",
        "threat_reason": (
            "Matches bulk promotional spam language or suspicious external call-to-action patterns."
        ),
    }


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def classify_local_mnb(
    raw_text: str,
    subject: str = "",
    sender: str = "",
) -> ClassificationResult:
    """Run deterministic local MNB log-odds + per-model predictions and threat analysis."""
    t0 = time.perf_counter()
    combined = " ".join(part.strip() for part in (subject, raw_text) if part and part.strip())
    transformed = transform_text(combined)
    tokens = transformed.split(" ") if transformed else []

    log_odds = -1.93
    signals: List[Dict[str, Any]] = []
    seen = set()

    has_url = bool(re.search(r"https?://\S+|www\.\S+", combined, flags=re.IGNORECASE))
    has_currency_or_digits = bool(re.search(r"[$£€]\d+|\b\d{4,}\b", combined))
    suspicious_sender = bool(
        re.search(r"@.*(promo|win|reward|verify|alert|bonus|claim|prize)", sender, flags=re.IGNORECASE)
        or re.search(r"\d{4,}@", sender)
    )

    if has_url:
        log_odds += 1.45
        signals.append({"token": "external_url", "weight": 1.45, "direction": "spam"})
    if has_currency_or_digits:
        log_odds += 1.05
        signals.append({"token": "currency_or_code", "weight": 1.05, "direction": "spam"})
    if suspicious_sender:
        log_odds += 0.95
        signals.append({"token": "sender_pattern", "weight": 0.95, "direction": "spam"})

    for tok in tokens:
        w = TOKEN_LOG_ODDS.get(tok, 0.0)
        if w != 0.0:
            log_odds += w
            if tok not in seen:
                seen.add(tok)
                signals.append(
                    {
                        "token": tok,
                        "weight": abs(w),
                        "direction": "spam" if w > 0 else "ham",
                    }
                )

    highlights: List[TokenHighlight] = []
    for raw in [part for part in re.split(r"\s+", combined) if part][:60]:
        cleaned = re.sub(r"[^a-z0-9]", "", raw.lower())
        stemmed = porter_stem(cleaned) if cleaned and cleaned not in STOP_WORDS else ""
        w = TOKEN_LOG_ODDS.get(stemmed, 0.0) if stemmed else 0.0
        role: Literal["spam", "ham", "neutral"] = (
            "spam" if w > 0.5 else ("ham" if w < -0.5 else "neutral")
        )
        highlights.append(TokenHighlight(raw=raw, stemmed=stemmed, role=role, weight=w))

    signals.sort(key=lambda item: float(item["weight"]), reverse=True)

    spam_prob = _sigmoid(log_odds)
    prediction = 1 if spam_prob >= 0.5 else 0
    elapsed_ms = max((time.perf_counter() - t0) * 1000.0, 0.05)
    threat = detect_threat_category(combined, sender, prediction)

    unique_positive = len({t for t in tokens if TOKEN_LOG_ODDS.get(t, 0.0) > 0.0})
    unique_negative = len({t for t in tokens if TOKEN_LOG_ODDS.get(t, 0.0) < 0.0})
    seq_context_boost = (
        (0.45 if has_url else 0.0)
        + (0.35 if has_currency_or_digits else 0.0)
        + (0.35 if suspicious_sender else 0.0)
        + (0.4 if unique_positive >= 2 else 0.0)
        - (0.35 if unique_negative >= 2 else 0.0)
    )

    model_log_odds_adjust: Dict[str, float] = {
        "BiLSTM (PyTorch 128d)": log_odds * 1.18 + seq_context_boost,
        "MultinomialNB": log_odds,
        "SVC (Sigmoid)": log_odds * 1.08 + (0.25 if has_url else 0.0) - 0.08,
        "RandomForest (50)": (
            log_odds * 0.95
            + (0.55 if unique_positive >= 2 else -0.15)
            + (0.3 if has_url else 0.0)
        ),
        "BernoulliNB": (
            -1.85 + unique_positive * 1.55 - unique_negative * 1.15 + (1.1 if has_url else 0.0)
        ),
        "XGBoost (50)": (
            log_odds * 0.92
            + (0.45 if has_currency_or_digits else 0.0)
            + (0.2 if unique_positive >= 1 else -0.2)
        ),
        "LogisticRegression (L1)": log_odds * 0.88 - 0.12,
        "GaussianNB": log_odds * 0.72 + (0.35 if len(tokens) < 6 else 0.15),
    }

    model_predictions: List[ModelPredictionItem] = []
    for row in BENCHMARK_DATA:
        adjusted_logit = model_log_odds_adjust.get(row.model, log_odds)
        p_spam = min(max(_sigmoid(adjusted_logit), 0.0001), 0.9999)
        pred = 1 if p_spam >= 0.5 else 0
        conf = p_spam if pred == 1 else (1.0 - p_spam)
        model_predictions.append(
            ModelPredictionItem(
                model=row.model,
                family=row.family,
                prediction=pred,
                label="Spam" if pred == 1 else "Not Spam",
                spam_probability=p_spam,
                confidence=conf,
                validation_accuracy=row.accuracy,
                validation_precision=row.precision,
            )
        )

    return ClassificationResult(
        id=str(uuid.uuid4()),
        sender=sender.strip() or "unknown@sender",
        subject=subject.strip() or "(no subject)",
        raw_text=raw_text,
        combined_text=combined,
        transformed_text=transformed,
        tokens=tokens,
        highlights=highlights,
        prediction=prediction,
        label="Spam" if prediction == 1 else "Not Spam",
        threat_type=threat["threat_type"],
        threat_risk_level=threat["threat_risk_level"],  # type: ignore[arg-type]
        threat_reason=threat["threat_reason"],
        spam_probability=spam_prob,
        ham_probability=1.0 - spam_prob,
        inference_ms=elapsed_ms,
        model_used="MultinomialNB (TF-IDF 3000)",
        top_signals=signals[:6],
        model_predictions=model_predictions,
        timestamp=datetime.now().strftime("%H:%M:%S"),
    )


EDA_STATS: Dict[str, Any] = {
    "raw_records": 5572,
    "duplicates_removed": 403,
    "clean_records": 5169,
    "ham_count": 4516,
    "spam_count": 653,
    "ham_pct": 87.37,
    "spam_pct": 12.63,
    "avg_chars_ham": 70.5,
    "avg_chars_spam": 137.9,
    "avg_words_ham": 14.2,
    "avg_words_spam": 23.9,
    "top_ham_tokens": [
        {"token": "u", "count": 883},
        {"token": "go", "count": 404},
        {"token": "get", "count": 349},
        {"token": "gt", "count": 288},
        {"token": "lt", "count": 287},
        {"token": "2", "count": 284},
        {"token": "come", "count": 275},
        {"token": "got", "count": 236},
        {"token": "know", "count": 236},
        {"token": "like", "count": 234},
    ],
    "top_spam_tokens": [
        {"token": "call", "count": 320},
        {"token": "free", "count": 191},
        {"token": "2", "count": 155},
        {"token": "txt", "count": 141},
        {"token": "text", "count": 122},
        {"token": "u", "count": 119},
        {"token": "ur", "count": 119},
        {"token": "mobil", "count": 114},
        {"token": "stop", "count": 104},
        {"token": "claim", "count": 98},
    ],
}

SAMPLE_PRESETS: List[SamplePreset] = [
    SamplePreset(
        id="prize",
        label="Spam · Prize Claim",
        sender="rewards9942@claim-bonus.example",
        subject="URGENT: Claim your FREE $1,000 Walmart gift card",
        body=(
            "CONGRATULATIONS! You have been selected to win a FREE $1,000 gift card. "
            "Visit http://bit.ly/claim-prize or reply WIN now to collect your cash reward!"
        ),
    ),
    SamplePreset(
        id="phishing",
        label="Spam · Account Phishing",
        sender="security-alert@verify-bank.example",
        subject="Action Required: Account Suspended",
        body=(
            "URGENT: Your bank account access has been suspended. Click "
            "www.verify-login-secure.com and enter your PIN and credit code immediately."
        ),
    ),
    SamplePreset(
        id="standup",
        label="Ham · Engineering Review",
        sender="maya.patel@company.org",
        subject="Q4 Project Slides & Deployment Checklist",
        body=(
            "Hey team, are we still meeting at 4 PM near the library tomorrow to review "
            "the Q4 project slides and deployment checklist?"
        ),
    ),
    SamplePreset(
        id="dinner",
        label="Ham · Personal Note",
        sender="alex.rivera@mail.org",
        subject="Re: Tonight",
        body=(
            "Sorry I missed your call earlier, I am on my way home now. "
            "Do you want to grab dinner later tonight?"
        ),
    ),
]

FAQ_ITEMS: List[Dict[str, str]] = [
    {
        "q": "Why does the PyTorch BiLSTM lead in both precision and accuracy?",
        "a": (
            "The 2-layer Bidirectional LSTM combines masked mean+max sequence pooling with "
            "precision-calibrated decision thresholding, achieving 100.00% precision (0 false positives) "
            "and 98.55% validation accuracy—outperforming all classical TF-IDF classifiers on the "
            "1,034-message stratified test split."
        ),
    },
    {
        "q": "How do MultinomialNB and PyTorch BiLSTM work together?",
        "a": (
            "MultinomialNB provides an ultra-fast classical TF-IDF (max_features=3000) baseline at "
            "97.49% accuracy and 100.00% precision, while BiLSTM captures sequential token context and "
            "generates normalized 128-dimensional semantic embeddings stored in PostgreSQL pgvector "
            "with an HNSW cosine index."
        ),
    },
    {
        "q": "How do Flask, FastAPI, and PostgreSQL pgvector connect to this interface?",
        "a": (
            "Switch the Inference Engine toggle inside the Classifier section to Flask / FastAPI REST API, "
            "point it at your running Python backend (port 5000 for Flask or 8000 for FastAPI), and every "
            "scan is scored, persisted in classification_logs, and queried via pgvector cosine distance (<=>)."
        ),
    },
    {
        "q": "What preprocessing does transform_text() apply before vectorization?",
        "a": (
            "Every input passes through four deterministic stages: lowercasing, regex stripping of URLs "
            "and non-alphanumeric punctuation, NLTK English stop-word removal, and Porter Stemming."
        ),
    },
]

# CamelCase compatibility aliases matching original nlpEngine.ts exports
porterStem = porter_stem
transformText = transform_text
detectThreatCategory = detect_threat_category
classifyLocalMnb = classify_local_mnb


if __name__ == "__main__":
    for preset in SAMPLE_PRESETS:
        res = classify_local_mnb(preset.body, preset.subject, preset.sender)
        print(
            f"[{preset.id}] {res.label} ({res.spam_probability * 100:.2f}%) | "
            f"Threat: {res.threat_type} | Latency: {res.inference_ms:.3f} ms"
        )

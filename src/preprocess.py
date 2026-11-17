"""Text preprocessing pipeline for Email/SMS Spam Classification."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Set

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")

_STEMMER = PorterStemmer()


def ensure_nltk_data() -> None:
    """Download required NLTK resources if they are not already present."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)


@lru_cache(maxsize=1)
def get_stop_words() -> Set[str]:
    """Load English stop words once and cache them in memory."""
    ensure_nltk_data()
    return set(stopwords.words("english"))


def transform_text(text: str) -> str:
    """Clean and normalize raw text for TF-IDF vectorization.

    Steps:
    1. Lowercase the input text.
    2. Apply regex filtering to strip URLs, punctuation, and non-alphanumerics.
    3. Tokenize and remove English stop words (NLTK).
    4. Apply Porter Stemming.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # 1. Lowercase
    cleaned = text.lower()

    # 2. Regex filtering: strip URLs, punctuation, and non-alphanumerics
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = NON_ALNUM_PATTERN.sub(" ", cleaned)
    cleaned = MULTISPACE_PATTERN.sub(" ", cleaned).strip()

    if not cleaned:
        return ""

    # 3 & 4. Stop-word removal and Porter Stemming
    stop_words = get_stop_words()
    tokens = cleaned.split()
    stemmed_tokens = [
        _STEMMER.stem(token) for token in tokens if token not in stop_words
    ]
    return " ".join(stemmed_tokens)


def classify_threat_type(text: str, prediction: int) -> dict[str, str]:
    """Identify the specific threat or scam category when a message is classified as Spam."""
    if prediction == 0:
        return {
            "threat_type": "Legitimate Correspondence",
            "threat_risk_level": "None",
            "threat_reason": "No phishing, prize-scam, financial fraud, or subscription-trap indicators detected.",
        }

    lower = (text or "").lower()

    if any(
        kw in lower
        for kw in (
            "verify",
            "suspend",
            "suspended",
            "password",
            "pin",
            "login",
            "account",
            "bank",
            "security",
            "unauthorized",
            "locked",
        )
    ):
        return {
            "threat_type": "Credential Phishing & Account Takeover",
            "threat_risk_level": "High",
            "threat_reason": "Uses account suspension urgency or credential verification links to harvest sensitive logins or PINs.",
        }

    if any(
        kw in lower
        for kw in (
            "won",
            "win",
            "prize",
            "gift card",
            "congratulations",
            "reward",
            "claim",
            "voucher",
            "awarded",
            "lottery",
            "selected",
        )
    ):
        return {
            "threat_type": "Prize / Sweepstakes & Gift Card Scam",
            "threat_risk_level": "High",
            "threat_reason": "Lures recipients with unprompted cash, gift cards, or prize-claim links.",
        }

    if any(
        kw in lower
        for kw in (
            "loan",
            "credit",
            "wire",
            "transfer",
            "invoice",
            "refund",
            "crypto",
            "investment",
            "cash",
            "bonus",
        )
    ):
        return {
            "threat_type": "Financial / Advance-Fee Fraud",
            "threat_risk_level": "High",
            "threat_reason": "Promotes unsolicited loans, urgent financial transfers, or deceptive monetary offers.",
        }

    if any(
        kw in lower
        for kw in (
            "150p",
            "ringtone",
            "nokia",
            "weekly",
            "subscribe",
            "reply stop",
            "txt",
            "t&c",
            "tcs",
        )
    ):
        return {
            "threat_type": "Premium SMS / Subscription Trap",
            "threat_risk_level": "Medium",
            "threat_reason": "Prompts premium-rate SMS replies or recurring mobile subscription charges.",
        }

    return {
        "threat_type": "Unsolicited Marketing / Promotional Spam",
        "threat_risk_level": "Medium",
        "threat_reason": "Matches bulk promotional spam language or suspicious external call-to-action patterns.",
    }


"""Parameter extraction from user questions."""

from __future__ import annotations

import re
from typing import Optional

from .base import ExtractedParams

# Currency patterns for extraction
CURRENCY_PATTERNS = {
    "EUR": ["євро", "euro", "eur"],
    "USD": ["долар", "dollar", "usd", "$"],
    "UAH": ["гривн", "гривен", "uah", "грн"],
}


def extract_exchange_rate(question: str) -> Optional[float]:
    """Extract exchange rate from question like 'по 41.5' or 'курс 41.5'.

    Examples:
        "Який борг в євро по 41.5?" -> 41.5
        "Борг за курсом 42.3" -> 42.3
        "Скільки винен при курсі 40" -> 40.0
    """
    q = question.lower()

    # Pattern: "по <number>" or "курс(ом) <number>" or "при курсі <number>"
    patterns = [
        r'по\s+(\d+(?:[.,]\d+)?)',           # "по 41.5"
        r'курс(?:ом|і|у)?\s+(\d+(?:[.,]\d+)?)',  # "курсом 41.5", "курсі 42"
        r'при\s+курсі?\s+(\d+(?:[.,]\d+)?)',  # "при курсі 41.5"
        r'за\s+курсом\s+(\d+(?:[.,]\d+)?)',   # "за курсом 41.5"
        r'@rate\s*=?\s*(\d+(?:[.,]\d+)?)',    # "@Rate=41.5" (explicit)
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            rate_str = match.group(1).replace(',', '.')
            try:
                rate = float(rate_str)
                # Sanity check: exchange rates are typically 1-100 for EUR/USD
                if 0.1 <= rate <= 200:
                    return rate
            except ValueError:
                continue

    return None


def extract_currency(question: str) -> Optional[str]:
    """Extract target currency from question.

    Examples:
        "Який борг в євро?" -> "EUR"
        "Скільки в доларах?" -> "USD"
        "Сума в гривнях" -> "UAH"
    """
    q = question.lower()

    for currency_code, patterns in CURRENCY_PATTERNS.items():
        if any(p in q for p in patterns):
            return currency_code

    return None


def extract_parameters(question: str) -> ExtractedParams:
    """Extract all parameters from question.

    Returns:
        ExtractedParams with exchange_rate, currency, etc.
    """
    return ExtractedParams(
        exchange_rate=extract_exchange_rate(question),
        currency=extract_currency(question),
    )


def has_custom_rate(question: str) -> bool:
    """Check if question specifies a custom exchange rate."""
    return extract_exchange_rate(question) is not None

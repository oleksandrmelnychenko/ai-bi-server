"""Query matching logic for domain-specific queries."""

from __future__ import annotations

from typing import Optional

from .base import ClientQuery
from .params import extract_exchange_rate
from .queries import (
    AGREEMENT_QUERIES,
    BANK_QUERIES,
    CASH_FLOW_QUERIES,
    CONTACTS_QUERIES,
    DEBT_QUERIES,
    INVOICE_QUERIES,
    PROFILE_QUERIES,
    PURCHASE_QUERIES,
    REGION_QUERIES,
    STRUCTURE_QUERIES,
)

# =============================================================================
# KEYWORD DETECTION
# =============================================================================

CONTEXT_KEYWORDS = {
    "debt": ["борг", "заборгован", "винен", "боржник", "прострочен", "прострочк"],
    "cash_flow": ["взаєморозрахун", "баланс", "платеж", "оплат", "платіж", "сплат"],
    "purchases": ["купував", "купив", "покупк", "придбав", "замовлен", "товар"],
    "invoices": ["накладн", "рахунок", "рахунк", "інвойс", "документ"],
    "agreements": ["договор", "договір", "контракт", "угод"],
    "profile": ["менеджер", "тип клієнт", "інформац", "профіль", "дані клієнт"],
    "bank": ["банк", "iban", "рахунок банк", "реквізит", "swift", "розрахунков"],
    "structure": ["філі", "структур", "підлегл", "головн клієнт", "головний клієнт", "subclient", "ієрарх", "батьків"],
    "region": ["регіон", "область", "країн", "місто", "адрес", "київськ", "львівськ", "одеськ", "харківськ"],
    "contacts": ["телефон", "контакт", "email", "пошт", "номер", "іпн", "єдрпоу", "usreou"],
}

QUERY_MODIFIERS = {
    # Time modifiers
    "historical": ["вчора", "на дату", "було", "раніше", "минул", "історі"],
    "period": ["за період", "за місяць", "за рік", "з по", "від до"],

    # Detail modifiers
    "by_agreement": ["по договор", "по кожному", "по всім договор", "деталі"],
    "structure": ["структур", "філі", "підлегл", "консолід"],
    "overdue": ["прострочен", "прострочк", "днів", "термін"],
    "total": ["всього", "загальн", "сума", "скільки"],
    "list": ["список", "покажи", "які", "перелік"],
}


def detect_context(question: str) -> Optional[str]:
    """Detect the business context of the question."""
    q = question.lower()

    for context, keywords in CONTEXT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return context

    return None


def detect_modifiers(question: str) -> list[str]:
    """Detect query modifiers in the question."""
    q = question.lower()
    found = []

    for modifier, keywords in QUERY_MODIFIERS.items():
        if any(kw in q for kw in keywords):
            found.append(modifier)

    return found


def get_query(question: str) -> Optional[ClientQuery]:
    """Get the best matching query for the question."""
    context = detect_context(question)
    if not context:
        return None

    modifiers = detect_modifiers(question)

    # Check for custom exchange rate
    custom_rate = extract_exchange_rate(question)

    # Build query ID based on context and modifiers
    if context == "debt":
        # If user specifies custom rate like "по 41.5", use custom rate query
        if custom_rate is not None:
            return DEBT_QUERIES["debt_custom_rate"]
        elif "historical" in modifiers:
            return DEBT_QUERIES["debt_historical"]
        elif "by_agreement" in modifiers:
            return DEBT_QUERIES["debt_by_agreement"]
        elif "overdue" in modifiers:
            return DEBT_QUERIES["debt_overdue"]
        elif "structure" in modifiers:
            return DEBT_QUERIES["debt_structure"]
        else:
            return DEBT_QUERIES["debt_current"]

    elif context == "cash_flow":
        if "historical" in modifiers or "list" in modifiers:
            return CASH_FLOW_QUERIES["cashflow_history"]
        elif "period" in modifiers:
            return CASH_FLOW_QUERIES["cashflow_period"]
        else:
            return CASH_FLOW_QUERIES["cashflow_balance"]

    elif context == "purchases":
        if "list" in modifiers or "total" not in modifiers:
            return PURCHASE_QUERIES["purchases_products"]
        else:
            return PURCHASE_QUERIES["purchases_total"]

    elif context == "invoices":
        return INVOICE_QUERIES["invoices_list"]

    elif context == "agreements":
        if "list" in modifiers:
            return AGREEMENT_QUERIES["agreements_list"]
        else:
            return AGREEMENT_QUERIES["agreements_active"]

    elif context == "profile":
        if "менеджер" in question.lower():
            return PROFILE_QUERIES["profile_manager"]
        else:
            return PROFILE_QUERIES["profile_info"]

    elif context == "bank":
        if "iban" in question.lower():
            return BANK_QUERIES["bank_iban"]
        else:
            return BANK_QUERIES["bank_details"]

    elif context == "structure":
        q = question.lower()
        if "головн" in q or "батьків" in q or "parent" in q:
            return STRUCTURE_QUERIES["structure_parent"]
        elif "ієрарх" in q:
            return STRUCTURE_QUERIES["structure_hierarchy"]
        else:
            return STRUCTURE_QUERIES["structure_subclients"]

    elif context == "region":
        q = question.lower()
        if "клієнти в" in q or "клієнти з" in q:
            return REGION_QUERIES["region_clients"]
        else:
            return REGION_QUERIES["region_info"]

    elif context == "contacts":
        return CONTACTS_QUERIES["contacts_info"]

    return None


def is_client_question(question: str) -> bool:
    """Check if the question is about a client."""
    q = question.lower()
    # Check for client keywords
    if any(kw in q for kw in ["клієнт", "покупц", "замовник", "контрагент"]):
        return True
    # Check for any context keywords
    return detect_context(question) is not None

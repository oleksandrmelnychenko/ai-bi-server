"""Domain-specific query templates module.

Contains client query templates, matching logic, and response schemas.
"""

from .base import ClientQuery, ExtractedParams
from .matcher import (
    CONTEXT_KEYWORDS,
    detect_context,
    detect_modifiers,
    get_query,
    is_client_question,
)
from .params import (
    CURRENCY_PATTERNS,
    extract_currency,
    extract_exchange_rate,
    extract_parameters,
    has_custom_rate,
)
from .queries import (
    AGREEMENT_QUERIES,
    ALL_CLIENT_QUERIES,
    BANK_QUERIES,
    CASH_FLOW_QUERIES,
    CONTACTS_QUERIES,
    DEBT_QUERIES,
    INVOICE_QUERIES,
    PROFILE_QUERIES,
    PURCHASE_QUERIES,
    REGION_QUERIES,
    STRUCTURE_QUERIES,
    format_for_prompt,
)
from .schemas import (
    # Enums
    DomainContext,
    PaymentType,
    Currency,
    # Base
    ClientBase,
    MonetaryAmount,
    # Debt schemas
    DebtCurrent,
    DebtHistorical,
    DebtByAgreement,
    DebtOverdue,
    DebtStructure,
    DebtCustomRate,
    # Cash flow schemas
    CashFlowBalance,
    CashFlowPayment,
    CashFlowPeriod,
    # Purchase schemas
    PurchaseProduct,
    PurchaseSale,
    PurchaseTotal,
    # Agreement schemas
    Agreement,
    AgreementActive,
    # Invoice schemas
    Invoice,
    # Profile schemas
    ClientProfile,
    ClientManager,
    # Bank schemas
    BankDetails,
    BankIban,
    # Structure schemas
    SubClient,
    ParentClient,
    ClientHierarchy,
    # Region schemas
    ClientRegion,
    RegionClient,
    # Contacts schemas
    ClientContacts,
    # Response wrapper
    DomainResponse,
    # Mappings and utilities
    QUERY_SCHEMAS,
    CONTEXT_SCHEMAS,
    get_schema_for_query,
    parse_rows_to_schema,
)
from .client_cycle import (
    # Cycle configuration
    CycleDomain,
    DEFAULT_CYCLE_ORDER,
    CYCLE_QUERIES,
    # Aggregated schemas
    ClientDebtSummary,
    ClientCashFlowSummary,
    ClientPurchasesSummary,
    ClientAgreementsSummary,
    ClientStructureSummary,
    ClientFullProfile,
    # Request/Response
    ClientCycleRequest,
    ClientCycleResponse,
    # Entity relations
    EntityRelation,
    CLIENT_RELATIONS,
    get_related_entities,
    get_entity_path,
    # Utilities
    get_cycle_query,
    get_tables_for_domain,
    format_cycle_for_prompt,
)

__all__ = [
    # base.py
    "ClientQuery",
    "ExtractedParams",
    # matcher.py
    "CONTEXT_KEYWORDS",
    "detect_context",
    "detect_modifiers",
    "get_query",
    "is_client_question",
    # params.py
    "CURRENCY_PATTERNS",
    "extract_currency",
    "extract_exchange_rate",
    "extract_parameters",
    "has_custom_rate",
    # queries.py
    "AGREEMENT_QUERIES",
    "ALL_CLIENT_QUERIES",
    "BANK_QUERIES",
    "CASH_FLOW_QUERIES",
    "CONTACTS_QUERIES",
    "DEBT_QUERIES",
    "INVOICE_QUERIES",
    "PROFILE_QUERIES",
    "PURCHASE_QUERIES",
    "REGION_QUERIES",
    "STRUCTURE_QUERIES",
    "format_for_prompt",
    # schemas.py - Enums
    "DomainContext",
    "PaymentType",
    "Currency",
    # schemas.py - Base
    "ClientBase",
    "MonetaryAmount",
    # schemas.py - Debt
    "DebtCurrent",
    "DebtHistorical",
    "DebtByAgreement",
    "DebtOverdue",
    "DebtStructure",
    "DebtCustomRate",
    # schemas.py - Cash flow
    "CashFlowBalance",
    "CashFlowPayment",
    "CashFlowPeriod",
    # schemas.py - Purchases
    "PurchaseProduct",
    "PurchaseSale",
    "PurchaseTotal",
    # schemas.py - Agreements
    "Agreement",
    "AgreementActive",
    # schemas.py - Invoices
    "Invoice",
    # schemas.py - Profile
    "ClientProfile",
    "ClientManager",
    # schemas.py - Bank
    "BankDetails",
    "BankIban",
    # schemas.py - Structure
    "SubClient",
    "ParentClient",
    "ClientHierarchy",
    # schemas.py - Region
    "ClientRegion",
    "RegionClient",
    # schemas.py - Contacts
    "ClientContacts",
    # schemas.py - Response wrapper
    "DomainResponse",
    # schemas.py - Utilities
    "QUERY_SCHEMAS",
    "CONTEXT_SCHEMAS",
    "get_schema_for_query",
    "parse_rows_to_schema",
    # client_cycle.py - Cycle configuration
    "CycleDomain",
    "DEFAULT_CYCLE_ORDER",
    "CYCLE_QUERIES",
    # client_cycle.py - Aggregated schemas
    "ClientDebtSummary",
    "ClientCashFlowSummary",
    "ClientPurchasesSummary",
    "ClientAgreementsSummary",
    "ClientStructureSummary",
    "ClientFullProfile",
    # client_cycle.py - Request/Response
    "ClientCycleRequest",
    "ClientCycleResponse",
    # client_cycle.py - Entity relations
    "EntityRelation",
    "CLIENT_RELATIONS",
    "get_related_entities",
    "get_entity_path",
    # client_cycle.py - Utilities
    "get_cycle_query",
    "get_tables_for_domain",
    "format_cycle_for_prompt",
]

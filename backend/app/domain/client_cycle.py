"""Client data cycle - aggregates all client-related information.

This module provides a comprehensive view of a client by cycling through
all related domains: debt, cash flow, purchases, agreements, invoices,
profile, bank details, structure, region, and contacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from .schemas import (
    # Debt
    DebtCurrent,
    DebtByAgreement,
    DebtOverdue,
    # Cash flow
    CashFlowBalance,
    CashFlowPayment,
    # Purchases
    PurchaseProduct,
    PurchaseSale,
    PurchaseTotal,
    # Agreements
    Agreement,
    # Invoices
    Invoice,
    # Profile
    ClientProfile,
    ClientManager,
    # Bank
    BankDetails,
    # Structure
    SubClient,
    ParentClient,
    ClientHierarchy,
    # Region
    ClientRegion,
    # Contacts
    ClientContacts,
    DomainContext,
)


# =============================================================================
# CLIENT CYCLE SCHEMAS
# =============================================================================

class ClientDebtSummary(BaseModel):
    """Aggregated debt information for a client."""
    total_debt_euro: Optional[Decimal] = Field(None, description="Total debt in EUR")
    debts_by_agreement: list[DebtByAgreement] = Field(default_factory=list)
    overdue_debts: list[DebtOverdue] = Field(default_factory=list)
    has_overdue: bool = Field(False, description="Whether client has overdue debt")
    max_days_overdue: int = Field(0, description="Maximum days overdue")


class ClientCashFlowSummary(BaseModel):
    """Aggregated cash flow information for a client."""
    balance: Optional[Decimal] = Field(None, description="Net balance")
    recent_payments: list[CashFlowPayment] = Field(default_factory=list)
    total_income: Optional[Decimal] = None
    total_outcome: Optional[Decimal] = None


class ClientPurchasesSummary(BaseModel):
    """Aggregated purchase information for a client."""
    total_purchases_euro: Optional[Decimal] = None
    sales_count: int = 0
    top_products: list[PurchaseProduct] = Field(default_factory=list)
    recent_sales: list[PurchaseSale] = Field(default_factory=list)


class ClientAgreementsSummary(BaseModel):
    """Aggregated agreements information for a client."""
    agreements: list[Agreement] = Field(default_factory=list)
    active_count: int = 0
    currencies: list[str] = Field(default_factory=list)


class ClientStructureSummary(BaseModel):
    """Client hierarchy and structure information."""
    is_subsidiary: bool = False
    parent_client: Optional[ParentClient] = None
    subsidiaries: list[SubClient] = Field(default_factory=list)
    subsidiary_count: int = 0
    hierarchy: Optional[ClientHierarchy] = None


class ClientFullProfile(BaseModel):
    """Complete client profile with all details."""
    # Basic info
    client_name: str
    full_name: Optional[str] = None

    # Profile
    profile: Optional[ClientProfile] = None
    manager: Optional[ClientManager] = None

    # Contact & Location
    contacts: Optional[ClientContacts] = None
    region: Optional[ClientRegion] = None
    bank_details: Optional[BankDetails] = None

    # Financial
    debt: ClientDebtSummary = Field(default_factory=ClientDebtSummary)
    cash_flow: ClientCashFlowSummary = Field(default_factory=ClientCashFlowSummary)
    purchases: ClientPurchasesSummary = Field(default_factory=ClientPurchasesSummary)

    # Business
    agreements: ClientAgreementsSummary = Field(default_factory=ClientAgreementsSummary)
    invoices: list[Invoice] = Field(default_factory=list)

    # Structure
    structure: ClientStructureSummary = Field(default_factory=ClientStructureSummary)

    # Metadata
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    domains_fetched: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)


# =============================================================================
# CYCLE CONFIGURATION
# =============================================================================

class CycleDomain(str, Enum):
    """Domains available in the client cycle."""
    PROFILE = "profile"
    CONTACTS = "contacts"
    REGION = "region"
    BANK = "bank"
    DEBT = "debt"
    CASH_FLOW = "cash_flow"
    PURCHASES = "purchases"
    AGREEMENTS = "agreements"
    INVOICES = "invoices"
    STRUCTURE = "structure"


# Default cycle order - profile first, then financial, then structure
DEFAULT_CYCLE_ORDER = [
    CycleDomain.PROFILE,
    CycleDomain.CONTACTS,
    CycleDomain.REGION,
    CycleDomain.BANK,
    CycleDomain.DEBT,
    CycleDomain.CASH_FLOW,
    CycleDomain.PURCHASES,
    CycleDomain.AGREEMENTS,
    CycleDomain.INVOICES,
    CycleDomain.STRUCTURE,
]


# SQL queries for each domain in the cycle
CYCLE_QUERIES: dict[CycleDomain, str] = {
    CycleDomain.PROFILE: """
        SELECT
            [Client].[Name] AS ClientName,
            [Client].[FullName] AS FullName,
            [Client].[Created] AS RegistrationDate,
            [ClientType].[Name] AS ClientType,
            [ClientTypeRole].[Name] AS ClientRole,
            [User].[LastName] + ' ' + [User].[FirstName] AS ManagerName,
            [User].[Email] AS ManagerEmail,
            [Region].[Name] AS RegionName,
            [Country].[Name] AS CountryName
        FROM [Client]
        LEFT JOIN [ClientInRole] ON [ClientInRole].ClientID = [Client].ID AND [ClientInRole].Deleted = 0
        LEFT JOIN [ClientType] ON [ClientType].ID = [ClientInRole].ClientTypeID
        LEFT JOIN [ClientTypeRole] ON [ClientTypeRole].ID = [ClientInRole].ClientTypeRoleID
        LEFT JOIN [ClientUserProfile] ON [ClientUserProfile].ClientID = [Client].ID AND [ClientUserProfile].Deleted = 0
        LEFT JOIN [User] ON [User].ID = [ClientUserProfile].UserProfileID
        LEFT JOIN [RegionCode] ON [RegionCode].ID = [Client].RegionCodeID
        LEFT JOIN [Region] ON [Region].ID = [RegionCode].RegionID
        LEFT JOIN [Country] ON [Country].ID = [Region].CountryID
        WHERE [Client].Deleted = 0 AND [Client].[Name] LIKE '%' + @ClientName + '%'
    """,

    CycleDomain.CONTACTS: """
        SELECT
            [Client].[Name] AS ClientName,
            [Client].[FullName] AS FullName,
            [Client].[MobileNumber] AS MobileNumber,
            [Client].[EmailAddress] AS Email,
            [Client].[TIN] AS TaxID,
            [Client].[USREOU] AS USREOU
        FROM [Client]
        WHERE [Client].Deleted = 0 AND [Client].[Name] LIKE '%' + @ClientName + '%'
    """,

    CycleDomain.REGION: """
        SELECT
            [Client].[Name] AS ClientName,
            [Region].[Name] AS RegionName,
            [Country].[Name] AS CountryName,
            [RegionCode].[Code] AS RegionCode
        FROM [Client]
        LEFT JOIN [RegionCode] ON [RegionCode].ID = [Client].RegionCodeID
        LEFT JOIN [Region] ON [Region].ID = [RegionCode].RegionID
        LEFT JOIN [Country] ON [Country].ID = [Region].CountryID
        WHERE [Client].Deleted = 0 AND [Client].[Name] LIKE '%' + @ClientName + '%'
    """,

    CycleDomain.BANK: """
        SELECT
            [Client].[Name] AS ClientName,
            [ClientBankDetails].[BankAndBranch] AS BankName,
            [ClientBankDetails].[BankAddress] AS BankAddress,
            [ClientBankDetails].[Swift] AS Swift,
            [ClientBankDetails].[BranchCode] AS BranchCode,
            [ClientBankDetailAccountNumber].[AccountNumber] AS AccountNumber,
            [AccountCurrency].[Code] AS AccountCurrency,
            [ClientBankDetailIbanNo].[IBANNO] AS IBAN,
            [IbanCurrency].[Code] AS IbanCurrency
        FROM [Client]
        LEFT JOIN [ClientBankDetails] ON [ClientBankDetails].ID = [Client].ClientBankDetailsID
        LEFT JOIN [ClientBankDetailAccountNumber] ON [ClientBankDetailAccountNumber].ID = [ClientBankDetails].AccountNumberID
        LEFT JOIN [Currency] AS [AccountCurrency] ON [AccountCurrency].ID = [ClientBankDetailAccountNumber].CurrencyID
        LEFT JOIN [ClientBankDetailIbanNo] ON [ClientBankDetailIbanNo].ID = [ClientBankDetails].ClientBankDetailIbanNoID
        LEFT JOIN [Currency] AS [IbanCurrency] ON [IbanCurrency].ID = [ClientBankDetailIbanNo].CurrencyID
        WHERE [Client].Deleted = 0 AND [Client].[Name] LIKE '%' + @ClientName + '%'
    """,

    CycleDomain.DEBT: """
        SELECT
            [Client].[Name] AS ClientName,
            [Agreement].[ID] AS AgreementID,
            [Agreement].[Name] AS AgreementName,
            [Currency].[Code] AS Currency,
            SUM([Debt].Total) AS DebtInCurrency,
            CONVERT(money, SUM(dbo.GetExchangedToEuroValue([Debt].Total, [Agreement].CurrencyID, GETDATE()))) AS DebtInEuro,
            DATEDIFF(DAY, MIN([Debt].[Created]), GETUTCDATE()) AS TotalDays,
            MAX([Agreement].[NumberDaysDebt]) AS AllowedDays
        FROM [ClientInDebt]
        LEFT JOIN [Debt] ON [Debt].ID = [ClientInDebt].DebtID
        LEFT JOIN [Agreement] ON [Agreement].ID = [ClientInDebt].AgreementID
        LEFT JOIN [Currency] ON [Currency].ID = [Agreement].CurrencyID
        LEFT JOIN [Client] ON [Client].ID = [ClientInDebt].ClientID
        WHERE [ClientInDebt].Deleted = 0 AND [Debt].Deleted = 0
          AND [Client].[Name] LIKE '%' + @ClientName + '%'
        GROUP BY [Client].[Name], [Agreement].[ID], [Agreement].[Name], [Currency].[Code]
        ORDER BY DebtInEuro DESC
    """,

    CycleDomain.CASH_FLOW: """
        ;WITH [AccountingCashFlow_CTE] AS (
            SELECT [ClientAgreement].ClientID,
                   'Outcome' AS PaymentType,
                   [OutcomePaymentOrder].FromDate AS PaymentDate,
                   [OutcomePaymentOrder].[Amount] * -1 AS Amount,
                   [Currency].[Code] AS Currency,
                   [dbo].[GetExchangedToEuroValue]([OutcomePaymentOrder].[Amount] * -1, [Currency].ID, GETUTCDATE()) AS EuroAmount
            FROM [OutcomePaymentOrder]
            LEFT JOIN [ClientAgreement] ON [ClientAgreement].ID = [OutcomePaymentOrder].ClientAgreementID
            LEFT JOIN [PaymentCurrencyRegister] ON [PaymentCurrencyRegister].[ID] = [OutcomePaymentOrder].[PaymentCurrencyRegisterID]
            LEFT JOIN [Currency] ON [Currency].[ID] = [PaymentCurrencyRegister].[CurrencyID]
            WHERE [OutcomePaymentOrder].Deleted = 0 AND [OutcomePaymentOrder].IsCanceled = 0

            UNION ALL

            SELECT [IncomePaymentOrder].ClientID,
                   'Income' AS PaymentType,
                   [IncomePaymentOrder].FromDate AS PaymentDate,
                   [IncomePaymentOrder].Amount AS Amount,
                   [Currency].[Code] AS Currency,
                   [IncomePaymentOrder].EuroAmount
            FROM [IncomePaymentOrder]
            LEFT JOIN [Currency] ON [Currency].ID = [IncomePaymentOrder].CurrencyID
            WHERE [IncomePaymentOrder].Deleted = 0 AND [IncomePaymentOrder].IsCanceled = 0
        )
        SELECT TOP 20
            [Client].[Name] AS ClientName,
            [AccountingCashFlow_CTE].PaymentType,
            [AccountingCashFlow_CTE].PaymentDate,
            [AccountingCashFlow_CTE].Amount,
            [AccountingCashFlow_CTE].Currency,
            [AccountingCashFlow_CTE].EuroAmount
        FROM [AccountingCashFlow_CTE]
        LEFT JOIN [Client] ON [Client].ID = [AccountingCashFlow_CTE].ClientID
        WHERE [Client].[Name] LIKE '%' + @ClientName + '%'
        ORDER BY [AccountingCashFlow_CTE].PaymentDate DESC
    """,

    CycleDomain.PURCHASES: """
        SELECT TOP 20
            [Client].[Name] AS ClientName,
            [Product].[Name] AS ProductName,
            [Product].[VendorCode] AS VendorCode,
            SUM([OrderItem].Qty) AS TotalQty,
            SUM([OrderItem].PricePerItem * [OrderItem].Qty) AS TotalAmount,
            MIN([Sale].Created) AS FirstPurchase,
            MAX([Sale].Created) AS LastPurchase
        FROM [Sale]
        LEFT JOIN [Order] ON [Order].ID = [Sale].OrderID
        LEFT JOIN [OrderItem] ON [OrderItem].OrderID = [Order].ID AND [OrderItem].Deleted = 0
        LEFT JOIN [Product] ON [Product].ID = [OrderItem].ProductID
        LEFT JOIN [ClientAgreement] ON [ClientAgreement].ID = [Sale].ClientAgreementID
        LEFT JOIN [Client] ON [Client].ID = [ClientAgreement].ClientID
        WHERE [Sale].Deleted = 0 AND [Client].[Name] LIKE '%' + @ClientName + '%'
        GROUP BY [Client].[Name], [Product].[Name], [Product].[VendorCode]
        ORDER BY TotalAmount DESC
    """,

    CycleDomain.AGREEMENTS: """
        SELECT
            [Client].[Name] AS ClientName,
            [Agreement].[ID] AS AgreementID,
            [Agreement].[Name] AS AgreementName,
            [Agreement].[Created] AS AgreementDate,
            [Currency].[Code] AS Currency,
            [Organization].[Name] AS OrganizationName,
            [Pricing].[Name] AS PricingName,
            [Agreement].[NumberDaysDebt] AS PaymentTermDays
        FROM [ClientAgreement]
        LEFT JOIN [Client] ON [Client].ID = [ClientAgreement].ClientID
        LEFT JOIN [Agreement] ON [Agreement].ID = [ClientAgreement].AgreementID
        LEFT JOIN [Currency] ON [Currency].ID = [Agreement].CurrencyID
        LEFT JOIN [Organization] ON [Organization].ID = [Agreement].OrganizationID
        LEFT JOIN [Pricing] ON [Pricing].ID = [Agreement].PricingID
        WHERE [ClientAgreement].Deleted = 0 AND [Agreement].Deleted = 0
          AND [Client].[Name] LIKE '%' + @ClientName + '%'
        ORDER BY [Agreement].[Created] DESC
    """,

    CycleDomain.INVOICES: """
        SELECT TOP 20
            [Client].[Name] AS ClientName,
            [SaleNumber].[Value] AS InvoiceNumber,
            [Sale].Created AS InvoiceDate,
            [Sale].Total AS Amount,
            [Currency].[Code] AS Currency,
            [BaseLifeCycleStatus].[Name] AS Status,
            [SaleInvoiceDocument].[ShippingAmount] AS ShippingCost
        FROM [Sale]
        LEFT JOIN [SaleNumber] ON [SaleNumber].ID = [Sale].SaleNumberID
        LEFT JOIN [SaleInvoiceDocument] ON [SaleInvoiceDocument].ID = [Sale].SaleInvoiceDocumentID
        LEFT JOIN [ClientAgreement] ON [ClientAgreement].ID = [Sale].ClientAgreementID
        LEFT JOIN [Agreement] ON [Agreement].ID = [ClientAgreement].AgreementID
        LEFT JOIN [Currency] ON [Currency].ID = [Agreement].CurrencyID
        LEFT JOIN [Client] ON [Client].ID = [ClientAgreement].ClientID
        LEFT JOIN [BaseLifeCycleStatus] ON [BaseLifeCycleStatus].ID = [Sale].LifeCycleStatusID
        WHERE [Sale].Deleted = 0 AND [Client].[Name] LIKE '%' + @ClientName + '%'
        ORDER BY [Sale].Created DESC
    """,

    CycleDomain.STRUCTURE: """
        SELECT
            [RootClient].[Name] AS RootClientName,
            [SubClient].[Name] AS SubClientName,
            [SubClient].[FullName] AS SubClientFullName,
            [SubClient].[Created] AS SubClientCreated,
            [Region].[Name] AS SubClientRegion,
            CASE WHEN [ParentLink].ID IS NOT NULL THEN 1 ELSE 0 END AS IsSubClient,
            [MainClient].[Name] AS MainClientName
        FROM [Client] AS [RootClient]
        LEFT JOIN [ClientSubClient] ON [ClientSubClient].RootClientID = [RootClient].ID AND [ClientSubClient].Deleted = 0
        LEFT JOIN [Client] AS [SubClient] ON [SubClient].ID = [ClientSubClient].SubClientID
        LEFT JOIN [RegionCode] ON [RegionCode].ID = [SubClient].RegionCodeID
        LEFT JOIN [Region] ON [Region].ID = [RegionCode].RegionID
        LEFT JOIN [ClientSubClient] AS [ParentLink] ON [ParentLink].SubClientID = [RootClient].ID AND [ParentLink].Deleted = 0
        LEFT JOIN [Client] AS [MainClient] ON [MainClient].ID = [RootClient].MainClientID
        WHERE [RootClient].Deleted = 0 AND [RootClient].[Name] LIKE '%' + @ClientName + '%'
        ORDER BY [SubClient].[Name]
    """,
}


# =============================================================================
# CYCLE REQUEST/RESPONSE
# =============================================================================

class ClientCycleRequest(BaseModel):
    """Request for client data cycle."""
    client_name: str = Field(..., min_length=1, description="Client name to search for")
    domains: list[CycleDomain] = Field(
        default_factory=lambda: list(DEFAULT_CYCLE_ORDER),
        description="Domains to fetch (in order)"
    )
    include_subsidiaries: bool = Field(False, description="Include subsidiary data")
    max_items_per_domain: int = Field(20, ge=1, le=100, description="Max items per domain")


class ClientCycleResponse(BaseModel):
    """Response from client data cycle."""
    client_name: str
    data: ClientFullProfile
    cycle_completed: bool = True
    domains_processed: list[str] = Field(default_factory=list)
    processing_time_ms: float = 0


# =============================================================================
# RELATED ENTITIES GRAPH
# =============================================================================

@dataclass
class EntityRelation:
    """Defines a relationship between entities."""
    from_entity: str
    to_entity: str
    relation_type: str  # "has_many", "belongs_to", "has_one"
    foreign_key: str
    description: str


# Client entity relationships
CLIENT_RELATIONS: list[EntityRelation] = [
    # Direct relations
    EntityRelation("Client", "ClientInDebt", "has_many", "ClientID", "Client debts"),
    EntityRelation("Client", "ClientAgreement", "has_many", "ClientID", "Client agreements"),
    EntityRelation("Client", "IncomePaymentOrder", "has_many", "ClientID", "Incoming payments"),
    EntityRelation("Client", "ClientUserProfile", "has_many", "ClientID", "Assigned managers"),
    EntityRelation("Client", "ClientInRole", "has_many", "ClientID", "Client roles/types"),
    EntityRelation("Client", "ClientBankDetails", "has_one", "ClientBankDetailsID", "Bank details"),
    EntityRelation("Client", "RegionCode", "belongs_to", "RegionCodeID", "Region"),

    # Structure relations
    EntityRelation("Client", "ClientSubClient", "has_many", "RootClientID", "Subsidiaries (as root)"),
    EntityRelation("Client", "ClientSubClient", "belongs_to", "SubClientID", "Parent (as subsidiary)"),
    EntityRelation("Client", "Client", "belongs_to", "MainClientID", "Main client"),

    # Through ClientAgreement
    EntityRelation("ClientAgreement", "Sale", "has_many", "ClientAgreementID", "Sales"),
    EntityRelation("ClientAgreement", "OutcomePaymentOrder", "has_many", "ClientAgreementID", "Outgoing payments"),
    EntityRelation("ClientAgreement", "Agreement", "belongs_to", "AgreementID", "Agreement details"),

    # Through Sale
    EntityRelation("Sale", "Order", "belongs_to", "OrderID", "Order"),
    EntityRelation("Order", "OrderItem", "has_many", "OrderID", "Order items"),
    EntityRelation("OrderItem", "Product", "belongs_to", "ProductID", "Product"),

    # Through Debt
    EntityRelation("ClientInDebt", "Debt", "belongs_to", "DebtID", "Debt details"),
    EntityRelation("ClientInDebt", "Agreement", "belongs_to", "AgreementID", "Agreement"),
]


def get_related_entities(entity: str, depth: int = 1) -> list[str]:
    """Get entities related to a given entity up to specified depth.

    Args:
        entity: Starting entity name
        depth: How many levels of relations to traverse

    Returns:
        List of related entity names
    """
    if depth <= 0:
        return []

    related = set()
    for relation in CLIENT_RELATIONS:
        if relation.from_entity == entity:
            related.add(relation.to_entity)
        elif relation.to_entity == entity:
            related.add(relation.from_entity)

    # Recursively get deeper relations
    if depth > 1:
        deeper = set()
        for rel_entity in related:
            deeper.update(get_related_entities(rel_entity, depth - 1))
        related.update(deeper)

    related.discard(entity)  # Remove self
    return sorted(related)


def get_entity_path(from_entity: str, to_entity: str) -> list[EntityRelation]:
    """Find the path of relations between two entities.

    Args:
        from_entity: Starting entity
        to_entity: Target entity

    Returns:
        List of EntityRelation representing the path
    """
    # BFS to find shortest path
    from collections import deque

    if from_entity == to_entity:
        return []

    visited = {from_entity}
    queue = deque([(from_entity, [])])

    while queue:
        current, path = queue.popleft()

        for relation in CLIENT_RELATIONS:
            next_entity = None
            if relation.from_entity == current and relation.to_entity not in visited:
                next_entity = relation.to_entity
            elif relation.to_entity == current and relation.from_entity not in visited:
                next_entity = relation.from_entity

            if next_entity:
                new_path = path + [relation]
                if next_entity == to_entity:
                    return new_path
                visited.add(next_entity)
                queue.append((next_entity, new_path))

    return []  # No path found


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_cycle_query(domain: CycleDomain) -> str:
    """Get the SQL query for a specific domain in the cycle."""
    return CYCLE_QUERIES.get(domain, "")


def get_tables_for_domain(domain: CycleDomain) -> list[str]:
    """Get the tables involved in a domain query."""
    tables_map = {
        CycleDomain.PROFILE: ["Client", "ClientInRole", "ClientType", "ClientTypeRole",
                             "ClientUserProfile", "User", "RegionCode", "Region", "Country"],
        CycleDomain.CONTACTS: ["Client"],
        CycleDomain.REGION: ["Client", "RegionCode", "Region", "Country"],
        CycleDomain.BANK: ["Client", "ClientBankDetails", "ClientBankDetailAccountNumber",
                          "ClientBankDetailIbanNo", "Currency"],
        CycleDomain.DEBT: ["Client", "ClientInDebt", "Debt", "Agreement", "Currency"],
        CycleDomain.CASH_FLOW: ["Client", "ClientAgreement", "OutcomePaymentOrder",
                               "IncomePaymentOrder", "PaymentCurrencyRegister", "Currency"],
        CycleDomain.PURCHASES: ["Client", "ClientAgreement", "Sale", "Order",
                               "OrderItem", "Product"],
        CycleDomain.AGREEMENTS: ["Client", "ClientAgreement", "Agreement", "Currency",
                                "Organization", "Pricing"],
        CycleDomain.INVOICES: ["Client", "ClientAgreement", "Sale", "SaleNumber",
                              "SaleInvoiceDocument", "Agreement", "Currency", "BaseLifeCycleStatus"],
        CycleDomain.STRUCTURE: ["Client", "ClientSubClient", "RegionCode", "Region"],
    }
    return tables_map.get(domain, [])


def format_cycle_for_prompt(domains: list[CycleDomain]) -> str:
    """Format cycle information for LLM prompt.

    Args:
        domains: List of domains to include

    Returns:
        Formatted string describing the cycle
    """
    lines = ["-- Client Data Cycle Configuration:"]

    for i, domain in enumerate(domains, 1):
        tables = get_tables_for_domain(domain)
        lines.append(f"-- {i}. {domain.value}: {', '.join(tables)}")

    lines.append("--")
    lines.append("-- The cycle fetches data in order, building a complete client profile.")

    return "\n".join(lines)

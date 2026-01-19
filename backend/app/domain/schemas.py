"""Pydantic schemas for different domain content types.

This module provides typed schemas for each business domain context:
- Debt (borhy) - client debt information
- CashFlow (vzayemorozrakhunky) - payment balances and history
- Purchases (pokupky) - products bought, sales
- Agreements (dohovory) - contracts
- Invoices (nakladni) - sales documents
- Profile (profil) - client information
- Bank (bank) - banking details
- Structure (struktura) - client hierarchy
- Region (region) - geographic info
- Contacts (kontakty) - contact details
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Generic, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================

class DomainContext(str, Enum):
    """Available domain contexts."""
    DEBT = "debt"
    CASH_FLOW = "cash_flow"
    PURCHASES = "purchases"
    AGREEMENTS = "agreements"
    INVOICES = "invoices"
    PROFILE = "profile"
    BANK = "bank"
    STRUCTURE = "structure"
    REGION = "region"
    CONTACTS = "contacts"
    UNKNOWN = "unknown"


class PaymentType(str, Enum):
    """Payment direction."""
    INCOME = "income"
    OUTCOME = "outcome"


class Currency(str, Enum):
    """Common currencies."""
    EUR = "EUR"
    USD = "USD"
    UAH = "UAH"
    PLN = "PLN"


# =============================================================================
# BASE SCHEMAS
# =============================================================================

class ClientBase(BaseModel):
    """Base client information present in most responses."""
    client_name: str = Field(..., alias="ClientName", description="Client name")

    class Config:
        populate_by_name = True


class MonetaryAmount(BaseModel):
    """Amount with optional currency."""
    amount: Decimal = Field(..., description="Monetary amount")
    currency: Optional[str] = Field(None, description="Currency code (EUR, USD, etc.)")


# =============================================================================
# DEBT SCHEMAS (borhy)
# =============================================================================

class DebtCurrent(ClientBase):
    """Current debt of a client."""
    total_debt_euro: Decimal = Field(..., alias="TotalDebtEuro", description="Total debt in EUR")


class DebtHistorical(ClientBase):
    """Debt as of a specific date."""
    total_debt_euro: Decimal = Field(..., alias="TotalDebtEuro", description="Total debt in EUR")
    as_of_date: date = Field(..., alias="AsOfDate", description="Date of debt snapshot")


class DebtByAgreement(ClientBase):
    """Debt breakdown by agreement."""
    agreement_id: int = Field(..., alias="AgreementID")
    agreement_name: str = Field(..., alias="AgreementName")
    currency: str = Field(..., alias="Currency")
    debt_in_currency: Decimal = Field(..., alias="DebtInCurrency")
    debt_in_euro: Decimal = Field(..., alias="DebtInEuro")


class DebtOverdue(ClientBase):
    """Overdue debt information."""
    debt_amount: Decimal = Field(..., alias="DebtAmount")
    debt_date: datetime = Field(..., alias="DebtDate")
    total_days: int = Field(..., alias="TotalDays")
    allowed_days: int = Field(..., alias="AllowedDays")
    days_overdue: int = Field(..., alias="DaysOverdue")


class DebtStructure(BaseModel):
    """Debt including all subsidiaries."""
    root_client_name: str = Field(..., alias="RootClientName")
    total_structure_debt: Decimal = Field(..., alias="TotalStructureDebt")

    class Config:
        populate_by_name = True


class DebtCustomRate(ClientBase):
    """Debt calculated with custom exchange rate."""
    debt_in_original_currency: Decimal = Field(..., alias="DebtInOriginalCurrency")
    debt_in_euro_at_custom_rate: Decimal = Field(..., alias="DebtInEuroAtCustomRate")
    exchange_rate: Decimal = Field(..., alias="ExchangeRate")


# =============================================================================
# CASH FLOW SCHEMAS (vzayemorozrakhunky)
# =============================================================================

class CashFlowBalance(ClientBase):
    """Client balance (payments summary)."""
    balance: Decimal = Field(..., alias="Balance", description="Net balance (positive = overpaid)")


class CashFlowPayment(ClientBase):
    """Single payment record."""
    payment_type: str = Field(..., alias="PaymentType")
    payment_date: datetime = Field(..., alias="PaymentDate")
    amount: Decimal = Field(..., alias="Amount")
    currency: str = Field(..., alias="Currency")
    euro_amount: Decimal = Field(..., alias="EuroAmount")


class CashFlowPeriod(ClientBase):
    """Payments summary for a period."""
    total_income: Decimal = Field(..., alias="TotalIncome")
    total_outcome: Decimal = Field(..., alias="TotalOutcome")


# =============================================================================
# PURCHASE SCHEMAS (pokupky)
# =============================================================================

class PurchaseProduct(ClientBase):
    """Product purchased by client."""
    product_name: str = Field(..., alias="ProductName")
    vendor_code: Optional[str] = Field(None, alias="VendorCode")
    total_qty: int = Field(..., alias="TotalQty")
    total_amount: Decimal = Field(..., alias="TotalAmount")
    first_purchase: datetime = Field(..., alias="FirstPurchase")
    last_purchase: datetime = Field(..., alias="LastPurchase")


class PurchaseSale(ClientBase):
    """Sale to a client."""
    sale_number: str = Field(..., alias="SaleNumber")
    sale_date: datetime = Field(..., alias="SaleDate")
    item_count: int = Field(..., alias="ItemCount")
    total_amount: Decimal = Field(..., alias="TotalAmount")
    currency: str = Field(..., alias="Currency")
    status: str = Field(..., alias="Status")


class PurchaseTotal(ClientBase):
    """Total purchases summary."""
    sales_count: int = Field(..., alias="SalesCount")
    total_amount: Decimal = Field(..., alias="TotalAmount")
    total_euro: Decimal = Field(..., alias="TotalEuro")


# =============================================================================
# AGREEMENT SCHEMAS (dohovory)
# =============================================================================

class Agreement(ClientBase):
    """Client agreement (contract)."""
    agreement_id: int = Field(..., alias="AgreementID")
    agreement_name: str = Field(..., alias="AgreementName")
    agreement_date: datetime = Field(..., alias="AgreementDate")
    currency: str = Field(..., alias="Currency")
    organization_name: Optional[str] = Field(None, alias="OrganizationName")
    pricing_name: Optional[str] = Field(None, alias="PricingName")
    payment_term_days: Optional[int] = Field(None, alias="PaymentTermDays")


class AgreementActive(ClientBase):
    """Active agreement for a client."""
    agreement_name: str = Field(..., alias="AgreementName")
    currency: str = Field(..., alias="Currency")
    payment_term_days: Optional[int] = Field(None, alias="PaymentTermDays")
    pricing_name: Optional[str] = Field(None, alias="PricingName")


# =============================================================================
# INVOICE SCHEMAS (nakladni)
# =============================================================================

class Invoice(ClientBase):
    """Sales invoice."""
    invoice_number: str = Field(..., alias="InvoiceNumber")
    invoice_date: datetime = Field(..., alias="InvoiceDate")
    amount: Decimal = Field(..., alias="Amount")
    currency: str = Field(..., alias="Currency")
    status: str = Field(..., alias="Status")
    shipping_cost: Optional[Decimal] = Field(None, alias="ShippingCost")


# =============================================================================
# PROFILE SCHEMAS (profil)
# =============================================================================

class ClientProfile(ClientBase):
    """Full client profile."""
    full_name: Optional[str] = Field(None, alias="FullName")
    registration_date: datetime = Field(..., alias="RegistrationDate")
    client_type: Optional[str] = Field(None, alias="ClientType")
    client_role: Optional[str] = Field(None, alias="ClientRole")
    manager_name: Optional[str] = Field(None, alias="ManagerName")
    region_name: Optional[str] = Field(None, alias="RegionName")
    country_name: Optional[str] = Field(None, alias="CountryName")


class ClientManager(ClientBase):
    """Client's manager information."""
    manager_name: str = Field(..., alias="ManagerName")
    manager_email: Optional[str] = Field(None, alias="ManagerEmail")


# =============================================================================
# BANK SCHEMAS (bank)
# =============================================================================

class BankDetails(ClientBase):
    """Client banking details."""
    bank_name: Optional[str] = Field(None, alias="BankName")
    bank_address: Optional[str] = Field(None, alias="BankAddress")
    swift: Optional[str] = Field(None, alias="Swift")
    branch_code: Optional[str] = Field(None, alias="BranchCode")
    account_number: Optional[str] = Field(None, alias="AccountNumber")
    account_currency: Optional[str] = Field(None, alias="AccountCurrency")
    iban: Optional[str] = Field(None, alias="IBAN")
    iban_currency: Optional[str] = Field(None, alias="IbanCurrency")


class BankIban(ClientBase):
    """Client IBAN only."""
    iban: str = Field(..., alias="IBAN")
    currency: Optional[str] = Field(None, alias="Currency")


# =============================================================================
# STRUCTURE SCHEMAS (struktura)
# =============================================================================

class SubClient(BaseModel):
    """Subsidiary client information."""
    root_client_name: str = Field(..., alias="RootClientName")
    sub_client_name: str = Field(..., alias="SubClientName")
    sub_client_full_name: Optional[str] = Field(None, alias="SubClientFullName")
    sub_client_created: Optional[datetime] = Field(None, alias="SubClientCreated")
    sub_client_region: Optional[str] = Field(None, alias="SubClientRegion")

    class Config:
        populate_by_name = True


class ParentClient(ClientBase):
    """Parent client for a subsidiary."""
    parent_client_name: Optional[str] = Field(None, alias="ParentClientName")
    parent_full_name: Optional[str] = Field(None, alias="ParentFullName")


class ClientHierarchy(ClientBase):
    """Client hierarchy information."""
    is_sub_client: bool = Field(..., alias="IsSubClient")
    has_sub_clients: bool = Field(..., alias="HasSubClients")
    sub_client_count: int = Field(..., alias="SubClientCount")
    main_client_name: Optional[str] = Field(None, alias="MainClientName")


# =============================================================================
# REGION SCHEMAS (region)
# =============================================================================

class ClientRegion(ClientBase):
    """Client geographic information."""
    region_name: Optional[str] = Field(None, alias="RegionName")
    country_name: Optional[str] = Field(None, alias="CountryName")
    region_code: Optional[str] = Field(None, alias="RegionCode")


class RegionClient(BaseModel):
    """Client in a region."""
    region_name: str = Field(..., alias="RegionName")
    client_name: str = Field(..., alias="ClientName")
    full_name: Optional[str] = Field(None, alias="FullName")
    client_type: Optional[str] = Field(None, alias="ClientType")

    class Config:
        populate_by_name = True


# =============================================================================
# CONTACTS SCHEMAS (kontakty)
# =============================================================================

class ClientContacts(ClientBase):
    """Client contact information."""
    full_name: Optional[str] = Field(None, alias="FullName")
    mobile_number: Optional[str] = Field(None, alias="MobileNumber")
    email: Optional[str] = Field(None, alias="Email")
    tax_id: Optional[str] = Field(None, alias="TaxID")
    usreou: Optional[str] = Field(None, alias="USREOU")


# =============================================================================
# GENERIC RESPONSE WRAPPER
# =============================================================================

T = TypeVar("T", bound=BaseModel)


class DomainResponse(BaseModel, Generic[T]):
    """Generic wrapper for domain-specific responses."""
    context: DomainContext = Field(..., description="Business domain context")
    query_id: str = Field(..., description="Query template ID used")
    data: list[T] = Field(default_factory=list, description="Typed result rows")
    row_count: int = Field(0, description="Number of rows returned")

    class Config:
        populate_by_name = True


# =============================================================================
# SCHEMA MAPPINGS
# =============================================================================

# Map query IDs to their response schemas
QUERY_SCHEMAS: dict[str, type[BaseModel]] = {
    # Debt
    "debt_current": DebtCurrent,
    "debt_historical": DebtHistorical,
    "debt_by_agreement": DebtByAgreement,
    "debt_overdue": DebtOverdue,
    "debt_structure": DebtStructure,
    "debt_custom_rate": DebtCustomRate,
    # Cash Flow
    "cashflow_balance": CashFlowBalance,
    "cashflow_history": CashFlowPayment,
    "cashflow_period": CashFlowPeriod,
    # Purchases
    "purchases_products": PurchaseProduct,
    "purchases_sales": PurchaseSale,
    "purchases_total": PurchaseTotal,
    # Agreements
    "agreements_list": Agreement,
    "agreements_active": AgreementActive,
    # Invoices
    "invoices_list": Invoice,
    # Profile
    "profile_info": ClientProfile,
    "profile_manager": ClientManager,
    # Bank
    "bank_details": BankDetails,
    "bank_iban": BankIban,
    # Structure
    "structure_subclients": SubClient,
    "structure_parent": ParentClient,
    "structure_hierarchy": ClientHierarchy,
    # Region
    "region_info": ClientRegion,
    "region_clients": RegionClient,
    # Contacts
    "contacts_info": ClientContacts,
}


# Map contexts to their primary schemas
CONTEXT_SCHEMAS: dict[DomainContext, list[type[BaseModel]]] = {
    DomainContext.DEBT: [DebtCurrent, DebtHistorical, DebtByAgreement, DebtOverdue, DebtStructure, DebtCustomRate],
    DomainContext.CASH_FLOW: [CashFlowBalance, CashFlowPayment, CashFlowPeriod],
    DomainContext.PURCHASES: [PurchaseProduct, PurchaseSale, PurchaseTotal],
    DomainContext.AGREEMENTS: [Agreement, AgreementActive],
    DomainContext.INVOICES: [Invoice],
    DomainContext.PROFILE: [ClientProfile, ClientManager],
    DomainContext.BANK: [BankDetails, BankIban],
    DomainContext.STRUCTURE: [SubClient, ParentClient, ClientHierarchy],
    DomainContext.REGION: [ClientRegion, RegionClient],
    DomainContext.CONTACTS: [ClientContacts],
}


def get_schema_for_query(query_id: str) -> Optional[type[BaseModel]]:
    """Get the Pydantic schema for a specific query ID."""
    return QUERY_SCHEMAS.get(query_id)


def parse_rows_to_schema(
    query_id: str,
    columns: list[str],
    rows: list[list[Any]]
) -> list[BaseModel]:
    """Parse raw query results into typed Pydantic models.

    Args:
        query_id: The query template ID
        columns: Column names from the query
        rows: Raw data rows

    Returns:
        List of Pydantic model instances
    """
    schema = QUERY_SCHEMAS.get(query_id)
    if not schema:
        return []

    results = []
    for row in rows:
        # Create dict from columns and row values
        data = dict(zip(columns, row))
        try:
            results.append(schema.model_validate(data))
        except Exception:
            # If validation fails, skip the row
            continue

    return results

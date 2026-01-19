"""Client-related query templates from GBA repository.

Covers all client business contexts:
- Debt (борги)
- Cash Flow (взаєморозрахунки)
- Purchases (покупки)
- Agreements (договори)
- Invoices (накладні, рахунки)
- Profile (тип, роль, менеджер)
- Structure (філії, підлеглі)
"""

from __future__ import annotations

from typing import Optional

from .base import ClientQuery, ExtractedParams

# =============================================================================
# DEBT QUERIES (борги)
# =============================================================================

DEBT_QUERIES = {
    "debt_current": ClientQuery(
        id="debt_current",
        context="debt",
        name_uk="Поточний борг клієнта",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    CONVERT(money, SUM(dbo.GetExchangedToEuroValue([Debt].Total, [Agreement].CurrencyID, GETDATE()))) AS TotalDebtEuro
FROM [ClientInDebt]
LEFT JOIN [Debt] ON [Debt].ID = [ClientInDebt].DebtID
LEFT JOIN [Agreement] ON [Agreement].ID = [ClientInDebt].AgreementID
LEFT JOIN [Client] ON [Client].ID = [ClientInDebt].ClientID
WHERE [ClientInDebt].Deleted = 0 AND [Debt].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
GROUP BY [Client].[Name]
""",
        tables=["Client", "ClientInDebt", "Debt", "Agreement"],
        notes="dbo.GetExchangedToEuroValue конвертує в EUR"
    ),

    "debt_historical": ClientQuery(
        id="debt_historical",
        context="debt",
        name_uk="Борг на певну дату",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    CONVERT(money, SUM(dbo.GetExchangedToEuroValue([Debt].Total, [Agreement].CurrencyID, @AsOfDate))) AS TotalDebtEuro,
    @AsOfDate AS AsOfDate
FROM [ClientInDebt]
LEFT JOIN [Debt] ON [Debt].ID = [ClientInDebt].DebtID
LEFT JOIN [Agreement] ON [Agreement].ID = [ClientInDebt].AgreementID
LEFT JOIN [Client] ON [Client].ID = [ClientInDebt].ClientID
WHERE [ClientInDebt].Deleted = 0 AND [Debt].Deleted = 0
  AND [Debt].[Created] <= @AsOfDate
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
GROUP BY [Client].[Name]
-- @AsOfDate = DATEADD(DAY, -1, GETDATE()) для вчора
""",
        tables=["Client", "ClientInDebt", "Debt", "Agreement"],
        notes="Фільтр Debt.Created <= @AsOfDate"
    ),

    "debt_by_agreement": ClientQuery(
        id="debt_by_agreement",
        context="debt",
        name_uk="Борг по договорам",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    [Agreement].[ID] AS AgreementID,
    [Agreement].[Name] AS AgreementName,
    [Currency].[Code] AS Currency,
    SUM([Debt].Total) AS DebtInCurrency,
    CONVERT(money, SUM(dbo.GetExchangedToEuroValue([Debt].Total, [Agreement].CurrencyID, GETDATE()))) AS DebtInEuro
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
        tables=["Client", "ClientInDebt", "Debt", "Agreement", "Currency"],
        notes="GROUP BY Agreement для розбивки по договорам"
    ),

    "debt_overdue": ClientQuery(
        id="debt_overdue",
        context="debt",
        name_uk="Прострочена заборгованість",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    [Debt].[Total] AS DebtAmount,
    [Debt].[Created] AS DebtDate,
    DATEDIFF(DAY, [Debt].[Created], GETUTCDATE()) AS TotalDays,
    [Agreement].[NumberDaysDebt] AS AllowedDays,
    CASE
      WHEN DATEDIFF(DAY, [Debt].[Created], GETUTCDATE()) > [Agreement].[NumberDaysDebt]
      THEN DATEDIFF(DAY, [Debt].[Created], GETUTCDATE()) - [Agreement].[NumberDaysDebt]
      ELSE 0
    END AS DaysOverdue
FROM [ClientInDebt]
LEFT JOIN [Debt] ON [Debt].ID = [ClientInDebt].DebtID
LEFT JOIN [Agreement] ON [Agreement].ID = [ClientInDebt].AgreementID
LEFT JOIN [Client] ON [Client].ID = [ClientInDebt].ClientID
WHERE [ClientInDebt].Deleted = 0 AND [Debt].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
ORDER BY DaysOverdue DESC
""",
        tables=["Client", "ClientInDebt", "Debt", "Agreement"],
        notes="DaysOverdue = фактичні дні - дозволені дні"
    ),

    "debt_structure": ClientQuery(
        id="debt_structure",
        context="debt",
        name_uk="Борг по структурі (з філіями)",
        sql="""
WITH SubClientDebts_CTE AS (
    SELECT SUM(Debt.Total) AS TotalSubDebt, ClientSubClient.RootClientID
    FROM ClientInDebt
    LEFT JOIN Debt ON Debt.ID = ClientInDebt.DebtID
    LEFT JOIN ClientSubClient ON ClientSubClient.SubClientID = ClientInDebt.ClientID
    WHERE ClientInDebt.Deleted = 0 AND Debt.Deleted = 0
    GROUP BY ClientSubClient.RootClientID
)
SELECT
    [Client].[Name] AS RootClientName,
    ISNULL(SubClientDebts_CTE.TotalSubDebt, 0) AS TotalStructureDebt
FROM [Client]
LEFT JOIN SubClientDebts_CTE ON SubClientDebts_CTE.RootClientID = [Client].ID
WHERE [Client].[Name] LIKE '%' + @ClientName + '%' AND [Client].Deleted = 0
""",
        tables=["Client", "ClientInDebt", "Debt", "ClientSubClient"],
        notes="CTE підсумовує борги всіх SubClient"
    ),

    "debt_custom_rate": ClientQuery(
        id="debt_custom_rate",
        context="debt",
        name_uk="Борг з користувацьким курсом",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    SUM([Debt].Total) AS DebtInOriginalCurrency,
    CONVERT(money, SUM([Debt].Total / @Rate)) AS DebtInEuroAtCustomRate,
    @Rate AS ExchangeRate
FROM [ClientInDebt]
LEFT JOIN [Debt] ON [Debt].ID = [ClientInDebt].DebtID
LEFT JOIN [Agreement] ON [Agreement].ID = [ClientInDebt].AgreementID
LEFT JOIN [Client] ON [Client].ID = [ClientInDebt].ClientID
WHERE [ClientInDebt].Deleted = 0 AND [Debt].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
GROUP BY [Client].[Name]
-- @Rate = user-provided exchange rate (e.g., 41.5)
""",
        tables=["Client", "ClientInDebt", "Debt", "Agreement"],
        notes="@Rate - курс введений користувачем (напр. 'по 41.5')"
    ),
}

# =============================================================================
# CASH FLOW QUERIES (взаєморозрахунки)
# =============================================================================

CASH_FLOW_QUERIES = {
    "cashflow_balance": ClientQuery(
        id="cashflow_balance",
        context="cash_flow",
        name_uk="Баланс клієнта (взаєморозрахунки)",
        sql="""
;WITH [AccountingCashFlow_CTE] AS (
    -- Outgoing payments (negative)
    SELECT [ClientAgreement].ClientID,
           SUM([dbo].[GetExchangedToEuroValue]([OutcomePaymentOrder].[Amount] * -1, [Currency].ID, GETUTCDATE())) AS [Amount]
    FROM [OutcomePaymentOrder]
    LEFT JOIN [ClientAgreement] ON [ClientAgreement].ID = [OutcomePaymentOrder].ClientAgreementID
    LEFT JOIN [Agreement] ON [Agreement].[ID] = [ClientAgreement].[AgreementID]
    LEFT JOIN [PaymentCurrencyRegister] ON [PaymentCurrencyRegister].[ID] = [OutcomePaymentOrder].[PaymentCurrencyRegisterID]
    LEFT JOIN [Currency] ON [Currency].[ID] = [PaymentCurrencyRegister].[CurrencyID]
    WHERE [OutcomePaymentOrder].Deleted = 0 AND [OutcomePaymentOrder].IsCanceled = 0
    GROUP BY [ClientAgreement].ClientID

    UNION ALL

    -- Incoming payments (positive)
    SELECT [IncomePaymentOrder].ClientID, SUM([IncomePaymentOrder].EuroAmount) AS [Amount]
    FROM [IncomePaymentOrder]
    WHERE [IncomePaymentOrder].Deleted = 0 AND [IncomePaymentOrder].IsCanceled = 0
    GROUP BY [IncomePaymentOrder].ClientID
)
SELECT
    [Client].[Name] AS ClientName,
    ROUND(SUM([AccountingCashFlow_CTE].Amount), 2) AS Balance
FROM [AccountingCashFlow_CTE]
LEFT JOIN [Client] ON [Client].ID = [AccountingCashFlow_CTE].ClientID
WHERE [Client].[Name] LIKE '%' + @ClientName + '%'
GROUP BY [Client].[Name]
""",
        tables=["Client", "ClientAgreement", "OutcomePaymentOrder", "IncomePaymentOrder", "Currency"],
        notes="UNION сумує платежі: вихідні (-) + вхідні (+)"
    ),

    "cashflow_history": ClientQuery(
        id="cashflow_history",
        context="cash_flow",
        name_uk="Історія платежів клієнта",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    'Outcome' AS PaymentType,
    [OutcomePaymentOrder].FromDate AS PaymentDate,
    [OutcomePaymentOrder].Amount * -1 AS Amount,
    [Currency].[Code] AS Currency,
    [dbo].[GetExchangedToEuroValue]([OutcomePaymentOrder].[Amount] * -1, [Currency].ID, [OutcomePaymentOrder].FromDate) AS EuroAmount
FROM [OutcomePaymentOrder]
LEFT JOIN [ClientAgreement] ON [ClientAgreement].ID = [OutcomePaymentOrder].ClientAgreementID
LEFT JOIN [Client] ON [Client].ID = [ClientAgreement].ClientID
LEFT JOIN [PaymentCurrencyRegister] ON [PaymentCurrencyRegister].ID = [OutcomePaymentOrder].PaymentCurrencyRegisterID
LEFT JOIN [Currency] ON [Currency].ID = [PaymentCurrencyRegister].CurrencyID
WHERE [OutcomePaymentOrder].Deleted = 0 AND [Client].[Name] LIKE '%' + @ClientName + '%'

UNION ALL

SELECT
    [Client].[Name] AS ClientName,
    'Income' AS PaymentType,
    [IncomePaymentOrder].FromDate AS PaymentDate,
    [IncomePaymentOrder].Amount AS Amount,
    [Currency].[Code] AS Currency,
    [IncomePaymentOrder].EuroAmount
FROM [IncomePaymentOrder]
LEFT JOIN [Client] ON [Client].ID = [IncomePaymentOrder].ClientID
LEFT JOIN [Currency] ON [Currency].ID = [IncomePaymentOrder].CurrencyID
WHERE [IncomePaymentOrder].Deleted = 0 AND [Client].[Name] LIKE '%' + @ClientName + '%'

ORDER BY PaymentDate DESC
""",
        tables=["Client", "ClientAgreement", "OutcomePaymentOrder", "IncomePaymentOrder", "Currency"],
        notes="UNION для об'єднання вхідних і вихідних платежів"
    ),

    "cashflow_period": ClientQuery(
        id="cashflow_period",
        context="cash_flow",
        name_uk="Платежі за період",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    SUM(CASE WHEN [IncomePaymentOrder].ID IS NOT NULL THEN [IncomePaymentOrder].EuroAmount ELSE 0 END) AS TotalIncome,
    SUM(CASE WHEN [OutcomePaymentOrder].ID IS NOT NULL
        THEN [dbo].[GetExchangedToEuroValue]([OutcomePaymentOrder].[Amount], [Currency].ID, GETUTCDATE())
        ELSE 0 END) AS TotalOutcome
FROM [Client]
LEFT JOIN [IncomePaymentOrder] ON [IncomePaymentOrder].ClientID = [Client].ID
    AND [IncomePaymentOrder].FromDate BETWEEN @FromDate AND @ToDate
    AND [IncomePaymentOrder].Deleted = 0
LEFT JOIN [ClientAgreement] ON [ClientAgreement].ClientID = [Client].ID
LEFT JOIN [OutcomePaymentOrder] ON [OutcomePaymentOrder].ClientAgreementID = [ClientAgreement].ID
    AND [OutcomePaymentOrder].FromDate BETWEEN @FromDate AND @ToDate
    AND [OutcomePaymentOrder].Deleted = 0
LEFT JOIN [PaymentCurrencyRegister] ON [PaymentCurrencyRegister].ID = [OutcomePaymentOrder].PaymentCurrencyRegisterID
LEFT JOIN [Currency] ON [Currency].ID = [PaymentCurrencyRegister].CurrencyID
WHERE [Client].[Name] LIKE '%' + @ClientName + '%'
GROUP BY [Client].[Name]
""",
        tables=["Client", "ClientAgreement", "OutcomePaymentOrder", "IncomePaymentOrder", "Currency"],
        notes="@FromDate, @ToDate - період для фільтрації"
    ),
}

# =============================================================================
# PURCHASE QUERIES (покупки клієнта)
# =============================================================================

PURCHASE_QUERIES = {
    "purchases_products": ClientQuery(
        id="purchases_products",
        context="purchases",
        name_uk="Які товари купував клієнт",
        sql="""
SELECT
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
WHERE [Sale].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
GROUP BY [Client].[Name], [Product].[Name], [Product].[VendorCode]
ORDER BY TotalAmount DESC
""",
        tables=["Client", "ClientAgreement", "Sale", "Order", "OrderItem", "Product"],
        notes="Групування по Product для списку товарів"
    ),

    "purchases_sales": ClientQuery(
        id="purchases_sales",
        context="purchases",
        name_uk="Продажі клієнту",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    [SaleNumber].[Value] AS SaleNumber,
    [Sale].Created AS SaleDate,
    COUNT([OrderItem].ID) AS ItemCount,
    SUM([OrderItem].PricePerItem * [OrderItem].Qty) AS TotalAmount,
    [Currency].[Code] AS Currency,
    [BaseLifeCycleStatus].[Name] AS Status
FROM [Sale]
LEFT JOIN [SaleNumber] ON [SaleNumber].ID = [Sale].SaleNumberID
LEFT JOIN [Order] ON [Order].ID = [Sale].OrderID
LEFT JOIN [OrderItem] ON [OrderItem].OrderID = [Order].ID AND [OrderItem].Deleted = 0
LEFT JOIN [ClientAgreement] ON [ClientAgreement].ID = [Sale].ClientAgreementID
LEFT JOIN [Agreement] ON [Agreement].ID = [ClientAgreement].AgreementID
LEFT JOIN [Currency] ON [Currency].ID = [Agreement].CurrencyID
LEFT JOIN [Client] ON [Client].ID = [ClientAgreement].ClientID
LEFT JOIN [BaseLifeCycleStatus] ON [BaseLifeCycleStatus].ID = [Sale].LifeCycleStatusID
WHERE [Sale].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
GROUP BY [Client].[Name], [SaleNumber].[Value], [Sale].Created, [Currency].[Code], [BaseLifeCycleStatus].[Name]
ORDER BY [Sale].Created DESC
""",
        tables=["Client", "ClientAgreement", "Sale", "SaleNumber", "Order", "OrderItem", "Currency"],
        notes="Список продажів (накладних) клієнту"
    ),

    "purchases_total": ClientQuery(
        id="purchases_total",
        context="purchases",
        name_uk="Загальна сума покупок",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    COUNT(DISTINCT [Sale].ID) AS SalesCount,
    SUM([OrderItem].PricePerItem * [OrderItem].Qty) AS TotalAmount,
    SUM([dbo].[GetExchangedToEuroValue](
        [OrderItem].PricePerItem * [OrderItem].Qty,
        [Agreement].CurrencyID,
        GETDATE()
    )) AS TotalEuro
FROM [Sale]
LEFT JOIN [Order] ON [Order].ID = [Sale].OrderID
LEFT JOIN [OrderItem] ON [OrderItem].OrderID = [Order].ID AND [OrderItem].Deleted = 0
LEFT JOIN [ClientAgreement] ON [ClientAgreement].ID = [Sale].ClientAgreementID
LEFT JOIN [Agreement] ON [Agreement].ID = [ClientAgreement].AgreementID
LEFT JOIN [Client] ON [Client].ID = [ClientAgreement].ClientID
WHERE [Sale].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
GROUP BY [Client].[Name]
""",
        tables=["Client", "ClientAgreement", "Sale", "Order", "OrderItem", "Agreement"],
        notes="Загальна статистика покупок клієнта"
    ),
}

# =============================================================================
# AGREEMENT QUERIES (договори)
# =============================================================================

AGREEMENT_QUERIES = {
    "agreements_list": ClientQuery(
        id="agreements_list",
        context="agreements",
        name_uk="Договори клієнта",
        sql="""
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
WHERE [ClientAgreement].Deleted = 0
  AND [Agreement].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
ORDER BY [Agreement].[Created] DESC
""",
        tables=["Client", "ClientAgreement", "Agreement", "Currency", "Organization", "Pricing"],
        notes="Список всіх активних договорів клієнта"
    ),

    "agreements_active": ClientQuery(
        id="agreements_active",
        context="agreements",
        name_uk="Активний договір клієнта",
        sql="""
SELECT TOP 1
    [Client].[Name] AS ClientName,
    [Agreement].[Name] AS AgreementName,
    [Currency].[Code] AS Currency,
    [Agreement].[NumberDaysDebt] AS PaymentTermDays,
    [Pricing].[Name] AS PricingName
FROM [ClientAgreement]
LEFT JOIN [Client] ON [Client].ID = [ClientAgreement].ClientID
LEFT JOIN [Agreement] ON [Agreement].ID = [ClientAgreement].AgreementID
LEFT JOIN [Currency] ON [Currency].ID = [Agreement].CurrencyID
LEFT JOIN [Pricing] ON [Pricing].ID = [Agreement].PricingID
WHERE [ClientAgreement].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
ORDER BY [ClientAgreement].[Created] DESC
""",
        tables=["Client", "ClientAgreement", "Agreement", "Currency", "Pricing"],
        notes="TOP 1 для останнього активного договору"
    ),
}

# =============================================================================
# INVOICE QUERIES (накладні, рахунки)
# =============================================================================

INVOICE_QUERIES = {
    "invoices_list": ClientQuery(
        id="invoices_list",
        context="invoices",
        name_uk="Накладні клієнта",
        sql="""
SELECT
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
WHERE [Sale].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
ORDER BY [Sale].Created DESC
""",
        tables=["Client", "ClientAgreement", "Sale", "SaleNumber", "SaleInvoiceDocument", "Currency"],
        notes="Список накладних (Sale) клієнта"
    ),
}

# =============================================================================
# PROFILE QUERIES (профіль клієнта)
# =============================================================================

PROFILE_QUERIES = {
    "profile_info": ClientQuery(
        id="profile_info",
        context="profile",
        name_uk="Інформація про клієнта",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    [Client].[FullName] AS FullName,
    [Client].[Created] AS RegistrationDate,
    [ClientType].[Name] AS ClientType,
    [ClientTypeRole].[Name] AS ClientRole,
    [User].[LastName] + ' ' + [User].[FirstName] AS ManagerName,
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
WHERE [Client].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
""",
        tables=["Client", "ClientInRole", "ClientType", "ClientTypeRole", "ClientUserProfile", "User", "Region", "Country"],
        notes="Основна інформація про клієнта"
    ),

    "profile_manager": ClientQuery(
        id="profile_manager",
        context="profile",
        name_uk="Менеджер клієнта",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    [User].[LastName] + ' ' + [User].[FirstName] AS ManagerName,
    [User].[Email] AS ManagerEmail
FROM [Client]
LEFT JOIN [ClientUserProfile] ON [ClientUserProfile].ClientID = [Client].ID AND [ClientUserProfile].Deleted = 0
LEFT JOIN [User] ON [User].ID = [ClientUserProfile].UserProfileID
WHERE [Client].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
""",
        tables=["Client", "ClientUserProfile", "User"],
        notes="Менеджер відповідальний за клієнта"
    ),
}

# =============================================================================
# BANK QUERIES (банківські реквізити)
# =============================================================================

BANK_QUERIES = {
    "bank_details": ClientQuery(
        id="bank_details",
        context="bank",
        name_uk="Банківські реквізити клієнта",
        sql="""
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
WHERE [Client].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
""",
        tables=["Client", "ClientBankDetails", "ClientBankDetailAccountNumber", "ClientBankDetailIbanNo", "Currency"],
        notes="Банк, рахунок, IBAN клієнта"
    ),

    "bank_iban": ClientQuery(
        id="bank_iban",
        context="bank",
        name_uk="IBAN клієнта",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    [ClientBankDetailIbanNo].[IBANNO] AS IBAN,
    [Currency].[Code] AS Currency
FROM [Client]
LEFT JOIN [ClientBankDetails] ON [ClientBankDetails].ID = [Client].ClientBankDetailsID
LEFT JOIN [ClientBankDetailIbanNo] ON [ClientBankDetailIbanNo].ID = [ClientBankDetails].ClientBankDetailIbanNoID
LEFT JOIN [Currency] ON [Currency].ID = [ClientBankDetailIbanNo].CurrencyID
WHERE [Client].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
""",
        tables=["Client", "ClientBankDetails", "ClientBankDetailIbanNo", "Currency"],
        notes="Тільки IBAN рахунок"
    ),
}

# =============================================================================
# STRUCTURE QUERIES (структура клієнта - філії)
# =============================================================================

STRUCTURE_QUERIES = {
    "structure_subclients": ClientQuery(
        id="structure_subclients",
        context="structure",
        name_uk="Філії/підлеглі клієнти",
        sql="""
SELECT
    [RootClient].[Name] AS RootClientName,
    [SubClient].[Name] AS SubClientName,
    [SubClient].[FullName] AS SubClientFullName,
    [SubClient].[Created] AS SubClientCreated,
    [Region].[Name] AS SubClientRegion
FROM [Client] AS [RootClient]
LEFT JOIN [ClientSubClient] ON [ClientSubClient].RootClientID = [RootClient].ID AND [ClientSubClient].Deleted = 0
LEFT JOIN [Client] AS [SubClient] ON [SubClient].ID = [ClientSubClient].SubClientID
LEFT JOIN [RegionCode] ON [RegionCode].ID = [SubClient].RegionCodeID
LEFT JOIN [Region] ON [Region].ID = [RegionCode].RegionID
WHERE [RootClient].Deleted = 0
  AND [RootClient].[Name] LIKE '%' + @ClientName + '%'
ORDER BY [SubClient].[Name]
""",
        tables=["Client", "ClientSubClient", "RegionCode", "Region"],
        notes="Список всіх підлеглих клієнтів (філій)"
    ),

    "structure_parent": ClientQuery(
        id="structure_parent",
        context="structure",
        name_uk="Головний клієнт",
        sql="""
SELECT
    [SubClient].[Name] AS ClientName,
    [RootClient].[Name] AS ParentClientName,
    [RootClient].[FullName] AS ParentFullName
FROM [Client] AS [SubClient]
LEFT JOIN [ClientSubClient] ON [ClientSubClient].SubClientID = [SubClient].ID AND [ClientSubClient].Deleted = 0
LEFT JOIN [Client] AS [RootClient] ON [RootClient].ID = [ClientSubClient].RootClientID
WHERE [SubClient].Deleted = 0
  AND [SubClient].[Name] LIKE '%' + @ClientName + '%'
""",
        tables=["Client", "ClientSubClient"],
        notes="Знайти головного клієнта для філії"
    ),

    "structure_hierarchy": ClientQuery(
        id="structure_hierarchy",
        context="structure",
        name_uk="Ієрархія клієнта",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    [Client].[IsSubClient] AS IsSubClient,
    CASE WHEN [ClientSubClient].ID IS NOT NULL THEN 1 ELSE 0 END AS HasSubClients,
    (SELECT COUNT(*) FROM [ClientSubClient] cs WHERE cs.RootClientID = [Client].ID AND cs.Deleted = 0) AS SubClientCount,
    [MainClient].[Name] AS MainClientName
FROM [Client]
LEFT JOIN [ClientSubClient] ON [ClientSubClient].RootClientID = [Client].ID AND [ClientSubClient].Deleted = 0
LEFT JOIN [Client] AS [MainClient] ON [MainClient].ID = [Client].MainClientID
WHERE [Client].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
GROUP BY [Client].[Name], [Client].[IsSubClient], [ClientSubClient].ID, [Client].ID, [MainClient].[Name]
""",
        tables=["Client", "ClientSubClient"],
        notes="Повна ієрархія: чи є філією, скільки має підлеглих"
    ),
}

# =============================================================================
# REGION QUERIES (регіон клієнта)
# =============================================================================

REGION_QUERIES = {
    "region_info": ClientQuery(
        id="region_info",
        context="region",
        name_uk="Регіон клієнта",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    [Region].[Name] AS RegionName,
    [Country].[Name] AS CountryName,
    [RegionCode].[Code] AS RegionCode
FROM [Client]
LEFT JOIN [RegionCode] ON [RegionCode].ID = [Client].RegionCodeID
LEFT JOIN [Region] ON [Region].ID = [RegionCode].RegionID
LEFT JOIN [Country] ON [Country].ID = [Region].CountryID
WHERE [Client].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
""",
        tables=["Client", "RegionCode", "Region", "Country"],
        notes="Регіон і країна клієнта"
    ),

    "region_clients": ClientQuery(
        id="region_clients",
        context="region",
        name_uk="Клієнти в регіоні",
        sql="""
SELECT
    [Region].[Name] AS RegionName,
    [Client].[Name] AS ClientName,
    [Client].[FullName] AS FullName,
    [ClientType].[Name] AS ClientType
FROM [Client]
LEFT JOIN [RegionCode] ON [RegionCode].ID = [Client].RegionCodeID
LEFT JOIN [Region] ON [Region].ID = [RegionCode].RegionID
LEFT JOIN [ClientInRole] ON [ClientInRole].ClientID = [Client].ID AND [ClientInRole].Deleted = 0
LEFT JOIN [ClientType] ON [ClientType].ID = [ClientInRole].ClientTypeID
WHERE [Client].Deleted = 0
  AND [Region].[Name] LIKE '%' + @RegionName + '%'
ORDER BY [Client].[Name]
""",
        tables=["Client", "RegionCode", "Region", "ClientInRole", "ClientType"],
        notes="Всі клієнти з певного регіону"
    ),
}

# =============================================================================
# CONTACTS QUERIES (контактні дані)
# =============================================================================

CONTACTS_QUERIES = {
    "contacts_info": ClientQuery(
        id="contacts_info",
        context="contacts",
        name_uk="Контакти клієнта",
        sql="""
SELECT
    [Client].[Name] AS ClientName,
    [Client].[FullName] AS FullName,
    [Client].[MobileNumber] AS MobileNumber,
    [Client].[EmailAddress] AS Email,
    [Client].[TIN] AS TaxID,
    [Client].[USREOU] AS USREOU
FROM [Client]
WHERE [Client].Deleted = 0
  AND [Client].[Name] LIKE '%' + @ClientName + '%'
""",
        tables=["Client"],
        notes="Телефон, email, ІПН клієнта"
    ),
}

# =============================================================================
# ALL QUERIES COMBINED
# =============================================================================

ALL_CLIENT_QUERIES = {
    **DEBT_QUERIES,
    **CASH_FLOW_QUERIES,
    **PURCHASE_QUERIES,
    **AGREEMENT_QUERIES,
    **INVOICE_QUERIES,
    **PROFILE_QUERIES,
    **BANK_QUERIES,
    **STRUCTURE_QUERIES,
    **REGION_QUERIES,
    **CONTACTS_QUERIES,
}


# =============================================================================
# FORMATTING
# =============================================================================

def format_for_prompt(query: ClientQuery, params: Optional[ExtractedParams] = None) -> str:
    """Format query for LLM prompt.

    Args:
        query: The query template
        params: Optional extracted parameters (exchange rate, etc.)
    """
    lines = [
        f"-- Pattern: {query.name_uk}",
        f"-- Context: {query.context}",
        f"-- Tables: {', '.join(query.tables)}",
        f"-- Note: {query.notes}",
    ]

    # Add extracted parameters if present
    if params:
        if params.exchange_rate is not None:
            lines.append(f"-- IMPORTANT: User specified exchange rate @Rate = {params.exchange_rate}")
        if params.currency:
            lines.append(f"-- Target currency: {params.currency}")

    lines.append(query.sql.strip())

    return "\n".join(lines)

# Client Domain Schema

## Core Tables

### Client (36 columns)
Main customer/client entity.

| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **NetUID** | uniqueidentifier | Global unique ID |
| **Name** | nvarchar | Short name |
| **FullName** | nvarchar | Full legal name |
| **FirstName** | nvarchar | Contact first name |
| **LastName** | nvarchar | Contact last name |
| **MiddleName** | nvarchar | Contact middle name |
| **MobileNumber** | nvarchar | Phone number |
| **EmailAddress** | nvarchar | Email |
| **ICQ** | nvarchar | ICQ (legacy) |
| **TIN** | nvarchar | Tax ID (ІПН) |
| **USREOU** | nvarchar | Company registration (ЄДРПОУ) |
| **SourceAmgCode** | nvarchar | Legacy AMG system code |
| **SourceFenixCode** | nvarchar | Legacy Fenix system code |
| **RegionCodeID** | int | FK → RegionCode |
| **CountryID** | int | FK → Country |
| **MainClientID** | int | FK → Client (parent) |
| **ClientBankDetailsID** | int | FK → ClientBankDetails |
| **TermsOfDeliveryID** | int | FK → TermsOfDelivery |
| **PackingMarkingID** | int | FK → PackingMarking |
| **PackingMarkingPaymentID** | int | FK → PackingMarkingPayment |
| **IsActive** | bit | Active flag |
| **IsSubClient** | bit | Is a subsidiary |
| **IsTemporaryClient** | bit | Temporary client |
| **IsTradePoint** | bit | Trade point flag |
| **IsForRetail** | bit | Retail client |
| **ClearCartAfterDays** | int | Cart expiry days |
| **OrderExpireDays** | int | Order expiry days |
| **Created** | datetime | Creation date |
| **Deleted** | bit | Soft delete flag |

---

### ClientAgreement (8 columns)
Links Client to Agreement (договір).

| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **ClientID** | int | FK → Client |
| **AgreementID** | int | FK → Agreement |
| **NetUID** | uniqueidentifier | Global ID |
| **OriginalClientAmgCode** | nvarchar | Original client AMG code |
| **OriginalClientFenixCode** | nvarchar | Original client Fenix code |
| **CurrentAmount** | money | Current balance |
| **Deleted** | bit | Soft delete |

---

### Agreement (31 columns)
Contract/agreement with client.

| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **Name** | nvarchar | Agreement name |
| **Number** | nvarchar | Agreement number |
| **NetUID** | uniqueidentifier | Global ID |
| **OrganizationID** | int | FK → Organization |
| **CurrencyID** | int | FK → Currency |
| **PricingID** | int | FK → Pricing |
| **ProviderPricingID** | int | FK → ProviderPricing |
| **PromotionalPricingID** | int | FK → Pricing (promo) |
| **NumberDaysDebt** | int | **Payment term days** |
| **AmountDebt** | money | Credit limit |
| **IsControlNumberDaysDebt** | bit | Enforce payment terms |
| **IsControlAmountDebt** | bit | Enforce credit limit |
| **IsPrePayment** | bit | Requires prepayment |
| **IsPrePaymentFull** | bit | Full prepayment |
| **PrePaymentPercentages** | decimal | Prepayment % |
| **DeferredPayment** | bit | Deferred payment |
| **IsActive** | bit | Active flag |
| **IsDefault** | bit | Default agreement |
| **ForReSale** | bit | For resale |
| **IsAccounting** | bit | Accounting flag |
| **IsManagementAccounting** | bit | Management accounting |
| **WithVATAccounting** | bit | VAT accounting |
| **Created** | datetime | Creation date |

---

### ClientInRole (4 columns)
Client type/role assignment.

| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **ClientID** | int | FK → Client |
| **ClientTypeID** | int | FK → ClientType |
| **ClientTypeRoleID** | int | FK → ClientTypeRole |
| **Deleted** | bit | Soft delete |

---

### ClientType (8 columns)
Types of clients (Покупець, Постачальник, etc.)

| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **Name** | nvarchar | Type name |
| **Type** | int | Type code |
| **NetUID** | uniqueidentifier | Global ID |
| **ClientTypeIcon** | nvarchar | Icon |
| **AllowMultiple** | bit | Allow multiple roles |
| **Created** | datetime | Creation date |
| **Deleted** | bit | Soft delete |

---

### ClientTypeRole (8 columns)
Roles within client types.

| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **ClientTypeID** | int | FK → ClientType |
| **Name** | nvarchar | Role name |
| **Description** | nvarchar | Description |
| **NetUID** | uniqueidentifier | Global ID |
| **OrderExpireDays** | int | Order expiry days |
| **Created** | datetime | Creation date |
| **Deleted** | bit | Soft delete |

---

## Hierarchy & Structure

### ClientSubClient (3 columns)
Parent-child client relationships.

| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **RootClientID** | int | FK → Client (parent) |
| **SubClientID** | int | FK → Client (child) |
| **Deleted** | bit | Soft delete |

---

### ClientUserProfile (3 columns)
Manager assignment.

| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **ClientID** | int | FK → Client |
| **UserProfileID** | int | FK → User (manager) |
| **Deleted** | bit | Soft delete |

---

## Banking

### ClientBankDetails
Bank account information.

| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **BankAndBranch** | nvarchar | Bank name |
| **BankAddress** | nvarchar | Bank address |
| **Swift** | nvarchar | SWIFT code |
| **BranchCode** | nvarchar | Branch code |
| **AccountNumberID** | int | FK → ClientBankDetailAccountNumber |
| **ClientBankDetailIbanNoID** | int | FK → ClientBankDetailIbanNo |

### ClientBankDetailAccountNumber
| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **AccountNumber** | nvarchar | Account number |
| **CurrencyID** | int | FK → Currency |

### ClientBankDetailIbanNo
| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **IBANNO** | nvarchar | IBAN |
| **CurrencyID** | int | FK → Currency |

---

## Debt & Payments

### ClientInDebt (7 columns)
Links client to debt records.

| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **ClientID** | int | FK → Client |
| **AgreementID** | int | FK → Agreement |
| **DebtID** | int | FK → Debt |
| **SaleID** | int | FK → Sale (source) |
| **ReSaleID** | int | FK → ReSale (source) |
| **Created** | datetime | Creation date |
| **Deleted** | bit | Soft delete |

### Debt (5 columns)
Debt records.

| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **Total** | money | **Debt amount** |
| **Created** | datetime | **Debt start date** |
| **NetUID** | uniqueidentifier | Global ID |
| **Deleted** | bit | Soft delete |

### OutcomePaymentOrder (35 columns)
Outgoing payments.

| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **ClientAgreementID** | int | FK → ClientAgreement |
| **ClientID** | int | FK → Client |
| **Amount** | money | Payment amount |
| **EuroAmount** | money | EUR equivalent |
| **AfterExchangeAmount** | money | After exchange |
| **FromDate** | datetime | Payment date |
| **IsCanceled** | bit | Canceled flag |
| **Comment** | nvarchar | Notes |
| **Deleted** | bit | Soft delete |

### IncomePaymentOrder (34 columns)
Incoming payments.

| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **ClientAgreementID** | int | FK → ClientAgreement |
| **ClientID** | int | FK → Client |
| **Amount** | money | Payment amount |
| **EuroAmount** | money | EUR equivalent |
| **FromDate** | datetime | Payment date |
| **CurrencyID** | int | FK → Currency |
| **IsCanceled** | bit | Canceled flag |
| **Comment** | nvarchar | Notes |
| **Deleted** | bit | Soft delete |

---

## Location

### RegionCode
| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **RegionID** | int | FK → Region |
| **Code** | nvarchar | Region code |
| **Value** | nvarchar | Value |
| **Deleted** | bit | Soft delete |

### Region
| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **Name** | nvarchar | Region name |
| **CountryID** | int | FK → Country |

### Country
| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **Name** | nvarchar | Country name |

---

## Sales

### Sale (40 columns)
Sales/invoices.

| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **ClientAgreementID** | int | FK → ClientAgreement |
| **OrderID** | int | FK → Order |
| **SaleNumberID** | int | FK → SaleNumber |
| **Total** | money | Sale total |
| **Created** | datetime | Sale date |
| **ChangedToInvoice** | datetime | Invoice date |
| **LifeCycleStatusID** | int | FK → BaseLifeCycleStatus |
| **PaymentStatusID** | int | FK → BaseSalePaymentStatus |
| **IsImported** | bit | Imported flag |
| **Deleted** | bit | Soft delete |

### Order (14 columns)
| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **ClientAgreementID** | int | FK → ClientAgreement |
| **UserID** | int | FK → User |
| **OrderStatus** | int | Status code |
| **Created** | datetime | Order date |
| **Deleted** | bit | Soft delete |

### OrderItem (21 columns)
| Column | Type | Description |
|--------|------|-------------|
| **ID** | int | Primary key |
| **OrderID** | int | FK → Order |
| **ProductID** | int | FK → Product |
| **Qty** | decimal | Quantity |
| **PricePerItem** | money | Unit price |
| **DiscountAmount** | money | Discount |
| **ExchangeRateAmount** | decimal | Exchange rate |
| **Deleted** | bit | Soft delete |

---

## Entity Relationship Diagram

```
                              ┌─────────────┐
                              │   Country   │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │   Region    │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │ RegionCode  │
                              └──────┬──────┘
                                     │
┌─────────────┐              ┌───────▼───────┐              ┌─────────────┐
│ ClientType  │◄─────────────│    Client     │──────────────►│ClientSubCli │
└─────────────┘              │               │              │ (hierarchy) │
      │                      │  Name         │              └─────────────┘
      │                      │  FullName     │                     ▲
      ▼                      │  TIN/USREOU   │                     │
┌─────────────┐              │  Phone/Email  │                     │
│ClientTypeRol│              └───────┬───────┘              ┌──────┴──────┐
└─────────────┘                      │                      │   Client    │
      │                              │                      │ (SubClient) │
      ▼                              │                      └─────────────┘
┌─────────────┐              ┌───────▼───────┐
│ClientInRole │◄─────────────│ClientAgreement│
└─────────────┘              └───────┬───────┘
                                     │
                              ┌──────▼──────┐
                              │  Agreement  │
                              │             │
                              │NumberDaysDeb│
                              │ CurrencyID  │
                              │ PricingID   │
                              └──────┬──────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
       ┌──────▼──────┐        ┌──────▼──────┐        ┌──────▼──────┐
       │ClientInDebt │        │    Sale     │        │OutcomePayme │
       └──────┬──────┘        └──────┬──────┘        │  ntOrder    │
              │                      │               └─────────────┘
       ┌──────▼──────┐        ┌──────▼──────┐
       │    Debt     │        │   Order     │
       │             │        └──────┬──────┘
       │ Total       │               │
       │ Created     │        ┌──────▼──────┐
       └─────────────┘        │  OrderItem  │
                              │             │
                              │ ProductID   │
                              │ Qty, Price  │
                              └─────────────┘
```

---

## Key UDF Functions

| Function | Description |
|----------|-------------|
| `dbo.GetExchangedToEuroValue(amount, currencyId, date)` | Convert to EUR |
| `dbo.GetExchangeRateByCurrencyIdAndCode(id, code, date)` | Get exchange rate |
| `dbo.GetDefaultCalculatedProductPriceWithSharesAndVat(...)` | Calculate product price |
| `dbo.GetCalculatedProductPriceWithSharesAndVat(...)` | Calculate price with shares |

---

## Common Filters

All tables use soft delete pattern:
```sql
WHERE [TableName].Deleted = 0
```

Multi-language support via Translation tables:
```sql
LEFT JOIN [ClientTypeTranslation]
  ON [ClientTypeTranslation].ClientTypeID = [ClientType].ID
  AND [ClientTypeTranslation].CultureCode = @Culture
  AND [ClientTypeTranslation].Deleted = 0
```

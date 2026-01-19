-- Curated SQL queries for few-shot learning
-- These are the CORRECT business patterns from GBA repository

-- Table: curated_queries
-- Columns: id, question_uk, sql, tables, notes

INSERT INTO curated_queries (question_uk, sql, tables, notes) VALUES

-- ============================================================
-- DEBT: Current total
-- ============================================================
('Який борг у клієнта? Скільки винен клієнт? Заборгованість клієнта зараз.',
'SELECT
    [Client].[Name] AS ClientName,
    CONVERT(money, SUM(dbo.GetExchangedToEuroValue([Debt].Total, [Agreement].CurrencyID, GETDATE()))) AS TotalDebtEuro
FROM [ClientInDebt]
LEFT JOIN [Debt] ON [Debt].ID = [ClientInDebt].DebtID
LEFT JOIN [Agreement] ON [Agreement].ID = [ClientInDebt].AgreementID
LEFT JOIN [Client] ON [Client].ID = [ClientInDebt].ClientID
WHERE [ClientInDebt].Deleted = 0 AND [Debt].Deleted = 0
  AND [Client].[Name] LIKE ''%'' + @ClientName + ''%''
GROUP BY [Client].[Name]',
'Client,ClientInDebt,Debt,Agreement',
'dbo.GetExchangedToEuroValue конвертує в EUR'),

-- ============================================================
-- DEBT: At specific date (yesterday)
-- ============================================================
('Який борг у клієнта на вчора? Борг на дату. Історичний борг.',
'SELECT
    [Client].[Name] AS ClientName,
    CONVERT(money, SUM(dbo.GetExchangedToEuroValue([Debt].Total, [Agreement].CurrencyID, @AsOfDate))) AS TotalDebtEuro
FROM [ClientInDebt]
LEFT JOIN [Debt] ON [Debt].ID = [ClientInDebt].DebtID
LEFT JOIN [Agreement] ON [Agreement].ID = [ClientInDebt].AgreementID
LEFT JOIN [Client] ON [Client].ID = [ClientInDebt].ClientID
WHERE [ClientInDebt].Deleted = 0 AND [Debt].Deleted = 0
  AND [Debt].[Created] <= @AsOfDate
  AND [Client].[Name] LIKE ''%'' + @ClientName + ''%''
GROUP BY [Client].[Name]',
'Client,ClientInDebt,Debt,Agreement',
'@AsOfDate = DATEADD(DAY, -1, GETDATE()) для вчора'),

-- ============================================================
-- DEBT: By all agreements (detailed)
-- ============================================================
('Який борг по всім договорам? Борг по кожному договору. Деталізація боргу.',
'SELECT
    [Client].[Name] AS ClientName,
    [Agreement].[ID] AS AgreementID,
    [Currency].[Code] AS Currency,
    SUM([Debt].Total) AS DebtInCurrency,
    CONVERT(money, SUM(dbo.GetExchangedToEuroValue([Debt].Total, [Agreement].CurrencyID, GETDATE()))) AS DebtInEuro
FROM [ClientInDebt]
LEFT JOIN [Debt] ON [Debt].ID = [ClientInDebt].DebtID
LEFT JOIN [Agreement] ON [Agreement].ID = [ClientInDebt].AgreementID
LEFT JOIN [Currency] ON [Currency].ID = [Agreement].CurrencyID
LEFT JOIN [Client] ON [Client].ID = [ClientInDebt].ClientID
WHERE [ClientInDebt].Deleted = 0 AND [Debt].Deleted = 0
  AND [Client].[Name] LIKE ''%'' + @ClientName + ''%''
GROUP BY [Client].[Name], [Agreement].[ID], [Currency].[Code]
ORDER BY DebtInEuro DESC',
'Client,ClientInDebt,Debt,Agreement,Currency',
'Групування по Agreement для розбивки по договорам'),

-- ============================================================
-- DEBT: With days overdue
-- ============================================================
('Скільки днів прострочки? Прострочені борги. Прострочена заборгованість.',
'SELECT
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
  AND [Client].[Name] LIKE ''%'' + @ClientName + ''%''
ORDER BY DaysOverdue DESC',
'Client,ClientInDebt,Debt,Agreement',
'NumberDaysDebt = дозволена відстрочка'),

-- ============================================================
-- DEBT: Client structure (with sub-clients)
-- ============================================================
('Борг по структурі клієнта? Консолідований борг. Борг з філіями.',
'WITH SubClientDebts_CTE AS (
    SELECT SUM(Debt.Total) AS TotalDebt, ClientSubClient.RootClientID
    FROM ClientInDebt
    LEFT JOIN Debt ON Debt.ID = ClientInDebt.DebtID
    LEFT JOIN ClientSubClient ON ClientSubClient.SubClientID = ClientInDebt.ClientID
    WHERE ClientInDebt.Deleted = 0
    GROUP BY ClientSubClient.RootClientID
)
SELECT
    [Client].[Name] AS RootClientName,
    ISNULL(SubClientDebts_CTE.TotalDebt, 0) AS TotalStructureDebt
FROM [Client]
LEFT JOIN SubClientDebts_CTE ON SubClientDebts_CTE.RootClientID = [Client].ID
WHERE [Client].[Name] LIKE ''%'' + @ClientName + ''%''',
'Client,ClientInDebt,Debt,ClientSubClient',
'CTE для суми боргів підлеглих клієнтів');

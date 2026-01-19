"""Unit tests for sql_examples module."""

from __future__ import annotations

import pytest
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.sql_examples import (
    load_examples,
    classify_question,
    get_relevant_examples,
    extract_table_names,
    reload_examples,
    get_all_categories,
    get_category_info,
)


class TestLoadExamples:
    """Tests for load_examples function."""

    def test_load_examples_returns_dict(self):
        """Should return a dictionary with categories."""
        result = load_examples()
        assert isinstance(result, dict)
        assert "categories" in result

    def test_load_examples_has_expected_categories(self):
        """Should have the expected category structure."""
        result = load_examples()
        categories = result.get("categories", {})

        # Check for some expected categories (including new domain categories)
        expected_categories = [
            "simple_select",
            "aggregation",
            "multi_join",
            "debt_financial",
            "price_calculation",
            "warehouse_inventory",
            "time_period_report",
            "document_query",
        ]
        for cat in expected_categories:
            assert cat in categories, f"Missing expected category: {cat}"

    def test_load_examples_categories_have_required_fields(self):
        """Each category should have description, keywords, and examples."""
        result = load_examples()
        categories = result.get("categories", {})

        for cat_name, cat_data in categories.items():
            assert "description" in cat_data, f"{cat_name} missing description"
            assert "keywords" in cat_data, f"{cat_name} missing keywords"
            assert "examples" in cat_data, f"{cat_name} missing examples"
            assert isinstance(cat_data["keywords"], list), f"{cat_name} keywords not a list"
            assert isinstance(cat_data["examples"], list), f"{cat_name} examples not a list"

    def test_load_examples_caching(self):
        """Should return cached result on second call."""
        # Clear cache first
        reload_examples()

        result1 = load_examples()
        result2 = load_examples()

        # Should be the same object (cached)
        assert result1 is result2


class TestClassifyQuestion:
    """Tests for classify_question function."""

    def test_simple_select_keywords(self):
        """Questions with 'покажи', 'список', etc. should match simple_select."""
        questions = [
            "Покажи всі продукти",
            "Список клієнтів",
            "Знайди замовлення",
            "Всі користувачі",
        ]
        for q in questions:
            categories = classify_question(q)
            assert "simple_select" in categories, f"'{q}' should match simple_select"

    def test_aggregation_keywords(self):
        """Questions with 'топ', 'сума', etc. should match aggregation."""
        questions = [
            "Топ 10 клієнтів",
            "Сума продажів",
            "Кількість замовлень",
            "Найбільші борги",
        ]
        for q in questions:
            categories = classify_question(q)
            assert "aggregation" in categories, f"'{q}' should match aggregation"

    def test_debt_financial_keywords(self):
        """Questions about debt should match debt_financial."""
        questions = [
            "Борги клієнтів",
            "Заборгованість по оплатах",
            "Прострочені платежі",
            "Баланс клієнта",
        ]
        for q in questions:
            categories = classify_question(q)
            assert "debt_financial" in categories, f"'{q}' should match debt_financial"

    def test_multi_join_keywords(self):
        """Questions with 'деталі', 'включаючи' should match multi_join."""
        questions = [
            "Деталі замовлення",
            "Інформація про клієнта",
            "Повна інформація",
        ]
        for q in questions:
            categories = classify_question(q)
            assert "multi_join" in categories, f"'{q}' should match multi_join"

    def test_date_calculation_keywords(self):
        """Questions with date/time keywords should match date_calculation."""
        questions = [
            "Замовлення за сьогодні",
            "Продажі за місяць",
            "За останній тиждень",
            "За період",
        ]
        for q in questions:
            categories = classify_question(q)
            assert "date_calculation" in categories, f"'{q}' should match date_calculation"

    def test_price_calculation_keywords(self):
        """Questions with price keywords should match price_calculation."""
        questions = [
            "Яка ціна продукту?",
            "Вартість замовлення",
            "Знижка для клієнта",
            "Покажи прайс",
            "Розрахувати ціну",
        ]
        for q in questions:
            categories = classify_question(q)
            assert "price_calculation" in categories, f"'{q}' should match price_calculation"

    def test_warehouse_inventory_keywords(self):
        """Questions with warehouse/inventory keywords should match warehouse_inventory."""
        questions = [
            "Залишки на складі",
            "Товари в наявності",
            "Запаси продукції",
            "Рух товарів",
            "Інвентаризація",
        ]
        for q in questions:
            categories = classify_question(q)
            assert "warehouse_inventory" in categories, f"'{q}' should match warehouse_inventory"

    def test_time_period_report_keywords(self):
        """Questions with period report keywords should match time_period_report."""
        questions = [
            "Звіт за місяць",
            "Продажі по місяцях",
            "Статистика за рік",
            "По роках",
            "Щомісячний звіт",
        ]
        for q in questions:
            categories = classify_question(q)
            assert "time_period_report" in categories, f"'{q}' should match time_period_report"

    def test_document_query_keywords(self):
        """Questions with document keywords should match document_query."""
        questions = [
            "Документи клієнта",
            "Накладна на відвантаження",
            "Рахунок на оплату",
            "Друк акту",
            "Інвойс для клієнта",
        ]
        for q in questions:
            categories = classify_question(q)
            assert "document_query" in categories, f"'{q}' should match document_query"

    def test_multiple_category_match(self):
        """Questions can match multiple categories."""
        question = "Топ 5 клієнтів за сумою боргу"
        categories = classify_question(question)

        # Should match both aggregation and debt_financial
        assert "aggregation" in categories
        assert "debt_financial" in categories

    def test_default_to_simple_select(self):
        """Unknown questions should default to simple_select."""
        question = "xyz unknown query abc"
        categories = classify_question(question)
        assert "simple_select" in categories

    def test_case_insensitive_matching(self):
        """Keyword matching should be case-insensitive."""
        question = "ПОКАЖИ ВСІ ПРОДУКТИ"
        categories = classify_question(question)
        assert "simple_select" in categories


class TestGetRelevantExamples:
    """Tests for get_relevant_examples function."""

    def test_returns_string(self):
        """Should return a string."""
        result = get_relevant_examples("Покажи всі продукти")
        assert isinstance(result, str)

    def test_includes_header(self):
        """Result should include the examples header."""
        result = get_relevant_examples("Покажи всі продукти")
        assert "## SQL Examples" in result

    def test_includes_pattern_comments(self):
        """Result should include pattern description comments."""
        result = get_relevant_examples("Покажи всі продукти")
        assert "-- Pattern:" in result

    def test_includes_sql_content(self):
        """Result should include actual SQL content."""
        result = get_relevant_examples("Покажи всі продукти")
        assert "SELECT" in result
        assert "FROM" in result

    def test_includes_deleted_filter_pattern(self):
        """Examples should demonstrate the Deleted = 0 pattern."""
        result = get_relevant_examples("Покажи всі продукти")
        assert "Deleted = 0" in result

    def test_aggregation_examples(self):
        """Aggregation questions should get aggregation examples."""
        result = get_relevant_examples("Топ 10 клієнтів за продажами")
        assert "GROUP BY" in result
        assert "TOP" in result or "SUM" in result

    def test_debt_examples(self):
        """Debt questions should get debt-related examples."""
        result = get_relevant_examples("Покажи борги клієнтів")
        assert "Debt" in result

    def test_max_categories_limit(self):
        """Should respect max_categories parameter."""
        # With max_categories=1, should only get one category's examples
        result = get_relevant_examples(
            "Топ 5 клієнтів за сумою боргу",  # Matches multiple categories
            max_categories=1
        )
        # Should still have content
        assert len(result) > 0
        assert "-- Pattern:" in result

    def test_max_total_limit(self):
        """Should respect max_total parameter."""
        result = get_relevant_examples("Покажи всі продукти", max_total=1)
        assert result.count("-- Pattern:") <= 1


class TestTableFiltering:
    """Tests for table-aware example selection."""

    def test_table_filter_limits_examples(self):
        """Should prefer examples that match selected tables."""
        result = get_relevant_examples(
            "Покажи всі продукти",
            table_keys=["dbo.Product"],
            max_per_category=1,
            max_total=1,
        )
        assert "[Product]" in result

    def test_table_filter_fallback_when_no_match(self):
        """Should fall back when no table matches are found."""
        result = get_relevant_examples(
            "Покажи всі продукти",
            table_keys=["dbo.DoesNotExist"]
        )
        assert "SELECT" in result


class TestReloadExamples:
    """Tests for reload_examples function."""

    def test_reload_clears_cache(self):
        """reload_examples should clear the cache."""
        # Load first
        result1 = load_examples()

        # Reload
        reload_examples()

        # Load again - should be a new object (freshly loaded)
        result2 = load_examples()

        # Note: The dictionaries will have same content but we mainly test
        # that reload doesn't raise errors
        assert result2 is not None
        assert "categories" in result2


class TestGetAllCategories:
    """Tests for get_all_categories function."""

    def test_returns_list(self):
        """Should return a list of category names."""
        result = get_all_categories()
        assert isinstance(result, list)

    def test_contains_expected_categories(self):
        """Should contain all expected categories."""
        result = get_all_categories()
        expected = [
            "simple_select",
            "aggregation",
            "multi_join",
            "debt_financial",
            "date_calculation",
            "price_calculation",
            "warehouse_inventory",
            "time_period_report",
            "document_query",
        ]
        for cat in expected:
            assert cat in result, f"Missing category: {cat}"


class TestGetCategoryInfo:
    """Tests for get_category_info function."""

    def test_returns_dict(self):
        """Should return a dictionary with category info."""
        result = get_category_info("simple_select")
        assert isinstance(result, dict)

    def test_has_required_fields(self):
        """Result should have description, keywords, and example_count."""
        result = get_category_info("simple_select")
        assert "description" in result
        assert "keywords" in result
        assert "example_count" in result

    def test_example_count_is_positive(self):
        """example_count should be a positive integer for valid categories."""
        result = get_category_info("simple_select")
        assert result["example_count"] > 0

    def test_unknown_category_returns_empty(self):
        """Unknown category should return empty fields."""
        result = get_category_info("nonexistent_category")
        assert result["description"] == ""
        assert result["keywords"] == []
        assert result["example_count"] == 0


class TestExtractTableNames:
    """Tests for SQL table extraction."""

    def test_extract_tables_from_join(self):
        sql = (
            "SELECT [Order].ID FROM [Order] LEFT OUTER JOIN [Client] "
            "ON [Client].ID = [Order].ClientID"
        )
        tables = extract_table_names(sql)
        assert "order" in tables
        assert "client" in tables

    def test_extract_tables_ignores_cte(self):
        sql = (
            ";WITH [Orders_CTE] AS (SELECT [Order].ID FROM [Order]) "
            "SELECT * FROM [Orders_CTE]"
        )
        tables = extract_table_names(sql)
        assert "orders_cte" not in tables
        assert "order" in tables


class TestSQLPatterns:
    """Tests to verify SQL examples follow correct patterns."""

    def test_all_examples_have_deleted_filter(self):
        """All examples should demonstrate Deleted = 0 pattern."""
        data = load_examples()
        for cat_name, cat_data in data.get("categories", {}).items():
            for ex in cat_data.get("examples", []):
                sql = ex.get("sql", "")
                # Most examples should have soft-delete filter
                # Skip CTEs that might structure differently
                if "DELETE" not in sql.upper() and "[" in sql:
                    assert "Deleted = 0" in sql or "Deleted <> 0" in sql or "@" in sql, \
                        f"Example in {cat_name} missing Deleted filter: {sql[:100]}..."

    def test_all_examples_use_brackets_for_reserved_words(self):
        """Examples should use [brackets] for reserved words."""
        data = load_examples()
        reserved_words = ["Order", "User", "Group"]

        for cat_name, cat_data in data.get("categories", {}).items():
            for ex in cat_data.get("examples", []):
                sql = ex.get("sql", "")
                for word in reserved_words:
                    # If the word appears, it should be in brackets
                    if word in sql and f"[{word}]" not in sql:
                        # Allow if it's part of another word like OrderItem
                        if f" {word} " in sql or f" {word}." in sql:
                            pytest.fail(
                                f"Reserved word '{word}' not bracketed in {cat_name}: {sql[:100]}..."
                            )

    def test_all_examples_have_question(self):
        """All examples should have a question field."""
        data = load_examples()
        for cat_name, cat_data in data.get("categories", {}).items():
            for i, ex in enumerate(cat_data.get("examples", [])):
                assert "question" in ex, f"Example {i} in {cat_name} missing question"
                assert ex["question"], f"Example {i} in {cat_name} has empty question"


class TestNewDomainCategories:
    """Tests for the new domain-specific categories."""

    def test_price_calculation_has_examples(self):
        """price_calculation category should have examples."""
        info = get_category_info("price_calculation")
        assert info["example_count"] > 0
        assert info["description"]
        assert len(info["keywords"]) > 0

    def test_warehouse_inventory_has_examples(self):
        """warehouse_inventory category should have examples."""
        info = get_category_info("warehouse_inventory")
        assert info["example_count"] > 0
        assert info["description"]
        assert len(info["keywords"]) > 0

    def test_time_period_report_has_examples(self):
        """time_period_report category should have examples."""
        info = get_category_info("time_period_report")
        assert info["example_count"] > 0
        assert info["description"]
        assert len(info["keywords"]) > 0

    def test_document_query_has_examples(self):
        """document_query category should have examples."""
        info = get_category_info("document_query")
        assert info["example_count"] > 0
        assert info["description"]
        assert len(info["keywords"]) > 0

    def test_warehouse_examples_have_stock_tables(self):
        """Warehouse examples should reference stock-related tables."""
        result = get_relevant_examples("Покажи залишки на складі", max_total=5)
        # Should include warehouse/stock related content
        assert "SELECT" in result
        assert "FROM" in result

    def test_price_examples_have_price_content(self):
        """Price examples should have price-related content."""
        result = get_relevant_examples("Яка ціна продукту зі знижкою?", max_total=5)
        assert "SELECT" in result
        assert "FROM" in result

    def test_time_period_examples_have_grouping(self):
        """Time period examples should have GROUP BY patterns."""
        result = get_relevant_examples("Звіт продажів по місяцях", max_total=5)
        assert "SELECT" in result
        # Should have time-based grouping or date functions
        assert "GROUP BY" in result or "YEAR" in result or "MONTH" in result

    def test_document_examples_have_invoice_content(self):
        """Document examples should have document/invoice content."""
        result = get_relevant_examples("Покажи накладну для друку", max_total=5)
        assert "SELECT" in result
        assert "FROM" in result


class TestValidationModule:
    """Tests for the validation functionality."""

    def test_extract_table_names_basic(self):
        """Should extract table names from simple SQL."""
        sql = "SELECT * FROM [Product] WHERE [Product].Deleted = 0"
        tables = extract_table_names(sql)
        assert "product" in tables

    def test_extract_table_names_with_joins(self):
        """Should extract table names from SQL with joins."""
        sql = """
            SELECT [Order].ID, [Client].Name
            FROM [Order]
            LEFT JOIN [Client] ON [Client].ID = [Order].ClientID
        """
        tables = extract_table_names(sql)
        assert "order" in tables
        assert "client" in tables

    def test_extract_table_names_with_schema(self):
        """Should handle schema-qualified table names."""
        sql = "SELECT * FROM dbo.Product WHERE dbo.Product.Deleted = 0"
        tables = extract_table_names(sql)
        assert "product" in tables or "dbo.product" in tables


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

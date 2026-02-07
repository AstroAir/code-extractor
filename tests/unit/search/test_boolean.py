"""
Unit tests for boolean query parsing and evaluation.

Tests the BooleanQueryParser and BooleanQueryEvaluator classes
to ensure correct parsing of boolean expressions and evaluation
against file content.
"""

from pathlib import Path

import pytest

from pysearch.core.types import BooleanOperator, BooleanQuery, SearchItem
from pysearch.search.boolean import (
    BooleanQueryEvaluator,
    BooleanQueryParser,
    evaluate_boolean_query,
    parse_boolean_query,
)


class TestBooleanQueryParser:
    """Test the boolean query parser."""

    def test_simple_term(self):
        """Test parsing a simple term."""
        parser = BooleanQueryParser()
        query = parser.parse("test")

        assert query.term == "test"
        assert query.operator is None
        assert query.left is None
        assert query.right is None

    def test_quoted_term(self):
        """Test parsing a quoted term."""
        parser = BooleanQueryParser()
        query = parser.parse('"test term"')

        assert query.term == "test term"
        assert query.operator is None

    def test_and_operation(self):
        """Test parsing AND operation."""
        parser = BooleanQueryParser()
        query = parser.parse("foo AND bar")

        assert query.operator == BooleanOperator.AND
        assert query.left.term == "foo"
        assert query.right.term == "bar"

    def test_or_operation(self):
        """Test parsing OR operation."""
        parser = BooleanQueryParser()
        query = parser.parse("foo OR bar")

        assert query.operator == BooleanOperator.OR
        assert query.left.term == "foo"
        assert query.right.term == "bar"

    def test_not_operation(self):
        """Test parsing NOT operation."""
        parser = BooleanQueryParser()
        query = parser.parse("NOT test")

        assert query.operator == BooleanOperator.NOT
        assert query.left.term == "test"
        assert query.right is None

    def test_parentheses(self):
        """Test parsing with parentheses."""
        parser = BooleanQueryParser()
        query = parser.parse("(foo AND bar) OR baz")

        assert query.operator == BooleanOperator.OR
        assert query.left.operator == BooleanOperator.AND
        assert query.left.left.term == "foo"
        assert query.left.right.term == "bar"
        assert query.right.term == "baz"

    def test_complex_query(self):
        """Test parsing complex query with multiple operators."""
        parser = BooleanQueryParser()
        query = parser.parse("(async AND handler) AND NOT test")

        # Should be: ((async AND handler) AND (NOT test))
        assert query.operator == BooleanOperator.AND
        assert query.left.operator == BooleanOperator.AND
        assert query.left.left.term == "async"
        assert query.left.right.term == "handler"
        assert query.right.operator == BooleanOperator.NOT
        assert query.right.left.term == "test"

    def test_case_insensitive_operators(self):
        """Test that operators are case insensitive."""
        parser = BooleanQueryParser()
        query = parser.parse("foo and bar or baz")

        assert query.operator == BooleanOperator.OR
        assert query.left.operator == BooleanOperator.AND

    def test_operator_precedence(self):
        """Test operator precedence: NOT > AND > OR."""
        parser = BooleanQueryParser()
        query = parser.parse("foo OR bar AND NOT baz")

        # Should be: foo OR (bar AND (NOT baz))
        assert query.operator == BooleanOperator.OR
        assert query.left.term == "foo"
        assert query.right.operator == BooleanOperator.AND
        assert query.right.left.term == "bar"
        assert query.right.right.operator == BooleanOperator.NOT
        assert query.right.right.left.term == "baz"

    def test_empty_query_error(self):
        """Test error handling for empty query."""
        parser = BooleanQueryParser()
        with pytest.raises(ValueError, match="Empty query"):
            parser.parse("")

    def test_missing_parenthesis_error(self):
        """Test error handling for missing parenthesis."""
        parser = BooleanQueryParser()
        with pytest.raises(ValueError, match="Missing closing parenthesis"):
            parser.parse("(foo AND bar")

    def test_unexpected_token_error(self):
        """Test error handling for unexpected tokens."""
        parser = BooleanQueryParser()
        with pytest.raises(ValueError):
            parser.parse("foo AND")

    def test_convenience_function(self):
        """Test the convenience parse_boolean_query function."""
        query = parse_boolean_query("foo AND bar")

        assert query.operator == BooleanOperator.AND
        assert query.left.term == "foo"
        assert query.right.term == "bar"


class TestBooleanQueryEvaluator:
    """Test the boolean query evaluator."""

    def test_simple_term_match(self):
        """Test evaluation of simple term that matches."""
        evaluator = BooleanQueryEvaluator()
        query = BooleanQuery(term="function")
        content = "def function():\n    pass"

        assert evaluator.evaluate(query, content) is True

    def test_simple_term_no_match(self):
        """Test evaluation of simple term that doesn't match."""
        evaluator = BooleanQueryEvaluator()
        query = BooleanQuery(term="missing")
        content = "def function():\n    pass"

        assert evaluator.evaluate(query, content) is False

    def test_case_insensitive_matching(self):
        """Test that matching is case insensitive."""
        evaluator = BooleanQueryEvaluator()
        query = BooleanQuery(term="FUNCTION")
        content = "def function():\n    pass"

        assert evaluator.evaluate(query, content) is True

    def test_and_operation_both_match(self):
        """Test AND operation when both terms match."""
        evaluator = BooleanQueryEvaluator()
        query = BooleanQuery(
            operator=BooleanOperator.AND,
            left=BooleanQuery(term="def"),
            right=BooleanQuery(term="function"),
        )
        content = "def function():\n    pass"

        assert evaluator.evaluate(query, content) is True

    def test_and_operation_one_matches(self):
        """Test AND operation when only one term matches."""
        evaluator = BooleanQueryEvaluator()
        query = BooleanQuery(
            operator=BooleanOperator.AND,
            left=BooleanQuery(term="def"),
            right=BooleanQuery(term="missing"),
        )
        content = "def function():\n    pass"

        assert evaluator.evaluate(query, content) is False

    def test_or_operation_both_match(self):
        """Test OR operation when both terms match."""
        evaluator = BooleanQueryEvaluator()
        query = BooleanQuery(
            operator=BooleanOperator.OR,
            left=BooleanQuery(term="def"),
            right=BooleanQuery(term="function"),
        )
        content = "def function():\n    pass"

        assert evaluator.evaluate(query, content) is True

    def test_or_operation_one_matches(self):
        """Test OR operation when one term matches."""
        evaluator = BooleanQueryEvaluator()
        query = BooleanQuery(
            operator=BooleanOperator.OR,
            left=BooleanQuery(term="def"),
            right=BooleanQuery(term="missing"),
        )
        content = "def function():\n    pass"

        assert evaluator.evaluate(query, content) is True

    def test_or_operation_none_match(self):
        """Test OR operation when neither term matches."""
        evaluator = BooleanQueryEvaluator()
        query = BooleanQuery(
            operator=BooleanOperator.OR,
            left=BooleanQuery(term="missing1"),
            right=BooleanQuery(term="missing2"),
        )
        content = "def function():\n    pass"

        assert evaluator.evaluate(query, content) is False

    def test_not_operation_term_matches(self):
        """Test NOT operation when term matches."""
        evaluator = BooleanQueryEvaluator()
        query = BooleanQuery(operator=BooleanOperator.NOT, left=BooleanQuery(term="function"))
        content = "def function():\n    pass"

        assert evaluator.evaluate(query, content) is False

    def test_not_operation_term_no_match(self):
        """Test NOT operation when term doesn't match."""
        evaluator = BooleanQueryEvaluator()
        query = BooleanQuery(operator=BooleanOperator.NOT, left=BooleanQuery(term="missing"))
        content = "def function():\n    pass"

        assert evaluator.evaluate(query, content) is True

    def test_complex_query_evaluation(self):
        """Test evaluation of complex query."""
        evaluator = BooleanQueryEvaluator()
        # (async AND handler) NOT test
        query = BooleanQuery(
            operator=BooleanOperator.AND,
            left=BooleanQuery(
                operator=BooleanOperator.AND,
                left=BooleanQuery(term="async"),
                right=BooleanQuery(term="handler"),
            ),
            right=BooleanQuery(operator=BooleanOperator.NOT, left=BooleanQuery(term="test")),
        )

        # Content has async and handler but not test -> should match
        content = "async def request_handler():\n    return response"
        assert evaluator.evaluate(query, content) is True

        # Content has async, handler, and test -> should not match
        content = "async def test_handler():\n    return response"
        assert evaluator.evaluate(query, content) is False

    def test_missing_operand_errors(self):
        """Test error handling for missing operands."""
        evaluator = BooleanQueryEvaluator()

        # NOT without operand
        query = BooleanQuery(operator=BooleanOperator.NOT, left=None)
        with pytest.raises(ValueError, match="NOT operator requires an operand"):
            evaluator.evaluate(query, "test content")

        # AND without right operand
        query = BooleanQuery(
            operator=BooleanOperator.AND, left=BooleanQuery(term="test"), right=None
        )
        with pytest.raises(ValueError, match="AND operator requires two operands"):
            evaluator.evaluate(query, "test content")

        # OR without left operand
        query = BooleanQuery(
            operator=BooleanOperator.OR, left=None, right=BooleanQuery(term="test")
        )
        with pytest.raises(ValueError, match="OR operator requires two operands"):
            evaluator.evaluate(query, "test content")

    def test_convenience_function(self):
        """Test the convenience evaluate_boolean_query function."""
        query = BooleanQuery(
            operator=BooleanOperator.AND,
            left=BooleanQuery(term="def"),
            right=BooleanQuery(term="function"),
        )
        content = "def function():\n    pass"

        assert evaluate_boolean_query(query, content) is True

    def test_evaluate_with_items(self):
        """Test evaluation that returns SearchItems."""
        evaluator = BooleanQueryEvaluator()
        query = BooleanQuery(term="function")
        content = "def function():\n    pass"

        # Create mock SearchItem
        items = [
            SearchItem(
                file=Path("test.py"),
                start_line=1,
                end_line=2,
                lines=["def function():", "    pass"],
                match_spans=[(0, (0, 12))],
            )
        ]

        result = evaluator.evaluate_with_line_matches(query, content, items)
        assert len(result) == 1
        assert result[0].file == Path("test.py")

        # Test with non-matching query
        query = BooleanQuery(term="missing")
        result = evaluator.evaluate_with_line_matches(query, content, items)
        assert len(result) == 0


class TestBooleanQueryIntegration:
    """Integration tests for boolean query functionality."""

    def test_end_to_end_parsing_and_evaluation(self):
        """Test complete flow from string parsing to evaluation."""
        # Parse query
        query = parse_boolean_query("(async AND handler) AND NOT test")

        # Test content that should match
        content1 = "async def request_handler():\n    return response"
        assert evaluate_boolean_query(query, content1) is True

        # Test content that should not match (has test)
        content2 = "async def test_handler():\n    return response"
        assert evaluate_boolean_query(query, content2) is False

        # Test content that should not match (missing async)
        content3 = "def request_handler():\n    return response"
        assert evaluate_boolean_query(query, content3) is False

    def test_quoted_terms_in_evaluation(self):
        """Test that quoted terms work correctly in evaluation."""
        query = parse_boolean_query('"def main" AND "return"')

        # Should match
        content1 = "def main():\n    return 0"
        assert evaluate_boolean_query(query, content1) is True

        # Should not match (missing return)
        content2 = "def main():\n    pass"
        assert evaluate_boolean_query(query, content2) is False

    def test_complex_real_world_query(self):
        """Test complex query that might be used in practice."""
        query = parse_boolean_query(
            '("async def" OR "def async") AND (handler OR controller) AND NOT (test OR mock)'
        )

        # Should match: async function with handler, no test
        content1 = """
        async def user_handler(request):
            return process_user(request)
        """
        assert evaluate_boolean_query(query, content1) is True

        # Should not match: has test
        content2 = """
        async def test_user_handler(request):
            return process_user(request)
        """
        assert evaluate_boolean_query(query, content2) is False

        # Should not match: no handler or controller
        content3 = """
        async def process_user(request):
            return user_data
        """
        assert evaluate_boolean_query(query, content3) is False

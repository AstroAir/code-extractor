"""
Boolean query parsing and evaluation for pysearch.

This module provides functionality to parse and evaluate boolean queries with
logical operators (AND, OR, NOT) for advanced search operations.

Classes:
    BooleanQueryParser: Parser for boolean query strings
    BooleanQueryEvaluator: Evaluator for boolean queries against search results

Functions:
    parse_boolean_query: Parse a boolean query string into a BooleanQuery tree
    evaluate_boolean_query: Evaluate a boolean query against file search results

Example:
    >>> from pysearch.search.boolean import parse_boolean_query, evaluate_boolean_query
    >>>
    >>> # Parse boolean query
    >>> query = parse_boolean_query("(async AND handler) NOT test")
    >>>
    >>> # Evaluate against file content
    >>> matches = evaluate_boolean_query(query, file_content, line_starts)
"""

from __future__ import annotations

import re

from ..core.types import BooleanOperator, BooleanQuery, SearchItem


class BooleanQueryParser:
    """Parser for boolean query expressions."""

    def __init__(self) -> None:
        self.pos = 0
        self.tokens: list[str] = []

    def tokenize(self, query: str) -> list[str]:
        """Tokenize a boolean query string."""
        # Regular expression to match tokens: operators, parentheses, quoted strings, words
        token_pattern = r'(\bAND\b|\bOR\b|\bNOT\b|\(|\)|"[^"]*"|\w+)'
        tokens = re.findall(token_pattern, query, re.IGNORECASE)

        # Normalize operator tokens to uppercase
        normalized_tokens = []
        for token in tokens:
            if token.upper() in ["AND", "OR", "NOT"]:
                normalized_tokens.append(token.upper())
            else:
                normalized_tokens.append(token)

        return normalized_tokens

    def parse(self, query: str) -> BooleanQuery:
        """Parse a boolean query string into a BooleanQuery tree."""
        self.tokens = self.tokenize(query)
        self.pos = 0

        if not self.tokens:
            raise ValueError("Empty query")

        result = self._parse_or_expression()

        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")

        return result

    def _current_token(self) -> str | None:
        """Get the current token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume_token(self) -> str | None:
        """Consume and return the current token."""
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            self.pos += 1
            return token
        return None

    def _parse_or_expression(self) -> BooleanQuery:
        """Parse OR expressions (lowest precedence)."""
        left = self._parse_and_expression()

        while self._current_token() == "OR":
            self._consume_token()  # consume OR
            right = self._parse_and_expression()
            left = BooleanQuery(operator=BooleanOperator.OR, left=left, right=right)

        return left

    def _parse_and_expression(self) -> BooleanQuery:
        """Parse AND expressions (middle precedence).

        Also treats adjacent NOT/term tokens as implicit AND
        (e.g. ``handler NOT test`` → ``handler AND (NOT test)``).
        """
        left = self._parse_not_expression()

        while True:
            tok = self._current_token()
            if tok == "AND":
                self._consume_token()  # consume AND
                right = self._parse_not_expression()
                left = BooleanQuery(operator=BooleanOperator.AND, left=left, right=right)
            elif tok is not None and tok not in ("OR", ")"):
                # Implicit AND: next token is NOT or a term/parenthesised expr
                right = self._parse_not_expression()
                left = BooleanQuery(operator=BooleanOperator.AND, left=left, right=right)
            else:
                break

        return left

    def _parse_not_expression(self) -> BooleanQuery:
        """Parse NOT expressions (highest precedence)."""
        if self._current_token() == "NOT":
            self._consume_token()  # consume NOT
            operand = self._parse_primary()
            return BooleanQuery(operator=BooleanOperator.NOT, left=operand, right=None)

        return self._parse_primary()

    def _parse_primary(self) -> BooleanQuery:
        """Parse primary expressions (terms and parentheses)."""
        token = self._current_token()

        if token is None:
            raise ValueError("Unexpected end of query")

        if token == "(":
            self._consume_token()  # consume (
            result = self._parse_or_expression()
            if self._current_token() != ")":
                raise ValueError("Missing closing parenthesis")
            self._consume_token()  # consume )
            return result

        # Handle quoted terms
        if token.startswith('"') and token.endswith('"'):
            term = token[1:-1]  # Remove quotes
            self._consume_token()
            return BooleanQuery(term=term)

        # Handle regular terms
        if token not in ["AND", "OR", "NOT", "(", ")"]:
            self._consume_token()
            return BooleanQuery(term=token)

        raise ValueError(f"Unexpected token: {token}")


class BooleanQueryEvaluator:
    """Evaluator for boolean queries against file content."""

    def __init__(self) -> None:
        pass

    def evaluate(self, query: BooleanQuery, file_content: str) -> bool:
        """Evaluate a boolean query against file content."""
        if query.term is not None:
            # Leaf node - check if term exists in content
            return query.term.lower() in file_content.lower()

        if query.operator == BooleanOperator.NOT:
            if query.left is None:
                raise ValueError("NOT operator requires an operand")
            return not self.evaluate(query.left, file_content)

        if query.operator == BooleanOperator.AND:
            if query.left is None or query.right is None:
                raise ValueError("AND operator requires two operands")
            return self.evaluate(query.left, file_content) and self.evaluate(
                query.right, file_content
            )

        if query.operator == BooleanOperator.OR:
            if query.left is None or query.right is None:
                raise ValueError("OR operator requires two operands")
            return self.evaluate(query.left, file_content) or self.evaluate(
                query.right, file_content
            )

        raise ValueError(f"Unknown operator: {query.operator}")

    def evaluate_with_line_matches(
        self, query: BooleanQuery, file_content: str, existing_items: list[SearchItem]
    ) -> list[SearchItem]:
        """
        Evaluate a boolean query and return matching SearchItems.

        This method combines boolean logic with line-level matching,
        evaluating the boolean condition against the content window of each
        individual SearchItem rather than just the whole file.
        """
        if not existing_items:
            return []

        # Evaluate each item against its own content window rather than
        # short-circuiting on the whole file.  The whole-file check is
        # wrong for NOT queries: a block may satisfy NOT even when the
        # whole file does not.
        filtered_items: list[SearchItem] = []
        for item in existing_items:
            if not item.lines:
                continue
            item_content = "\n".join(item.lines)
            if self.evaluate(query, item_content):
                filtered_items.append(item)

        return filtered_items


def extract_terms(query: BooleanQuery) -> list[str]:
    """Extract all search terms from a BooleanQuery tree."""
    terms: list[str] = []
    if query.term is not None:
        terms.append(query.term)
    if query.left is not None:
        terms.extend(extract_terms(query.left))
    if query.right is not None:
        terms.extend(extract_terms(query.right))
    return terms


# Convenience functions
def parse_boolean_query(query: str) -> BooleanQuery:
    """Parse a boolean query string into a BooleanQuery tree."""
    parser = BooleanQueryParser()
    return parser.parse(query)


def evaluate_boolean_query(query: BooleanQuery, file_content: str) -> bool:
    """Evaluate a boolean query against file content."""
    evaluator = BooleanQueryEvaluator()
    return evaluator.evaluate(query, file_content)


def evaluate_boolean_query_with_items(
    query: BooleanQuery, file_content: str, existing_items: list[SearchItem]
) -> list[SearchItem]:
    """Evaluate a boolean query and return matching SearchItems."""
    evaluator = BooleanQueryEvaluator()
    return evaluator.evaluate_with_line_matches(query, file_content, existing_items)

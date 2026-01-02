#!/usr/bin/env python3
"""
Expression Normalization
Applies canonical ordering and cheap simplification rules to expressions
"""

import re
from typing import List, Tuple, Optional


def normalize_expression(expression: str) -> str:
    """
    Normalize expression by applying cheap rules and canonical ordering.
    
    Args:
        expression: GAMBA expression string
    
    Returns:
        Normalized expression
    """
    # Apply cheap rules first
    expr = apply_cheap_rules(expression)
    
    # Apply canonical ordering
    expr = canonical_order(expr)
    
    return expr


def apply_cheap_rules(expression: str) -> str:
    """
    Apply cheap simplification rules:
    - x ^ x = 0
    - x & x = x
    - x | x = x
    - x | 0 = x
    - x & 0 = 0
    - x ^ 0 = x
    - x + 0 = x
    - x - 0 = x
    - x * 1 = x
    - x * 0 = 0
    
    Args:
        expression: Expression string
    
    Returns:
        Simplified expression
    """
    expr = expression
    
    # Pattern: (x ^ x) -> 0
    expr = re.sub(r'\((\w+)\s*\^\s*\1\)', r'0', expr)
    
    # Pattern: (x & x) -> x
    expr = re.sub(r'\((\w+)\s*&\s*\1\)', r'\1', expr)
    
    # Pattern: (x | x) -> x
    expr = re.sub(r'\((\w+)\s*\|\s*\1\)', r'\1', expr)
    
    # Pattern: (x | 0) or (0 | x) -> x
    expr = re.sub(r'\((\w+)\s*\|\s*0\)', r'\1', expr)
    expr = re.sub(r'\(0\s*\|\s*(\w+)\)', r'\1', expr)
    
    # Pattern: (x & 0) or (0 & x) -> 0
    expr = re.sub(r'\((\w+)\s*&\s*0\)', r'0', expr)
    expr = re.sub(r'\(0\s*&\s*(\w+)\)', r'0', expr)
    
    # Pattern: (x ^ 0) or (0 ^ x) -> x
    expr = re.sub(r'\((\w+)\s*\^\s*0\)', r'\1', expr)
    expr = re.sub(r'\(0\s*\^\s*(\w+)\)', r'\1', expr)
    
    # Pattern: (x + 0) or (0 + x) -> x
    expr = re.sub(r'\((\w+)\s*\+\s*0\)', r'\1', expr)
    expr = re.sub(r'\(0\s*\+\s*(\w+)\)', r'\1', expr)
    
    # Pattern: (x - 0) -> x
    expr = re.sub(r'\((\w+)\s*-\s*0\)', r'\1', expr)
    
    # Pattern: (x * 1) or (1 * x) -> x
    expr = re.sub(r'\((\w+)\s*\*\s*1\)', r'\1', expr)
    expr = re.sub(r'\(1\s*\*\s*(\w+)\)', r'\1', expr)
    
    # Pattern: (x * 0) or (0 * x) -> 0
    expr = re.sub(r'\((\w+)\s*\*\s*0\)', r'0', expr)
    expr = re.sub(r'\(0\s*\*\s*(\w+)\)', r'0', expr)
    
    # Pattern: ~(~x) -> x
    expr = re.sub(r'~\(~(\w+)\)', r'\1', expr)
    
    return expr


def canonical_order(expression: str) -> str:
    """
    Apply canonical ordering to commutative operations.
    
    For commutative ops (and, or, xor, add), sort operands alphabetically.
    
    Note: This is a simplified version that doesn't break nested expressions.
    For now, we skip canonical ordering to avoid breaking valid expressions.
    
    Args:
        expression: Expression string
    
    Returns:
        Expression (unchanged for now to avoid breaking valid expressions)
    """
    # TODO: Implement proper canonical ordering that handles nested expressions
    # For now, return expression unchanged to avoid breaking valid expressions
    # The GAMBA simplifier will handle optimization anyway
    return expression


def normalize_expression_list(expressions: List[str]) -> List[str]:
    """
    Normalize a list of expressions.
    
    Args:
        expressions: List of expression strings
    
    Returns:
        List of normalized expressions
    """
    return [normalize_expression(expr) for expr in expressions]


def get_expression_complexity(expression: str) -> int:
    """
    Calculate complexity score for expression (number of operations).
    
    Args:
        expression: Expression string
    
    Returns:
        Complexity score (higher = more complex)
    """
    # Count operators
    operators = ['+', '-', '*', '&', '|', '^', '<<', '>>', '~']
    complexity = 0
    
    for op in operators:
        complexity += expression.count(op)
    
    return complexity


if __name__ == "__main__":
    # Test normalization
    test_expressions = [
        "(x ^ x)",
        "(x & x)",
        "(x | 0)",
        "(x + 0)",
        "(x * 1)",
        "(y + x)",
        "(x + y)",
        "(a & b)",
        "(b & a)",
    ]
    
    print("Normalization test:")
    for expr in test_expressions:
        normalized = normalize_expression(expr)
        print(f"  {expr} -> {normalized}")


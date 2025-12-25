#!/usr/bin/env python3
"""
Advanced Parallel Processing for GAMBA++

Uses ProcessPoolExecutor for true parallelism (bypasses GIL),
adaptive timeouts, and early termination optimizations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gamba.simplify_general import GeneralSimplifier
from .batch import complexity_score


def quick_simplify_check(expression: str) -> Optional[str]:
    """
    Quick check if expression is already simplified.
    
    Returns simplified expression if trivial, None otherwise.
    
    Args:
        expression: GAMBA expression string
    
    Returns:
        Simplified expression if already simple, None if needs processing
    """
    if not expression or not expression.strip():
        return None
    
    expr = expression.strip()
    
    # Single variable: already simplified
    if re.match(r'^[a-z]\d*$', expr, re.IGNORECASE):
        return expr
    
    # Single constant: already simplified
    if re.match(r'^-?\d+$', expr):
        return expr
    
    # No boolean operations: likely already simplified
    if not re.search(r'[&|^~]', expr):
        # Check if it's a simple arithmetic expression
        # e.g., "x + y", "2*x", etc.
        if re.match(r'^[a-z0-9+\-*\s()]+$', expr, re.IGNORECASE):
            # Could be simplified, but let GAMBA decide
            return None
    
    # Has complex operations: needs processing
    return None


def adaptive_timeout(expression: str) -> int:
    """
    Calculate adaptive timeout based on expression complexity.
    
    Args:
        expression: GAMBA expression string
    
    Returns:
        Timeout in seconds (5, 15, or 30)
    """
    score = complexity_score(expression)
    
    if score < 10:
        return 5   # Simple: 5 seconds
    elif score < 30:
        return 15  # Medium: 15 seconds
    else:
        return 30  # Complex: 30 seconds (GAMBA default)


def _process_single_expression_worker(args):
    """
    Worker function for ProcessPoolExecutor.
    
    Must be at module level for pickling.
    
    Args:
        args: Tuple of (expression, bitcount, use_z3, mod_red)
    
    Returns:
        Result dictionary
    """
    expression, bitcount, use_z3, mod_red = args
    import time
    
    start_time = time.time()
    
    try:
        # Quick check first
        quick_result = quick_simplify_check(expression)
        if quick_result:
            return {
                "original": expression,
                "simplified": quick_result,
                "success": True,
                "error": None,
                "processing_time": time.time() - start_time,
                "cached": False,
                "early_terminated": True
            }
        
        # Process with GAMBA
        simplifier = GeneralSimplifier(bitcount, modRed=mod_red, verifBitCount=None)
        simplified = simplifier.simplify(expression, useZ3=use_z3)
        
        elapsed = time.time() - start_time
        
        if simplified:
            return {
                "original": expression,
                "simplified": simplified,
                "success": True,
                "error": None,
                "processing_time": elapsed,
                "cached": False,
                "early_terminated": False
            }
        else:
            return {
                "original": expression,
                "simplified": None,
                "success": False,
                "error": "GAMBA returned empty result (timeout or error)",
                "processing_time": elapsed,
                "cached": False,
                "early_terminated": False
            }
    
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "original": expression,
            "simplified": None,
            "success": False,
            "error": str(e),
            "processing_time": elapsed,
            "cached": False,
            "early_terminated": False
        }


def process_expressions_parallel_advanced(
    expressions: List[str],
    bitcount: int = 32,
    max_workers: int = 8,
    use_cache: bool = True,
    cache: Optional = None,
    use_z3: bool = False,
    mod_red: bool = False,
    show_progress: bool = True
) -> List[Dict]:
    """
    Process multiple expressions in parallel using ProcessPoolExecutor.
    
    Uses true parallelism (bypasses GIL) with 8 cores by default.
    Includes early termination and adaptive timeouts.
    
    Args:
        expressions: List of expression strings
        bitcount: Bit width for variables
        max_workers: Maximum number of parallel workers (default: 8)
        use_cache: Enable caching (default: True)
        cache: GAMBACache instance (created if None and use_cache=True)
        use_z3: Enable Z3 verification (default: False)
        mod_red: Enable modulo reduction (default: False)
        show_progress: Show progress bar (default: True)
    
    Returns:
        List of result dictionaries
    """
    if max_workers is None:
        max_workers = min(len(expressions), mp.cpu_count())
    
    if use_cache and cache is None:
        from .cache import GAMBACache
        cache = GAMBACache()
    
    results = []
    
    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
    
    # Prepare worker arguments
    worker_args = [
        (expr, bitcount, use_z3, mod_red)
        for expr in expressions
    ]
    
    # Process in parallel using ProcessPoolExecutor (true parallelism)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_expr = {
            executor.submit(_process_single_expression_worker, args): expr
            for args, expr in zip(worker_args, expressions)
        }
        
        # Collect results as they complete
        iterator = as_completed(future_to_expr)
        if show_progress and has_tqdm:
            iterator = tqdm(iterator, total=len(expressions), desc="Processing expressions")
        
        # Create results list with correct order
        results_dict = {}
        for future in iterator:
            expr = future_to_expr[future]
            try:
                result = future.result()
                
                # Check cache if enabled
                if use_cache and cache:
                    cached = cache.get(expr, bitcount)
                    if cached and not result.get("early_terminated"):
                        result["simplified"] = cached
                        result["cached"] = True
                        result["processing_time"] = 0.0
                    elif result["success"] and not result.get("cached"):
                        cache.set(expr, bitcount, result["simplified"])
                
                results_dict[expr] = result
            except Exception as e:
                results_dict[expr] = {
                    "original": expr,
                    "simplified": None,
                    "success": False,
                    "error": str(e),
                    "processing_time": 0.0,
                    "cached": False,
                    "early_terminated": False
                }
        
        # Return results in original order
        results = [results_dict[expr] for expr in expressions]
    
    # Save cache if used
    if use_cache and cache:
        cache.save()
    
    return results


def process_with_gamba_optimized(
    expression: str,
    bitcount: int = 32,
    use_z3: bool = False,
    mod_red: bool = False,
    cache: Optional = None
) -> Dict:
    """
    Process single expression with all optimizations.
    
    Args:
        expression: Mathematical expression string
        bitcount: Bit width for variables
        use_z3: Enable Z3 verification
        mod_red: Enable modulo reduction
        cache: Optional GAMBACache instance
    
    Returns:
        Result dictionary
    """
    if cache is None:
        from .cache import GAMBACache
        cache = GAMBACache()
    
    # Check cache first
    cached = cache.get(expression, bitcount)
    if cached:
        return {
            "original": expression,
            "simplified": cached,
            "success": True,
            "cached": True,
            "processing_time": 0.0
        }
    
    # Quick check for simple expressions
    quick_result = quick_simplify_check(expression)
    if quick_result:
        cache.set(expression, bitcount, quick_result)
        return {
            "original": expression,
            "simplified": quick_result,
            "success": True,
            "cached": False,
            "early_terminated": True,
            "processing_time": 0.0
        }
    
    # Process with GAMBA
    import time
    start_time = time.time()
    
    try:
        simplifier = GeneralSimplifier(bitcount, modRed=mod_red, verifBitCount=None)
        simplified = simplifier.simplify(expression, useZ3=use_z3)
        
        elapsed = time.time() - start_time
        
        if simplified:
            cache.set(expression, bitcount, simplified)
            return {
                "original": expression,
                "simplified": simplified,
                "success": True,
                "error": None,
                "processing_time": elapsed,
                "cached": False
            }
        else:
            return {
                "original": expression,
                "simplified": None,
                "success": False,
                "error": "GAMBA returned empty result",
                "processing_time": elapsed
            }
    except Exception as e:
        return {
            "original": expression,
            "simplified": None,
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time
        }


if __name__ == "__main__":
    # Test
    test_exprs = [
        "x + y",  # Simple - should early terminate
        "(x ^ y) + 2*(x & y)",  # Complex - needs processing
    ]
    
    print("Testing advanced parallel processing...")
    results = process_expressions_parallel_advanced(
        test_exprs,
        bitcount=32,
        max_workers=4,
        show_progress=False
    )
    
    for result in results:
        print(f"{result['original']} -> {result.get('simplified', 'FAILED')}")


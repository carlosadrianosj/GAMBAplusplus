#!/usr/bin/env python3
"""
Parallel Processing Utilities for GAMBA++

Provides direct API calls to GAMBA (eliminating subprocess overhead)
and parallel processing capabilities.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Callable, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gamba.simplify_general import GeneralSimplifier

if TYPE_CHECKING:
    from .cache import GAMBACache


def process_with_gamba_direct(
    expression: str,
    bitcount: int = 32,
    timeout: int = 30,
    use_z3: bool = False,
    mod_red: bool = False
) -> Dict:
    """
    Process expression directly via GAMBA API (no subprocess overhead).
    
    Args:
        expression: Mathematical expression string (GAMBA format)
        bitcount: Bit width for variables (default: 32)
        timeout: Timeout in seconds (default: 30, GAMBA internal timeout)
        use_z3: Enable Z3 verification (default: False)
        mod_red: Enable modulo reduction (default: False)
    
    Returns:
        Dictionary with processing results:
        {
            "original": "...",
            "simplified": "...",
            "success": True/False,
            "error": "...",
            "processing_time": float
        }
    """
    import time
    
    start_time = time.time()
    
    try:
        # Create simplifier instance
        simplifier = GeneralSimplifier(bitcount, modRed=mod_red, verifBitCount=None)
        
        # GAMBA has internal timeout of 30 seconds
        # For longer timeouts, we'd need to modify GAMBA code
        simplified = simplifier.simplify(expression, useZ3=use_z3)
        
        elapsed = time.time() - start_time
        
        if simplified:
            return {
                "original": expression,
                "simplified": simplified,
                "success": True,
                "error": None,
                "processing_time": elapsed
            }
        else:
            return {
                "original": expression,
                "simplified": None,
                "success": False,
                "error": "GAMBA returned empty result (timeout or error)",
                "processing_time": elapsed
            }
    
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "original": expression,
            "simplified": None,
            "success": False,
            "error": str(e),
            "processing_time": elapsed
        }


def process_expressions_parallel(
    expressions: List[str],
    bitcount: int = 32,
    max_workers: Optional[int] = None,
    use_cache: bool = True,
    cache: Optional['GAMBACache'] = None,
    show_progress: bool = True
) -> List[Dict]:
    """
    Process multiple expressions in parallel.
    
    Args:
        expressions: List of expression strings
        bitcount: Bit width for variables
        max_workers: Maximum number of parallel workers (None = auto)
        use_cache: Enable caching (default: True)
        cache: GAMBACache instance (created if None and use_cache=True)
        show_progress: Show progress bar (default: True)
    
    Returns:
        List of result dictionaries
    """
    from .cache import GAMBACache
    
    if max_workers is None:
        max_workers = min(len(expressions), mp.cpu_count())
    
    if use_cache and cache is None:
        cache = GAMBACache()
    
    results = []
    
    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
    
    def process_single(expr: str) -> Dict:
        """Process single expression with optional caching"""
        if use_cache and cache:
            cached = cache.get(expr, bitcount)
            if cached:
                return {
                    "original": expr,
                    "simplified": cached,
                    "success": True,
                    "cached": True,
                    "processing_time": 0.0
                }
        
        result = process_with_gamba_direct(expr, bitcount)
        
        if use_cache and cache and result["success"]:
            cache.set(expr, bitcount, result["simplified"])
        
        return result
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_expr = {
            executor.submit(process_single, expr): expr
            for expr in expressions
        }
        
        # Collect results as they complete
        iterator = as_completed(future_to_expr)
        if show_progress and has_tqdm:
            iterator = tqdm(iterator, total=len(expressions), desc="Processing expressions")
        
        for future in iterator:
            expr = future_to_expr[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "original": expr,
                    "simplified": None,
                    "success": False,
                    "error": str(e),
                    "processing_time": 0.0
                })
    
    # Save cache if used
    if use_cache and cache:
        cache.save()
    
    return results


def process_mba_blocks_parallel(
    mba_blocks: List,
    converter_func: Callable,
    bitcount: int = 32,
    max_workers: Optional[int] = None,
    use_cache: bool = True,
    cache: Optional['GAMBACache'] = None,
    show_progress: bool = True
) -> List[Dict]:
    """
    Process multiple MBA blocks in parallel.
    
    Args:
        mba_blocks: List of MBABlock objects
        converter_func: Function to convert block to expression (convert_mba_block_to_expression)
        bitcount: Bit width for variables
        max_workers: Maximum number of parallel workers (None = auto)
        use_cache: Enable caching (default: True)
        cache: GAMBACache instance (created if None and use_cache=True)
        show_progress: Show progress bar (default: True)
    
    Returns:
        List of result dictionaries with block info and simplified expressions
    """
    from .cache import GAMBACache
    
    if max_workers is None:
        max_workers = min(len(mba_blocks), mp.cpu_count())
    
    if use_cache and cache is None:
        cache = GAMBACache()
    
    results = []
    
    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
    
    def process_block(block):
        """Process single MBA block"""
        # Convert block to expression
        expr_data = converter_func(block)
        
        if not expr_data:
            return {
                "block": block,
                "success": False,
                "error": "Failed to convert block to expression"
            }
        
        expression = expr_data['gamba_expression']
        
        # Check cache
        if use_cache and cache:
            cached = cache.get(expression, bitcount)
            if cached:
                return {
                    "block": block,
                    "expression_data": expr_data,
                    "original": expression,
                    "simplified": cached,
                    "success": True,
                    "cached": True
                }
        
        # Process with GAMBA
        result = process_with_gamba_direct(expression, bitcount)
        
        if use_cache and cache and result["success"]:
            cache.set(expression, bitcount, result["simplified"])
        
        return {
            "block": block,
            "expression_data": expr_data,
            **result
        }
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_block = {
            executor.submit(process_block, block): block
            for block in mba_blocks
        }
        
        iterator = as_completed(future_to_block)
        if show_progress and has_tqdm:
            iterator = tqdm(iterator, total=len(mba_blocks), desc="Processing MBA blocks")
        
        for future in iterator:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                block = future_to_block[future]
                results.append({
                    "block": block,
                    "success": False,
                    "error": str(e)
                })
    
    # Save cache if used
    if use_cache and cache:
        cache.save()
    
    return results


if __name__ == "__main__":
    # Test direct API
    test_expr = "(x ^ y) + 2*(x & y)"
    print(f"Testing direct API with: {test_expr}")
    result = process_with_gamba_direct(test_expr, bitcount=32)
    print(f"Result: {result}")


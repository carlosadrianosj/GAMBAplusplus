#!/usr/bin/env python3
"""
Batch Processing with Prioritization for GAMBA++

Provides intelligent batch processing with expression prioritization
and optimized function processing pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from .cache import GAMBACache
from .parallel import process_with_gamba_direct, process_expressions_parallel

if TYPE_CHECKING:
    from .cache import GAMBACache


def complexity_score(expression: str) -> float:
    """
    Calculate complexity score for an expression.
    
    Lower score = simpler expression (process first)
    Higher score = complex expression (process later)
    
    Args:
        expression: GAMBA expression string
    
    Returns:
        Complexity score (float)
    """
    if not expression:
        return 0.0
    
    score = 0.0
    
    # Length factor
    score += len(expression) * 0.1
    
    # Operator count (more operators = more complex)
    score += expression.count('(') * 2.0
    score += expression.count('&') * 1.5
    score += expression.count('|') * 1.5
    score += expression.count('^') * 1.5
    score += expression.count('~') * 1.0
    score += expression.count('+') * 0.5
    score += expression.count('-') * 0.5
    score += expression.count('*') * 1.0
    score += expression.count('**') * 3.0
    
    # Variable count (more variables = more complex)
    import re
    variables = set(re.findall(r'\b[a-z]\b', expression.lower()))
    score += len(variables) * 2.0
    
    # Nested parentheses (deep nesting = more complex)
    max_depth = 0
    current_depth = 0
    for char in expression:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1
    score += max_depth * 3.0
    
    return score


def prioritize_expressions(expressions: List[Dict]) -> List[Dict]:
    """
    Prioritize expressions by complexity (simpler first).
    
    Args:
        expressions: List of expression dictionaries with 'gamba_expression' key
    
    Returns:
        Sorted list (simplest first)
    """
    def get_score(expr_data):
        expr = expr_data.get('gamba_expression', '')
        return complexity_score(expr)
    
    return sorted(expressions, key=get_score)


def batch_process_expressions(
    expressions: List[str],
    bitcount: int = 32,
    max_workers: Optional[int] = None,
    use_cache: bool = True,
    prioritize: bool = True,
    cache: Optional[GAMBACache] = None,
    show_progress: bool = True
) -> List[Dict]:
    """
    Process expressions in batch with optional prioritization.
    
    Args:
        expressions: List of expression strings
        bitcount: Bit width for variables
        max_workers: Maximum number of parallel workers (None = auto)
        use_cache: Enable caching (default: True)
        prioritize: Sort by complexity before processing (default: True)
        cache: GAMBACache instance (created if None and use_cache=True)
        show_progress: Show progress bar (default: True)
    
    Returns:
        List of result dictionaries
    """
    # Prioritize if requested
    if prioritize:
        # Convert to dict format for prioritization
        expr_dicts = [{"gamba_expression": expr} for expr in expressions]
        prioritized = prioritize_expressions(expr_dicts)
        expressions = [d["gamba_expression"] for d in prioritized]
    
    # Process in parallel
    return process_expressions_parallel(
        expressions=expressions,
        bitcount=bitcount,
        max_workers=max_workers,
        use_cache=use_cache,
        cache=cache,
        show_progress=show_progress
    )


def process_function_optimized(
    asm_file: Path,
    parser_func,
    detector_func,
    converter_func,
    bitcount: int = 32,
    use_cache: bool = True,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    cache: Optional[GAMBACache] = None,
    show_progress: bool = True
) -> Dict:
    """
    Process a function through optimized pipeline.
    
    Pipeline:
    1. Parse assembly (sequential, fast)
    2. Detect MBA blocks (sequential, fast)
    3. Convert blocks to expressions (parallel if enabled)
    4. Process with GAMBA (parallel, cached)
    5. Aggregate results
    
    Args:
        asm_file: Path to assembly file
        parser_func: Function to parse assembly (parse_assembly)
        detector_func: Function to detect MBA blocks (detect_mba_blocks)
        converter_func: Function to convert blocks (convert_mba_block_to_expression)
        bitcount: Bit width for variables
        use_cache: Enable caching (default: True)
        parallel: Enable parallel processing (default: True)
        max_workers: Maximum number of parallel workers (None = auto)
        cache: GAMBACache instance (created if None and use_cache=True)
        show_progress: Show progress bar (default: True)
    
    Returns:
        Dictionary with processing results
    """
    import time
    
    start_time = time.time()
    
    if cache is None and use_cache:
        cache = GAMBACache()
    
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    results = {
        "function": str(asm_file),
        "success": False,
        "steps": {}
    }
    
    try:
        # Step 1: Parse assembly (sequential, fast)
        if show_progress:
            print("[1/4] Parsing assembly...")
        parsed = parser_func(asm_file)
        instructions = parsed["instructions"]
        results["steps"]["parsing"] = {
            "success": True,
            "instruction_count": len(instructions)
        }
        
        # Step 2: Detect MBA blocks (sequential, fast)
        if show_progress:
            print("[2/4] Detecting MBA blocks...")
        mba_blocks = detector_func(instructions)
        results["steps"]["detection"] = {
            "success": True,
            "blocks_found": len(mba_blocks)
        }
        
        if len(mba_blocks) == 0:
            results["success"] = True
            results["message"] = "No MBA blocks found"
            return results
        
        # Step 3: Convert blocks to expressions
        if show_progress:
            print(f"[3/4] Converting {len(mba_blocks)} blocks to expressions...")
        
        expressions_data = []
        if parallel:
            # Parallel conversion
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_block = {
                    executor.submit(converter_func, block): block
                    for block in mba_blocks
                }
                
                for future in as_completed(future_to_block):
                    expr_data = future.result()
                    if expr_data:
                        expressions_data.append(expr_data)
        else:
            # Sequential conversion
            for block in mba_blocks:
                expr_data = converter_func(block)
                if expr_data:
                    expressions_data.append(expr_data)
        
        results["steps"]["conversion"] = {
            "success": True,
            "expressions_converted": len(expressions_data)
        }
        
        if len(expressions_data) == 0:
            results["success"] = True
            results["message"] = "No expressions converted"
            return results
        
        # Step 4: Process with GAMBA (parallel, cached)
        if show_progress:
            print(f"[4/4] Processing {len(expressions_data)} expressions with GAMBA...")
        
        expressions = [expr['gamba_expression'] for expr in expressions_data]
        gamba_results = batch_process_expressions(
            expressions=expressions,
            bitcount=bitcount,
            max_workers=max_workers,
            use_cache=use_cache,
            prioritize=True,
            cache=cache,
            show_progress=show_progress
        )
        
        # Aggregate results
        successful = [r for r in gamba_results if r.get("success")]
        failed = [r for r in gamba_results if not r.get("success")]
        cached = [r for r in successful if r.get("cached", False)]
        
        results["steps"]["gamba_processing"] = {
            "success": True,
            "total": len(gamba_results),
            "successful": len(successful),
            "failed": len(failed),
            "cached": len(cached)
        }
        
        results["success"] = True
        results["expressions"] = gamba_results
        results["processing_time"] = time.time() - start_time
        
        if show_progress:
            print(f"\n✓ Processed {len(successful)}/{len(gamba_results)} expressions successfully")
            if cached:
                print(f"  ({len(cached)} from cache)")
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        results["processing_time"] = time.time() - start_time
    
    return results


if __name__ == "__main__":
    # Test complexity scoring
    test_exprs = [
        "x + y",
        "(x ^ y) + 2*(x & y)",
        "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",
    ]
    
    print("Testing complexity scoring:")
    for expr in test_exprs:
        score = complexity_score(expr)
        print(f"  {expr[:50]:50s} -> {score:.2f}")


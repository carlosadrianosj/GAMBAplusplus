#!/usr/bin/env python3
"""
Advanced Batch Processing for GAMBA++

Processes multiple expressions in batches within single GAMBA process
to reduce process creation overhead.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gamba.simplify_general import GeneralSimplifier
from .cache import GAMBACache
from .parallel_advanced import quick_simplify_check
from .simplifier_pool import SimplifierPool


def _process_batch_worker(args):
    """
    Worker function to process a batch of expressions in single process.
    
    Must be at module level for pickling.
    
    Args:
        args: Tuple of (expressions, bitcount, use_z3, mod_red)
    
    Returns:
        List of result dictionaries
    """
    expressions, bitcount, use_z3, mod_red = args
    import time
    import sys
    
    # Monkey-patch sys.exit to raise exception instead of exiting
    original_exit = sys.exit
    def safe_exit(code=None):
        if code is None or code == 0:
            return
        # Convert sys.exit() to exception
        error_msg = str(code) if isinstance(code, str) else f"Exit code: {code}"
        raise RuntimeError(f"GAMBA exit: {error_msg}")
    
    sys.exit = safe_exit
    
    results = []
    # Use verifBitCount=None to avoid slow exhaustive verification
    # that can cause timeouts in batch processing
    simplifier = GeneralSimplifier(bitcount, modRed=mod_red, verifBitCount=None)
    
    for expr in expressions:
        start_time = time.time()
        error_msg = None
        simplified = None
        
        try:
            # Quick check first
            quick_result = quick_simplify_check(expr)
            if quick_result:
                results.append({
                    "original": expr,
                    "simplified": quick_result,
                    "success": True,
                    "error": None,
                    "processing_time": time.time() - start_time,
                    "early_terminated": True
                })
                continue
            
            # Process with GAMBA (reusing same simplifier)
            # Use increased timeout (120s) for better success rate
            try:
                simplified = simplifier.simplify(expr, useZ3=use_z3, timeout=120)
            except RuntimeError as e:
                # Caught sys.exit() converted to RuntimeError
                error_msg = str(e).replace("GAMBA exit: ", "")
                simplified = None
            except Exception as e:
                # Other exceptions during simplification
                error_msg = str(e)
                simplified = None
            
            elapsed = time.time() - start_time
            
            if simplified:
                results.append({
                    "original": expr,
                    "simplified": simplified,
                    "success": True,
                    "error": None,
                    "processing_time": elapsed,
                    "early_terminated": False
                })
            else:
                if not error_msg:
                    error_msg = "GAMBA returned empty result"
                results.append({
                    "original": expr,
                    "simplified": None,
                    "success": False,
                    "error": error_msg,
                    "processing_time": elapsed,
                    "early_terminated": False
                })
        
        except Exception as e:
            results.append({
                "original": expr,
                "simplified": None,
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "early_terminated": False
            })
    
    # Restore original sys.exit
    sys.exit = original_exit
    
    return results


def process_expressions_batch_advanced(
    expressions: List[str],
    bitcount: int = 32,
    max_workers: int = 8,
    batch_size: int = 10,
    use_cache: bool = True,
    cache: Optional[GAMBACache] = None,
    use_z3: bool = False,
    mod_red: bool = False,
    show_progress: bool = True
) -> List[Dict]:
    """
    Process expressions in batches using ProcessPoolExecutor.
    
    Groups expressions into batches and processes each batch in a single
    GAMBA process, reducing process creation overhead.
    
    Args:
        expressions: List of expression strings
        bitcount: Bit width for variables
        max_workers: Maximum number of parallel workers (default: 8)
        batch_size: Number of expressions per batch (default: 10)
        use_cache: Enable caching (default: True)
        cache: GAMBACache instance (created if None and use_cache=True)
        use_z3: Enable Z3 verification (default: False)
        mod_red: Enable modulo reduction (default: False)
        show_progress: Show progress bar (default: True)
    
    Returns:
        List of result dictionaries
    """
    if use_cache and cache is None:
        cache = GAMBACache()
    
    if max_workers is None:
        max_workers = min(len(expressions), mp.cpu_count())
    
    # Split expressions into batches
    batches = []
    for i in range(0, len(expressions), batch_size):
        batch = expressions[i:i + batch_size]
        batches.append(batch)
    
    # Check cache for all expressions first
    cached_results = {}
    expressions_to_process = []
    expression_indices = []
    
    for i, expr in enumerate(expressions):
        if use_cache and cache:
            cached = cache.get(expr, bitcount)
            if cached:
                cached_results[i] = {
                    "original": expr,
                    "simplified": cached,
                    "success": True,
                    "cached": True,
                    "processing_time": 0.0
                }
                continue
        
        expressions_to_process.append(expr)
        expression_indices.append(i)
    
    if not expressions_to_process:
        # All expressions were cached
        return [cached_results[i] for i in range(len(expressions))]
    
    # Split remaining expressions into batches
    batches_to_process = []
    batch_indices = []
    
    for i in range(0, len(expressions_to_process), batch_size):
        batch = expressions_to_process[i:i + batch_size]
        batch_idx = expression_indices[i:i + batch_size]
        batches_to_process.append(batch)
        batch_indices.append(batch_idx)
    
    # Process batches in parallel
    results_dict = cached_results.copy()
    
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit batch tasks
        future_to_batch = {
            executor.submit(
                _process_batch_worker,
                (batch, bitcount, use_z3, mod_red)
            ): (batch, batch_idx)
            for batch, batch_idx in zip(batches_to_process, batch_indices)
        }
        
        # Collect results
        iterator = as_completed(future_to_batch)
        if show_progress and has_tqdm:
            iterator = tqdm(iterator, total=len(batches_to_process), desc="Processing batches")
        
        for future in iterator:
            batch, batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                
                # Map results back to original indices
                for result, orig_idx in zip(batch_results, batch_idx):
                    # Update cache if enabled
                    if use_cache and cache and result["success"]:
                        cache.set(result["original"], bitcount, result["simplified"])
                    
                    results_dict[orig_idx] = result
            
            except Exception as e:
                # Handle batch failure
                for orig_idx in batch_idx:
                    results_dict[orig_idx] = {
                        "original": expressions[orig_idx],
                        "simplified": None,
                        "success": False,
                        "error": f"Batch processing failed: {str(e)}",
                        "processing_time": 0.0
                    }
    
    # Save cache if used
    if use_cache and cache:
        cache.save()
    
    # Return results in original order
    return [results_dict.get(i, {"success": False}) for i in range(len(expressions))]


if __name__ == "__main__":
    # Test batch processing
    test_exprs = [
        "(x ^ y) + 2*(x & y)",
        "(x | y) - (x & y)",
        "~x + ~y + 1",
        "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",
    ] * 3  # 12 expressions total
    
    print(f"Testing batch processing with {len(test_exprs)} expressions...")
    results = process_expressions_batch_advanced(
        test_exprs,
        bitcount=32,
        max_workers=4,
        batch_size=4,
        show_progress=False
    )
    
    successful = sum(1 for r in results if r["success"])
    print(f"Successfully processed: {successful}/{len(results)}")


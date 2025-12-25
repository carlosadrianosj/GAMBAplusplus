#!/usr/bin/env python3
"""
Advanced Usage Examples for GAMBA++

Demonstrates all advanced performance optimizations including:
- ProcessPoolExecutor (8 cores)
- Thread-safe cache
- Adaptive timeouts
- Simplifier pooling
- Batch processing
- Early termination
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.cache import GAMBACache
from optimization.parallel_advanced import (
    process_expressions_parallel_advanced,
    process_with_gamba_optimized,
    adaptive_timeout,
    quick_simplify_check
)
from optimization.simplifier_pool import SimplifierPool
from optimization.batch_advanced import process_expressions_batch_advanced
from optimization.memoization import get_parse_cache_info


def example_processpool_8cores():
    """Example: ProcessPoolExecutor with 8 cores"""
    print("=" * 70)
    print("Example: ProcessPoolExecutor (8 cores)")
    print("=" * 70)
    
    expressions = [
        "(x ^ y) + 2*(x & y)",
        "(x | y) - (x & y)",
        "~x + ~y + 1",
        "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",
        "(x & 0xFF) | (y & 0xFF00)",
    ] * 2  # 10 expressions
    
    print(f"\nProcessing {len(expressions)} expressions with 8 workers...")
    
    results = process_expressions_parallel_advanced(
        expressions=expressions,
        bitcount=32,
        max_workers=8,
        use_cache=True,
        show_progress=True
    )
    
    successful = sum(1 for r in results if r["success"])
    early_terminated = sum(1 for r in results if r.get("early_terminated", False))
    
    print(f"\n✓ Successfully processed: {successful}/{len(expressions)}")
    if early_terminated > 0:
        print(f"  ({early_terminated} early terminated)")


def example_threadsafe_cache():
    """Example: Thread-safe cache in parallel processing"""
    print("\n" + "=" * 70)
    print("Example: Thread-Safe Cache")
    print("=" * 70)
    
    cache = GAMBACache(Path(".example_cache_advanced.json"))
    
    # Create expressions with duplicates
    base_exprs = [
        "(x ^ y) + 2*(x & y)",
        "(x | y) - (x & y)",
    ]
    expressions = base_exprs * 5  # 10 expressions, 5 duplicates each
    
    print(f"\nProcessing {len(expressions)} expressions ({len(base_exprs)} unique)...")
    print("Cache should handle duplicates correctly in parallel")
    
    results = process_expressions_parallel_advanced(
        expressions=expressions,
        bitcount=32,
        max_workers=8,
        use_cache=True,
        cache=cache,
        show_progress=True
    )
    
    cached = sum(1 for r in results if r.get("cached", False))
    stats = cache.get_stats()
    
    print(f"\n✓ Cache hits: {cached}/{len(expressions)}")
    print(f"  Cache hit rate: {stats['hit_rate_percent']}%")
    print(f"  Cache size: {stats['cache_size']} entries")


def example_adaptive_timeout():
    """Example: Adaptive timeout based on complexity"""
    print("\n" + "=" * 70)
    print("Example: Adaptive Timeout")
    print("=" * 70)
    
    expressions = [
        "x + y",  # Simple: 5s timeout
        "(x ^ y) + 2*(x & y)",  # Medium: 15s timeout
        "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",  # Complex: 30s timeout
    ]
    
    print("\nTimeout assignment:")
    for expr in expressions:
        timeout = adaptive_timeout(expr)
        print(f"  {timeout:2d}s  {expr[:50]}")


def example_early_termination():
    """Example: Early termination for simple expressions"""
    print("\n" + "=" * 70)
    print("Example: Early Termination")
    print("=" * 70)
    
    expressions = [
        "x + y",  # Should early terminate
        "x",  # Should early terminate
        "42",  # Should early terminate
        "(x ^ y) + 2*(x & y)",  # Needs processing
    ]
    
    print("\nEarly termination check:")
    for expr in expressions:
        quick = quick_simplify_check(expr)
        if quick:
            print(f"  ✓ {expr:30s} -> Early terminated: {quick}")
        else:
            print(f"  → {expr:30s} -> Needs GAMBA processing")


def example_simplifier_pool():
    """Example: Simplifier pool for reuse"""
    print("\n" + "=" * 70)
    print("Example: Simplifier Pool")
    print("=" * 70)
    
    expressions = [
        "(x ^ y) + 2*(x & y)",
        "(x | y) - (x & y)",
        "~x + ~y + 1",
    ]
    
    print("\nProcessing with simplifier pool...")
    
    with SimplifierPool(bitcount=32, pool_size=4) as pool:
        for expr in expressions:
            from optimization.simplifier_pool import process_with_pooled_simplifier
            result = process_with_pooled_simplifier(expr, pool)
            print(f"  {expr[:40]:40s} -> {result[:30]}")
        
        print(f"\n  Pool size: {pool.size()}")
        print(f"  Total created: {pool.total_created()}")


def example_batch_processing():
    """Example: Batch processing within GAMBA"""
    print("\n" + "=" * 70)
    print("Example: Batch Processing")
    print("=" * 70)
    
    expressions = [
        "(x ^ y) + 2*(x & y)",
        "(x | y) - (x & y)",
        "~x + ~y + 1",
        "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",
    ] * 3  # 12 expressions
    
    print(f"\nProcessing {len(expressions)} expressions in batches (batch_size=4)...")
    
    results = process_expressions_batch_advanced(
        expressions=expressions,
        bitcount=32,
        max_workers=8,
        batch_size=4,
        use_cache=True,
        show_progress=True
    )
    
    successful = sum(1 for r in results if r["success"])
    print(f"\n✓ Successfully processed: {successful}/{len(expressions)}")


def example_fully_optimized():
    """Example: All optimizations combined"""
    print("\n" + "=" * 70)
    print("Example: Fully Optimized Pipeline")
    print("=" * 70)
    
    expressions = [
        "x + y",  # Simple - early terminate
        "(x ^ y) + 2*(x & y)",  # Medium
        "(x | y) - (x & y)",  # Medium
        "~x + ~y + 1",  # Medium
        "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",  # Complex
    ] * 3  # 15 expressions with duplicates
    
    print(f"\nProcessing {len(expressions)} expressions with all optimizations:")
    print("  - ProcessPoolExecutor (8 workers)")
    print("  - Thread-safe cache")
    print("  - Early termination")
    print("  - Batch processing")
    
    results = process_expressions_batch_advanced(
        expressions=expressions,
        bitcount=32,
        max_workers=8,
        batch_size=5,
        use_cache=True,
        show_progress=True
    )
    
    successful = sum(1 for r in results if r["success"])
    cached = sum(1 for r in results if r.get("cached", False))
    early = sum(1 for r in results if r.get("early_terminated", False))
    
    print(f"\n✓ Results:")
    print(f"  Successful: {successful}/{len(expressions)}")
    print(f"  From cache: {cached}")
    print(f"  Early terminated: {early}")


if __name__ == "__main__":
    print("GAMBA++ Advanced Usage Examples")
    print("=" * 70)
    
    # Run examples
    example_processpool_8cores()
    example_threadsafe_cache()
    example_adaptive_timeout()
    example_early_termination()
    example_simplifier_pool()
    example_batch_processing()
    example_fully_optimized()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


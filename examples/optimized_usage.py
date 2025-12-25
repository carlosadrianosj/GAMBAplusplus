#!/usr/bin/env python3
"""
Optimized Usage Examples for GAMBA++

Demonstrates performance optimizations including caching, parallel processing,
and batch operations.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.cache import GAMBACache
from optimization.parallel import process_with_gamba_direct, process_expressions_parallel
from optimization.batch import (
    process_function_optimized,
    prioritize_expressions,
    batch_process_expressions,
    complexity_score
)
from assembly.x86_64 import parse_assembly, detect_mba_blocks, convert_mba_block_to_expression


def example_cache_usage():
    """Example: Using cache for repeated expressions"""
    print("=" * 70)
    print("Example: Cache Usage")
    print("=" * 70)
    
    cache = GAMBACache(cache_file=Path(".example_cache.json"))
    
    expressions = [
        "(x ^ y) + 2*(x & y)",
        "(x | y) - (x & y)",
        "(x ^ y) + 2*(x & y)",  # Duplicate - should be cached
    ]
    
    print("\nProcessing expressions:")
    for i, expr in enumerate(expressions, 1):
        print(f"\n[{i}] {expr}")
        
        # Check cache first
        cached = cache.get(expr, bitcount=32)
        if cached:
            print(f"  ✓ From cache: {cached}")
        else:
            # Process with GAMBA
            result = process_with_gamba_direct(expr, bitcount=32)
            if result["success"]:
                print(f"  ✓ Simplified: {result['simplified']}")
                cache.set(expr, 32, result['simplified'])
            else:
                print(f"  ✗ Error: {result.get('error')}")
    
    # Show cache statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate_percent']}%")
    print(f"  Cache Size: {stats['cache_size']}")


def example_parallel_processing():
    """Example: Parallel processing of multiple expressions"""
    print("\n" + "=" * 70)
    print("Example: Parallel Processing")
    print("=" * 70)
    
    expressions = [
        "(x ^ y) + 2*(x & y)",
        "(x | y) - (x & y)",
        "~x + ~y + 1",
        "(x & 0xFF) | (y & 0xFF00)",
        "(x ^ y) ^ (x & y)",
    ]
    
    print(f"\nProcessing {len(expressions)} expressions in parallel...")
    
    results = process_expressions_parallel(
        expressions=expressions,
        bitcount=32,
        max_workers=4,
        use_cache=True,
        show_progress=True
    )
    
    print("\nResults:")
    for expr, result in zip(expressions, results):
        if result["success"]:
            cached = " (cached)" if result.get("cached") else ""
            print(f"  ✓ {expr[:40]:40s} -> {result['simplified']}{cached}")
        else:
            print(f"  ✗ {expr[:40]:40s} -> Error: {result.get('error', 'Unknown')}")


def example_batch_processing():
    """Example: Batch processing with prioritization"""
    print("\n" + "=" * 70)
    print("Example: Batch Processing with Prioritization")
    print("=" * 70)
    
    expressions = [
        "x + y",  # Simple
        "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",  # Complex
        "(x ^ y) + 2*(x & y)",  # Medium
        "~x + ~y + 1",  # Simple
    ]
    
    print("\nOriginal order:")
    for i, expr in enumerate(expressions, 1):
        score = complexity_score(expr)
        print(f"  {i}. {expr[:50]:50s} (complexity: {score:.2f})")
    
    # Prioritize
    expr_dicts = [{"gamba_expression": expr} for expr in expressions]
    prioritized = prioritize_expressions(expr_dicts)
    
    print("\nPrioritized order (simplest first):")
    for i, expr_data in enumerate(prioritized, 1):
        expr = expr_data["gamba_expression"]
        score = complexity_score(expr)
        print(f"  {i}. {expr[:50]:50s} (complexity: {score:.2f})")
    
    # Process in batch
    print("\nProcessing with batch processing...")
    results = batch_process_expressions(
        expressions=expressions,
        bitcount=32,
        prioritize=True,
        use_cache=True,
        show_progress=True
    )
    
    successful = sum(1 for r in results if r["success"])
    print(f"\n✓ Successfully processed {successful}/{len(results)} expressions")


def example_optimized_function_processing():
    """Example: Optimized function processing pipeline"""
    print("\n" + "=" * 70)
    print("Example: Optimized Function Processing")
    print("=" * 70)
    
    # Example assembly file (would need actual file)
    asm_file = Path("example_x86.asm")
    
    if not asm_file.exists():
        print(f"\nNote: {asm_file} not found. This is a template example.")
        print("\nTo use this example:")
        print("1. Create an assembly file with x86-64 instructions")
        print("2. Update the asm_file path above")
        print("3. Run this script")
        return
    
    print(f"\nProcessing function: {asm_file}")
    
    result = process_function_optimized(
        asm_file=asm_file,
        parser_func=parse_assembly,
        detector_func=detect_mba_blocks,
        converter_func=convert_mba_block_to_expression,
        bitcount=32,
        use_cache=True,
        parallel=True,
        show_progress=True
    )
    
    if result["success"]:
        print(f"\n✓ Processing completed in {result.get('processing_time', 0):.2f}s")
        print(f"  Steps completed: {list(result['steps'].keys())}")
        
        if "expressions" in result:
            successful = sum(1 for r in result["expressions"] if r.get("success"))
            print(f"  Expressions processed: {successful}/{len(result['expressions'])}")
    else:
        print(f"\n✗ Processing failed: {result.get('error', 'Unknown error')}")


def example_complexity_scoring():
    """Example: Expression complexity scoring"""
    print("\n" + "=" * 70)
    print("Example: Complexity Scoring")
    print("=" * 70)
    
    expressions = [
        "x + y",
        "(x ^ y) + 2*(x & y)",
        "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",
        "~x + ~y + 1",
        "(x & 0xFF) | (y & 0xFF00) | (z & 0xFF0000)",
    ]
    
    print("\nComplexity scores (lower = simpler, process first):")
    for expr in expressions:
        score = complexity_score(expr)
        print(f"  {score:6.2f}  {expr}")


if __name__ == "__main__":
    print("GAMBA++ Optimized Usage Examples")
    print("=" * 70)
    
    # Run examples
    example_cache_usage()
    example_parallel_processing()
    example_batch_processing()
    example_complexity_scoring()
    example_optimized_function_processing()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


#!/usr/bin/env python3
"""
Performance Benchmark for GAMBA++ Optimizations

Compares optimized vs non-optimized implementations to measure speedup.
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimization.cache import GAMBACache
from optimization.parallel import process_with_gamba_direct, process_expressions_parallel
from optimization.batch import batch_process_expressions
from gamba.simplify_general import GeneralSimplifier


# Test expressions of varying complexity
TEST_EXPRESSIONS = [
    # Simple expressions
    "x + y",
    "x ^ y",
    "x & y",
    "x | y",
    
    # Medium complexity
    "(x ^ y) + 2*(x & y)",
    "(x | y) - (x & y)",
    "~x + ~y + 1",
    
    # Complex expressions
    "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",
    "(x & 0xFF) | (y & 0xFF00) | (z & 0xFF0000)",
    "((x ^ y) + 2*(x & y)) + ((x | y) - (x & y))",
]


def benchmark_direct_api_vs_subprocess():
    """
    Benchmark: Direct API vs Subprocess (simulated)
    
    Note: We can't easily benchmark subprocess without the old code,
    but we can measure the direct API performance.
    """
    print("=" * 70)
    print("Benchmark: Direct API Performance")
    print("=" * 70)
    
    results = []
    
    for expr in TEST_EXPRESSIONS[:5]:  # Test first 5
        times = []
        for _ in range(3):  # Run 3 times for average
            start = time.time()
            result = process_with_gamba_direct(expr, bitcount=32)
            elapsed = time.time() - start
            if result["success"]:
                times.append(elapsed)
        
        if times:
            avg_time = sum(times) / len(times)
            results.append({
                "expression": expr,
                "average_time": avg_time,
                "runs": len(times)
            })
            print(f"  {expr[:50]:50s} -> {avg_time:.3f}s")
    
    return results


def benchmark_parallel_vs_sequential():
    """Benchmark: Parallel vs Sequential Processing"""
    print("\n" + "=" * 70)
    print("Benchmark: Parallel vs Sequential Processing")
    print("=" * 70)
    
    expressions = TEST_EXPRESSIONS[:5]
    
    # Sequential processing
    print("\nSequential processing...")
    start = time.time()
    sequential_results = []
    for expr in expressions:
        result = process_with_gamba_direct(expr, bitcount=32)
        sequential_results.append(result)
    sequential_time = time.time() - start
    
    # Parallel processing
    print("Parallel processing...")
    start = time.time()
    parallel_results = process_expressions_parallel(
        expressions=expressions,
        bitcount=32,
        max_workers=4,
        use_cache=False,  # Disable cache for fair comparison
        show_progress=False
    )
    parallel_time = time.time() - start
    
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    
    print(f"\nResults:")
    print(f"  Sequential time: {sequential_time:.3f}s")
    print(f"  Parallel time:   {parallel_time:.3f}s")
    print(f"  Speedup:         {speedup:.2f}x")
    
    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
        "expressions": len(expressions)
    }


def benchmark_cache_performance():
    """Benchmark: Cache Hit Performance"""
    print("\n" + "=" * 70)
    print("Benchmark: Cache Performance")
    print("=" * 70)
    
    cache = GAMBACache(cache_file=Path(".benchmark_cache.json"))
    cache.clear()  # Start fresh
    
    expressions = TEST_EXPRESSIONS[:5]
    
    # First run: populate cache
    print("\nFirst run (populating cache)...")
    start = time.time()
    for expr in expressions:
        result = process_with_gamba_direct(expr, bitcount=32)
        if result["success"]:
            cache.set(expr, 32, result["simplified"])
    first_run_time = time.time() - start
    
    # Second run: use cache
    print("Second run (using cache)...")
    start = time.time()
    cached_count = 0
    for expr in expressions:
        cached = cache.get(expr, bitcount=32)
        if cached:
            cached_count += 1
        else:
            result = process_with_gamba_direct(expr, bitcount=32)
            if result["success"]:
                cache.set(expr, 32, result["simplified"])
    second_run_time = time.time() - start
    
    speedup = first_run_time / second_run_time if second_run_time > 0 else 0
    
    stats = cache.get_stats()
    
    print(f"\nResults:")
    print(f"  First run time:  {first_run_time:.3f}s")
    print(f"  Second run time: {second_run_time:.3f}s")
    print(f"  Speedup:         {speedup:.2f}x")
    print(f"  Cache hits:      {stats['hits']}")
    print(f"  Cache hit rate:  {stats['hit_rate_percent']}%")
    
    return {
        "first_run_time": first_run_time,
        "second_run_time": second_run_time,
        "speedup": speedup,
        "cache_hits": stats['hits'],
        "cache_hit_rate": stats['hit_rate_percent']
    }


def benchmark_batch_processing():
    """Benchmark: Batch Processing with Prioritization"""
    print("\n" + "=" * 70)
    print("Benchmark: Batch Processing")
    print("=" * 70)
    
    expressions = TEST_EXPRESSIONS
    
    # Without prioritization
    print("\nWithout prioritization...")
    start = time.time()
    results_no_priority = batch_process_expressions(
        expressions=expressions,
        bitcount=32,
        prioritize=False,
        use_cache=False,
        show_progress=False
    )
    time_no_priority = time.time() - start
    
    # With prioritization
    print("With prioritization...")
    start = time.time()
    results_with_priority = batch_process_expressions(
        expressions=expressions,
        bitcount=32,
        prioritize=True,
        use_cache=False,
        show_progress=False
    )
    time_with_priority = time.time() - start
    
    # Note: Prioritization may not always be faster, but it processes
    # simpler expressions first which can be useful for early results
    
    print(f"\nResults:")
    print(f"  Without priority: {time_no_priority:.3f}s")
    print(f"  With priority:     {time_with_priority:.3f}s")
    
    return {
        "time_no_priority": time_no_priority,
        "time_with_priority": time_with_priority,
        "expressions": len(expressions)
    }


def run_all_benchmarks():
    """Run all benchmarks and generate report"""
    print("GAMBA++ Performance Benchmarks")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {}
    }
    
    # Run benchmarks
    try:
        results["benchmarks"]["direct_api"] = benchmark_direct_api_vs_subprocess()
    except Exception as e:
        print(f"Error in direct API benchmark: {e}")
        results["benchmarks"]["direct_api"] = {"error": str(e)}
    
    try:
        results["benchmarks"]["parallel_vs_sequential"] = benchmark_parallel_vs_sequential()
    except Exception as e:
        print(f"Error in parallel benchmark: {e}")
        results["benchmarks"]["parallel_vs_sequential"] = {"error": str(e)}
    
    try:
        results["benchmarks"]["cache"] = benchmark_cache_performance()
    except Exception as e:
        print(f"Error in cache benchmark: {e}")
        results["benchmarks"]["cache"] = {"error": str(e)}
    
    try:
        results["benchmarks"]["batch"] = benchmark_batch_processing()
    except Exception as e:
        print(f"Error in batch benchmark: {e}")
        results["benchmarks"]["batch"] = {"error": str(e)}
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    
    if "parallel_vs_sequential" in results["benchmarks"]:
        ps = results["benchmarks"]["parallel_vs_sequential"]
        if "speedup" in ps:
            print(f"Parallel Processing Speedup: {ps['speedup']:.2f}x")
    
    if "cache" in results["benchmarks"]:
        cache = results["benchmarks"]["cache"]
        if "speedup" in cache:
            print(f"Cache Speedup: {cache['speedup']:.2f}x")
            print(f"Cache Hit Rate: {cache.get('cache_hit_rate', 0):.1f}%")
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    run_all_benchmarks()


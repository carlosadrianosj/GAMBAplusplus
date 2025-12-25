#!/usr/bin/env python3
"""
Advanced Performance Benchmark for GAMBA++

Tests all advanced optimizations and measures combined speedup.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimization.cache import GAMBACache
from optimization.parallel import process_expressions_parallel
from optimization.parallel_advanced import (
    process_expressions_parallel_advanced,
    quick_simplify_check
)
from optimization.batch_advanced import process_expressions_batch_advanced
from optimization.simplifier_pool import SimplifierPool
from gamba.simplify_general import GeneralSimplifier


# Test expressions
TEST_EXPRESSIONS = [
    # Simple (should early terminate)
    "x + y",
    "x ^ y",
    
    # Medium
    "(x ^ y) + 2*(x & y)",
    "(x | y) - (x & y)",
    "~x + ~y + 1",
    
    # Complex
    "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",
    "(x & 0xFF) | (y & 0xFF00)",
] * 5  # 35 expressions total


def benchmark_baseline(expressions: List[str]) -> Dict:
    """Baseline: Original GAMBA sequential"""
    print("\n[Baseline] Original GAMBA sequential...")
    start = time.time()
    successful = 0
    
    for expr in expressions:
        try:
            simplifier = GeneralSimplifier(32, False, None)
            result = simplifier.simplify(expr, useZ3=False)
            if result:
                successful += 1
        except:
            pass
    
    elapsed = time.time() - start
    return {
        "time": elapsed,
        "successful": successful,
        "method": "Baseline (sequential)"
    }


def benchmark_threadpool(expressions: List[str]) -> Dict:
    """ThreadPoolExecutor (current implementation)"""
    print("\n[ThreadPool] Current ThreadPoolExecutor (4 workers)...")
    start = time.time()
    results = process_expressions_parallel(
        expressions, bitcount=32, max_workers=4, use_cache=False, show_progress=False
    )
    elapsed = time.time() - start
    successful = sum(1 for r in results if r["success"])
    
    return {
        "time": elapsed,
        "successful": successful,
        "method": "ThreadPoolExecutor (4 workers)"
    }


def benchmark_processpool(expressions: List[str]) -> Dict:
    """ProcessPoolExecutor (8 workers) - true parallelism"""
    print("\n[ProcessPool] ProcessPoolExecutor (8 workers)...")
    start = time.time()
    results = process_expressions_parallel_advanced(
        expressions, bitcount=32, max_workers=8, use_cache=False, show_progress=False
    )
    elapsed = time.time() - start
    successful = sum(1 for r in results if r["success"])
    early_terminated = sum(1 for r in results if r.get("early_terminated", False))
    
    return {
        "time": elapsed,
        "successful": successful,
        "method": "ProcessPoolExecutor (8 workers)",
        "early_terminated": early_terminated
    }


def benchmark_with_cache(expressions: List[str]) -> Dict:
    """ProcessPoolExecutor with cache"""
    print("\n[Cache] ProcessPoolExecutor (8 workers) + Cache...")
    cache = GAMBACache(Path(".benchmark_cache_advanced.json"))
    cache.clear()
    
    start = time.time()
    results = process_expressions_parallel_advanced(
        expressions, bitcount=32, max_workers=8, use_cache=True, cache=cache, show_progress=False
    )
    elapsed = time.time() - start
    successful = sum(1 for r in results if r["success"])
    cached = sum(1 for r in results if r.get("cached", False))
    stats = cache.get_stats()
    
    return {
        "time": elapsed,
        "successful": successful,
        "method": "ProcessPoolExecutor (8 workers) + Cache",
        "cached": cached,
        "cache_hit_rate": stats["hit_rate_percent"]
    }


def benchmark_batch_processing(expressions: List[str]) -> Dict:
    """Batch processing within GAMBA"""
    print("\n[Batch] Batch processing (8 workers, batch_size=5)...")
    start = time.time()
    results = process_expressions_batch_advanced(
        expressions, bitcount=32, max_workers=8, batch_size=5, use_cache=False, show_progress=False
    )
    elapsed = time.time() - start
    successful = sum(1 for r in results if r["success"])
    
    return {
        "time": elapsed,
        "successful": successful,
        "method": "Batch processing (8 workers)"
    }


def benchmark_fully_optimized(expressions: List[str]) -> Dict:
    """All optimizations combined"""
    print("\n[Optimized] All optimizations (8 workers + cache + batch)...")
    cache = GAMBACache(Path(".benchmark_cache_full.json"))
    cache.clear()
    
    start = time.time()
    results = process_expressions_batch_advanced(
        expressions, bitcount=32, max_workers=8, batch_size=5,
        use_cache=True, cache=cache, show_progress=False
    )
    elapsed = time.time() - start
    successful = sum(1 for r in results if r["success"])
    cached = sum(1 for r in results if r.get("cached", False))
    stats = cache.get_stats()
    
    return {
        "time": elapsed,
        "successful": successful,
        "method": "Fully Optimized (8 workers + cache + batch)",
        "cached": cached,
        "cache_hit_rate": stats["hit_rate_percent"]
    }


def run_advanced_benchmarks():
    """Run all advanced benchmarks"""
    print("=" * 70)
    print("GAMBA++ Advanced Performance Benchmarks")
    print("=" * 70)
    print(f"\nTest expressions: {len(TEST_EXPRESSIONS)}")
    print(f"Expressions: {TEST_EXPRESSIONS[:3]}... (+ {len(TEST_EXPRESSIONS)-3} more)\n")
    
    results = {}
    
    # Run benchmarks
    try:
        results["baseline"] = benchmark_baseline(TEST_EXPRESSIONS)
    except Exception as e:
        print(f"Error in baseline: {e}")
        results["baseline"] = {"time": 0, "method": "Error"}
    
    try:
        results["threadpool"] = benchmark_threadpool(TEST_EXPRESSIONS)
    except Exception as e:
        print(f"Error in threadpool: {e}")
        results["threadpool"] = {"time": 0, "method": "Error"}
    
    try:
        results["processpool"] = benchmark_processpool(TEST_EXPRESSIONS)
    except Exception as e:
        print(f"Error in processpool: {e}")
        results["processpool"] = {"time": 0, "method": "Error"}
    
    try:
        results["with_cache"] = benchmark_with_cache(TEST_EXPRESSIONS)
    except Exception as e:
        print(f"Error in cache: {e}")
        results["with_cache"] = {"time": 0, "method": "Error"}
    
    try:
        results["batch"] = benchmark_batch_processing(TEST_EXPRESSIONS)
    except Exception as e:
        print(f"Error in batch: {e}")
        results["batch"] = {"time": 0, "method": "Error"}
    
    try:
        results["optimized"] = benchmark_fully_optimized(TEST_EXPRESSIONS)
    except Exception as e:
        print(f"Error in optimized: {e}")
        results["optimized"] = {"time": 0, "method": "Error"}
    
    # Print results
    print("\n" + "=" * 70)
    print("ADVANCED BENCHMARK RESULTS")
    print("=" * 70)
    
    baseline_time = results["baseline"]["time"]
    
    print(f"\n{'Method':<50} {'Time (s)':<12} {'Speedup':<12} {'Success':<10}")
    print("-" * 85)
    
    for key, result in results.items():
        method = result["method"]
        time_val = result["time"]
        speedup = baseline_time / time_val if time_val > 0 else 0
        success = result["successful"]
        
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
        
        print(f"{method:<50} {time_val:<12.3f} {speedup_str:<12} {success}/{len(TEST_EXPRESSIONS)}")
        
        if "early_terminated" in result:
            print(f"  └─ Early terminated: {result['early_terminated']}")
        if "cached" in result:
            print(f"  └─ Cached: {result['cached']}/{len(TEST_EXPRESSIONS)}")
        if "cache_hit_rate" in result:
            print(f"  └─ Cache hit rate: {result['cache_hit_rate']:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUMMARY")
    print("=" * 70)
    
    if "processpool" in results and results["processpool"]["time"] > 0:
        pp = results["processpool"]
        speedup = baseline_time / pp["time"]
        print(f"\n✓ ProcessPoolExecutor (8 workers) vs Baseline: {speedup:.2f}x faster")
    
    if "optimized" in results and results["optimized"]["time"] > 0:
        opt = results["optimized"]
        speedup = baseline_time / opt["time"]
        print(f"✓ Fully Optimized vs Baseline: {speedup:.2f}x faster")
    
    if "threadpool" in results and "processpool" in results:
        if results["threadpool"]["time"] > 0 and results["processpool"]["time"] > 0:
            speedup = results["threadpool"]["time"] / results["processpool"]["time"]
            print(f"✓ ProcessPool vs ThreadPool: {speedup:.2f}x faster")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    run_advanced_benchmarks()


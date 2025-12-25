#!/usr/bin/env python3
"""
Compare GAMBA++ Optimized vs Original GAMBA Performance

Measures speedup achieved by optimizations.
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Original GAMBA path
ORIGINAL_GAMBA_PATH = Path(__file__).parent.parent.parent.parent / "keep-simple" / "GAMBA"

# GAMBA++ optimized
from optimization.cache import GAMBACache
from optimization.parallel import process_with_gamba_direct, process_expressions_parallel
from gamba.simplify_general import GeneralSimplifier


# Test expressions of varying complexity
TEST_EXPRESSIONS = [
    # Simple
    "x + y",
    "x ^ y",
    "x & y",
    
    # Medium
    "(x ^ y) + 2*(x & y)",
    "(x | y) - (x & y)",
    "~x + ~y + 1",
    
    # Complex
    "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",
    "(x & 0xFF) | (y & 0xFF00)",
]


def benchmark_original_gamba_subprocess(expressions: List[str], bitcount: int = 32) -> Dict:
    """Benchmark original GAMBA using subprocess (simulated old way)"""
    print("\n[Original GAMBA] Testing subprocess approach...")
    
    original_script = ORIGINAL_GAMBA_PATH / "src" / "simplify_general.py"
    
    if not original_script.exists():
        print(f"  [!] Original GAMBA script not found at {original_script}")
        print(f"  [!] Skipping subprocess benchmark")
        return {"time": 0, "successful": 0, "method": "subprocess (not available)"}
    
    start_time = time.time()
    successful = 0
    
    for expr in expressions:
        try:
            cmd = [
                sys.executable,
                str(original_script),
                "-b", str(bitcount),
                expr
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(original_script.parent.parent)
            )
            
            if result.returncode == 0:
                successful += 1
        except Exception as e:
            pass
    
    elapsed = time.time() - start_time
    
    return {
        "time": elapsed,
        "successful": successful,
        "method": "subprocess",
        "expressions": len(expressions)
    }


def benchmark_original_gamba_direct(expressions: List[str], bitcount: int = 32) -> Dict:
    """Benchmark original GAMBA using direct API (same as GAMBA++ but without optimizations)"""
    print("\n[Original GAMBA] Testing direct API (no optimizations)...")
    
    start_time = time.time()
    successful = 0
    
    for expr in expressions:
        try:
            simplifier = GeneralSimplifier(bitcount, modRed=False, verifBitCount=None)
            result = simplifier.simplify(expr, useZ3=False)
            if result:
                successful += 1
        except Exception as e:
            pass
    
    elapsed = time.time() - start_time
    
    return {
        "time": elapsed,
        "successful": successful,
        "method": "direct API (no cache, no parallel)",
        "expressions": len(expressions)
    }


def benchmark_gambaplusplus_sequential(expressions: List[str], bitcount: int = 32) -> Dict:
    """Benchmark GAMBA++ with direct API but sequential (no parallel)"""
    print("\n[GAMBA++] Testing direct API sequential (no cache, no parallel)...")
    
    start_time = time.time()
    successful = 0
    
    for expr in expressions:
        result = process_with_gamba_direct(expr, bitcount=bitcount, use_z3=False)
        if result["success"]:
            successful += 1
    
    elapsed = time.time() - start_time
    
    return {
        "time": elapsed,
        "successful": successful,
        "method": "GAMBA++ direct API sequential",
        "expressions": len(expressions)
    }


def benchmark_gambaplusplus_cached(expressions: List[str], bitcount: int = 32) -> Dict:
    """Benchmark GAMBA++ with cache"""
    print("\n[GAMBA++] Testing with cache...")
    
    cache = GAMBACache(cache_file=Path(".benchmark_cache_temp.json"))
    cache.clear()  # Start fresh
    
    # First run: populate cache
    start_time = time.time()
    successful = 0
    
    for expr in expressions:
        cached = cache.get(expr, bitcount)
        if cached:
            successful += 1
        else:
            result = process_with_gamba_direct(expr, bitcount=bitcount)
            if result["success"]:
                cache.set(expr, bitcount, result["simplified"])
                successful += 1
    
    first_run_time = time.time() - start_time
    
    # Second run: use cache
    start_time = time.time()
    cached_count = 0
    
    for expr in expressions:
        cached = cache.get(expr, bitcount)
        if cached:
            cached_count += 1
            successful += 1
    
    second_run_time = time.time() - start_time
    
    stats = cache.get_stats()
    
    return {
        "time": first_run_time,
        "second_run_time": second_run_time,
        "successful": successful,
        "method": "GAMBA++ with cache",
        "cache_hits": stats["hits"],
        "cache_hit_rate": stats["hit_rate_percent"],
        "expressions": len(expressions)
    }


def benchmark_gambaplusplus_parallel(expressions: List[str], bitcount: int = 32, max_workers: int = 4) -> Dict:
    """Benchmark GAMBA++ with parallel processing"""
    print(f"\n[GAMBA++] Testing parallel processing ({max_workers} workers)...")
    
    start_time = time.time()
    
    results = process_expressions_parallel(
        expressions=expressions,
        bitcount=bitcount,
        max_workers=max_workers,
        use_cache=False,  # Disable cache for fair comparison
        show_progress=False
    )
    
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    
    return {
        "time": elapsed,
        "successful": successful,
        "method": f"GAMBA++ parallel ({max_workers} workers)",
        "expressions": len(expressions)
    }


def benchmark_gambaplusplus_optimized(expressions: List[str], bitcount: int = 32, max_workers: int = 4) -> Dict:
    """Benchmark GAMBA++ with all optimizations (cache + parallel)"""
    print(f"\n[GAMBA++] Testing with all optimizations (cache + parallel, {max_workers} workers)...")
    
    cache = GAMBACache(cache_file=Path(".benchmark_cache_optimized.json"))
    cache.clear()  # Start fresh
    
    start_time = time.time()
    
    results = process_expressions_parallel(
        expressions=expressions,
        bitcount=bitcount,
        max_workers=max_workers,
        use_cache=True,
        cache=cache,
        show_progress=False
    )
    
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    cached = sum(1 for r in results if r.get("cached", False))
    
    stats = cache.get_stats()
    
    return {
        "time": elapsed,
        "successful": successful,
        "method": "GAMBA++ optimized (cache + parallel)",
        "cache_hits": stats["hits"],
        "cache_hit_rate": stats["hit_rate_percent"],
        "cached_in_run": cached,
        "expressions": len(expressions)
    }


def run_comparison():
    """Run complete comparison"""
    print("=" * 70)
    print("GAMBA++ vs Original GAMBA Performance Comparison")
    print("=" * 70)
    print(f"\nTest expressions: {len(TEST_EXPRESSIONS)}")
    print(f"Expressions: {TEST_EXPRESSIONS[:3]}... (+ {len(TEST_EXPRESSIONS)-3} more)\n")
    
    results = {}
    
    # Benchmark 1: Original GAMBA direct API (baseline)
    results["original_direct"] = benchmark_original_gamba_direct(TEST_EXPRESSIONS)
    
    # Benchmark 2: GAMBA++ sequential (direct API, no optimizations)
    results["gambaplusplus_sequential"] = benchmark_gambaplusplus_sequential(TEST_EXPRESSIONS)
    
    # Benchmark 3: GAMBA++ with cache
    results["gambaplusplus_cached"] = benchmark_gambaplusplus_cached(TEST_EXPRESSIONS)
    
    # Benchmark 4: GAMBA++ parallel
    import multiprocessing as mp
    max_workers = min(4, mp.cpu_count())
    results["gambaplusplus_parallel"] = benchmark_gambaplusplus_parallel(TEST_EXPRESSIONS, max_workers=max_workers)
    
    # Benchmark 5: GAMBA++ fully optimized
    results["gambaplusplus_optimized"] = benchmark_gambaplusplus_optimized(TEST_EXPRESSIONS, max_workers=max_workers)
    
    # Try original subprocess if available
    try:
        results["original_subprocess"] = benchmark_original_gamba_subprocess(TEST_EXPRESSIONS)
    except:
        pass
    
    # Print comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 70)
    
    baseline_time = results["original_direct"]["time"]
    
    print(f"\n{'Method':<50} {'Time (s)':<12} {'Speedup':<12} {'Success':<10}")
    print("-" * 85)
    
    for key, result in results.items():
        method = result["method"]
        time_val = result["time"]
        speedup = baseline_time / time_val if time_val > 0 else 0
        success = result["successful"]
        
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
        
        print(f"{method:<50} {time_val:<12.3f} {speedup_str:<12} {success}/{result['expressions']}")
        
        if "cache_hit_rate" in result:
            print(f"  └─ Cache hit rate: {result['cache_hit_rate']:.1f}%")
        if "second_run_time" in result:
            second_speedup = baseline_time / result["second_run_time"] if result["second_run_time"] > 0 else 0
            print(f"  └─ Second run (cached): {result['second_run_time']:.3f}s ({second_speedup:.2f}x speedup)")
    
    # Calculate improvements
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUMMARY")
    print("=" * 70)
    
    if "gambaplusplus_optimized" in results:
        opt = results["gambaplusplus_optimized"]
        speedup = baseline_time / opt["time"] if opt["time"] > 0 else 0
        print(f"\n✓ GAMBA++ Optimized vs Original Direct API: {speedup:.2f}x faster")
    
    if "gambaplusplus_parallel" in results:
        par = results["gambaplusplus_parallel"]
        speedup = baseline_time / par["time"] if par["time"] > 0 else 0
        print(f"✓ GAMBA++ Parallel vs Original: {speedup:.2f}x faster")
    
    if "gambaplusplus_cached" in results and "second_run_time" in results["gambaplusplus_cached"]:
        cached = results["gambaplusplus_cached"]
        speedup = baseline_time / cached["second_run_time"] if cached["second_run_time"] > 0 else 0
        print(f"✓ GAMBA++ Cached (2nd run) vs Original: {speedup:.2f}x faster")
    
    if "original_subprocess" in results and results["original_subprocess"]["time"] > 0:
        sub = results["original_subprocess"]
        if "gambaplusplus_optimized" in results:
            opt = results["gambaplusplus_optimized"]
            speedup = sub["time"] / opt["time"] if opt["time"] > 0 else 0
            print(f"✓ GAMBA++ Optimized vs Original Subprocess: {speedup:.2f}x faster")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_comparison()


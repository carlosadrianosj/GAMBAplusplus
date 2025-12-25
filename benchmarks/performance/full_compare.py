#!/usr/bin/env python3
"""
Full Comparison: GAMBA++ vs Original GAMBA

Comprehensive test with many expressions including duplicates to test cache.
"""

import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimization.parallel_advanced import process_expressions_parallel_advanced
from optimization.batch_advanced import process_expressions_batch_advanced
from optimization.cache import GAMBACache
from gamba.simplify_general import GeneralSimplifier

# Large set of test expressions with duplicates
BASE_EXPRESSIONS = [
    # Simple
    "x + y",
    "x ^ y",
    "x & y",
    "x | y",
    
    # Medium
    "(x ^ y) + 2*(x & y)",
    "(x | y) - (x & y)",
    "~x + ~y + 1",
    "(x << 1) ^ (y >> 1)",
    "x + (y & ~x)",
    
    # High
    "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",
    "(x & y) + (x & ~y) + (~x & y) + (~x & ~y)",
    "(x | y) + (x & y) + (x ^ y)",
    "((x ^ y) << 1) + ((x & y) << 2)",
    
    # Multi-variable
    "x + y + z",
    "(x ^ y) + (y ^ z) + (z ^ x)",
    "(x & y) | (y & z) | (z & x)",
    "((x & y) | (x & z)) + ((y & z) | (y & x))",
    "x + (y & z) + (z & x)",
    
    # Complex multi-variable
    "((x & y) | (y & z)) + ((x & z) | (z & y))",
    "(x ^ y ^ z) + 2*((x & y) | (y & z) | (z & x))",
    "((x | y) & (y | z)) + ((x | z) & (z | y))",
    "x + y + z + (x & y) + (y & z) + (z & x)",
]

# Create list with duplicates (5x each)
FULL_EXPRESSIONS = BASE_EXPRESSIONS * 5  # 100 expressions total (20 unique)


def run_original_gamba(expressions):
    """Run original GAMBA via subprocess"""
    gamba_path = Path(__file__).parent.parent.parent.parent / "keep-simple" / "GAMBA" / "src" / "simplify_general.py"
    
    if not gamba_path.exists():
        return 0.0, []
    
    start = time.time()
    successful = 0
    
    for i, expr in enumerate(expressions):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(expressions)}...", end='\r')
        try:
            result = subprocess.run(
                [sys.executable, str(gamba_path), expr, "-b", "32"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and "simplified to" in result.stdout:
                successful += 1
        except:
            pass
    
    print()  # New line after progress
    elapsed = time.time() - start
    return elapsed, successful


def run_gamba_plusplus_sequential(expressions):
    """Run GAMBA++ sequential"""
    start = time.time()
    successful = 0
    
    import os
    import sys
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    for i, expr in enumerate(expressions):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(expressions)}...", end='\r')
        try:
            simplifier = GeneralSimplifier(32, False, None)
            result = simplifier.simplify(expr, useZ3=False)
            if result and result.strip():
                successful += 1
        except:
            pass
    
    print()  # New line after progress
    sys.stderr.close()
    sys.stderr = old_stderr
    
    elapsed = time.time() - start
    return elapsed, successful


def run_gamba_plusplus_parallel(expressions):
    """Run GAMBA++ parallel (8 cores)"""
    print("  Processing with 8 workers...")
    start = time.time()
    results = process_expressions_parallel_advanced(
        expressions, bitcount=32, max_workers=8, use_cache=False, show_progress=True
    )
    elapsed = time.time() - start
    successful = sum(1 for r in results if r["success"])
    return elapsed, successful


def run_gamba_plusplus_optimized(expressions):
    """Run GAMBA++ fully optimized"""
    print("  Processing with 8 workers + cache + batch...")
    cache = GAMBACache(Path(".full_compare_cache.json"))
    cache.clear()
    
    start = time.time()
    results = process_expressions_batch_advanced(
        expressions, bitcount=32, max_workers=8, batch_size=10,
        use_cache=True, cache=cache, show_progress=True
    )
    elapsed = time.time() - start
    successful = sum(1 for r in results if r["success"])
    cached = sum(1 for r in results if r.get("cached", False))
    stats = cache.get_stats()
    return elapsed, successful, cached, stats


def main():
    print("=" * 80)
    print("FULL COMPARISON: GAMBA++ vs ORIGINAL GAMBA")
    print("=" * 80)
    print(f"\nTest expressions: {len(FULL_EXPRESSIONS)}")
    print(f"Unique expressions: {len(BASE_EXPRESSIONS)}")
    print(f"Duplicates: {len(FULL_EXPRESSIONS) - len(BASE_EXPRESSIONS)} (to test cache)")
    print(f"Expected cache hits: {len(FULL_EXPRESSIONS) - len(BASE_EXPRESSIONS)}\n")
    
    # 1. Original GAMBA
    print("[1/4] Original GAMBA (Sequential)...")
    time_orig, success_orig = run_original_gamba(FULL_EXPRESSIONS)
    print(f"  ✓ Time: {time_orig:.3f}s, Success: {success_orig}/{len(FULL_EXPRESSIONS)}")
    
    # 2. GAMBA++ Sequential
    print("\n[2/4] GAMBA++ Sequential...")
    time_seq, success_seq = run_gamba_plusplus_sequential(FULL_EXPRESSIONS)
    print(f"  ✓ Time: {time_seq:.3f}s, Success: {success_seq}/{len(FULL_EXPRESSIONS)}")
    
    # 3. GAMBA++ Parallel
    print("\n[3/4] GAMBA++ Parallel (8 cores, no cache)...")
    time_par, success_par = run_gamba_plusplus_parallel(FULL_EXPRESSIONS)
    print(f"  ✓ Time: {time_par:.3f}s, Success: {success_par}/{len(FULL_EXPRESSIONS)}")
    
    # 4. GAMBA++ Optimized
    print("\n[4/4] GAMBA++ Fully Optimized (8 cores + cache + batch)...")
    time_opt, success_opt, cached, cache_stats = run_gamba_plusplus_optimized(FULL_EXPRESSIONS)
    print(f"  ✓ Time: {time_opt:.3f}s, Success: {success_opt}/{len(FULL_EXPRESSIONS)}")
    print(f"  ✓ Cache hits: {cached}/{len(FULL_EXPRESSIONS)} ({cache_stats.get('hit_rate_percent', 0):.1f}% hit rate)")
    print(f"  ✓ Cache size: {cache_stats.get('cache_size', 0)} entries")
    
    # Summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Method':<55} {'Time (s)':<12} {'Speedup':<12} {'Success':<10}")
    print("-" * 90)
    
    baseline = time_orig
    for name, t, s in [
        ("Original GAMBA", time_orig, success_orig),
        ("GAMBA++ Sequential", time_seq, success_seq),
        ("GAMBA++ Parallel (8 cores)", time_par, success_par),
        ("GAMBA++ Optimized (cache+batch)", time_opt, success_opt),
    ]:
        speedup = baseline / t if t > 0 else 0
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
        print(f"{name:<55} {t:<12.3f} {speedup_str:<12} {s}/{len(FULL_EXPRESSIONS)}")
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT BREAKDOWN")
    print("=" * 80)
    
    if time_orig > 0 and time_seq > 0:
        speedup = time_orig / time_seq
        print(f"✓ GAMBA++ Sequential vs Original: {speedup:.2f}x faster")
    
    if time_seq > 0 and time_par > 0:
        speedup = time_seq / time_par
        print(f"✓ Parallel (8 cores) vs Sequential: {speedup:.2f}x faster")
    
    if time_par > 0 and time_opt > 0:
        speedup = time_par / time_opt
        print(f"✓ Optimized (cache+batch) vs Parallel: {speedup:.2f}x faster")
    
    if time_orig > 0 and time_opt > 0:
        speedup = time_orig / time_opt
        print(f"✓ GAMBA++ Optimized vs Original: {speedup:.2f}x faster (TOTAL)")
    
    print("\n" + "=" * 80)
    print(f"Cache Effectiveness: {cached} expressions from cache ({cache_stats.get('hit_rate_percent', 0):.1f}% hit rate)")
    print("=" * 80)


if __name__ == "__main__":
    main()


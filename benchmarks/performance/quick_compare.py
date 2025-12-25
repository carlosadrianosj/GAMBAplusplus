#!/usr/bin/env python3
"""
Quick Comparison: GAMBA++ vs Original GAMBA

Simplified version for faster testing with a smaller set of expressions.
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

# Test expressions - mix of complexities
TEST_EXPRESSIONS = [
    # Simple
    "x + y",
    "x ^ y",
    "x & y",
    "x | y",
    
    # Medium
    "(x ^ y) + 2*(x & y)",
    "(x | y) - (x & y)",
    "~x + ~y + 1",
    
    # High
    "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",
    "(x & y) + (x & ~y) + (~x & y) + (~x & ~y)",
    "(x | y) + (x & y) + (x ^ y)",
    
    # Multi-variable
    "x + y + z",
    "(x ^ y) + (y ^ z) + (z ^ x)",
    "(x & y) | (y & z) | (z & x)",
] * 2  # 26 expressions total


def run_original_gamba(expressions):
    """Run original GAMBA via subprocess"""
    gamba_path = Path(__file__).parent.parent.parent.parent / "keep-simple" / "GAMBA" / "src" / "simplify_general.py"
    
    if not gamba_path.exists():
        return 0.0, []
    
    start = time.time()
    successful = 0
    
    for expr in expressions:
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
    
    for expr in expressions:
        try:
            simplifier = GeneralSimplifier(32, False, None)
            result = simplifier.simplify(expr, useZ3=False)
            if result and result.strip():
                successful += 1
        except:
            pass
    
    sys.stderr.close()
    sys.stderr = old_stderr
    
    elapsed = time.time() - start
    return elapsed, successful


def run_gamba_plusplus_parallel(expressions):
    """Run GAMBA++ parallel (8 cores)"""
    start = time.time()
    results = process_expressions_parallel_advanced(
        expressions, bitcount=32, max_workers=8, use_cache=False, show_progress=False
    )
    elapsed = time.time() - start
    successful = sum(1 for r in results if r["success"])
    return elapsed, successful


def run_gamba_plusplus_optimized(expressions):
    """Run GAMBA++ fully optimized"""
    cache = GAMBACache(Path(".quick_compare_cache.json"))
    cache.clear()
    
    start = time.time()
    results = process_expressions_batch_advanced(
        expressions, bitcount=32, max_workers=8, batch_size=5,
        use_cache=True, cache=cache, show_progress=False
    )
    elapsed = time.time() - start
    successful = sum(1 for r in results if r["success"])
    stats = cache.get_stats()
    return elapsed, successful, stats


def main():
    print("=" * 70)
    print("QUICK COMPARISON: GAMBA++ vs ORIGINAL GAMBA")
    print("=" * 70)
    print(f"\nTest expressions: {len(TEST_EXPRESSIONS)}")
    print(f"Unique expressions: {len(TEST_EXPRESSIONS) // 2}\n")
    
    # 1. Original GAMBA
    print("[1/4] Original GAMBA (Sequential)...")
    time_orig, success_orig = run_original_gamba(TEST_EXPRESSIONS)
    print(f"  Time: {time_orig:.3f}s, Success: {success_orig}/{len(TEST_EXPRESSIONS)}")
    
    # 2. GAMBA++ Sequential
    print("\n[2/4] GAMBA++ Sequential...")
    time_seq, success_seq = run_gamba_plusplus_sequential(TEST_EXPRESSIONS)
    print(f"  Time: {time_seq:.3f}s, Success: {success_seq}/{len(TEST_EXPRESSIONS)}")
    
    # 3. GAMBA++ Parallel
    print("\n[3/4] GAMBA++ Parallel (8 cores)...")
    time_par, success_par = run_gamba_plusplus_parallel(TEST_EXPRESSIONS)
    print(f"  Time: {time_par:.3f}s, Success: {success_par}/{len(TEST_EXPRESSIONS)}")
    
    # 4. GAMBA++ Optimized
    print("\n[4/4] GAMBA++ Fully Optimized (8 cores + cache + batch)...")
    time_opt, success_opt, cache_stats = run_gamba_plusplus_optimized(TEST_EXPRESSIONS)
    print(f"  Time: {time_opt:.3f}s, Success: {success_opt}/{len(TEST_EXPRESSIONS)}")
    if cache_stats:
        print(f"  Cache hits: {cache_stats.get('hits', 0)}, Hit rate: {cache_stats.get('hit_rate_percent', 0):.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Method':<50} {'Time (s)':<12} {'Speedup':<12} {'Success':<10}")
    print("-" * 85)
    
    baseline = time_orig
    for name, t, s in [
        ("Original GAMBA", time_orig, success_orig),
        ("GAMBA++ Sequential", time_seq, success_seq),
        ("GAMBA++ Parallel (8 cores)", time_par, success_par),
        ("GAMBA++ Optimized", time_opt, success_opt),
    ]:
        speedup = baseline / t if t > 0 else 0
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
        print(f"{name:<50} {t:<12.3f} {speedup_str:<12} {s}/{len(TEST_EXPRESSIONS)}")
    
    print("\n" + "=" * 70)
    if time_orig > 0 and time_opt > 0:
        speedup = time_orig / time_opt
        print(f"✓ GAMBA++ Optimized is {speedup:.2f}x faster than Original GAMBA")
    if time_seq > 0 and time_par > 0:
        speedup = time_seq / time_par
        print(f"✓ Parallel processing is {speedup:.2f}x faster than sequential")
    print("=" * 70)


if __name__ == "__main__":
    main()


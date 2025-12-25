#!/usr/bin/env python3
"""
Comprehensive Comparison: GAMBA++ vs Original GAMBA

Tests both versions with complex, large-scale expressions and compares:
- Processing time
- Success rate
- Results correctness
- Speedup achieved
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "keep-simple" / "GAMBA" / "src"))

from optimization.cache import GAMBACache
from optimization.parallel_advanced import process_expressions_parallel_advanced
from optimization.batch_advanced import process_expressions_batch_advanced
from gamba.simplify_general import GeneralSimplifier

# Import original GAMBA
try:
    from simplify_general import GeneralSimplifier as OriginalGeneralSimplifier
    HAS_ORIGINAL = True
except ImportError:
    HAS_ORIGINAL = False
    print("Warning: Original GAMBA not found. Will use subprocess method.")


# Complex test expressions - larger and more complex than previous tests
COMPLEX_EXPRESSIONS = [
    # Simple (should early terminate in GAMBA++)
    "x + y",
    "x ^ y",
    "x & y",
    "x | y",
    
    # Medium complexity
    "(x ^ y) + 2*(x & y)",
    "(x | y) - (x & y)",
    "~x + ~y + 1",
    "(x & 0xFF) | (y & 0xFF00)",
    "(x << 1) ^ (y >> 1)",
    "x + (y & ~x)",
    
    # High complexity - nested operations
    "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",
    "((x | y) & (x ^ y)) + ((~x | y) & (x | ~y))",
    "(x & y) + (x & ~y) + (~x & y) + (~x & ~y)",
    "((x ^ y) << 1) + ((x & y) << 2)",
    "(x | y) + (x & y) + (x ^ y)",
    
    # Very high complexity - deeply nested
    "((x & y) + (x & ~y)) + ((~x & y) + (~x & ~y))",
    "((x ^ y) + 2*(x & y)) + ((x | y) - (x & y))",
    "((x & 0xFF) | (y & 0xFF00)) + ((x & 0xFF00) | (y & 0xFF))",
    
    # Multiple variables
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
    "((x ^ y) + (y ^ z)) + ((x & y) + (y & z))",
    
    # Very complex multi-variable
    "((x ^ y) + (y ^ z) + (z ^ x)) + 2*((x & y) | (y & z) | (z & x))",
    "((x & y) | (y & z) | (z & x)) + ((x | y) & (y | z) & (z | x))",
    "x + y + z + (x & y & z) + (x | y | z)",
] * 1  # 30 expressions total (30 unique)


def run_original_gamba_subprocess(expressions: List[str], bitcount: int = 32) -> Tuple[float, List[Dict]]:
    """
    Run original GAMBA via subprocess (command line).
    
    Args:
        expressions: List of expressions
        bitcount: Bit width
    
    Returns:
        Tuple of (time_elapsed, results)
    """
    gamba_path = Path(__file__).parent.parent.parent.parent / "keep-simple" / "GAMBA" / "src" / "simplify_general.py"
    
    if not gamba_path.exists():
        return 0.0, []
    
    results = []
    start = time.time()
    
    for expr in expressions:
        try:
            cmd = [
                sys.executable,
                str(gamba_path),
                expr,
                "-b", str(bitcount)
            ]
            
            proc_start = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            proc_time = time.time() - proc_start
            
            if result.returncode == 0:
                # Parse output
                output = result.stdout
                if "simplified to" in output:
                    simplified = output.split("simplified to")[-1].strip()
                    results.append({
                        "original": expr,
                        "simplified": simplified,
                        "success": True,
                        "error": None,
                        "processing_time": proc_time
                    })
                else:
                    results.append({
                        "original": expr,
                        "simplified": None,
                        "success": False,
                        "error": "No simplified result in output",
                        "processing_time": proc_time
                    })
            else:
                results.append({
                    "original": expr,
                    "simplified": None,
                    "success": False,
                    "error": result.stderr[:100] if result.stderr else "Unknown error",
                    "processing_time": proc_time
                })
        except subprocess.TimeoutExpired:
            results.append({
                "original": expr,
                "simplified": None,
                "success": False,
                "error": "Timeout",
                "processing_time": 30.0
            })
        except Exception as e:
            results.append({
                "original": expr,
                "simplified": None,
                "success": False,
                "error": str(e)[:100],
                "processing_time": 0.0
            })
    
    elapsed = time.time() - start
    return elapsed, results


def run_original_gamba_direct(expressions: List[str], bitcount: int = 32) -> Tuple[float, List[Dict]]:
    """
    Run original GAMBA using direct API (if available).
    
    Args:
        expressions: List of expressions
        bitcount: Bit width
    
    Returns:
        Tuple of (time_elapsed, results)
    """
    if not HAS_ORIGINAL:
        return run_original_gamba_subprocess(expressions, bitcount)
    
    results = []
    start = time.time()
    
    for expr in expressions:
        expr_start = time.time()
        try:
            # Suppress stdout/stderr to avoid printing errors
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                simplifier = OriginalGeneralSimplifier(bitcount, modRed=False, verifBitCount=None)
                simplified = simplifier.simplify(expr, useZ3=False)
            
            elapsed = time.time() - expr_start
            
            if simplified and simplified.strip():
                results.append({
                    "original": expr,
                    "simplified": simplified.strip(),
                    "success": True,
                    "error": None,
                    "processing_time": elapsed
                })
            else:
                results.append({
                    "original": expr,
                    "simplified": None,
                    "success": False,
                    "error": "GAMBA returned empty result or parsing error",
                    "processing_time": elapsed
                })
        except Exception as e:
            error_msg = str(e)
            # Filter out common parsing errors
            if "Could not parse" in error_msg or "No simplification" in error_msg:
                error_msg = "Parsing/simplification failed"
            results.append({
                "original": expr,
                "simplified": None,
                "success": False,
                "error": error_msg[:100],
                "processing_time": time.time() - expr_start
            })
    
    elapsed = time.time() - start
    return elapsed, results


def run_gamba_plusplus_sequential(expressions: List[str], bitcount: int = 32) -> Tuple[float, List[Dict]]:
    """
    Run GAMBA++ sequential (baseline comparison).
    
    Args:
        expressions: List of expressions
        bitcount: Bit width
    
    Returns:
        Tuple of (time_elapsed, results)
    """
    import io
    import contextlib
    
    results = []
    start = time.time()
    
    for expr in expressions:
        expr_start = time.time()
        try:
            # Suppress stdout/stderr to avoid printing errors
            f = io.StringIO()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                simplifier = GeneralSimplifier(bitcount, modRed=False, verifBitCount=None)
                simplified = simplifier.simplify(expr, useZ3=False)
            
            elapsed = time.time() - expr_start
            
            if simplified and simplified.strip():
                results.append({
                    "original": expr,
                    "simplified": simplified.strip(),
                    "success": True,
                    "error": None,
                    "processing_time": elapsed
                })
            else:
                results.append({
                    "original": expr,
                    "simplified": None,
                    "success": False,
                    "error": "GAMBA returned empty result",
                    "processing_time": elapsed
                })
        except Exception as e:
            error_msg = str(e)
            if "Could not parse" in error_msg or "No simplification" in error_msg:
                error_msg = "Parsing/simplification failed"
            results.append({
                "original": expr,
                "simplified": None,
                "success": False,
                "error": error_msg[:100],
                "processing_time": time.time() - expr_start
            })
    
    elapsed = time.time() - start
    return elapsed, results


def run_gamba_plusplus_parallel(expressions: List[str], bitcount: int = 32, max_workers: int = 8) -> Tuple[float, List[Dict]]:
    """
    Run GAMBA++ with ProcessPoolExecutor (8 cores).
    
    Args:
        expressions: List of expressions
        bitcount: Bit width
        max_workers: Number of workers
    
    Returns:
        Tuple of (time_elapsed, results)
    """
    start = time.time()
    results = process_expressions_parallel_advanced(
        expressions=expressions,
        bitcount=bitcount,
        max_workers=max_workers,
        use_cache=False,  # Disable cache for fair comparison
        show_progress=True
    )
    elapsed = time.time() - start
    return elapsed, results


def run_gamba_plusplus_optimized(expressions: List[str], bitcount: int = 32, max_workers: int = 8) -> Tuple[float, List[Dict]]:
    """
    Run GAMBA++ with all optimizations (cache + parallel + batch).
    
    Args:
        expressions: List of expressions
        bitcount: Bit width
        max_workers: Number of workers
    
    Returns:
        Tuple of (time_elapsed, results)
    """
    cache = GAMBACache(Path(".compare_cache.json"))
    cache.clear()
    
    start = time.time()
    results = process_expressions_batch_advanced(
        expressions=expressions,
        bitcount=bitcount,
        max_workers=max_workers,
        batch_size=10,
        use_cache=True,
        cache=cache,
        show_progress=True
    )
    elapsed = time.time() - start
    
    stats = cache.get_stats()
    return elapsed, results, stats


def compare_results(original_results: List[Dict], gpp_results: List[Dict]) -> Dict:
    """
    Compare results between original GAMBA and GAMBA++.
    
    Args:
        original_results: Results from original GAMBA
        gpp_results: Results from GAMBA++
    
    Returns:
        Comparison statistics
    """
    if len(original_results) != len(gpp_results):
        return {"error": "Result count mismatch"}
    
    matches = 0
    both_success = 0
    only_original = 0
    only_gpp = 0
    neither = 0
    
    for orig, gpp in zip(original_results, gpp_results):
        orig_success = orig.get("success", False)
        gpp_success = gpp.get("success", False)
        
        if orig_success and gpp_success:
            both_success += 1
            # Compare simplified results (normalize whitespace)
            orig_simpl = orig.get("simplified", "").strip()
            gpp_simpl = gpp.get("simplified", "").strip()
            
            if orig_simpl == gpp_simpl:
                matches += 1
        elif orig_success:
            only_original += 1
        elif gpp_success:
            only_gpp += 1
        else:
            neither += 1
    
    return {
        "total": len(original_results),
        "matches": matches,
        "both_success": both_success,
        "only_original": only_original,
        "only_gpp": only_gpp,
        "neither": neither,
        "match_rate": matches / both_success * 100 if both_success > 0 else 0
    }


def run_comprehensive_comparison():
    """Run comprehensive comparison between GAMBA++ and original GAMBA"""
    print("=" * 80)
    print("COMPREHENSIVE GAMBA++ vs ORIGINAL GAMBA COMPARISON")
    print("=" * 80)
    print(f"\nTest expressions: {len(COMPLEX_EXPRESSIONS)}")
    print(f"Unique expressions: {len(COMPLEX_EXPRESSIONS) // 3}")
    print(f"Complexity: Simple to Very High")
    print(f"Variables: 1-3 variables per expression\n")
    
    results = {}
    
    # 1. Original GAMBA (sequential) - use subprocess to avoid stdout pollution
    print("\n" + "=" * 80)
    print("[1/5] Original GAMBA (Sequential)")
    print("=" * 80)
    print("Processing expressions...")
    try:
        time_orig, results_orig = run_original_gamba_subprocess(COMPLEX_EXPRESSIONS)
        results["original"] = {
            "time": time_orig,
            "results": results_orig,
            "successful": sum(1 for r in results_orig if r["success"]),
            "method": "Original GAMBA (Sequential)"
        }
        print(f"✓ Completed in {time_orig:.3f}s")
        print(f"  Successful: {results['original']['successful']}/{len(COMPLEX_EXPRESSIONS)}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        results["original"] = {"time": 0, "results": [], "successful": 0}
    
    # 2. GAMBA++ Sequential (baseline)
    print("\n" + "=" * 80)
    print("[2/5] GAMBA++ Sequential (Baseline)")
    print("=" * 80)
    print("Processing expressions (errors will be suppressed)...")
    try:
        import os
        import sys
        # Redirect stderr to /dev/null to suppress error messages
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        time_gpp_seq, results_gpp_seq = run_gamba_plusplus_sequential(COMPLEX_EXPRESSIONS)
        
        # Restore stderr
        sys.stderr.close()
        sys.stderr = old_stderr
        
        results["gpp_sequential"] = {
            "time": time_gpp_seq,
            "results": results_gpp_seq,
            "successful": sum(1 for r in results_gpp_seq if r["success"]),
            "method": "GAMBA++ Sequential"
        }
        print(f"✓ Completed in {time_gpp_seq:.3f}s")
        print(f"  Successful: {results['gpp_sequential']['successful']}/{len(COMPLEX_EXPRESSIONS)}")
    except Exception as e:
        # Restore stderr in case of error
        if 'old_stderr' in locals():
            sys.stderr = old_stderr
        print(f"✗ Error: {e}")
        results["gpp_sequential"] = {"time": 0, "results": [], "successful": 0}
    
    # 3. GAMBA++ Parallel (8 cores, no cache)
    print("\n" + "=" * 80)
    print("[3/5] GAMBA++ Parallel (8 cores, no cache)")
    print("=" * 80)
    try:
        time_gpp_par, results_gpp_par = run_gamba_plusplus_parallel(COMPLEX_EXPRESSIONS, max_workers=8)
        results["gpp_parallel"] = {
            "time": time_gpp_par,
            "results": results_gpp_par,
            "successful": sum(1 for r in results_gpp_par if r["success"]),
            "method": "GAMBA++ Parallel (8 cores)"
        }
        print(f"✓ Completed in {time_gpp_par:.3f}s")
        print(f"  Successful: {results['gpp_parallel']['successful']}/{len(COMPLEX_EXPRESSIONS)}")
    except Exception as e:
        print(f"✗ Error: {e}")
        results["gpp_parallel"] = {"time": 0, "results": [], "successful": 0}
    
    # 4. GAMBA++ Optimized (cache + parallel + batch)
    print("\n" + "=" * 80)
    print("[4/5] GAMBA++ Fully Optimized (8 cores + cache + batch)")
    print("=" * 80)
    try:
        time_gpp_opt, results_gpp_opt, cache_stats = run_gamba_plusplus_optimized(COMPLEX_EXPRESSIONS, max_workers=8)
        results["gpp_optimized"] = {
            "time": time_gpp_opt,
            "results": results_gpp_opt,
            "successful": sum(1 for r in results_gpp_opt if r["success"]),
            "method": "GAMBA++ Fully Optimized",
            "cache_stats": cache_stats
        }
        print(f"✓ Completed in {time_gpp_opt:.3f}s")
        print(f"  Successful: {results['gpp_optimized']['successful']}/{len(COMPLEX_EXPRESSIONS)}")
        if cache_stats:
            print(f"  Cache hits: {cache_stats.get('hits', 0)}")
            print(f"  Cache hit rate: {cache_stats.get('hit_rate_percent', 0):.1f}%")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        results["gpp_optimized"] = {"time": 0, "results": [], "successful": 0}
    
    # 5. Results comparison
    print("\n" + "=" * 80)
    print("[5/5] Results Comparison")
    print("=" * 80)
    
    if results["original"]["results"] and results["gpp_optimized"]["results"]:
        comparison = compare_results(
            results["original"]["results"],
            results["gpp_optimized"]["results"]
        )
        print(f"\nResult Matching:")
        print(f"  Total expressions: {comparison['total']}")
        print(f"  Both successful: {comparison['both_success']}")
        print(f"  Results match: {comparison['matches']}/{comparison['both_success']} ({comparison['match_rate']:.1f}%)")
        print(f"  Only original: {comparison['only_original']}")
        print(f"  Only GAMBA++: {comparison['only_gpp']}")
        print(f"  Neither: {comparison['neither']}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    baseline_time = results["original"]["time"]
    
    print(f"\n{'Method':<50} {'Time (s)':<15} {'Speedup':<15} {'Success':<10}")
    print("-" * 90)
    
    for key in ["original", "gpp_sequential", "gpp_parallel", "gpp_optimized"]:
        if key in results and results[key]["time"] > 0:
            method = results[key]["method"]
            time_val = results[key]["time"]
            speedup = baseline_time / time_val if time_val > 0 else 0
            success = results[key]["successful"]
            
            speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
            
            print(f"{method:<50} {time_val:<15.3f} {speedup_str:<15} {success}/{len(COMPLEX_EXPRESSIONS)}")
            
            if key == "gpp_optimized" and "cache_stats" in results[key]:
                stats = results[key]["cache_stats"]
                if stats:
                    print(f"  └─ Cache: {stats.get('hits', 0)} hits ({stats.get('hit_rate_percent', 0):.1f}% hit rate)")
    
    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    
    if results["original"]["time"] > 0 and results["gpp_optimized"]["time"] > 0:
        speedup = results["original"]["time"] / results["gpp_optimized"]["time"]
        print(f"\n✓ GAMBA++ Optimized is {speedup:.2f}x faster than Original GAMBA")
    
    if results["gpp_sequential"]["time"] > 0 and results["gpp_parallel"]["time"] > 0:
        speedup = results["gpp_sequential"]["time"] / results["gpp_parallel"]["time"]
        print(f"✓ Parallel processing is {speedup:.2f}x faster than sequential")
    
    if results["gpp_parallel"]["time"] > 0 and results["gpp_optimized"]["time"] > 0:
        speedup = results["gpp_parallel"]["time"] / results["gpp_optimized"]["time"]
        print(f"✓ Optimized (cache+batch) is {speedup:.2f}x faster than parallel only")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    run_comprehensive_comparison()


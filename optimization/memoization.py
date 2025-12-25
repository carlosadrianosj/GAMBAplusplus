#!/usr/bin/env python3
"""
Memoization for GAMBA++

Caches parsed ASTs to avoid re-parsing identical expressions.
"""

import sys
from pathlib import Path
from functools import lru_cache
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from gamba.utils.parse import parse
    HAS_GAMBA = True
except ImportError:
    HAS_GAMBA = False


# Global cache for parsed ASTs
_parse_cache_size = 1000


def set_parse_cache_size(size: int):
    """Set the size of the parse cache"""
    global _parse_cache_size
    _parse_cache_size = size
    # Recreate cached function with new size
    global cached_parse
    cached_parse = lru_cache(maxsize=size)(_parse_impl)


def _parse_impl(expression: str, bitcount: int, modred: bool, refine: bool, check: bool):
    """Internal parse implementation"""
    if not HAS_GAMBA:
        return None
    
    try:
        return parse(expression, bitcount, modred, refine, check)
    except Exception:
        return None


# Create cached parse function
cached_parse = lru_cache(maxsize=_parse_cache_size)(_parse_impl)


def clear_parse_cache():
    """Clear the parse cache"""
    cached_parse.cache_clear()


def get_parse_cache_info() -> dict:
    """Get parse cache statistics"""
    cache_info = cached_parse.cache_info()
    return {
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "maxsize": cache_info.maxsize,
        "currsize": cache_info.currsize,
        "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses) * 100
        if (cache_info.hits + cache_info.misses) > 0 else 0
    }


if __name__ == "__main__":
    # Test memoization
    if HAS_GAMBA:
        test_expr = "(x ^ y) + 2*(x & y)"
        
        # First parse (cache miss)
        import time
        start = time.time()
        result1 = cached_parse(test_expr, 32, False, True, True)
        time1 = time.time() - start
        
        # Second parse (cache hit)
        start = time.time()
        result2 = cached_parse(test_expr, 32, False, True, True)
        time2 = time.time() - start
        
        print(f"First parse:  {time1:.6f}s")
        print(f"Second parse: {time2:.6f}s (cached)")
        print(f"Speedup: {time1 / time2:.2f}x" if time2 > 0 else "N/A")
        
        info = get_parse_cache_info()
        print(f"Cache info: {info}")
    else:
        print("GAMBA not available for testing")


"""
GAMBA++ Performance Optimization Module

Provides caching, parallel processing, and batch operations for improved performance.
"""

from .cache import GAMBACache
from .parallel import (
    process_with_gamba_direct,
    process_expressions_parallel,
    process_mba_blocks_parallel
)
from .batch import (
    process_function_optimized,
    prioritize_expressions,
    batch_process_expressions
)
from .parallel_advanced import (
    process_expressions_parallel_advanced,
    process_with_gamba_optimized,
    quick_simplify_check,
    adaptive_timeout
)
from .simplifier_pool import SimplifierPool, process_with_pooled_simplifier
from .batch_advanced import process_expressions_batch_advanced
from .memoization import (
    cached_parse,
    clear_parse_cache,
    get_parse_cache_info,
    set_parse_cache_size
)
from .vectorization import (
    generate_truth_table_vectorized,
    compute_result_vector_vectorized
)
from .jit_optimizations import (
    is_jit_available,
    compute_result_vector_jit,
    truth_table_generation_jit
)

__all__ = [
    # Basic optimizations
    'GAMBACache',
    'process_with_gamba_direct',
    'process_expressions_parallel',
    'process_mba_blocks_parallel',
    'process_function_optimized',
    'prioritize_expressions',
    'batch_process_expressions',
    # Advanced optimizations
    'process_expressions_parallel_advanced',
    'process_with_gamba_optimized',
    'quick_simplify_check',
    'adaptive_timeout',
    'SimplifierPool',
    'process_with_pooled_simplifier',
    'process_expressions_batch_advanced',
    'cached_parse',
    'clear_parse_cache',
    'get_parse_cache_info',
    'set_parse_cache_size',
    'generate_truth_table_vectorized',
    'compute_result_vector_vectorized',
    'is_jit_available',
]


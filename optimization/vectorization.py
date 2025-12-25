#!/usr/bin/env python3
"""
Vectorization Optimizations for GAMBA++

Uses NumPy vectorization for truth table generation and array operations.
"""

import numpy as np
from typing import List, Tuple


def generate_truth_table_vectorized(vnumber: int) -> np.ndarray:
    """
    Generate truth table using NumPy vectorization.
    
    Much faster than Python loops for large numbers of variables.
    
    Args:
        vnumber: Number of variables
    
    Returns:
        NumPy array of shape (2^vnumber, vnumber) with truth table
    """
    if vnumber == 0:
        return np.array([[]], dtype=np.uint64)
    
    size = 2 ** vnumber
    
    # Vectorized generation using broadcasting
    indices = np.arange(size, dtype=np.uint64)[:, np.newaxis]
    bits = np.arange(vnumber, dtype=np.uint64)[np.newaxis, :]
    
    # Extract bits using bitwise operations
    truth_table = (indices >> bits) & 1
    
    return truth_table.astype(np.uint64)


def compute_result_vector_vectorized(
    eval_func,
    vnumber: int,
    modulus: int
) -> np.ndarray:
    """
    Compute result vector using vectorized operations.
    
    Args:
        eval_func: Function that evaluates expression for given variable values
        vnumber: Number of variables
        modulus: Modulus for operations
    
    Returns:
        NumPy array with results for all truth value combinations
    """
    truth_table = generate_truth_table_vectorized(vnumber)
    size = 2 ** vnumber
    
    # Vectorized evaluation
    results = np.zeros(size, dtype=np.uint64)
    
    for i in range(size):
        results[i] = eval_func(truth_table[i]) % modulus
    
    return results


def vectorized_modulo_reduction(values: np.ndarray, modulus: int) -> np.ndarray:
    """
    Apply modulo reduction to array of values using NumPy.
    
    Args:
        values: NumPy array of values
        modulus: Modulus
    
    Returns:
        Array with modulo reduction applied
    """
    return np.mod(values, modulus, dtype=np.uint64)


def vectorized_linear_combination(
    coefficients: np.ndarray,
    basis_expressions: np.ndarray,
    modulus: int
) -> np.ndarray:
    """
    Compute linear combination using vectorized operations.
    
    Args:
        coefficients: Array of coefficients
        basis_expressions: Array of basis expression values
        modulus: Modulus
    
    Returns:
        Result of linear combination
    """
    # Vectorized dot product with modulo
    result = np.dot(coefficients, basis_expressions)
    return np.mod(result, modulus, dtype=np.uint64)


if __name__ == "__main__":
    # Test vectorization
    print("Testing vectorized truth table generation...")
    
    for vnum in [2, 3, 4]:
        import time
        
        # NumPy vectorized
        start = time.time()
        table = generate_truth_table_vectorized(vnum)
        time_vec = time.time() - start
        
        # Python loop (for comparison)
        start = time.time()
        table_py = []
        for i in range(2**vnum):
            row = []
            n = i
            for j in range(vnum):
                row.append(n & 1)
                n = n >> 1
            table_py.append(row)
        time_py = time.time() - start
        
        speedup = time_py / time_vec if time_vec > 0 else 0
        print(f"  {vnum} variables: {time_vec:.6f}s (vectorized) vs {time_py:.6f}s (Python) = {speedup:.2f}x faster")


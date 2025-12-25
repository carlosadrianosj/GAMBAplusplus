#!/usr/bin/env python3
"""
Simplifier Pool for GAMBA++

Reuses GeneralSimplifier instances to avoid creation overhead.
"""

import sys
import queue
import threading
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gamba.simplify_general import GeneralSimplifier


class SimplifierPool:
    """
    Pool of reusable GeneralSimplifier instances.
    
    Reduces overhead of creating new simplifier instances for each expression.
    """
    
    def __init__(self, bitcount: int = 32, pool_size: int = 8, mod_red: bool = False):
        """
        Initialize simplifier pool.
        
        Args:
            bitcount: Bit width for variables
            pool_size: Number of simplifiers in pool (default: 8, matches cores)
            mod_red: Enable modulo reduction
        """
        self.bitcount = bitcount
        self.pool_size = pool_size
        self.mod_red = mod_red
        self.pool = queue.Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self.created = 0
        
        # Pre-populate pool
        for _ in range(pool_size):
            simplifier = GeneralSimplifier(bitcount, modRed=mod_red, verifBitCount=None)
            self.pool.put(simplifier)
            self.created += 1
    
    def get(self, timeout: Optional[float] = None) -> GeneralSimplifier:
        """
        Get a simplifier from the pool.
        
        Args:
            timeout: Maximum time to wait for available simplifier (None = wait forever)
        
        Returns:
            GeneralSimplifier instance
        
        Raises:
            queue.Empty: If timeout expires
        """
        try:
            return self.pool.get(timeout=timeout)
        except queue.Empty:
            # If pool is empty and we haven't reached max, create a new one
            with self.lock:
                if self.created < self.pool_size * 2:  # Allow up to 2x pool size
                    simplifier = GeneralSimplifier(self.bitcount, modRed=self.mod_red, verifBitCount=None)
                    self.created += 1
                    return simplifier
            # Otherwise wait again
            return self.pool.get(timeout=timeout)
    
    def put(self, simplifier: GeneralSimplifier):
        """
        Return a simplifier to the pool.
        
        Args:
            simplifier: GeneralSimplifier instance to return
        """
        try:
            self.pool.put_nowait(simplifier)
        except queue.Full:
            # Pool is full, discard this simplifier
            pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Clear pool
        while not self.pool.empty():
            try:
                self.pool.get_nowait()
            except queue.Empty:
                break
    
    def size(self) -> int:
        """Get current pool size (available simplifiers)"""
        return self.pool.qsize()
    
    def total_created(self) -> int:
        """Get total number of simplifiers created"""
        return self.created


def process_with_pooled_simplifier(
    expression: str,
    pool: SimplifierPool,
    use_z3: bool = False
) -> str:
    """
    Process expression using a simplifier from the pool.
    
    Args:
        expression: GAMBA expression string
        pool: SimplifierPool instance
        use_z3: Enable Z3 verification
    
    Returns:
        Simplified expression or empty string
    """
    simplifier = pool.get()
    try:
        result = simplifier.simplify(expression, useZ3=use_z3)
        return result if result else ""
    finally:
        pool.put(simplifier)


if __name__ == "__main__":
    # Test pool
    print("Testing SimplifierPool...")
    
    with SimplifierPool(bitcount=32, pool_size=4) as pool:
        expressions = [
            "(x ^ y) + 2*(x & y)",
            "(x | y) - (x & y)",
            "~x + ~y + 1",
        ]
        
        for expr in expressions:
            result = process_with_pooled_simplifier(expr, pool)
            print(f"{expr} -> {result}")
        
        print(f"Pool size: {pool.size()}")
        print(f"Total created: {pool.total_created()}")


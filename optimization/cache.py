#!/usr/bin/env python3
"""
GAMBA Result Cache System

Provides persistent caching of GAMBA simplification results to avoid
reprocessing identical expressions.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from threading import Lock
from collections import defaultdict


class GAMBACache:
    """
    Cache system for GAMBA simplification results.
    
    Uses SHA256 hashing for cache keys and JSON for persistent storage.
    """
    
    def __init__(self, cache_file: Optional[Path] = None):
        """
        Initialize cache.
        
        Args:
            cache_file: Path to cache file (default: .gamba_cache.json)
        """
        if cache_file is None:
            cache_file = Path(".gamba_cache.json")
        
        self.cache_file = Path(cache_file)
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "created_at": datetime.now().isoformat()
        }
        self.cache = self._load_cache()
        self.lock = Lock()  # Global lock for cache operations
        self.key_locks = defaultdict(Lock)  # Per-key locks for granular locking
    
    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    # Load stats if available
                    if "stats" in data:
                        self.stats.update(data["stats"])
                    return data.get("cache", {})
            except (json.JSONDecodeError, IOError) as e:
                print(f"[!] Warning: Failed to load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            data = {
                "cache": self.cache,
                "stats": self.stats,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"[!] Warning: Failed to save cache: {e}")
    
    def _get_key(self, expression: str, bitcount: int) -> str:
        """
        Generate cache key from expression and bitcount.
        
        Args:
            expression: GAMBA expression string
            bitcount: Bit width for variables
        
        Returns:
            SHA256 hash as hex string
        """
        key_string = f"{expression}:{bitcount}"
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def get(self, expression: str, bitcount: int = 32) -> Optional[str]:
        """
        Get cached result for expression (thread-safe).
        
        Args:
            expression: GAMBA expression string
            bitcount: Bit width for variables
        
        Returns:
            Simplified expression if cached, None otherwise
        """
        key = self._get_key(expression, bitcount)
        
        # Use per-key lock for granular locking
        with self.key_locks[key]:
            result = self.cache.get(key)
            
            if result is not None:
                with self.lock:  # Lock for stats update
                    self.stats["hits"] += 1
                return result
            
            with self.lock:  # Lock for stats update
                self.stats["misses"] += 1
            return None
    
    def set(self, expression: str, bitcount: int, simplified: str):
        """
        Cache simplification result (thread-safe).
        
        Args:
            expression: Original GAMBA expression
            bitcount: Bit width for variables
            simplified: Simplified expression result
        """
        if not simplified:
            return
        
        key = self._get_key(expression, bitcount)
        
        # Use per-key lock for granular locking
        with self.key_locks[key]:
            self.cache[key] = simplified
            
            with self.lock:  # Lock for stats update
                self.stats["sets"] += 1
                sets_count = self.stats["sets"]
            
            # Auto-save every 100 sets (outside lock to avoid blocking)
            if sets_count % 100 == 0:
                self._save_cache()
    
    def clear(self):
        """Clear all cached results"""
        self.cache.clear()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "created_at": datetime.now().isoformat()
        }
        self._save_cache()
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            "cache_size": len(self.cache),
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2)
        }
    
    def __len__(self):
        """Return number of cached entries"""
        return len(self.cache)
    
    def __contains__(self, key):
        """Check if expression is cached (requires bitcount)"""
        # This is a simplified check - full check requires bitcount
        return False
    
    def save(self):
        """Manually save cache to disk"""
        self._save_cache()


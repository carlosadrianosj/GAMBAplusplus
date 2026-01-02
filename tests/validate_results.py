#!/usr/bin/env python3
"""
Result Validation
Validates test results and generates reports
"""

import json
from pathlib import Path
from typing import Dict, List


def validate_test_results(test_dir: Path) -> Dict:
    """
    Validate test results for an architecture.
    
    Args:
        test_dir: Test directory path
    
    Returns:
        Validation results dictionary
    """
    output_dir = test_dir / "output"
    
    validation = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required files
    required_files = [
        'cfg_before.png',
        'cfg_after.png',
        'cfg_comparison.png',
        'mba_blocks.json',
        'simplified_expressions.json',
        'test_report.json'
    ]
    
    for filename in required_files:
        filepath = output_dir / filename
        if not filepath.exists():
            validation['valid'] = False
            validation['errors'].append(f"Missing file: {filename}")
    
    # Validate JSON files
    if (output_dir / "test_report.json").exists():
        try:
            with open(output_dir / "test_report.json", 'r') as f:
                report = json.load(f)
            
            metrics = report.get('metrics', {})
            
            # Check metrics
            if metrics.get('mba_block_count', 0) == 0:
                validation['warnings'].append("No MBA blocks detected")
            
            if metrics.get('simplified_count', 0) == 0:
                validation['warnings'].append("No expressions simplified")
            
            if metrics.get('basic_block_count_before', 0) == 0:
                validation['errors'].append("CFG not generated")
                validation['valid'] = False
            
        except json.JSONDecodeError as e:
            validation['valid'] = False
            validation['errors'].append(f"Invalid JSON: {e}")
    
    return validation


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_results.py <test_directory>")
        sys.exit(1)
    
    test_dir = Path(sys.argv[1])
    validation = validate_test_results(test_dir)
    
    if validation['valid']:
        print("✓ Validation passed")
    else:
        print("✗ Validation failed")
        for error in validation['errors']:
            print(f"  Error: {error}")
    
    if validation['warnings']:
        for warning in validation['warnings']:
            print(f"  Warning: {warning}")


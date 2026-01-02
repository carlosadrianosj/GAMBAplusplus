#!/usr/bin/env python3
"""
Run all architecture tests
Main test runner that executes tests for all architectures
"""

import os
import sys
import json
from pathlib import Path
from test_runner import ArchitectureTest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_architecture_tests(arch_filter: Optional[str] = None):
    """
    Run tests for all architectures or a specific one.
    
    Args:
        arch_filter: Optional architecture name to test only one
    """
    architectures = ['x86_64', 'arm32', 'arm64', 'mips32', 'mips64',
                     'powerpc32', 'powerpc64', 'riscv32', 'riscv64']
    
    if arch_filter and arch_filter != 'all':
        architectures = [arch_filter]
    
    results = {}
    tests_dir = Path(__file__).parent
    
    print("=" * 80)
    print("GAMBA++ Architecture Test Suite")
    print("=" * 80)
    print()
    
    for arch in architectures:
        print(f"\n[{arch}] Testing {arch}...")
        print("-" * 80)
        
        test_dir = tests_dir / arch
        if not test_dir.exists():
            print(f"  [!] Test directory not found: {test_dir}")
            results[arch] = {'status': 'directory_not_found'}
            continue
        
        test = ArchitectureTest(arch, test_dir)
        
        # Build phase
        if not test.build():
            results[arch] = {'status': 'build_failed'}
            continue
        
        # Analysis phase
        try:
            analysis = test.analyze_with_gamba()
            
            # Validation phase
            if test.validate_results(analysis):
                results[arch] = {
                    'status': 'passed',
                    'analysis': analysis
                }
                print(f"  ✓ {arch} test PASSED")
            else:
                results[arch] = {
                    'status': 'validation_failed',
                    'analysis': analysis
                }
                print(f"  ✗ {arch} test FAILED (validation)")
        except Exception as e:
            results[arch] = {
                'status': 'analysis_failed',
                'error': str(e)
            }
            print(f"  ✗ {arch} test FAILED (analysis error)")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r.get('status') == 'passed')
    failed = len(results) - passed
    
    print(f"\nTotal: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    print("\nResults by architecture:")
    for arch, result in results.items():
        status = result.get('status', 'unknown')
        status_symbol = '✓' if status == 'passed' else '✗'
        print(f"  {status_symbol} {arch}: {status}")
    
    # Save summary
    summary_path = tests_dir / "test_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nSummary saved to: {summary_path}")
    
    return results


if __name__ == "__main__":
    arch_filter = os.environ.get('TEST_ARCH', 'all')
    results = run_architecture_tests(arch_filter)
    
    # Exit with error code if any tests failed
    failed_count = sum(1 for r in results.values() if r.get('status') != 'passed')
    sys.exit(0 if failed_count == 0 else 1)


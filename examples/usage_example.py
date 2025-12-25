#!/usr/bin/env python3
"""
Example usage of GAMBA++ for assembly-to-expression conversion
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from assembly.x86_64 import parse_assembly, detect_mba_blocks, convert_mba_block_to_expression
from assembly.arm import parse_assembly as parse_arm_assembly, detect_mba_blocks as detect_arm_mba_blocks
from gamba.simplify_general import GeneralSimplifier


def example_x86_64():
    """Example: Process x86-64 assembly"""
    print("=" * 70)
    print("Example: x86-64 Assembly Processing")
    print("=" * 70)
    
    # Example assembly file path (replace with actual file)
    asm_file = Path("example_x86.asm")
    
    if not asm_file.exists():
        print(f"Note: {asm_file} not found. This is a template example.")
        print("\nTo use this example:")
        print("1. Create an assembly file with x86-64 instructions")
        print("2. Update the asm_file path above")
        print("3. Run this script")
        return
    
    # Parse assembly
    result = parse_assembly(asm_file)
    instructions = result["instructions"]
    print(f"Parsed {len(instructions)} instructions")
    
    # Detect MBA blocks
    mba_blocks = detect_mba_blocks(instructions, min_boolean_chain=5)
    print(f"Found {len(mba_blocks)} MBA blocks")
    
    # Convert to expressions and simplify
    simplifier = GeneralSimplifier(64, False, None)
    
    for i, block in enumerate(mba_blocks, 1):
        print(f"\nMBA Block {i}:")
        print(f"  Address: 0x{block.start_address:X} - 0x{block.end_address:X}")
        print(f"  Pattern: {block.pattern_type}")
        
        expr_data = convert_mba_block_to_expression(block)
        if expr_data:
            print(f"  Expression: {expr_data['gamba_expression']}")
            
            # Simplify with GAMBA
            simplified = simplifier.simplify(expr_data['gamba_expression'])
            print(f"  Simplified: {simplified}")


def example_arm():
    """Example: Process ARM assembly"""
    print("\n" + "=" * 70)
    print("Example: ARM Assembly Processing")
    print("=" * 70)
    
    # Example assembly file path (replace with actual file)
    asm_file = Path("example_arm.asm")
    
    if not asm_file.exists():
        print(f"Note: {asm_file} not found. This is a template example.")
        print("\nTo use this example:")
        print("1. Create an assembly file with ARM instructions")
        print("2. Update the asm_file path above")
        print("3. Run this script")
        return
    
    # Parse ARM64 assembly
    result = parse_arm_assembly(asm_file, arch="arm64")
    instructions = result["instructions"]
    print(f"Parsed {len(instructions)} instructions")
    
    # Detect MBA blocks
    mba_blocks = detect_arm_mba_blocks(instructions, min_boolean_chain=5)
    print(f"Found {len(mba_blocks)} MBA blocks")
    
    # Convert to expressions
    from assembly.arm.converter import convert_mba_block_to_expression
    
    for i, block in enumerate(mba_blocks, 1):
        print(f"\nMBA Block {i}:")
        print(f"  Address: 0x{block.start_address:X} - 0x{block.end_address:X}")
        print(f"  Pattern: {block.pattern_type}")
        
        expr_data = convert_mba_block_to_expression(block)
        if expr_data:
            print(f"  Expression: {expr_data['gamba_expression']}")


def example_gamba_simplification():
    """Example: Direct GAMBA simplification"""
    print("\n" + "=" * 70)
    print("Example: Direct GAMBA Simplification")
    print("=" * 70)
    
    expressions = [
        "(x ^ y) + 2*(x & y)",  # Should simplify to x + y
        "(x | y) - (x & y)",     # Should simplify to x ^ y
        "~x + ~y + 1",           # Should simplify to ~(x | y)
    ]
    
    simplifier = GeneralSimplifier(64, False, None)
    
    for expr in expressions:
        simplified = simplifier.simplify(expr)
        print(f"\nOriginal:   {expr}")
        print(f"Simplified: {simplified}")


if __name__ == "__main__":
    print("GAMBA++ Usage Examples")
    print("=" * 70)
    
    # Run examples
    example_gamba_simplification()
    example_x86_64()
    example_arm()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


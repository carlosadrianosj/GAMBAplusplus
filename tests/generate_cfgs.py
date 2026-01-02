#!/usr/bin/env python3
"""
Generate CFGs for all test binaries
Creates CFG visualizations before and after MBA simplification
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from assembly.x86_64.parser import parse_assembly as parse_x86
from assembly.arm.parser import parse_assembly as parse_arm
from assembly.x86_64.detector import detect_mba_blocks as detect_mba_x86
from assembly.arm.detector import detect_mba_blocks as detect_mba_arm
from assembly.x86_64.converter import convert_mba_block_to_expression as convert_x86
from assembly.arm.converter import convert_mba_block_to_expression as convert_arm
from assembly.common.cfg import build_cfg
from assembly.common.mba_function_detector import extract_function_boundaries_from_symbols
from visualization.cfg_renderer import render_cfg, render_cfg_comparison
from optimization.batch_advanced import process_expressions_batch_advanced
from assembly.common.normalization import normalize_expression_list


def generate_cfgs_for_architecture(arch: str, test_dir: Path):
    """Generate CFGs for a specific architecture"""
    print(f"\n{'='*80}")
    print(f"Generating CFGs for {arch}")
    print(f"{'='*80}")
    
    asm_path = test_dir / "test_mba.asm"
    symbols_path = test_dir / "test_mba.symbols"
    output_dir = test_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    if not asm_path.exists():
        print(f"  [!] Assembly file not found: {asm_path}")
        print(f"  [!] Run build.sh first to compile the test binary")
        return False
    
    # Variables to track state for JSON saving even on errors
    instructions = []
    mba_blocks = []
    expressions = []
    expressions_data = []
    simplified_blocks = []
    cfg_before = None
    cfg_after = None
    function_boundaries = []
    arch_type = arch  # Default to parameter
    
    try:
        # Save JSON even on error - initialize early
        def save_json_on_error():
            """Save JSON with current state even if there's an error"""
            try:
                results_data = {
                    'architecture': arch_type if 'arch_type' in locals() else arch,
                    'summary': {
                        'instruction_count': len(instructions) if 'instructions' in locals() else 0,
                        'mba_block_count': len(mba_blocks) if 'mba_blocks' in locals() else 0,
                        'expression_count': len(expressions) if 'expressions' in locals() else 0,
                        'simplified_count': sum(1 for e in expressions_data if e.get('success', False)) if 'expressions_data' in locals() else 0,
                        'basic_block_count': len(cfg_before.blocks) if 'cfg_before' in locals() and cfg_before else 0,
                        'basic_block_count_after': len(cfg_after.blocks) if 'cfg_after' in locals() and cfg_after else 0,
                        'function_count': len(function_boundaries) if 'function_boundaries' in locals() else 0
                    },
                    'mba_blocks': [
                        {
                            'start': getattr(b, 'start_address', 0),
                            'end': getattr(b, 'end_address', 0),
                            'pattern': getattr(b, 'pattern_type', 'unknown'),
                            'instruction_count': getattr(b, 'instruction_count', lambda: 0)()
                        }
                        for b in (mba_blocks if 'mba_blocks' in locals() else [])
                    ],
                    'expressions': (expressions_data if 'expressions_data' in locals() else []),
                    'simplified_blocks': (simplified_blocks if 'simplified_blocks' in locals() else []),
                    'functions': (function_boundaries if 'function_boundaries' in locals() else [])
                }
                with open(output_dir / "cfg_analysis.json", 'w') as f:
                    json.dump(results_data, f, indent=2)
            except:
                pass
        
        # Detect architecture from assembly file
        with open(asm_path, 'r') as f:
            first_lines = ''.join(f.readlines()[:20])  # Read more lines for better detection
        
        # Determine architecture - check for ARM64 first (more specific patterns)
        if 'aarch64' in first_lines or 'elf64-littleaarch64' in first_lines:
            arch_type = 'arm64'
            parse_assembly = parse_arm
            detect_mba = detect_mba_arm
            convert_mba = convert_arm
            print(f"      Detected architecture: ARM64")
        elif 'w0' in first_lines or 'w1' in first_lines or 'w2' in first_lines:
            # ARM64 uses w0-w30 registers
            arch_type = 'arm64'
            parse_assembly = parse_arm
            detect_mba = detect_mba_arm
            convert_mba = convert_arm
            print(f"      Detected architecture: ARM64 (from register names)")
        elif 'x86' in first_lines.lower() or 'eax' in first_lines or 'rax' in first_lines or 'ebx' in first_lines:
            arch_type = 'x86_64'
            parse_assembly = parse_x86
            detect_mba = detect_mba_x86
            convert_mba = convert_x86
            print(f"      Detected architecture: x86_64")
        else:
            # Default to x86_64
            arch_type = 'x86_64'
            parse_assembly = parse_x86
            detect_mba = detect_mba_x86
            convert_mba = convert_x86
            print(f"      Using default architecture: x86_64")
        
        # Parse assembly
        print(f"  [1/6] Parsing assembly...")
        if arch_type == 'arm64':
            parsed = parse_assembly(asm_path, arch='arm64')
        else:
            parsed = parse_assembly(asm_path)
        instructions = parsed.get("instructions", [])
        print(f"      ✓ Parsed {len(instructions)} instructions")
        
        # Extract function boundaries
        print(f"  [2/6] Extracting function boundaries...")
        function_boundaries = []
        if symbols_path.exists():
            function_boundaries = extract_function_boundaries_from_symbols(str(symbols_path))
            print(f"      ✓ Found {len(function_boundaries)} functions")
        else:
            print(f"      ⚠ Symbols file not found, analyzing as single function")
        
        # Detect MBA blocks with lower thresholds for better detection
        print(f"  [3/6] Detecting MBA blocks...")
        # Use lower thresholds to detect smaller MBA patterns
        mba_blocks = detect_mba(instructions, min_boolean_chain=3, min_arithmetic_boolean=5)
        print(f"      ✓ Detected {len(mba_blocks)} MBA blocks")
        
        if len(mba_blocks) == 0:
            print(f"      ⚠ No MBA blocks detected, generating CFG anyway")
        
        # Build CFG before simplification
        print(f"  [4/6] Building CFG before simplification...")
        cfg_before = build_cfg(instructions, arch=arch_type)
        print(f"      ✓ CFG has {len(cfg_before.blocks)} basic blocks")
        
        # Render CFG before with analysis
        print(f"      Rendering CFG before...")
        cfg_before_path = output_dir / "cfg_before.png"
        from visualization.cfg_renderer import render_cfg_with_analysis
        analysis_data = {
            'summary': {
                'instruction_count': len(instructions),
                'mba_block_count': len(mba_blocks),
                'basic_block_count': len(cfg_before.blocks),
                'function_count': len(function_boundaries)
            }
        }
        render_cfg_with_analysis(cfg_before, mba_blocks, analysis_data, cfg_before_path)
        print(f"      ✓ Saved to {cfg_before_path}")
        
        # Convert MBA blocks to expressions
        print(f"  [5/6] Converting MBA blocks to expressions...")
        # expressions_data and expressions already initialized above
        expressions_data = []
        expressions = []
        
        for i, mba_block in enumerate(mba_blocks):
            try:
                expr_data = convert_mba(mba_block)
                if expr_data:
                    expr = expr_data.get('gamba_expression', '')
                    if expr:
                        expressions.append(expr)
                        expressions_data.append({
                            'block_index': i,
                            'block': {
                                'start': getattr(mba_block, 'start_address', 0),
                                'end': getattr(mba_block, 'end_address', 0),
                                'pattern': getattr(mba_block, 'pattern_type', 'unknown'),
                                'instruction_count': getattr(mba_block, 'instruction_count', lambda: 0)()
                            },
                            'original': expr,
                            'simplified': None,
                            'success': False
                        })
            except Exception as e:
                print(f"      ⚠ Failed to convert block {i}: {e}")
        
        print(f"      ✓ Converted {len(expressions)} expressions")
        
        # Simplify expressions
        simplified_blocks = []
        simplification_error = None
        if expressions:
            print(f"      Simplifying expressions with GAMBA++...")
            try:
                # Normalize before simplification
                normalized = normalize_expression_list(expressions)
                
                # Simplify - wrap in try/except to catch any errors
                try:
                    # Simplify with verifBitCount=None to avoid slow exhaustive verification
                    # that causes timeouts. Simplification still works correctly.
                    results = process_expressions_batch_advanced(
                        expressions=normalized,
                        bitcount=32,
                        max_workers=4,
                        use_cache=True,
                        show_progress=False,
                        mod_red=False  # Disable modulo reduction for faster processing
                    )
                except (SystemExit, KeyboardInterrupt):
                    raise
                except Exception as e:
                    # If simplification fails completely, mark all as failed
                    print(f"      ⚠ Simplification failed: {e}")
                    simplification_error = str(e)
                    results = [{'success': False, 'error': str(e)}] * len(expressions)
                
                # Update expressions_data
                for i, result in enumerate(results):
                    if i < len(expressions_data):
                        if result.get('success'):
                            simplified = result.get('simplified', '')
                            expressions_data[i]['simplified'] = simplified
                            expressions_data[i]['success'] = True
                            
                            # Create simplified block representation
                            original_block = expressions_data[i]['block']
                            simplified_blocks.append({
                                'start': original_block['start'],
                                'end': original_block['end'],
                                'original_expr': expressions_data[i]['original'],
                                'simplified_expr': simplified,
                                'complexity_reduction': len(expressions_data[i]['original']) - len(simplified)
                            })
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            expressions_data[i]['error'] = error_msg
                            if not simplification_error:  # Only print individual errors if not a global failure
                                print(f"      ⚠ Expression {i} failed: {error_msg}")
                
                success_count = sum(1 for r in results if r.get('success'))
                print(f"      ✓ Simplified {success_count}/{len(expressions)} expressions")
            except (Exception, SystemExit) as e:
                simplification_error = str(e)
                print(f"      ⚠ Error during simplification: {e}")
                import traceback
                traceback.print_exc()
                # Continue anyway - we'll create CFG with attempted MBA blocks
                # Mark all expressions as attempted but failed
                for expr_data in expressions_data:
                    if not expr_data.get('success'):
                        expr_data['error'] = simplification_error
        else:
            print(f"      ⚠ No expressions to simplify")
        
        # Build CFG after simplification
        # Create simplified CFG that shows the reduction in complexity
        print(f"  [6/6] Building CFG after simplification...")
        
        from assembly.common.cfg import CFG, BasicBlock
        from assembly.common.capstone_wrapper import CapstoneInstruction
        
        # Create a mapping of MBA blocks by address (both simplified and attempted)
        simplified_by_address = {}
        mba_attempted_by_address = {}  # Track MBA blocks even if simplification failed
        
        for expr_data in expressions_data:
            block_info = expr_data.get('block', {})
            block_start = block_info.get('start', 0)
            
            if expr_data.get('success'):
                # Successfully simplified
                simplified_by_address[block_start] = {
                    'original_expr': expr_data.get('original', ''),
                    'simplified_expr': expr_data.get('simplified', ''),
                    'complexity_reduction': len(expr_data.get('original', '')) - len(expr_data.get('simplified', ''))
                }
            else:
                # Simplification attempted but failed - still mark for visual difference
                mba_attempted_by_address[block_start] = {
                    'original_expr': expr_data.get('original', ''),
                    'error': expr_data.get('error', 'Simplification failed'),
                    'instruction_count': block_info.get('instruction_count', 0)
                }
        
        # Build simplified blocks - replace MBA instructions with simplified representation
        simplified_blocks_list = []
        for block in cfg_before.blocks:
            # Check if this block contains a simplified MBA block
            block_has_simplified = False
            simplified_info = None
            mba_attempted_info = None
            
            # First check for successfully simplified blocks
            for mba_start, info in simplified_by_address.items():
                # Check if MBA block overlaps with this basic block
                if (block.start_address <= mba_start < block.end_address):
                    block_has_simplified = True
                    simplified_info = info
                    break
            
            # If not simplified, check if MBA was attempted
            if not block_has_simplified:
                for mba_start, info in mba_attempted_by_address.items():
                    if (block.start_address <= mba_start < block.end_address):
                        mba_attempted_info = info
                        break
            
            if block_has_simplified and simplified_info:
                # Create simplified block - replace MBA instructions with simplified expression
                # Create a synthetic instruction showing the simplified result
                class SimplifiedInstruction:
                    def __init__(self, address, simplified_expr):
                        self.address = address
                        self.mnemonic = "simplified"
                        self.op_str = simplified_expr
                        self.size = 4
                        self.is_branch = False
                        self.is_call = False
                        self.is_return = False
                        self.branch_target = None
                
                # Keep non-MBA instructions and add simplified instruction
                simplified_insts = []
                mba_start_addr = None
                for mba_start, info in simplified_by_address.items():
                    if block.start_address <= mba_start < block.end_address:
                        mba_start_addr = mba_start
                        break
                
                # Add instructions before MBA
                for inst in block.instructions:
                    if inst.address < mba_start_addr:
                        simplified_insts.append(inst)
                
                # Add simplified instruction
                if simplified_info['simplified_expr']:
                    # Truncate long expressions for display
                    simplified_expr = simplified_info['simplified_expr']
                    if len(simplified_expr) > 60:
                        simplified_expr = simplified_expr[:57] + "..."
                    simplified_insts.append(SimplifiedInstruction(
                        mba_start_addr or block.start_address,
                        simplified_expr
                    ))
                
                # Add instructions after MBA (if any)
                for inst in block.instructions:
                    if inst.address >= (mba_start_addr or 0) + 16:  # Approximate MBA block size
                        simplified_insts.append(inst)
                
                # If no instructions, keep original but mark as simplified
                if not simplified_insts:
                    simplified_insts = block.instructions
                
                new_block = BasicBlock(block.start_address, simplified_insts)
                new_block.predecessors = block.predecessors
                new_block.successors = block.successors
                new_block.simplified = True  # Mark as simplified
                new_block.simplified_info = simplified_info
                simplified_blocks_list.append(new_block)
            elif mba_attempted_info:
                # MBA block was attempted but failed - show compact representation
                class CompactMBAInstruction:
                    def __init__(self, address, original_expr, inst_count):
                        self.address = address
                        self.mnemonic = "mba_attempted"
                        # Show compact representation: original expression (truncated) + note
                        expr_preview = original_expr[:40] + "..." if len(original_expr) > 40 else original_expr
                        self.op_str = f"{expr_preview} (simplification failed)"
                        self.size = 4
                        self.is_branch = False
                        self.is_call = False
                        self.is_return = False
                        self.branch_target = None
                
                # Find MBA block start address
                mba_start_addr = None
                for mba_start in mba_attempted_by_address.keys():
                    if block.start_address <= mba_start < block.end_address:
                        mba_start_addr = mba_start
                        break
                
                # Create compact block - keep non-MBA instructions, replace MBA with compact representation
                compact_insts = []
                
                # Add instructions before MBA
                for inst in block.instructions:
                    if mba_start_addr and inst.address < mba_start_addr:
                        compact_insts.append(inst)
                
                # Add compact MBA representation (1 instruction instead of many)
                if mba_start_addr:
                    compact_insts.append(CompactMBAInstruction(
                        mba_start_addr,
                        mba_attempted_info['original_expr'],
                        mba_attempted_info['instruction_count']
                    ))
                
                # Add instructions after MBA
                mba_end_addr = mba_start_addr + (mba_attempted_info['instruction_count'] * 4) if mba_start_addr else 0
                for inst in block.instructions:
                    if mba_end_addr and inst.address >= mba_end_addr:
                        compact_insts.append(inst)
                
                # If no instructions, keep original but mark
                if not compact_insts:
                    compact_insts = block.instructions
                
                new_block = BasicBlock(block.start_address, compact_insts)
                new_block.predecessors = block.predecessors
                new_block.successors = block.successors
                new_block.simplified = False
                new_block.mba_attempted = True
                new_block.mba_info = mba_attempted_info
                simplified_blocks_list.append(new_block)
            else:
                # Block not simplified, keep as is
                new_block = BasicBlock(block.start_address, block.instructions)
                new_block.predecessors = block.predecessors
                new_block.successors = block.successors
                new_block.simplified = False
                simplified_blocks_list.append(new_block)
        
        if simplified_blocks_list:
            cfg_after = CFG(simplified_blocks_list[0])
            for block in simplified_blocks_list[1:]:
                cfg_after.add_block(block)
        else:
            cfg_after = cfg_before
        
        simplified_count = sum(1 for b in simplified_blocks_list if getattr(b, 'simplified', False))
        print(f"      ✓ CFG after has {len(cfg_after.blocks)} basic blocks ({simplified_count} simplified)")
        
        # Render CFG after (with simplified blocks marked)
        print(f"      Rendering CFG after...")
        cfg_after_path = output_dir / "cfg_after.png"
        # Pass empty list for mba_blocks since they're simplified
        analysis_data_after = {
            'summary': {
                'instruction_count': len(instructions),
                'mba_block_count': 0,  # All simplified
                'basic_block_count': len(cfg_after.blocks),
                'simplified_count': sum(1 for e in expressions_data if e.get('success'))
            }
        }
        render_cfg_with_analysis(cfg_after, [], analysis_data_after, cfg_after_path)
        print(f"      ✓ Saved to {cfg_after_path}")
        
        # Render comparison with proper alignment
        print(f"      Rendering comparison...")
        cfg_comparison_path = output_dir / "cfg_comparison.png"
        render_cfg_comparison(cfg_before, cfg_after, mba_blocks, [], cfg_comparison_path,
                            title=f"CFG Comparison: {arch_type.upper()} | Before vs After MBA Simplification")
        print(f"      ✓ Saved to {cfg_comparison_path}")
        
        # Save results with correct data - ensure this happens even if there were errors
        print(f"  [7/7] Saving analysis data...")
        results_data = {
            'architecture': arch_type,  # Use detected architecture, not parameter
            'summary': {
                'instruction_count': len(instructions),
                'mba_block_count': len(mba_blocks),
                'expression_count': len(expressions),
                'simplified_count': sum(1 for e in expressions_data if e.get('success', False)),
                'basic_block_count': len(cfg_before.blocks) if cfg_before else 0,
                'basic_block_count_after': len(cfg_after.blocks) if cfg_after else 0,
                'function_count': len(function_boundaries)
            },
            'mba_blocks': [
                {
                    'start': getattr(b, 'start_address', 0),
                    'end': getattr(b, 'end_address', 0),
                    'pattern': getattr(b, 'pattern_type', 'unknown'),
                    'instruction_count': getattr(b, 'instruction_count', lambda: 0)()
                }
                for b in mba_blocks
            ],
            'expressions': [
                {
                    'block_index': e.get('block_index', -1),
                    'block': e.get('block', {}),
                    'original': e.get('original', ''),
                    'simplified': e.get('simplified', ''),
                    'success': e.get('success', False),
                    'error': e.get('error', '')
                }
                for e in expressions_data
            ],
            'simplified_blocks': simplified_blocks,
            'functions': function_boundaries
        }
        
        # Add simplification error if any
        if 'simplification_error' in locals() and simplification_error:
            results_data['summary']['simplification_error'] = simplification_error
        
        # Save JSON
        with open(output_dir / "cfg_analysis.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"      ✓ Saved analysis to {output_dir / 'cfg_analysis.json'}")
        
        print(f"\n  ✅ CFG generation complete for {arch}")
        print(f"     - CFG before: {cfg_before_path}")
        print(f"     - CFG after: {cfg_after_path}")
        print(f"     - Comparison: {cfg_comparison_path}")
        print(f"     - Analysis: {output_dir / 'cfg_analysis.json'}")
        
        return True
        
    except (Exception, SystemExit, KeyboardInterrupt) as e:
        print(f"  [!] Error generating CFGs: {e}")
        import traceback
        traceback.print_exc()
        
        # Save partial results even on error
        save_json_on_error()
        print(f"  ✓ Saved partial results to {output_dir / 'cfg_analysis.json'}")
        
        if isinstance(e, (SystemExit, KeyboardInterrupt)):
            raise
        return False


def main():
    """Generate CFGs for all architectures"""
    tests_dir = Path(__file__).parent
    architectures = ['x86_64', 'arm32', 'arm64', 'mips32', 'mips64',
                     'powerpc32', 'powerpc64', 'riscv32', 'riscv64']
    
    print("="*80)
    print("GAMBA++ CFG Generation for Test Binaries")
    print("="*80)
    print("\nThis script will:")
    print("  1. Parse assembly from compiled test binaries")
    print("  2. Detect MBA blocks")
    print("  3. Generate CFG before simplification")
    print("  4. Simplify MBA expressions")
    print("  5. Generate CFG after simplification")
    print("  6. Create comparison visualizations")
    print()
    
    results = {}
    
    for arch in architectures:
        test_dir = tests_dir / arch
        if not test_dir.exists():
            print(f"\n[SKIP] {arch}: Test directory not found")
            results[arch] = 'directory_not_found'
            continue
        
        success = generate_cfgs_for_architecture(arch, test_dir)
        results[arch] = 'success' if success else 'failed'
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results.values() if r == 'success')
    print(f"\nSuccessfully generated CFGs for {success_count}/{len(results)} architectures")
    
    print("\nResults:")
    for arch, result in results.items():
        status = "✅" if result == 'success' else "❌"
        print(f"  {status} {arch}: {result}")
    
    print(f"\nCFG files saved in: tests/*/output/")
    print("  - cfg_before.png: CFG with MBA blocks highlighted")
    print("  - cfg_after.png: CFG after simplification")
    print("  - cfg_comparison.png: Side-by-side comparison")
    print("  - cfg_analysis.json: Detailed analysis data")


if __name__ == "__main__":
    main()


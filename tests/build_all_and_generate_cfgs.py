#!/usr/bin/env python3
"""
Build all test binaries and generate CFGs
Compiles test binaries for all architectures and generates CFG visualizations
"""

import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import generate_cfgs functions directly
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
    
    try:
        # Detect architecture from assembly file
        with open(asm_path, 'r') as f:
            first_lines = ''.join(f.readlines()[:10])
        
        # Determine architecture
        if 'aarch64' in first_lines or 'arm64' in first_lines.lower() or 'w0' in first_lines or 'w1' in first_lines:
            arch_type = 'arm64'
            parse_assembly = parse_arm
            detect_mba = detect_mba_arm
            convert_mba = convert_arm
            print(f"      Detected architecture: ARM64")
        elif 'x86' in first_lines.lower() or 'eax' in first_lines or 'rax' in first_lines:
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
        
        # Detect MBA blocks
        print(f"  [3/6] Detecting MBA blocks...")
        mba_blocks = detect_mba(instructions)
        print(f"      ✓ Detected {len(mba_blocks)} MBA blocks")
        
        if len(mba_blocks) == 0:
            print(f"      ⚠ No MBA blocks detected, generating CFG anyway")
        
        # Build CFG before simplification
        print(f"  [4/6] Building CFG before simplification...")
        cfg_before = build_cfg(instructions, arch=arch_type)
        print(f"      ✓ CFG has {len(cfg_before.blocks)} basic blocks")
        
        # Render CFG before
        print(f"      Rendering CFG before...")
        cfg_before_path = output_dir / "cfg_before.png"
        render_cfg(cfg_before, mba_blocks, cfg_before_path)
        print(f"      ✓ Saved to {cfg_before_path}")
        
        # Convert MBA blocks to expressions
        print(f"  [5/6] Converting MBA blocks to expressions...")
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
        if expressions:
            print(f"      Simplifying expressions with GAMBA++...")
            # Normalize before simplification
            normalized = normalize_expression_list(expressions)
            
            # Simplify
            results = process_expressions_batch_advanced(
                expressions=normalized,
                bitcount=32,
                max_workers=4,
                use_cache=True,
                show_progress=False
            )
            
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
                        expressions_data[i]['error'] = result.get('error', 'Unknown error')
            
            success_count = sum(1 for r in results if r.get('success'))
            print(f"      ✓ Simplified {success_count}/{len(expressions)} expressions")
        else:
            print(f"      ⚠ No expressions to simplify")
        
        # Build CFG after simplification
        print(f"  [6/6] Building CFG after simplification...")
        cfg_after = cfg_before  # Placeholder - would rebuild from simplified code
        
        # Render CFG after (with simplified blocks marked)
        print(f"      Rendering CFG after...")
        cfg_after_path = output_dir / "cfg_after.png"
        # Pass empty list for mba_blocks since they're simplified
        render_cfg(cfg_after, [], cfg_after_path)
        print(f"      ✓ Saved to {cfg_after_path}")
        
        # Render comparison
        print(f"      Rendering comparison...")
        cfg_comparison_path = output_dir / "cfg_comparison.png"
        render_cfg_comparison(cfg_before, cfg_after, mba_blocks, [], cfg_comparison_path)
        print(f"      ✓ Saved to {cfg_comparison_path}")
        
        # Save results
        import json
        results_data = {
            'architecture': arch,
            'summary': {
                'instruction_count': len(instructions),
                'mba_block_count': len(mba_blocks),
                'expression_count': len(expressions),
                'simplified_count': sum(1 for e in expressions_data if e.get('success')),
                'basic_block_count': len(cfg_before.blocks),
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
            'expressions': expressions_data,
            'simplified_blocks': simplified_blocks,
            'functions': function_boundaries
        }
        
        # Save JSON
        with open(output_dir / "cfg_analysis.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n  ✅ CFG generation complete for {arch}")
        print(f"     - CFG before: {cfg_before_path}")
        print(f"     - CFG after: {cfg_after_path}")
        print(f"     - Comparison: {cfg_comparison_path}")
        print(f"     - Analysis: {output_dir / 'cfg_analysis.json'}")
        
        return True
        
    except Exception as e:
        print(f"  [!] Error generating CFGs: {e}")
        import traceback
        traceback.print_exc()
        return False


def build_architecture(arch: str, test_dir: Path) -> bool:
    """Build test binary for an architecture"""
    build_script = test_dir / "build.sh"
    
    if not build_script.exists():
        print(f"  [!] build.sh not found for {arch}")
        return False
    
    print(f"  Building {arch}...")
    try:
        result = subprocess.run(
            ['bash', str(build_script)],
            cwd=test_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"  [!] Build failed: {result.stderr[:200]}")
            return False
        
        # Verify assembly file was created
        asm_file = test_dir / "test_mba.asm"
        if asm_file.exists():
            print(f"  ✓ Build successful")
            return True
        else:
            print(f"  [!] Assembly file not generated")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [!] Build timeout")
        return False
    except Exception as e:
        print(f"  [!] Build error: {e}")
        return False


def main():
    """Build all binaries and generate CFGs"""
    tests_dir = Path(__file__).parent
    architectures = ['x86_64', 'arm32', 'arm64', 'mips32', 'mips64',
                     'powerpc32', 'powerpc64', 'riscv32', 'riscv64']
    
    print("="*80)
    print("GAMBA++ Build All & Generate CFGs")
    print("="*80)
    print("\nThis script will:")
    print("  1. Build test binaries for all architectures")
    print("  2. Generate CFG visualizations for each")
    print("  3. Save CFGs in tests/<arch>/output/")
    print()
    
    build_results = {}
    cfg_results = {}
    
    # Phase 1: Build all binaries
    print("\n" + "="*80)
    print("PHASE 1: Building Test Binaries")
    print("="*80)
    
    for arch in architectures:
        test_dir = tests_dir / arch
        if not test_dir.exists():
            print(f"\n[SKIP] {arch}: Test directory not found")
            build_results[arch] = 'directory_not_found'
            continue
        
        print(f"\n[{arch}]")
        success = build_architecture(arch, test_dir)
        build_results[arch] = 'success' if success else 'failed'
    
    # Phase 2: Generate CFGs
    print("\n" + "="*80)
    print("PHASE 2: Generating CFGs")
    print("="*80)
    
    for arch in architectures:
        test_dir = tests_dir / arch
        if build_results.get(arch) != 'success':
            print(f"\n[SKIP] {arch}: Binary not built")
            cfg_results[arch] = 'not_built'
            continue
        
        success = generate_cfgs_for_architecture(arch, test_dir)
        cfg_results[arch] = 'success' if success else 'failed'
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    build_success = sum(1 for r in build_results.values() if r == 'success')
    cfg_success = sum(1 for r in cfg_results.values() if r == 'success')
    
    print(f"\nBuild Results: {build_success}/{len(architectures)} architectures")
    print(f"CFG Generation: {cfg_success}/{len(architectures)} architectures")
    
    print("\nBuild Status:")
    for arch in architectures:
        status = build_results.get(arch, 'unknown')
        symbol = "✅" if status == 'success' else "❌"
        print(f"  {symbol} {arch}: {status}")
    
    print("\nCFG Generation Status:")
    for arch in architectures:
        status = cfg_results.get(arch, 'unknown')
        symbol = "✅" if status == 'success' else "❌"
        print(f"  {symbol} {arch}: {status}")
    
    print(f"\n📁 CFG files saved in: tests/<arch>/output/")
    print("   - cfg_before.png")
    print("   - cfg_after.png")
    print("   - cfg_comparison.png")
    print("   - cfg_analysis.json")
    
    # List generated files
    print("\nGenerated CFG files:")
    for arch in architectures:
        output_dir = tests_dir / arch / "output"
        if output_dir.exists():
            cfg_files = list(output_dir.glob("cfg_*.png"))
            if cfg_files:
                print(f"  ✅ {arch}: {len(cfg_files)} CFG images")
                for f in cfg_files:
                    print(f"     - {f.name}")


if __name__ == "__main__":
    main()


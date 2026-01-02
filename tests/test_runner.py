#!/usr/bin/env python3
"""
Test Runner for Architecture Tests
Executes GAMBA++ analysis on compiled test binaries
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ArchitectureTest:
    """Test runner for a specific architecture"""
    
    def __init__(self, arch: str, test_dir: Path):
        self.arch = arch
        self.test_dir = Path(test_dir)
        self.binary_path = self.test_dir / "test_mba.o"
        self.asm_path = self.test_dir / "test_mba.asm"
        self.symbols_path = self.test_dir / "test_mba.symbols"
        self.output_dir = self.test_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
    
    def build(self) -> bool:
        """Build Docker image and compile binary"""
        try:
            build_script = self.test_dir / "build.sh"
            if not build_script.exists():
                print(f"  [!] build.sh not found for {self.arch}")
                return False
            
            print(f"  Building {self.arch}...")
            result = subprocess.run(
                ['bash', str(build_script)],
                cwd=self.test_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"  [!] Build failed: {result.stderr}")
                return False
            
            # Verify artifacts exist
            if not self.asm_path.exists():
                print(f"  [!] Assembly file not generated")
                return False
            
            print(f"  ✓ Build successful")
            return True
        except Exception as e:
            print(f"  [!] Build error: {e}")
            return False
    
    def analyze_with_gamba(self) -> Dict:
        """Run GAMBA++ analysis inside Docker container"""
        try:
            print(f"  Analyzing {self.arch} with GAMBA++...")
            
            # Import GAMBA++ modules
            from assembly.x86_64.parser import parse_assembly as parse_x86
            from assembly.arm.parser import parse_assembly as parse_arm
            from assembly.x86_64.detector import detect_mba_blocks as detect_mba_x86
            from assembly.arm.detector import detect_mba_blocks as detect_mba_arm
            from assembly.x86_64.converter import convert_mba_block_to_expression as convert_x86
            from assembly.arm.converter import convert_mba_block_to_expression as convert_arm
            from assembly.common.cfg import build_cfg
            from assembly.common.mba_function_detector import detect_mba_functions, extract_function_boundaries_from_symbols
            from visualization.cfg_renderer import render_cfg, render_cfg_comparison
            from optimization.batch_advanced import process_expressions_batch_advanced
            from assembly.common.normalization import normalize_expression
            
            # Determine architecture
            arch_map = {
                'x86_64': 'x86_64',
                'arm64': 'arm64',
                'arm32': 'arm32',
                'mips32': 'x86_64',  # Use x86 parser as template for now
                'mips64': 'x86_64',
                'powerpc32': 'x86_64',
                'powerpc64': 'x86_64',
                'riscv32': 'x86_64',
                'riscv64': 'x86_64',
            }
            arch_type = arch_map.get(self.arch, 'x86_64')
            
            # Parse assembly
            if arch_type in ['x86_64'] or self.arch in ['mips32', 'mips64', 'powerpc32', 'powerpc64', 'riscv32', 'riscv64']:
                parsed = parse_x86(self.asm_path)
                detect_mba = detect_mba_x86
                convert_mba = convert_x86
            else:
                parsed = parse_arm(self.asm_path, arch=arch_type)
                detect_mba = detect_mba_arm
                convert_mba = convert_arm
            
            instructions = parsed.get("instructions", [])
            print(f"    Parsed {len(instructions)} instructions")
            
            # Extract function boundaries
            function_boundaries = []
            if self.symbols_path.exists():
                function_boundaries = extract_function_boundaries_from_symbols(str(self.symbols_path))
                print(f"    Found {len(function_boundaries)} functions")
            
            # Detect MBA blocks
            mba_blocks = detect_mba(instructions)
            print(f"    Detected {len(mba_blocks)} MBA blocks")
            
            # Build CFG before simplification
            cfg_before = build_cfg(instructions, arch=arch_type)
            print(f"    Built CFG with {len(cfg_before.blocks)} basic blocks")
            
            # Render CFG before
            cfg_before_path = self.output_dir / "cfg_before.png"
            render_cfg(cfg_before, mba_blocks, cfg_before_path)
            print(f"    Rendered CFG before: {cfg_before_path}")
            
            # Convert MBA blocks to expressions
            expressions_data = []
            expressions = []
            for mba_block in mba_blocks:
                expr_data = convert_mba(mba_block) if convert_mba else None
                if expr_data:
                    expr = expr_data.get('gamba_expression', '')
                    if expr:
                        expressions.append(expr)
                        expressions_data.append({
                            'block': {
                                'start': getattr(mba_block, 'start_address', 0),
                                'end': getattr(mba_block, 'end_address', 0),
                                'pattern': getattr(mba_block, 'pattern_type', 'unknown')
                            },
                            'original': expr,
                            'simplified': None
                        })
            
            print(f"    Converted {len(expressions)} expressions")
            
            # Simplify expressions
            if expressions:
                # Normalize before simplification
                from assembly.common.normalization import normalize_expression_list
                normalized = normalize_expression_list(expressions)
                
                # Simplify with GAMBA++
                results = process_expressions_batch_advanced(
                    expressions=normalized,
                    bitcount=32,
                    max_workers=4,
                    use_cache=True,
                    show_progress=False
                )
                
                # Update expressions_data with simplified results
                for i, result in enumerate(results):
                    if i < len(expressions_data):
                        if result.get('success'):
                            expressions_data[i]['simplified'] = result.get('simplified', '')
                        else:
                            expressions_data[i]['error'] = result.get('error', 'Unknown error')
                
                print(f"    Simplified {sum(1 for r in results if r.get('success'))} expressions")
            
            # Generate simplified assembly representation (simplified CFG)
            # For now, we'll use the same CFG but mark simplified blocks
            cfg_after = cfg_before  # In real implementation, would rebuild from simplified code
            
            # Render CFG after
            cfg_after_path = self.output_dir / "cfg_after.png"
            render_cfg(cfg_after, [], cfg_after_path)  # No MBA blocks after simplification
            print(f"    Rendered CFG after: {cfg_after_path}")
            
            # Render comparison
            cfg_comparison_path = self.output_dir / "cfg_comparison.png"
            render_cfg_comparison(cfg_before, cfg_after, mba_blocks, [], cfg_comparison_path)
            print(f"    Rendered CFG comparison: {cfg_comparison_path}")
            
            # Calculate metrics
            metrics = {
                'instruction_count': len(instructions),
                'mba_block_count': len(mba_blocks),
                'mba_expression_count': len(expressions),
                'simplified_count': sum(1 for r in results if r.get('success')) if expressions else 0,
                'basic_block_count_before': len(cfg_before.blocks),
                'basic_block_count_after': len(cfg_after.blocks),
            }
            
            # Save results
            results_data = {
                'architecture': self.arch,
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
                'metrics': metrics,
                'functions': [
                    {
                        'name': f.get('name', 'unknown'),
                        'start': f.get('start', 0),
                        'end': f.get('end', 0)
                    }
                    for f in function_boundaries
                ]
            }
            
            # Save JSON outputs
            with open(self.output_dir / "mba_blocks.json", 'w') as f:
                json.dump(results_data['mba_blocks'], f, indent=2)
            
            with open(self.output_dir / "simplified_expressions.json", 'w') as f:
                json.dump(results_data['expressions'], f, indent=2)
            
            with open(self.output_dir / "test_report.json", 'w') as f:
                json.dump(results_data, f, indent=2)
            
            return results_data
            
        except Exception as e:
            print(f"  [!] Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def validate_results(self, results: Dict) -> bool:
        """Validate analysis results"""
        if 'error' in results:
            return False
        
        # Check MBA detection
        if results.get('metrics', {}).get('mba_block_count', 0) == 0:
            print(f"  [!] No MBA blocks detected")
            return False
        
        # Check simplification
        metrics = results.get('metrics', {})
        if metrics.get('simplified_count', 0) == 0:
            print(f"  [!] No expressions simplified")
            return False
        
        # Check CFG generation
        if metrics.get('basic_block_count_before', 0) == 0:
            print(f"  [!] CFG not generated")
            return False
        
        # Check output files
        required_files = [
            'cfg_before.png',
            'cfg_after.png',
            'cfg_comparison.png',
            'mba_blocks.json',
            'simplified_expressions.json',
            'test_report.json'
        ]
        
        for filename in required_files:
            if not (self.output_dir / filename).exists():
                print(f"  [!] Missing output file: {filename}")
                return False
        
        print(f"  ✓ Validation passed")
        return True


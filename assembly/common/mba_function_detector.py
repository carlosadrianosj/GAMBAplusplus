#!/usr/bin/env python3
"""
MBA Function Detection
Detects functions containing MBA patterns
"""

from typing import List, Dict, Optional
from .cfg import CFG, BasicBlock
from ..x86_64.detector import detect_mba_blocks as detect_mba_x86
from ..arm.detector import detect_mba_blocks as detect_mba_arm


class MBAFunction:
    """Represents a function containing MBA patterns"""
    
    def __init__(self, function_name: str, start_address: int, end_address: int,
                 instructions: List, mba_blocks: List, cfg: Optional[CFG] = None):
        self.function_name = function_name
        self.start_address = start_address
        self.end_address = end_address
        self.instructions = instructions
        self.mba_blocks = mba_blocks
        self.cfg = cfg
        self.mba_block_count = len(mba_blocks)
        self.instruction_count = len(instructions)
        self.mba_ratio = self.mba_block_count / self.instruction_count if self.instruction_count > 0 else 0
    
    def __repr__(self):
        return f"MBAFunction({self.function_name}, 0x{self.start_address:X}-0x{self.end_address:X}, {self.mba_block_count} MBA blocks)"


def detect_mba_functions(instructions: List, arch: str = 'x86_64',
                        function_boundaries: Optional[List[Dict]] = None) -> List[MBAFunction]:
    """
    Detect functions containing MBA patterns.
    
    Args:
        instructions: List of instruction objects
        arch: Architecture name (x86_64, arm64, etc.)
        function_boundaries: Optional list of function boundaries with 'name', 'start', 'end'
    
    Returns:
        List of MBAFunction objects
    """
    mba_functions = []
    
    # If function boundaries provided, analyze each function separately
    if function_boundaries:
        for func_info in function_boundaries:
            func_name = func_info.get('name', f"func_{func_info['start']:X}")
            start_addr = func_info['start']
            end_addr = func_info['end']
            
            # Extract function instructions
            func_instructions = [
                inst for inst in instructions
                if start_addr <= getattr(inst, 'address', 0) < end_addr
            ]
            
            if not func_instructions:
                continue
            
            # Detect MBA blocks in function
            mba_blocks = detect_mba_blocks_in_instructions(func_instructions, arch)
            
            if mba_blocks:
                # Build CFG for function
                from .cfg import build_cfg
                cfg = build_cfg(func_instructions, arch)
                
                # Associate MBA blocks with basic blocks
                associate_mba_with_blocks(cfg, mba_blocks)
                
                mba_function = MBAFunction(
                    function_name=func_name,
                    start_address=start_addr,
                    end_address=end_addr,
                    instructions=func_instructions,
                    mba_blocks=mba_blocks,
                    cfg=cfg
                )
                mba_functions.append(mba_function)
    else:
        # No function boundaries - analyze entire instruction list as one function
        mba_blocks = detect_mba_blocks_in_instructions(instructions, arch)
        
        if mba_blocks:
            from .cfg import build_cfg
            cfg = build_cfg(instructions, arch)
            associate_mba_with_blocks(cfg, mba_blocks)
            
            start_addr = instructions[0].address if instructions else 0
            end_addr = instructions[-1].address + getattr(instructions[-1], 'size', 0) if instructions else 0
            
            mba_function = MBAFunction(
                function_name="unknown",
                start_address=start_addr,
                end_address=end_addr,
                instructions=instructions,
                mba_blocks=mba_blocks,
                cfg=cfg
            )
            mba_functions.append(mba_function)
    
    return mba_functions


def detect_mba_blocks_in_instructions(instructions: List, arch: str) -> List:
    """
    Detect MBA blocks using architecture-specific detector.
    
    Args:
        instructions: List of instructions
        arch: Architecture name
    
    Returns:
        List of MBABlock objects
    """
    if arch in ['x86_64', 'x86_32', 'x86']:
        return detect_mba_x86(instructions)
    elif arch in ['arm64', 'arm32', 'arm', 'thumb']:
        return detect_mba_arm(instructions)
    else:
        # Default: try x86_64 detector
        return detect_mba_x86(instructions)


def associate_mba_with_blocks(cfg: CFG, mba_blocks: List):
    """
    Associate MBA blocks with basic blocks in CFG.
    
    Args:
        cfg: Control Flow Graph
        mba_blocks: List of MBA blocks
    """
    for mba_block in mba_blocks:
        mba_start = getattr(mba_block, 'start_address', 0)
        mba_end = getattr(mba_block, 'end_address', 0)
        
        # Find basic blocks that overlap with MBA block
        for block in cfg.blocks:
            if (block.start_address <= mba_start < block.end_address or
                block.start_address < mba_end <= block.end_address or
                (mba_start <= block.start_address and mba_end >= block.end_address)):
                block.mba_blocks.append(mba_block)


def extract_function_boundaries_from_symbols(symbols_file: str) -> List[Dict]:
    """
    Extract function boundaries from nm/objdump symbols output.
    
    Args:
        symbols_file: Path to symbols file
    
    Returns:
        List of function dictionaries with 'name', 'start', 'end'
    """
    import re
    from pathlib import Path
    
    functions = []
    
    if not Path(symbols_file).exists():
        return functions
    
    with open(symbols_file, 'r') as f:
        for line in f:
            # nm format: address type name
            # Example: 0000000000000000 T mba_simple
            match = re.match(r'([0-9a-fA-F]+)\s+([Tt])\s+(.+)', line)
            if match:
                address = int(match.group(1), 16)
                func_name = match.group(3).strip()
                functions.append({
                    'name': func_name,
                    'start': address,
                    'end': address + 100  # Approximate, would need size from objdump
                })
    
    return functions


if __name__ == "__main__":
    # Test function detection
    class TestInst:
        def __init__(self, address, mnemonic, operands):
            self.address = address
            self.mnemonic = mnemonic
            self.operands = operands
            self.size = 4
    
    test_instructions = [
        TestInst(0x1000, 'mov', ['eax', 'ebx']),
        TestInst(0x1004, 'xor', ['eax', 'ecx']),
        TestInst(0x1008, 'and', ['eax', 'edx']),
        TestInst(0x100C, 'or', ['eax', 'esi']),
        TestInst(0x1010, 'xor', ['eax', 'edi']),
        TestInst(0x1014, 'ret', []),
    ]
    
    functions = detect_mba_functions(test_instructions, arch='x86_64')
    print(f"Found {len(functions)} functions with MBA:")
    for func in functions:
        print(f"  {func}")


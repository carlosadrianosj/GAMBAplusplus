#!/usr/bin/env python3
"""
ARM MBA Pattern Detector
Identifies sequences of ARM instructions that form MBA (Mixed Boolean-Arithmetic) expressions
"""

from typing import List, Dict, Optional
from .parser import Instruction


class MBABlock:
    """Represents a block of ARM instructions that form an MBA expression"""
    
    def __init__(self, start_address: int, end_address: int, 
                 instructions: List[Instruction], pattern_type: str):
        self.start_address = start_address
        self.end_address = end_address
        self.instructions = instructions
        self.pattern_type = pattern_type  # "boolean_chain" | "arithmetic_boolean" | "comparison_chain"
        self.registers_used = self._extract_registers()
    
    def _extract_registers(self) -> set:
        """Extract all registers used in this block"""
        registers = set()
        for inst in self.instructions:
            for op in inst.operands:
                regs = self._extract_register_names(op)
                registers.update(regs)
        return registers
    
    def _extract_register_names(self, operand: str) -> List[str]:
        """Extract register names from operand string"""
        import re
        
        # ARM32 registers: r0-r15, sp, lr, pc
        # ARM64 registers: x0-x30, w0-w30, sp, lr, pc
        reg_patterns = [
            r'\b(r\d+|sp|lr|pc)\b',  # ARM32
            r'\b([xw]\d+|sp|lr|pc)\b',  # ARM64
        ]
        
        registers = []
        for pattern in reg_patterns:
            matches = re.findall(pattern, operand, re.IGNORECASE)
            registers.extend(matches)
        
        return registers
    
    def size(self) -> int:
        """Get size of block in bytes"""
        return self.end_address - self.start_address
    
    def instruction_count(self) -> int:
        """Get number of instructions in block"""
        return len(self.instructions)
    
    def __repr__(self):
        return f"MBABlock(0x{self.start_address:X}-0x{self.end_address:X}, {len(self.instructions)} insts, {self.pattern_type})"


def is_boolean_instruction(mnemonic: str) -> bool:
    """Check if ARM instruction is a boolean operation"""
    boolean_ops = {
        'and', 'orr', 'eor', 'bic', 'mvn',  # ARM32/ARM64
        'eon', 'orn',  # ARM64 only
    }
    return mnemonic in boolean_ops


def is_arithmetic_instruction(mnemonic: str) -> bool:
    """Check if ARM instruction is arithmetic"""
    arithmetic_ops = {
        'add', 'sub', 'mul', 'madd', 'msub', 'neg',  # ARM64
        'mla', 'rsb',  # ARM32
        'lsl', 'lsr', 'asr', 'ror',  # Shift operations
    }
    return mnemonic in arithmetic_ops


def is_comparison_instruction(mnemonic: str) -> bool:
    """Check if ARM instruction is comparison"""
    comparison_ops = {
        'cmp', 'cmn', 'tst', 'teq',  # ARM32/ARM64
        'cset', 'csel', 'csinc',  # ARM64 conditional set/select
    }
    return mnemonic in comparison_ops


def is_control_flow_instruction(mnemonic: str) -> bool:
    """Check if ARM instruction is control flow"""
    control_ops = {
        'b', 'bl', 'br', 'blr',  # Branch
        'ret', 'retx',  # Return
        'cbz', 'cbnz',  # Conditional branch
    }
    return mnemonic in control_ops


def detect_mba_blocks(instructions: List[Instruction], 
                     min_boolean_chain: int = 10,
                     min_arithmetic_boolean: int = 15) -> List[MBABlock]:
    """
    Detect MBA blocks in ARM instruction list.
    
    Args:
        instructions: List of Instruction objects
        min_boolean_chain: Minimum length for boolean chain pattern
        min_arithmetic_boolean: Minimum length for arithmetic-boolean pattern
    
    Returns:
        List of MBABlock objects
    """
    mba_blocks = []
    i = 0
    
    while i < len(instructions):
        # Try to detect different MBA patterns
        
        # Pattern 1: Arithmetic-Boolean pattern (add/sub + mul + boolean chain)
        block = detect_arithmetic_boolean_pattern(instructions, i, min_arithmetic_boolean)
        if block:
            mba_blocks.append(block)
            i = find_instruction_index(instructions, block.end_address)
            continue
        
        # Pattern 2: Long boolean chain
        block = detect_boolean_chain_pattern(instructions, i, min_boolean_chain)
        if block:
            mba_blocks.append(block)
            i = find_instruction_index(instructions, block.end_address)
            continue
        
        # Pattern 3: Comparison chain (complex comparisons using boolean operations)
        block = detect_comparison_chain_pattern(instructions, i)
        if block:
            mba_blocks.append(block)
            i = find_instruction_index(instructions, block.end_address)
            continue
        
        i += 1
    
    return mba_blocks


def detect_arithmetic_boolean_pattern(instructions: List[Instruction], 
                                     start_idx: int, min_length: int) -> Optional[MBABlock]:
    """
    Detect pattern: add/sub + mul + boolean chain
    
    This is a common MBA pattern where arithmetic is mixed with boolean operations.
    """
    if start_idx >= len(instructions):
        return None
    
    if start_idx + 1 >= len(instructions):
        return None
    
    inst1 = instructions[start_idx]
    inst2 = instructions[start_idx + 1] if start_idx + 1 < len(instructions) else None
    
    # Check for add/sub followed by mul/madd
    if inst1.mnemonic in ['add', 'sub'] and inst2 and inst2.mnemonic in ['mul', 'madd', 'msub']:
        # Found start of pattern, continue until end
        end_idx = find_pattern_end(instructions, start_idx + 2, 
                                   is_boolean_or_arithmetic, 
                                   min_length - 2)
        
        if end_idx > start_idx + min_length:
            block_instructions = instructions[start_idx:end_idx]
            return MBABlock(
                start_address=inst1.address,
                end_address=instructions[end_idx - 1].address + 4,  # ARM instructions are 4 bytes
                instructions=block_instructions,
                pattern_type="arithmetic_boolean"
            )
    
    return None


def detect_boolean_chain_pattern(instructions: List[Instruction], 
                                start_idx: int, min_length: int) -> Optional[MBABlock]:
    """
    Detect long chain of boolean operations (and/orr/eor/bic/mvn)
    """
    if start_idx >= len(instructions):
        return None
    
    # Count consecutive boolean instructions
    boolean_count = 0
    end_idx = start_idx
    
    for i in range(start_idx, len(instructions)):
        if is_boolean_instruction(instructions[i].mnemonic):
            boolean_count += 1
            end_idx = i + 1
        elif boolean_count >= min_length:
            # Found end of boolean chain
            break
        elif boolean_count > 0:
            # Chain broken too early
            break
    
    if boolean_count >= min_length:
        block_instructions = instructions[start_idx:end_idx]
        return MBABlock(
            start_address=instructions[start_idx].address,
            end_address=instructions[end_idx - 1].address + 4,
            instructions=block_instructions,
            pattern_type="boolean_chain"
        )
    
    return None


def detect_comparison_chain_pattern(instructions: List[Instruction], 
                                   start_idx: int) -> Optional[MBABlock]:
    """
    Detect complex comparison using boolean operations (cset/csel after boolean chain)
    """
    if start_idx >= len(instructions):
        return None
    
    # Look for boolean chain ending with cset/csel/csinc
    boolean_count = 0
    end_idx = start_idx
    
    for i in range(start_idx, len(instructions)):
        if is_boolean_instruction(instructions[i].mnemonic):
            boolean_count += 1
            end_idx = i + 1
        elif is_comparison_instruction(instructions[i].mnemonic):
            # Found comparison instruction after boolean chain
            if boolean_count >= 5:  # At least 5 boolean ops before comparison
                end_idx = i + 1
                block_instructions = instructions[start_idx:end_idx]
                return MBABlock(
                    start_address=instructions[start_idx].address,
                    end_address=instructions[end_idx - 1].address + 4,
                    instructions=block_instructions,
                    pattern_type="comparison_chain"
                )
            break
        elif boolean_count > 0:
            # Chain broken
            break
    
    return None


def is_boolean_or_arithmetic(mnemonic: str) -> bool:
    """Check if instruction is boolean or arithmetic"""
    return is_boolean_instruction(mnemonic) or is_arithmetic_instruction(mnemonic)


def find_pattern_end(instructions: List[Instruction], start_idx: int,
                    predicate, min_length: int) -> int:
    """
    Find end of pattern starting at start_idx.
    
    Args:
        instructions: List of instructions
        start_idx: Starting index
        predicate: Function to check if instruction matches pattern
        min_length: Minimum length of pattern
    
    Returns:
        Index of end of pattern
    """
    count = 0
    end_idx = start_idx
    
    for i in range(start_idx, len(instructions)):
        if predicate(instructions[i].mnemonic):
            count += 1
            end_idx = i + 1
        elif count >= min_length:
            # Pattern ended
            break
        elif count > 0:
            # Pattern broken too early
            break
    
    return end_idx if count >= min_length else start_idx


def find_instruction_index(instructions: List[Instruction], address: int) -> int:
    """Find index of instruction with given address"""
    for i, inst in enumerate(instructions):
        if inst.address >= address:
            return i
    return len(instructions)


if __name__ == "__main__":
    import sys
    from .parser import parse_assembly
    
    if len(sys.argv) < 2:
        print("Usage: python detector.py <assembly_file.asm> [arm32|arm64]")
        sys.exit(1)
    
    asm_file = sys.argv[1]
    arch = sys.argv[2] if len(sys.argv) > 2 else "arm64"
    
    try:
        parsed = parse_assembly(asm_file, arch=arch)
        instructions = parsed["instructions"]
        
        print(f"Analyzing {len(instructions)} instructions for MBA patterns...")
        
        mba_blocks = detect_mba_blocks(instructions)
        
        print(f"\nFound {len(mba_blocks)} MBA blocks:")
        for i, block in enumerate(mba_blocks, 1):
            print(f"\n{i}. {block}")
            print(f"   Pattern: {block.pattern_type}")
            print(f"   Size: {block.size()} bytes")
            print(f"   Instructions: {block.instruction_count()}")
            print(f"   Registers: {', '.join(sorted(block.registers_used))}")
            print(f"   First instruction: {block.instructions[0]}")
            if len(block.instructions) > 1:
                print(f"   Last instruction: {block.instructions[-1]}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


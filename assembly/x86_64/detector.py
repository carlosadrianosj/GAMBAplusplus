#!/usr/bin/env python3
"""
MBA Pattern Detector
Identifies sequences of instructions that form MBA (Mixed Boolean-Arithmetic) expressions
"""

from typing import List, Dict, Optional
from .parser import Instruction


class MBABlock:
    """Represents a block of instructions that form an MBA expression"""
    
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
                # Extract register names (simple heuristic)
                regs = self._extract_register_names(op)
                registers.update(regs)
        return registers
    
    def _extract_register_names(self, operand: str) -> List[str]:
        """Extract register names from operand string"""
        # Common x86-64 registers
        reg_patterns = [
            r'\b(rax|eax|ax|al|ah)\b',
            r'\b(rbx|ebx|bx|bl|bh)\b',
            r'\b(rcx|ecx|cx|cl|ch)\b',
            r'\b(rdx|edx|dx|dl|dh)\b',
            r'\b(rsi|esi|si|sil)\b',
            r'\b(rdi|edi|di|dil)\b',
            r'\b(rbp|ebp|bp|bpl)\b',
            r'\b(rsp|esp|sp|spl)\b',
            r'\b(r8|r8d|r8w|r8b)\b',
            r'\b(r9|r9d|r9w|r9b)\b',
            r'\b(r10|r10d|r10w|r10b)\b',
            r'\b(r11|r11d|r11w|r11b)\b',
            r'\b(r12|r12d|r12w|r12b)\b',
            r'\b(r13|r13d|r13w|r13b)\b',
            r'\b(r14|r14d|r14w|r14b)\b',
            r'\b(r15|r15d|r15w|r15b)\b',
        ]
        
        import re
        registers = []
        for pattern in reg_patterns:
            matches = re.findall(pattern, operand, re.IGNORECASE)
            registers.extend(matches)
        
        return registers
    
    def size(self) -> int:
        """Get size of block in bytes (approximate)"""
        return self.end_address - self.start_address
    
    def instruction_count(self) -> int:
        """Get number of instructions in block"""
        return len(self.instructions)
    
    def __repr__(self):
        return f"MBABlock(0x{self.start_address:X}-0x{self.end_address:X}, {len(self.instructions)} insts, {self.pattern_type})"


def is_boolean_instruction(mnemonic: str) -> bool:
    """Check if instruction is a boolean operation"""
    boolean_ops = {'and', 'or', 'xor', 'not', 'test'}
    return mnemonic in boolean_ops


def is_arithmetic_instruction(mnemonic: str) -> bool:
    """Check if instruction is arithmetic"""
    arithmetic_ops = {'add', 'sub', 'imul', 'mul', 'lea', 'inc', 'dec', 'neg'}
    return mnemonic in arithmetic_ops


def is_shift_instruction(mnemonic: str) -> bool:
    """Check if instruction is a shift operation"""
    shift_ops = {'shl', 'sal', 'shr', 'sar', 'rol', 'ror'}
    return mnemonic in shift_ops


def is_comparison_instruction(mnemonic: str) -> bool:
    """Check if instruction is comparison"""
    comparison_ops = {'cmp', 'test', 'setz', 'setnz', 'setl', 'setle', 'setg', 'setge',
                      'sete', 'setne', 'seta', 'setae', 'setb', 'setbe', 'setc', 'setnc',
                      'seto', 'setno', 'setp', 'setnp', 'sets', 'setns'}
    return mnemonic in comparison_ops or mnemonic.startswith('set')


def is_conditional_move_instruction(mnemonic: str) -> bool:
    """Check if instruction is conditional move"""
    return mnemonic.startswith('cmov')


def is_control_flow_instruction(mnemonic: str) -> bool:
    """Check if instruction is control flow"""
    control_ops = {'jmp', 'je', 'jne', 'jl', 'jle', 'jg', 'jge', 'call', 'ret', 'retn'}
    return mnemonic in control_ops


def detect_mba_blocks(instructions: List[Instruction], 
                     min_boolean_chain: int = 10,
                     min_arithmetic_boolean: int = 15) -> List[MBABlock]:
    """
    Detect MBA blocks in instruction list.
    
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
        
        # Pattern 1: Arithmetic-Boolean pattern (lea + imul + boolean chain)
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
    Detect pattern: lea [reg-N] + imul + boolean chain
    
    This is a common MBA pattern where arithmetic is mixed with boolean operations.
    """
    if start_idx >= len(instructions):
        return None
    
    # Look for lea [reg-N] pattern
    if start_idx + 1 >= len(instructions):
        return None
    
    inst1 = instructions[start_idx]
    inst2 = instructions[start_idx + 1] if start_idx + 1 < len(instructions) else None
    
    # Check for lea [reg-N] followed by imul
    if inst1.mnemonic == 'lea' and inst2 and inst2.mnemonic == 'imul':
        # Found start of pattern, continue until end
        end_idx = find_pattern_end(instructions, start_idx + 2, 
                                   is_boolean_or_arithmetic, 
                                   min_length - 2)
        
        if end_idx > start_idx + min_length:
            block_instructions = instructions[start_idx:end_idx]
            return MBABlock(
                start_address=inst1.address,
                end_address=instructions[end_idx - 1].address + 10,  # Approximate
                instructions=block_instructions,
                pattern_type="arithmetic_boolean"
            )
    
    return None


def detect_boolean_chain_pattern(instructions: List[Instruction], 
                                start_idx: int, min_length: int) -> Optional[MBABlock]:
    """
    Detect long chain of boolean operations (and/or/xor/not)
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
            end_address=instructions[end_idx - 1].address + 10,  # Approximate
            instructions=block_instructions,
            pattern_type="boolean_chain"
        )
    
    return None


def detect_comparison_chain_pattern(instructions: List[Instruction], 
                                   start_idx: int) -> Optional[MBABlock]:
    """
    Detect complex comparison using boolean operations (setz/setnz after boolean chain)
    """
    if start_idx >= len(instructions):
        return None
    
    # Look for boolean chain ending with setz/setnz/setl/etc
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
                    end_address=instructions[end_idx - 1].address + 10,
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
    from assembly_parser import parse_assembly
    
    if len(sys.argv) < 2:
        print("Usage: python mba_detector.py <assembly_file.asm>")
        sys.exit(1)
    
    asm_file = sys.argv[1]
    
    try:
        parsed = parse_assembly(asm_file)
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


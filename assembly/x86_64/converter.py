#!/usr/bin/env python3
"""
Expression Converter
Converts sequences of assembly instructions to GAMBA-compatible mathematical expressions
"""

from typing import List, Dict, Optional, Set
from .parser import Instruction
from .detector import MBABlock


class RegisterTracker:
    """Tracks register values through instruction sequences"""
    
    def __init__(self):
        self.registers = {}  # register -> expression string
        self.temp_counter = 0
    
    def get_temp_var(self) -> str:
        """Get a temporary variable name"""
        var = f"t{self.temp_counter}"
        self.temp_counter += 1
        return var
    
    def set_register(self, reg: str, expression: str):
        """Set register value to expression"""
        self.registers[reg] = expression
    
    def get_register(self, reg: str) -> Optional[str]:
        """Get current expression for register"""
        return self.registers.get(reg)
    
    def clear(self):
        """Clear all register values"""
        self.registers.clear()
        self.temp_counter = 0


def normalize_register_name(reg: str) -> str:
    """Normalize register name (eax -> reg32, rax -> reg64)"""
    # For GAMBA, we'll use simple variable names
    # Map registers to variables: eax -> x, edx -> y, etc.
    reg_map = {
        'eax': 'x', 'rax': 'x',
        'ebx': 'y', 'rbx': 'y',
        'ecx': 'z', 'rcx': 'z',
        'edx': 'w', 'rdx': 'w',
        'esi': 'a', 'rsi': 'a',
        'edi': 'b', 'rdi': 'b',
        'ebp': 'c', 'rbp': 'c',
        'esp': 'd', 'rsp': 'd',
    }
    
    reg_lower = reg.lower()
    return reg_map.get(reg_lower, reg_lower)


def extract_constant(operand: str) -> Optional[int]:
    """Extract constant value from operand"""
    import re
    
    # Try hex: 0x1234 or 1234h
    hex_match = re.search(r'0x([0-9a-fA-F]+)', operand)
    if hex_match:
        return int(hex_match.group(1), 16)
    
    hex_match = re.search(r'([0-9a-fA-F]+)h', operand, re.IGNORECASE)
    if hex_match:
        return int(hex_match.group(1), 16)
    
    # Try decimal
    dec_match = re.search(r'\b(\d+)\b', operand)
    if dec_match:
        return int(dec_match.group(1))
    
    return None


def convert_instruction_to_expression(inst: Instruction, tracker: RegisterTracker) -> Optional[str]:
    """
    Convert a single instruction to part of an expression.
    
    Returns:
        Expression string or None if instruction doesn't contribute to expression
    """
    mnemonic = inst.mnemonic
    
    if len(inst.operands) == 0:
        return None
    
    # Handle different instruction types
    if mnemonic == 'mov':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            
            # Check if source is constant
            const = extract_constant(src)
            if const is not None:
                tracker.set_register(dst, str(const))
            else:
                # Source is register or memory
                src_reg = normalize_register_name(src)
                src_expr = tracker.get_register(src_reg)
                if src_expr:
                    tracker.set_register(dst, src_expr)
                else:
                    tracker.set_register(dst, src_reg)
        return None
    
    elif mnemonic == 'lea':
        # lea dst, [src+offset] -> dst = src + offset
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src_expr = inst.operands[1]
            
            # Parse [reg+offset] or [reg-N]
            import re
            match = re.search(r'\[([^\]]+)\]', src_expr)
            if match:
                inner = match.group(1)
                # Simple case: [reg-N] or [reg+N]
                reg_match = re.search(r'(\w+)\s*([+-])\s*(\d+)', inner)
                if reg_match:
                    reg = normalize_register_name(reg_match.group(1))
                    op = reg_match.group(2)
                    offset = reg_match.group(3)
                    
                    reg_expr = tracker.get_register(reg) or reg
                    expr = f"({reg_expr} {op} {offset})"
                    tracker.set_register(dst, expr)
                    return expr
        return None
    
    elif mnemonic == 'imul':
        # imul dst, src -> dst = dst * src
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            
            dst_expr = tracker.get_register(dst) or dst
            const = extract_constant(src)
            if const is not None:
                src_expr = str(const)
            else:
                src_reg = normalize_register_name(src)
                src_expr = tracker.get_register(src_reg) or src_reg
            
            expr = f"({dst_expr} * {src_expr})"
            tracker.set_register(dst, expr)
            return expr
        return None
    
    elif mnemonic == 'and':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            
            dst_expr = tracker.get_register(dst) or dst
            const = extract_constant(src)
            if const is not None:
                src_expr = hex(const)
            else:
                src_reg = normalize_register_name(src)
                src_expr = tracker.get_register(src_reg) or src_reg
            
            expr = f"({dst_expr} & {src_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    elif mnemonic == 'or':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            
            dst_expr = tracker.get_register(dst) or dst
            const = extract_constant(src)
            if const is not None:
                src_expr = hex(const)
            else:
                src_reg = normalize_register_name(src)
                src_expr = tracker.get_register(src_reg) or src_reg
            
            expr = f"({dst_expr} | {src_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    elif mnemonic == 'xor':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            
            dst_expr = tracker.get_register(dst) or dst
            const = extract_constant(src)
            if const is not None:
                src_expr = hex(const)
            else:
                src_reg = normalize_register_name(src)
                src_expr = tracker.get_register(src_reg) or src_reg
            
            expr = f"({dst_expr} ^ {src_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    elif mnemonic == 'not':
        if len(inst.operands) >= 1:
            dst = normalize_register_name(inst.operands[0])
            dst_expr = tracker.get_register(dst) or dst
            
            expr = f"(~{dst_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    elif mnemonic == 'add':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            
            dst_expr = tracker.get_register(dst) or dst
            const = extract_constant(src)
            if const is not None:
                src_expr = str(const)
            else:
                src_reg = normalize_register_name(src)
                src_expr = tracker.get_register(src_reg) or src_reg
            
            expr = f"({dst_expr} + {src_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    elif mnemonic == 'sub':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            
            dst_expr = tracker.get_register(dst) or dst
            const = extract_constant(src)
            if const is not None:
                src_expr = str(const)
            else:
                src_reg = normalize_register_name(src)
                src_expr = tracker.get_register(src_reg) or src_reg
            
            expr = f"({dst_expr} - {src_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    return None


def convert_mba_block_to_expression(block: MBABlock) -> Optional[Dict]:
    """
    Convert MBA block to GAMBA expression.
    
    Args:
        block: MBABlock to convert
    
    Returns:
        Dictionary with expression data or None if conversion fails
    """
    tracker = RegisterTracker()
    expressions = []
    
    # Identify input registers (registers used before being set)
    input_registers = set()
    defined_registers = set()
    
    for inst in block.instructions:
        if len(inst.operands) > 0:
            dst_reg = normalize_register_name(inst.operands[0])
            if inst.mnemonic in ['mov', 'lea', 'imul', 'and', 'or', 'xor', 'not', 'add', 'sub']:
                defined_registers.add(dst_reg)
            
            # Check operands for input registers
            for op in inst.operands:
                regs = extract_registers_from_operand(op)
                for reg in regs:
                    if reg not in defined_registers:
                        input_registers.add(reg)
    
    # Track through instructions
    for inst in block.instructions:
        expr = convert_instruction_to_expression(inst, tracker)
        if expr:
            expressions.append(expr)
    
    # Find output register (last register that was modified)
    output_register = None
    for inst in reversed(block.instructions):
        if len(inst.operands) > 0:
            reg = normalize_register_name(inst.operands[0])
            if reg in tracker.registers:
                output_register = reg
                break
    
    if not output_register:
        return None
    
    final_expression = tracker.get_register(output_register)
    if not final_expression:
        return None
    
    return {
        "original_block": block,
        "gamba_expression": final_expression,
        "output_register": output_register,
        "input_registers": list(input_registers),
        "intermediate_expressions": expressions
    }


def extract_registers_from_operand(operand: str) -> List[str]:
    """Extract register names from operand string"""
    import re
    from mba_detector import MBABlock
    
    # Use MBABlock's register extraction method
    temp_block = MBABlock(0, 0, [], "temp")
    return temp_block._extract_register_names(operand)


if __name__ == "__main__":
    import sys
    from assembly_parser import parse_assembly
    from mba_detector import detect_mba_blocks
    
    if len(sys.argv) < 2:
        print("Usage: python expression_converter.py <assembly_file.asm>")
        sys.exit(1)
    
    asm_file = sys.argv[1]
    
    try:
        parsed = parse_assembly(asm_file)
        instructions = parsed["instructions"]
        
        mba_blocks = detect_mba_blocks(instructions)
        
        print(f"Converting {len(mba_blocks)} MBA blocks to expressions...\n")
        
        for i, block in enumerate(mba_blocks[:5], 1):  # Process first 5 blocks
            print(f"Block {i}:")
            result = convert_mba_block_to_expression(block)
            if result:
                print(f"  Expression: {result['gamba_expression']}")
                print(f"  Output register: {result['output_register']}")
                print(f"  Input registers: {', '.join(result['input_registers'])}")
            else:
                print("  Failed to convert")
            print()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


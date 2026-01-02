#!/usr/bin/env python3
"""
Expression Converter
Converts sequences of assembly instructions to GAMBA-compatible mathematical expressions
"""

from typing import List, Dict, Optional, Set
from .parser import Instruction
from .detector import MBABlock


class RegisterTracker:
    """Tracks register values through instruction sequences with subregister modeling"""
    
    def __init__(self):
        self.registers = {}  # register -> expression string
        self.register_widths = {}  # register -> width in bits (8, 16, 32, 64)
        self.temp_counter = 0
    
    def get_temp_var(self) -> str:
        """Get a temporary variable name"""
        var = f"t{self.temp_counter}"
        self.temp_counter += 1
        return var
    
    def set_register(self, reg: str, expression: str, width: int = 64):
        """Set register value to expression with width tracking"""
        self.registers[reg] = expression
        self.register_widths[reg] = width
        
        # Handle subregister writes in x86-64
        # Writing to eax clears upper 32 bits of rax
        if width == 32:
            base_reg = self._get_base_register_64(reg)
            if base_reg:
                # Clear upper bits: rax = (rax & 0xFFFFFFFF00000000) | eax
                # For GAMBA, we'll use zero-extend: rax = zext(eax, 64)
                self.registers[base_reg] = f"zext({expression}, 64)"
                self.register_widths[base_reg] = 64
    
    def get_register(self, reg: str) -> Optional[str]:
        """Get current expression for register"""
        return self.registers.get(reg)
    
    def get_register_width(self, reg: str) -> int:
        """Get register width in bits"""
        return self.register_widths.get(reg, 64)
    
    def _get_base_register_64(self, reg: str) -> Optional[str]:
        """Get 64-bit base register for 32-bit subregister"""
        reg_lower = reg.lower()
        subreg_map = {
            'eax': 'rax', 'ebx': 'rbx', 'ecx': 'rcx', 'edx': 'rdx',
            'esi': 'rsi', 'edi': 'rdi', 'ebp': 'rbp', 'esp': 'rsp',
            'r8d': 'r8', 'r9d': 'r9', 'r10d': 'r10', 'r11d': 'r11',
            'r12d': 'r12', 'r13d': 'r13', 'r14d': 'r14', 'r15d': 'r15'
        }
        return subreg_map.get(reg_lower)
    
    def clear(self):
        """Clear all register values"""
        self.registers.clear()
        self.register_widths.clear()
        self.temp_counter = 0


def normalize_register_name(reg: str) -> str:
    """Normalize register name (eax -> reg32, rax -> reg64)"""
    # For GAMBA, we'll use simple variable names
    # Map registers to variables: eax -> x, edx -> y, etc.
    # Handle subregisters: al/ah map to same base as ax/eax/rax
    reg_map = {
        # 64-bit registers
        'rax': 'x', 'rbx': 'y', 'rcx': 'z', 'rdx': 'w',
        'rsi': 'a', 'rdi': 'b', 'rbp': 'c', 'rsp': 'd',
        'r8': 'r8', 'r9': 'r9', 'r10': 'r10', 'r11': 'r11',
        'r12': 'r12', 'r13': 'r13', 'r14': 'r14', 'r15': 'r15',
        # 32-bit subregisters
        'eax': 'x', 'ebx': 'y', 'ecx': 'z', 'edx': 'w',
        'esi': 'a', 'edi': 'b', 'ebp': 'c', 'esp': 'd',
        'r8d': 'r8', 'r9d': 'r9', 'r10d': 'r10', 'r11d': 'r11',
        'r12d': 'r12', 'r13d': 'r13', 'r14d': 'r14', 'r15d': 'r15',
        # 16-bit subregisters
        'ax': 'x', 'bx': 'y', 'cx': 'z', 'dx': 'w',
        'si': 'a', 'di': 'b', 'bp': 'c', 'sp': 'd',
        'r8w': 'r8', 'r9w': 'r9', 'r10w': 'r10', 'r11w': 'r11',
        'r12w': 'r12', 'r13w': 'r13', 'r14w': 'r14', 'r15w': 'r15',
        # 8-bit subregisters
        'al': 'x', 'bl': 'y', 'cl': 'z', 'dl': 'w',
        'ah': 'x', 'bh': 'y', 'ch': 'z', 'dh': 'w',
        'sil': 'a', 'dil': 'b', 'bpl': 'c', 'spl': 'd',
        'r8b': 'r8', 'r9b': 'r9', 'r10b': 'r10', 'r11b': 'r11',
        'r12b': 'r12', 'r13b': 'r13', 'r14b': 'r14', 'r15b': 'r15',
    }
    
    reg_lower = reg.lower()
    return reg_map.get(reg_lower, reg_lower)


def get_register_width(reg: str) -> int:
    """Get register width in bits"""
    reg_lower = reg.lower()
    
    # 64-bit
    if reg_lower.startswith('r') and len(reg_lower) <= 3 and reg_lower[1:].isdigit():
        return 64
    if reg_lower in ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp']:
        return 64
    
    # 32-bit
    if reg_lower.endswith('d') and reg_lower[:-1].startswith('r') and reg_lower[1:-1].isdigit():
        return 32
    if reg_lower in ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp']:
        return 32
    
    # 16-bit
    if reg_lower.endswith('w') and reg_lower[:-1].startswith('r') and reg_lower[1:-1].isdigit():
        return 16
    if reg_lower in ['ax', 'bx', 'cx', 'dx', 'si', 'di', 'bp', 'sp']:
        return 16
    
    # 8-bit
    if reg_lower.endswith('b') and reg_lower[:-1].startswith('r') and reg_lower[1:-1].isdigit():
        return 8
    if reg_lower.endswith('l') and reg_lower[:-1] in ['si', 'di', 'bp', 'sp']:
        return 8
    if reg_lower in ['al', 'bl', 'cl', 'dl', 'ah', 'bh', 'ch', 'dh']:
        return 8
    
    return 64  # Default


def get_operand_expression(operand: str, tracker: RegisterTracker, default_width: int = 64) -> str:
    """Get expression for operand (register, constant, or memory)"""
    mem = parse_memory_operand(operand)
    if mem:
        if mem['base'] in ['rbp', 'ebp', 'rsp', 'esp']:
            offset = mem['offset']
            return f"stack_var_{hex(abs(offset))}"
        else:
            base_expr = tracker.get_register(normalize_register_name(mem['base'])) or mem['base'] if mem['base'] else '0'
            return f"mem[{base_expr} + {mem['offset']}]"
    
    const = extract_constant(operand)
    if const is not None:
        return str(const)
    
    reg = normalize_register_name(operand)
    return tracker.get_register(reg) or reg


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


def parse_memory_operand(operand: str) -> Optional[Dict]:
    """
    Parse memory operand like [rbp-0x10] or [rsp+0x20]
    
    Returns:
        Dict with 'base', 'offset', 'index', 'scale' or None if not memory
    """
    import re
    
    # Match [base+offset] or [base-offset] or [base+index*scale+offset]
    match = re.search(r'\[([^\]]+)\]', operand)
    if not match:
        return None
    
    inner = match.group(1)
    result = {'base': None, 'offset': 0, 'index': None, 'scale': 1}
    
    # Parse offset (hex or decimal)
    offset_match = re.search(r'([+-])\s*(0x[0-9a-fA-F]+|\d+)', inner)
    if offset_match:
        sign = offset_match.group(1)
        value_str = offset_match.group(2)
        if value_str.startswith('0x'):
            offset = int(value_str, 16)
        else:
            offset = int(value_str)
        if sign == '-':
            offset = -offset
        result['offset'] = offset
    
    # Parse base register
    base_match = re.search(r'\b(rbp|rbx|rsp|r12|r13|r14|r15|ebp|ebx|esp)\b', inner, re.IGNORECASE)
    if base_match:
        result['base'] = base_match.group(1).lower()
    
    # Parse index*scale
    index_match = re.search(r'(\w+)\s*\*\s*(\d+)', inner)
    if index_match:
        result['index'] = index_match.group(1).lower()
        result['scale'] = int(index_match.group(2))
    
    return result if result['base'] or result['index'] else None


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
            dst_width = get_register_width(inst.operands[0])
            
            # Check if source is memory
            mem = parse_memory_operand(src)
            if mem:
                # Map stack variable to symbolic name
                if mem['base'] in ['rbp', 'ebp', 'rsp', 'esp']:
                    offset = mem['offset']
                    var_name = f"stack_var_{hex(abs(offset))}"
                    tracker.set_register(dst, var_name, width=dst_width)
                    return var_name
                else:
                    # Other memory access - use generic memory expression
                    base_expr = tracker.get_register(normalize_register_name(mem['base'])) or mem['base'] if mem['base'] else '0'
                    expr = f"mem[{base_expr} + {mem['offset']}]"
                    tracker.set_register(dst, expr, width=dst_width)
                    return expr
            
            # Check if source is constant
            const = extract_constant(src)
            if const is not None:
                tracker.set_register(dst, str(const), width=dst_width)
            else:
                # Source is register
                src_reg = normalize_register_name(src)
                src_expr = tracker.get_register(src_reg)
                if src_expr:
                    tracker.set_register(dst, src_expr, width=dst_width)
                else:
                    tracker.set_register(dst, src_reg, width=dst_width)
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
            dst_width = get_register_width(inst.operands[0])
            
            dst_expr = tracker.get_register(dst) or dst
            src_expr = get_operand_expression(src, tracker, dst_width)
            
            expr = f"({dst_expr} * {src_expr})"
            tracker.set_register(dst, expr, width=dst_width)
            return expr
        return None
    
    elif mnemonic == 'and':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            dst_width = get_register_width(inst.operands[0])
            
            dst_expr = tracker.get_register(dst) or dst
            const = extract_constant(src)
            if const is not None:
                src_expr = hex(const)
            else:
                src_expr = get_operand_expression(src, dst_width)
            
            expr = f"({dst_expr} & {src_expr})"
            tracker.set_register(dst, expr, width=dst_width)
            return expr
    
    elif mnemonic == 'or':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            dst_width = get_register_width(inst.operands[0])
            
            dst_expr = tracker.get_register(dst) or dst
            const = extract_constant(src)
            if const is not None:
                src_expr = hex(const)
            else:
                src_expr = get_operand_expression(src, dst_width)
            
            expr = f"({dst_expr} | {src_expr})"
            tracker.set_register(dst, expr, width=dst_width)
            return expr
    
    elif mnemonic == 'xor':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            dst_width = get_register_width(inst.operands[0])
            
            dst_expr = tracker.get_register(dst) or dst
            const = extract_constant(src)
            if const is not None:
                src_expr = hex(const)
            else:
                src_expr = get_operand_expression(src, dst_width)
            
            expr = f"({dst_expr} ^ {src_expr})"
            tracker.set_register(dst, expr, width=dst_width)
            return expr
    
    elif mnemonic == 'not':
        if len(inst.operands) >= 1:
            dst = normalize_register_name(inst.operands[0])
            dst_width = get_register_width(inst.operands[0])
            dst_expr = tracker.get_register(dst) or dst
            
            expr = f"(~{dst_expr})"
            tracker.set_register(dst, expr, width=dst_width)
            return expr
    
    elif mnemonic == 'add':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            dst_width = get_register_width(inst.operands[0])
            
            dst_expr = tracker.get_register(dst) or dst
            src_expr = get_operand_expression(src, tracker, dst_width)
            
            expr = f"({dst_expr} + {src_expr})"
            tracker.set_register(dst, expr, width=dst_width)
            return expr
    
    elif mnemonic == 'sub':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            dst_width = get_register_width(inst.operands[0])
            
            dst_expr = tracker.get_register(dst) or dst
            src_expr = get_operand_expression(src, tracker, dst_width)
            
            expr = f"({dst_expr} - {src_expr})"
            tracker.set_register(dst, expr, width=dst_width)
            return expr
    
    # Shift instructions
    elif mnemonic in ['shl', 'sal']:  # shl and sal are the same
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            dst_width = get_register_width(inst.operands[0])
            
            dst_expr = tracker.get_register(dst) or dst
            const = extract_constant(src)
            if const is not None:
                shift_amount = const
            else:
                src_reg = normalize_register_name(src)
                src_expr = tracker.get_register(src_reg) or src_reg
                shift_amount = src_expr
            
            expr = f"({dst_expr} << {shift_amount})"
            tracker.set_register(dst, expr, width=dst_width)
            return expr
    
    elif mnemonic == 'shr':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            dst_width = get_register_width(inst.operands[0])
            
            dst_expr = tracker.get_register(dst) or dst
            const = extract_constant(src)
            if const is not None:
                shift_amount = const
            else:
                src_reg = normalize_register_name(src)
                src_expr = tracker.get_register(src_reg) or src_reg
                shift_amount = src_expr
            
            expr = f"({dst_expr} >> {shift_amount})"
            tracker.set_register(dst, expr, width=dst_width)
            return expr
    
    elif mnemonic == 'sar':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            dst_width = get_register_width(inst.operands[0])
            
            dst_expr = tracker.get_register(dst) or dst
            const = extract_constant(src)
            if const is not None:
                shift_amount = const
            else:
                src_reg = normalize_register_name(src)
                src_expr = tracker.get_register(src_reg) or src_reg
                shift_amount = src_expr
            
            # Arithmetic right shift (sign-extending)
            expr = f"({dst_expr} >> {shift_amount})"  # GAMBA will handle sign extension
            tracker.set_register(dst, expr, width=dst_width)
            return expr
    
    # Rotate instructions
    elif mnemonic == 'rol':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            dst_width = get_register_width(inst.operands[0])
            
            dst_expr = tracker.get_register(dst) or dst
            const = extract_constant(src)
            if const is not None:
                rotate_amount = const
            else:
                src_reg = normalize_register_name(src)
                src_expr = tracker.get_register(src_reg) or src_reg
                rotate_amount = src_expr
            
            # Rotate left: (x << n) | (x >> (width - n)) & mask
            # Simplified: rol(x, n) = (x << n) | (x >> (width - n))
            expr = f"(({dst_expr} << {rotate_amount}) | (({dst_expr} >> ({dst_width} - {rotate_amount})) & ((1 << {rotate_amount}) - 1)))"
            tracker.set_register(dst, expr, width=dst_width)
            return expr
    
    elif mnemonic == 'ror':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            dst_width = get_register_width(inst.operands[0])
            
            dst_expr = tracker.get_register(dst) or dst
            const = extract_constant(src)
            if const is not None:
                rotate_amount = const
            else:
                src_reg = normalize_register_name(src)
                src_expr = tracker.get_register(src_reg) or src_reg
                rotate_amount = src_expr
            
            # Rotate right: (x >> n) | (x << (width - n)) & mask
            expr = f"(({dst_expr} >> {rotate_amount}) | (({dst_expr} << ({dst_width} - {rotate_amount})) & (((1 << {dst_width}) - 1) ^ ((1 << ({dst_width} - {rotate_amount})) - 1))))"
            tracker.set_register(dst, expr, width=dst_width)
            return expr
    
    # Arithmetic sugar instructions
    elif mnemonic == 'neg':
        if len(inst.operands) >= 1:
            dst = normalize_register_name(inst.operands[0])
            dst_width = get_register_width(inst.operands[0])
            dst_expr = tracker.get_register(dst) or dst
            
            # neg x = 0 - x
            expr = f"(0 - {dst_expr})"
            tracker.set_register(dst, expr, width=dst_width)
            return expr
    
    elif mnemonic == 'inc':
        if len(inst.operands) >= 1:
            dst = normalize_register_name(inst.operands[0])
            dst_width = get_register_width(inst.operands[0])
            dst_expr = tracker.get_register(dst) or dst
            
            # inc x = x + 1
            expr = f"({dst_expr} + 1)"
            tracker.set_register(dst, expr, width=dst_width)
            return expr
    
    elif mnemonic == 'dec':
        if len(inst.operands) >= 1:
            dst = normalize_register_name(inst.operands[0])
            dst_width = get_register_width(inst.operands[0])
            dst_expr = tracker.get_register(dst) or dst
            
            # dec x = x - 1
            expr = f"({dst_expr} - 1)"
            tracker.set_register(dst, expr, width=dst_width)
            return expr
    
    # Comparison instructions
    elif mnemonic == 'cmp':
        # cmp a, b sets flags based on (a - b)
        # For MBA, we can track the comparison result
        if len(inst.operands) >= 2:
            left = inst.operands[0]
            right = inst.operands[1]
            
            left_expr = tracker.get_register(normalize_register_name(left)) or normalize_register_name(left)
            const = extract_constant(right)
            if const is not None:
                right_expr = str(const)
            else:
                right_expr = tracker.get_register(normalize_register_name(right)) or normalize_register_name(right)
            
            # Store comparison result in a temp variable for later use
            temp = tracker.get_temp_var()
            expr = f"({left_expr} - {right_expr})"
            tracker.set_register(temp, expr)
            return expr
    
    elif mnemonic == 'test':
        # test a, b sets flags based on (a & b)
        if len(inst.operands) >= 2:
            left = inst.operands[0]
            right = inst.operands[1]
            
            left_expr = tracker.get_register(normalize_register_name(left)) or normalize_register_name(left)
            const = extract_constant(right)
            if const is not None:
                right_expr = hex(const)
            else:
                right_expr = tracker.get_register(normalize_register_name(right)) or normalize_register_name(right)
            
            temp = tracker.get_temp_var()
            expr = f"({left_expr} & {right_expr})"
            tracker.set_register(temp, expr)
            return expr
    
    # Set condition codes (materialize boolean from flags)
    elif mnemonic.startswith('set'):
        # sete, setne, setl, setg, etc.
        if len(inst.operands) >= 1:
            dst = normalize_register_name(inst.operands[0])
            condition = mnemonic[3:]  # Extract condition (e, ne, l, g, etc.)
            
            # Map condition codes
            condition_map = {
                'e': 'eq', 'ne': 'ne', 'z': 'eq', 'nz': 'ne',
                'l': 'lt', 'le': 'le', 'g': 'gt', 'ge': 'ge',
                's': 'lt', 'ns': 'ge', 'o': 'overflow', 'no': 'no_overflow'
            }
            cond_expr = condition_map.get(condition, condition)
            
            # Materialize boolean: ite(condition, 1, 0)
            expr = f"ite({cond_expr}_flag, 1, 0)"
            tracker.set_register(dst, expr, width=8)  # setcc sets 8-bit register
            return expr
    
    # Conditional move
    elif mnemonic.startswith('cmov'):
        # cmove, cmovne, cmovl, etc.
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            dst_width = get_register_width(inst.operands[0])
            
            condition = mnemonic[4:]  # Extract condition (e, ne, l, g, etc.)
            dst_expr = tracker.get_register(dst) or dst
            src_expr = get_operand_expression(src, tracker, dst_width)
            
            # Conditional move: ite(condition, src, dst)
            # Map condition codes to GAMBA expressions
            condition_map = {
                'e': 'eq', 'ne': 'ne', 'z': 'eq', 'nz': 'ne',
                'l': 'lt', 'le': 'le', 'g': 'gt', 'ge': 'ge',
                's': 'lt', 'ns': 'ge', 'o': 'overflow', 'no': 'no_overflow'
            }
            cond_expr = condition_map.get(condition, condition)
            expr = f"ite({cond_expr}_flag, {src_expr}, {dst_expr})"
            tracker.set_register(dst, expr, width=dst_width)
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


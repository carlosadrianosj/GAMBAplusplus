#!/usr/bin/env python3
"""
ARM Expression Converter
Converts sequences of ARM assembly instructions to GAMBA-compatible mathematical expressions
"""

from typing import List, Dict, Optional, Set
from .parser import Instruction
from .detector import MBABlock


class RegisterTracker:
    """Tracks register values through ARM instruction sequences"""
    
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
    """Normalize ARM register name to GAMBA variable"""
    # Map ARM registers to variables: r0/x0 -> x, r1/x1 -> y, etc.
    import re
    
    # Extract register number
    match = re.match(r'([xwr])(\d+)', reg.lower())
    if match:
        reg_type = match.group(1)
        reg_num = int(match.group(2))
        
        # Map to simple variables
        var_map = ['x', 'y', 'z', 'w', 'a', 'b', 'c', 'd']
        if reg_num < len(var_map):
            return var_map[reg_num]
    
    # Special registers
    if reg.lower() in ['sp', 'r13']:
        return 'sp'
    if reg.lower() in ['lr', 'r14']:
        return 'lr'
    if reg.lower() in ['pc', 'r15']:
        return 'pc'
    
    return reg.lower()


def extract_constant(operand: str) -> Optional[int]:
    """Extract constant value from ARM operand"""
    import re
    
    # Try hex: #0x1234 or #1234
    hex_match = re.search(r'#0x([0-9a-fA-F]+)', operand, re.IGNORECASE)
    if hex_match:
        return int(hex_match.group(1), 16)
    
    # Try decimal: #123
    dec_match = re.search(r'#(\d+)', operand)
    if dec_match:
        return int(dec_match.group(1))
    
    # Try without # prefix
    hex_match = re.search(r'0x([0-9a-fA-F]+)', operand, re.IGNORECASE)
    if hex_match:
        return int(hex_match.group(1), 16)
    
    dec_match = re.search(r'\b(\d+)\b', operand)
    if dec_match:
        return int(dec_match.group(1))
    
    return None


def parse_shift_operation(operand: str) -> tuple:
    """
    Parse shift operation from operand (e.g., "lsl #2", "lsr x1")
    Returns: (shift_type, shift_amount_or_reg)
    """
    import re
    
    shift_patterns = {
        'lsl': 'lsl',
        'lsr': 'lsr',
        'asr': 'asr',
        'ror': 'ror',
    }
    
    for shift, name in shift_patterns.items():
        match = re.search(rf'{shift}\s+(.+)', operand, re.IGNORECASE)
        if match:
            shift_val = match.group(1).strip()
            const = extract_constant(shift_val)
            if const is not None:
                return (name, const)
            else:
                # Register shift
                reg = normalize_register_name(shift_val)
                return (name, reg)
    
    return (None, None)


def convert_instruction_to_expression(inst: Instruction, tracker: RegisterTracker) -> Optional[str]:
    """
    Convert a single ARM instruction to part of an expression.
    
    Returns:
        Expression string or None if instruction doesn't contribute to expression
    """
    mnemonic = inst.mnemonic
    
    if len(inst.operands) == 0:
        return None
    
    # Handle different instruction types
    
    # MOV: mov dst, src -> dst = src
    if mnemonic in ['mov', 'movz', 'movn', 'movk']:
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            
            const = extract_constant(src)
            if const is not None:
                tracker.set_register(dst, str(const))
            else:
                src_reg = normalize_register_name(src)
                src_expr = tracker.get_register(src_reg)
                if src_expr:
                    tracker.set_register(dst, src_expr)
                else:
                    tracker.set_register(dst, src_reg)
        return None
    
    # ADD: add dst, src1, src2 -> dst = src1 + src2
    # Supports embedded shifts: add x0, x1, x2, lsl #3
    elif mnemonic == 'add':
        if len(inst.operands) >= 3:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            
            # get_operand_expression handles embedded shifts automatically
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            
            expr = f"({src1_expr} + {src2_expr})"
            tracker.set_register(dst, expr)
            return expr
        elif len(inst.operands) >= 2:
            # add dst, src -> dst = dst + src
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            
            dst_expr = tracker.get_register(dst) or dst
            src_expr = get_operand_expression(src, tracker)
            
            expr = f"({dst_expr} + {src_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    # SUB: sub dst, src1, src2 -> dst = src1 - src2
    # Supports embedded shifts
    elif mnemonic == 'sub':
        if len(inst.operands) >= 3:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            
            expr = f"({src1_expr} - {src2_expr})"
            tracker.set_register(dst, expr)
            return expr
        elif len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            
            dst_expr = tracker.get_register(dst) or dst
            src_expr = get_operand_expression(src, tracker)
            
            expr = f"({dst_expr} - {src_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    # MUL: mul dst, src1, src2 -> dst = src1 * src2
    elif mnemonic == 'mul':
        if len(inst.operands) >= 3:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            
            expr = f"({src1_expr} * {src2_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    # MADD: madd dst, src1, src2, src3 -> dst = src1 * src2 + src3
    elif mnemonic == 'madd':
        if len(inst.operands) >= 4:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            src3 = inst.operands[3]
            
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            src3_expr = get_operand_expression(src3, tracker)
            
            expr = f"(({src1_expr} * {src2_expr}) + {src3_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    # MSUB: msub dst, src1, src2, src3 -> dst = src3 - src1 * src2
    elif mnemonic == 'msub':
        if len(inst.operands) >= 4:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            src3 = inst.operands[3]
            
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            src3_expr = get_operand_expression(src3, tracker)
            
            expr = f"({src3_expr} - ({src1_expr} * {src2_expr}))"
            tracker.set_register(dst, expr)
            return expr
    
    # AND: and dst, src1, src2 -> dst = src1 & src2
    elif mnemonic == 'and':
        if len(inst.operands) >= 3:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            
            expr = f"({src1_expr} & {src2_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    # ORR: orr dst, src1, src2 -> dst = src1 | src2
    elif mnemonic == 'orr':
        if len(inst.operands) >= 3:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            
            expr = f"({src1_expr} | {src2_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    # EOR: eor dst, src1, src2 -> dst = src1 ^ src2
    elif mnemonic == 'eor':
        if len(inst.operands) >= 3:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            
            expr = f"({src1_expr} ^ {src2_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    # BIC: bic dst, src1, src2 -> dst = src1 & ~src2
    elif mnemonic == 'bic':
        if len(inst.operands) >= 3:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            
            expr = f"({src1_expr} & (~{src2_expr}))"
            tracker.set_register(dst, expr)
            return expr
    
    # MVN: mvn dst, src -> dst = ~src
    elif mnemonic == 'mvn':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            
            src_expr = get_operand_expression(src, tracker)
            expr = f"(~{src_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    # EON: eon dst, src1, src2 -> dst = src1 ^ ~src2 (ARM64)
    elif mnemonic == 'eon':
        if len(inst.operands) >= 3:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            
            expr = f"({src1_expr} ^ (~{src2_expr}))"
            tracker.set_register(dst, expr)
            return expr
    
    # ORN: orn dst, src1, src2 -> dst = src1 | ~src2 (ARM64)
    elif mnemonic == 'orn':
        if len(inst.operands) >= 3:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            
            expr = f"({src1_expr} | (~{src2_expr}))"
            tracker.set_register(dst, expr)
            return expr
    
    # NEG: neg dst, src -> dst = -src
    elif mnemonic == 'neg':
        if len(inst.operands) >= 2:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            
            src_expr = get_operand_expression(src, tracker)
            expr = f"(-{src_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    # Bitfield instructions (AArch64)
    # UBFX: ubfx dst, src, lsb, width -> dst = (src >> lsb) & ((1 << width) - 1)
    elif mnemonic == 'ubfx':
        if len(inst.operands) >= 4:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            lsb = extract_constant(inst.operands[2])
            width = extract_constant(inst.operands[3])
            
            if lsb is not None and width is not None:
                src_expr = get_operand_expression(src, tracker)
                mask = (1 << width) - 1
                expr = f"(({src_expr} >> {lsb}) & {hex(mask)})"
                tracker.set_register(dst, expr)
                return expr
    
    # SBFX: sbfx dst, src, lsb, width -> signed bitfield extract
    elif mnemonic == 'sbfx':
        if len(inst.operands) >= 4:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            lsb = extract_constant(inst.operands[2])
            width = extract_constant(inst.operands[3])
            
            if lsb is not None and width is not None:
                src_expr = get_operand_expression(src, tracker)
                mask = (1 << width) - 1
                # Sign-extend: extract, shift left to sign bit, then arithmetic shift right
                expr = f"((({src_expr} >> {lsb}) & {hex(mask)}) << (32 - {lsb} - {width})) >> (32 - {lsb} - {width})"
                tracker.set_register(dst, expr)
                return expr
    
    # BFI: bfi dst, src, lsb, width -> bitfield insert
    elif mnemonic == 'bfi':
        if len(inst.operands) >= 4:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            lsb = extract_constant(inst.operands[2])
            width = extract_constant(inst.operands[3])
            
            if lsb is not None and width is not None:
                dst_expr = tracker.get_register(dst) or dst
                src_expr = get_operand_expression(src, tracker)
                mask = ((1 << width) - 1) << lsb
                # Clear bits in dst, then insert src
                expr = f"(({dst_expr} & {hex(~mask)}) | (({src_expr} & {hex((1 << width) - 1)}) << {lsb}))"
                tracker.set_register(dst, expr)
                return expr
    
    # BFXIL: bfxil dst, src, lsb, width -> bitfield extract and insert low
    elif mnemonic == 'bfxil':
        if len(inst.operands) >= 4:
            dst = normalize_register_name(inst.operands[0])
            src = inst.operands[1]
            lsb = extract_constant(inst.operands[2])
            width = extract_constant(inst.operands[3])
            
            if lsb is not None and width is not None:
                dst_expr = tracker.get_register(dst) or dst
                src_expr = get_operand_expression(src, tracker)
                mask = (1 << width) - 1
                # Extract from src and insert into low bits of dst
                expr = f"(({dst_expr} & {hex(~mask)}) | (({src_expr} >> {lsb}) & {hex(mask)}))"
                tracker.set_register(dst, expr)
                return expr
    
    # EXTR: extr dst, src1, src2, lsb -> extract from concatenated src1:src2
    elif mnemonic == 'extr':
        if len(inst.operands) >= 4:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            lsb = extract_constant(inst.operands[3])
            
            if lsb is not None:
                src1_expr = get_operand_expression(src1, tracker)
                src2_expr = get_operand_expression(src2, tracker)
                # Extract from concatenated register: (src1 << 32) | src2, then shift right
                expr = f"((({src1_expr} << 32) | {src2_expr}) >> {lsb})"
                tracker.set_register(dst, expr)
                return expr
    
    # Conditional selection (AArch64)
    # CSEL: csel dst, src1, src2, cond -> dst = cond ? src1 : src2
    elif mnemonic == 'csel':
        if len(inst.operands) >= 4:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            cond = inst.operands[3].lower()  # Condition code (eq, ne, lt, gt, etc.)
            
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            
            # Map ARM condition codes to GAMBA expressions
            cond_map = {
                'eq': 'eq', 'ne': 'ne', 'cs': 'ge', 'cc': 'lt',
                'mi': 'lt', 'pl': 'ge', 'vs': 'overflow', 'vc': 'no_overflow',
                'hi': 'gt', 'ls': 'le', 'ge': 'ge', 'lt': 'lt',
                'gt': 'gt', 'le': 'le', 'al': 'true', 'nv': 'false'
            }
            cond_expr = cond_map.get(cond, cond)
            expr = f"ite({cond_expr}_flag, {src1_expr}, {src2_expr})"
            tracker.set_register(dst, expr)
            return expr
    
    # CSINC: csinc dst, src1, src2, cond -> dst = cond ? src1 : src2 + 1
    elif mnemonic == 'csinc':
        if len(inst.operands) >= 4:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            cond = inst.operands[3].lower()
            
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            
            cond_map = {
                'eq': 'eq', 'ne': 'ne', 'cs': 'ge', 'cc': 'lt',
                'mi': 'lt', 'pl': 'ge', 'vs': 'overflow', 'vc': 'no_overflow',
                'hi': 'gt', 'ls': 'le', 'ge': 'ge', 'lt': 'lt',
                'gt': 'gt', 'le': 'le', 'al': 'true', 'nv': 'false'
            }
            cond_expr = cond_map.get(cond, cond)
            expr = f"ite({cond_expr}_flag, {src1_expr}, ({src2_expr} + 1))"
            tracker.set_register(dst, expr)
            return expr
    
    # CSINV: csinv dst, src1, src2, cond -> dst = cond ? src1 : ~src2
    elif mnemonic == 'csinv':
        if len(inst.operands) >= 4:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            cond = inst.operands[3].lower()
            
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            
            cond_map = {
                'eq': 'eq', 'ne': 'ne', 'cs': 'ge', 'cc': 'lt',
                'mi': 'lt', 'pl': 'ge', 'vs': 'overflow', 'vc': 'no_overflow',
                'hi': 'gt', 'ls': 'le', 'ge': 'ge', 'lt': 'lt',
                'gt': 'gt', 'le': 'le', 'al': 'true', 'nv': 'false'
            }
            cond_expr = cond_map.get(cond, cond)
            expr = f"ite({cond_expr}_flag, {src1_expr}, (~{src2_expr}))"
            tracker.set_register(dst, expr)
            return expr
    
    # CSNEG: csneg dst, src1, src2, cond -> dst = cond ? src1 : -src2
    elif mnemonic == 'csneg':
        if len(inst.operands) >= 4:
            dst = normalize_register_name(inst.operands[0])
            src1 = inst.operands[1]
            src2 = inst.operands[2]
            cond = inst.operands[3].lower()
            
            src1_expr = get_operand_expression(src1, tracker)
            src2_expr = get_operand_expression(src2, tracker)
            
            cond_map = {
                'eq': 'eq', 'ne': 'ne', 'cs': 'ge', 'cc': 'lt',
                'mi': 'lt', 'pl': 'ge', 'vs': 'overflow', 'vc': 'no_overflow',
                'hi': 'gt', 'ls': 'le', 'ge': 'ge', 'lt': 'lt',
                'gt': 'gt', 'le': 'le', 'al': 'true', 'nv': 'false'
            }
            cond_expr = cond_map.get(cond, cond)
            expr = f"ite({cond_expr}_flag, {src1_expr}, (-{src2_expr}))"
            tracker.set_register(dst, expr)
            return expr
    
    return None


def get_operand_expression(operand: str, tracker: RegisterTracker) -> str:
    """Get expression for an operand (register, constant, or shifted register)"""
    # Check for embedded shift operations (e.g., "x1, lsl #3" or "x2, lsl x3")
    shift_type, shift_val = parse_shift_operation(operand)
    
    if shift_type:
        # Extract base register (before the shift)
        import re
        # Match register at start: x0, r1, w2, etc.
        base_match = re.match(r'([xwr]\d+|[a-z]+)', operand.lower())
        if base_match:
            base_reg = normalize_register_name(base_match.group(1))
            base_expr = tracker.get_register(base_reg) or base_reg
            
            if isinstance(shift_val, int):
                # Constant shift
                if shift_type == 'lsl':
                    return f"({base_expr} << {shift_val})"
                elif shift_type == 'lsr':
                    return f"({base_expr} >> {shift_val})"
                elif shift_type == 'asr':
                    # Arithmetic shift right (sign-extending)
                    return f"({base_expr} >> {shift_val})"
                elif shift_type == 'ror':
                    # Rotate right - simplified for now
                    return f"(({base_expr} >> {shift_val}) | (({base_expr} << (32 - {shift_val})) & (((1 << 32) - 1) ^ ((1 << (32 - {shift_val})) - 1))))"
                return base_expr
            else:
                # Register shift
                shift_expr = tracker.get_register(shift_val) or shift_val
                if shift_type == 'lsl':
                    return f"({base_expr} << {shift_expr})"
                elif shift_type == 'lsr':
                    return f"({base_expr} >> {shift_expr})"
                elif shift_type == 'asr':
                    return f"({base_expr} >> {shift_expr})"
                elif shift_type == 'ror':
                    return f"(({base_expr} >> {shift_expr}) | (({base_expr} << (32 - {shift_expr})) & (((1 << 32) - 1) ^ ((1 << (32 - {shift_expr})) - 1))))"
                return base_expr
    
    # Check for constant
    const = extract_constant(operand)
    if const is not None:
        return str(const)
    
    # Check for register
    reg = normalize_register_name(operand)
    reg_expr = tracker.get_register(reg)
    if reg_expr:
        return reg_expr
    
    return reg


def convert_mba_block_to_expression(block: MBABlock) -> Optional[Dict]:
    """
    Convert ARM MBA block to GAMBA expression.
    
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
            if inst.mnemonic in ['mov', 'movz', 'movn', 'movk', 'add', 'sub', 'mul', 'madd', 'msub', 
                                'and', 'orr', 'eor', 'bic', 'mvn', 'eon', 'orn', 'neg',
                                'ubfx', 'sbfx', 'bfi', 'bfxil', 'extr',
                                'csel', 'csinc', 'csinv', 'csneg']:
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
    
    # ARM32/ARM64 register patterns
    reg_patterns = [
        r'\b([xwr]\d+|sp|lr|pc)\b',
    ]
    
    registers = []
    for pattern in reg_patterns:
        matches = re.findall(pattern, operand, re.IGNORECASE)
        registers.extend(matches)
    
    return registers


if __name__ == "__main__":
    import sys
    from .parser import parse_assembly
    from .detector import detect_mba_blocks
    
    if len(sys.argv) < 2:
        print("Usage: python converter.py <assembly_file.asm> [arm32|arm64]")
        sys.exit(1)
    
    asm_file = sys.argv[1]
    arch = sys.argv[2] if len(sys.argv) > 2 else "arm64"
    
    try:
        parsed = parse_assembly(asm_file, arch=arch)
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


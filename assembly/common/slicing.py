#!/usr/bin/env python3
"""
Dataflow Slicing
Implements backward slicing and mini-SSA conversion for MBA blocks
"""

from typing import List, Dict, Set, Optional
from collections import defaultdict


class DataflowSlice:
    """Represents a dataflow slice from a sink register"""
    
    def __init__(self, sink_register: str, instructions: List, variables: Set[str]):
        self.sink_register = sink_register
        self.instructions = instructions
        self.variables = variables
        self.input_variables = variables.copy()


def build_dependency_graph(instructions: List) -> Dict[str, Set[str]]:
    """
    Build dependency graph: register -> set of registers it depends on
    
    Args:
        instructions: List of instruction objects with mnemonic and operands
    
    Returns:
        Dictionary mapping registers to their dependencies
    """
    dependencies = defaultdict(set)
    defined_regs = set()
    
    for inst in instructions:
        if not hasattr(inst, 'mnemonic') or not hasattr(inst, 'operands'):
            continue
        
        mnemonic = inst.mnemonic
        operands = inst.operands
        
        if len(operands) == 0:
            continue
        
        # Get destination register (first operand for most instructions)
        dst_reg = None
        if mnemonic in ['mov', 'add', 'sub', 'mul', 'imul', 'and', 'or', 'xor', 
                       'shl', 'shr', 'sar', 'rol', 'ror', 'lea', 'neg', 'inc', 'dec',
                       'sete', 'setne', 'setl', 'setg', 'cmove', 'cmovne']:
            if len(operands) > 0:
                dst_reg = operands[0].lower()
        
        # Get source registers (all other operands)
        src_regs = set()
        for i, op in enumerate(operands):
            if i == 0 and dst_reg:
                continue  # Skip destination
            
            # Extract register names from operand
            regs = extract_registers_from_operand(op)
            src_regs.update(regs)
        
        # Add dependencies
        if dst_reg:
            dependencies[dst_reg] = src_regs.copy()
            defined_regs.add(dst_reg)
    
    return dict(dependencies)


def extract_registers_from_operand(operand: str) -> List[str]:
    """Extract register names from operand string"""
    import re
    
    # Common patterns for registers
    reg_patterns = [
        r'\b(rax|eax|ax|al|rbx|ebx|bx|bl|rcx|ecx|cx|cl|rdx|edx|dx|dl)\b',
        r'\b(rsi|esi|si|rdi|edi|di|rbp|ebp|bp|rsp|esp|sp)\b',
        r'\b(r\d+|r\d+d|r\d+w|r\d+b)\b',  # x86-64 extended registers
        r'\b([xwr]\d+|sp|lr|pc)\b',  # ARM registers
    ]
    
    registers = []
    for pattern in reg_patterns:
        matches = re.findall(pattern, operand, re.IGNORECASE)
        registers.extend(matches)
    
    return registers


def backward_slice(sink_register: str, instructions: List, 
                   dependencies: Dict[str, Set[str]]) -> DataflowSlice:
    """
    Perform backward slice from sink register.
    
    Args:
        sink_register: Target register to slice from
        instructions: List of instructions
        dependencies: Dependency graph
    
    Returns:
        DataflowSlice containing relevant instructions and variables
    """
    relevant_regs = {sink_register.lower()}
    relevant_instructions = []
    worklist = [sink_register.lower()]
    
    # Backward traversal
    while worklist:
        reg = worklist.pop()
        
        # Find instructions that define this register
        for inst in reversed(instructions):
            if not hasattr(inst, 'mnemonic') or not hasattr(inst, 'operands'):
                continue
            
            if len(inst.operands) == 0:
                continue
            
            dst_reg = inst.operands[0].lower()
            
            if dst_reg == reg:
                # This instruction defines the register
                if inst not in relevant_instructions:
                    relevant_instructions.insert(0, inst)
                
                # Add source registers to worklist
                if reg in dependencies:
                    for src_reg in dependencies[reg]:
                        if src_reg not in relevant_regs:
                            relevant_regs.add(src_reg)
                            worklist.append(src_reg)
                break
    
    # Sort instructions by original order
    relevant_instructions.sort(key=lambda x: getattr(x, 'address', 0))
    
    return DataflowSlice(
        sink_register=sink_register,
        instructions=relevant_instructions,
        variables=relevant_regs
    )


def convert_to_mini_ssa(instructions: List) -> List[Dict]:
    """
    Convert instructions to mini-SSA form within a basic block.
    
    Each register definition gets a unique version number.
    
    Args:
        instructions: List of instructions
    
    Returns:
        List of SSA instructions with versioned registers
    """
    ssa_instructions = []
    register_versions = defaultdict(int)
    register_map = {}  # original -> current version
    
    for inst in instructions:
        if not hasattr(inst, 'mnemonic') or not hasattr(inst, 'operands'):
            continue
        
        mnemonic = inst.mnemonic
        operands = inst.operands
        
        if len(operands) == 0:
            continue
        
        # Process operands
        ssa_operands = []
        dst_reg = None
        
        for i, op in enumerate(operands):
            # Extract registers from operand
            regs = extract_registers_from_operand(op)
            
            if i == 0 and mnemonic in ['mov', 'add', 'sub', 'mul', 'imul', 
                                      'and', 'or', 'xor', 'shl', 'shr', 'sar']:
                # Destination register
                dst_reg = op.lower()
                # Increment version for new definition
                register_versions[dst_reg] += 1
                register_map[dst_reg] = register_versions[dst_reg]
                ssa_operands.append(f"{dst_reg}.{register_versions[dst_reg]}")
            else:
                # Source operands - use current version
                ssa_op = op
                for reg in regs:
                    version = register_map.get(reg.lower(), 0)
                    if version > 0:
                        ssa_op = ssa_op.replace(reg, f"{reg}.{version}")
                ssa_operands.append(ssa_op)
        
        ssa_instructions.append({
            'mnemonic': mnemonic,
            'operands': ssa_operands,
            'original': inst
        })
    
    return ssa_instructions


def slice_mba_block(instructions: List, sink_register: Optional[str] = None) -> DataflowSlice:
    """
    Slice MBA block to reduce variable count.
    
    Args:
        instructions: List of instructions in MBA block
        sink_register: Target register (if None, use last defined register)
    
    Returns:
        DataflowSlice with reduced instruction set
    """
    # Build dependency graph
    dependencies = build_dependency_graph(instructions)
    
    # Find sink register if not provided
    if sink_register is None:
        # Use last register that was defined
        for inst in reversed(instructions):
            if hasattr(inst, 'operands') and len(inst.operands) > 0:
                sink_register = inst.operands[0].lower()
                break
    
    if sink_register is None:
        # No sink found, return all instructions
        return DataflowSlice(
            sink_register="unknown",
            instructions=instructions,
            variables=set()
        )
    
    # Perform backward slice
    slice_result = backward_slice(sink_register, instructions, dependencies)
    
    return slice_result


if __name__ == "__main__":
    # Test slicing
    class TestInst:
        def __init__(self, mnemonic, operands, address=0):
            self.mnemonic = mnemonic
            self.operands = operands
            self.address = address
    
    test_instructions = [
        TestInst('mov', ['eax', 'ebx'], 0x1000),
        TestInst('add', ['ecx', 'eax', 'edx'], 0x1001),
        TestInst('xor', ['eax', 'ecx'], 0x1002),
        TestInst('mov', ['edx', '0x10'], 0x1003),
    ]
    
    slice_result = slice_mba_block(test_instructions, 'eax')
    print(f"Sink: {slice_result.sink_register}")
    print(f"Variables: {slice_result.variables}")
    print(f"Instructions: {len(slice_result.instructions)}")
    
    ssa = convert_to_mini_ssa(slice_result.instructions)
    print(f"\nSSA form:")
    for ssa_inst in ssa:
        print(f"  {ssa_inst['mnemonic']} {', '.join(ssa_inst['operands'])}")


#!/usr/bin/env python3
"""
Control Flow Graph (CFG) Construction
Builds CFG from instructions using Capstone for accurate decoding
"""

from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from .capstone_wrapper import CapstoneDecoder, create_decoder, CapstoneInstruction


class BasicBlock:
    """Represents a basic block in the CFG"""
    
    def __init__(self, start_address: int, instructions: List):
        self.start_address = start_address
        self.instructions = instructions
        self.end_address = start_address + sum(inst.size for inst in instructions) if instructions else start_address
        self.predecessors: List['BasicBlock'] = []
        self.successors: List['BasicBlock'] = []
        self.mba_blocks = []  # MBA blocks associated with this basic block
    
    def add_predecessor(self, block: 'BasicBlock'):
        """Add predecessor block"""
        if block not in self.predecessors:
            self.predecessors.append(block)
    
    def add_successor(self, block: 'BasicBlock'):
        """Add successor block"""
        if block not in self.successors:
            self.successors.append(block)
    
    def __repr__(self):
        return f"BasicBlock(0x{self.start_address:X}-0x{self.end_address:X}, {len(self.instructions)} insts)"


class CFG:
    """Control Flow Graph"""
    
    def __init__(self, entry_block: BasicBlock):
        self.entry_block = entry_block
        self.blocks: List[BasicBlock] = [entry_block]
        self.address_to_block: Dict[int, BasicBlock] = {entry_block.start_address: entry_block}
    
    def add_block(self, block: BasicBlock):
        """Add block to CFG"""
        if block not in self.blocks:
            self.blocks.append(block)
            self.address_to_block[block.start_address] = block
    
    def get_block_at_address(self, address: int) -> Optional[BasicBlock]:
        """Get block containing address"""
        for block in self.blocks:
            if block.start_address <= address < block.end_address:
                return block
        return None
    
    def get_block_by_start(self, address: int) -> Optional[BasicBlock]:
        """Get block by start address"""
        return self.address_to_block.get(address)


class CFGBuilder:
    """Builds CFG from instructions"""
    
    def __init__(self, arch: str = 'x86_64'):
        self.decoder = create_decoder(arch)
        self.arch = arch
    
    def build_cfg_from_bytes(self, code: bytes, base_address: int = 0) -> CFG:
        """
        Build CFG from binary code.
        
        Args:
            code: Binary code bytes
            base_address: Base address
        
        Returns:
            CFG object
        """
        # Decode all instructions
        instructions = self.decoder.decode_bytes(code, base_address)
        
        return self.build_cfg_from_instructions(instructions)
    
    def build_cfg_from_instructions(self, instructions: List[CapstoneInstruction]) -> CFG:
        """
        Build CFG from decoded instructions.
        
        Args:
            instructions: List of CapstoneInstruction objects
        
        Returns:
            CFG object
        """
        if not instructions:
            # Empty CFG
            empty_block = BasicBlock(0, [])
            return CFG(empty_block)
        
        # Identify basic block boundaries
        # Entry points: first instruction, targets of branches
        entry_points = {instructions[0].address}
        branch_targets = set()
        
        for i, inst in enumerate(instructions):
            if inst.is_branch and inst.branch_target:
                branch_targets.add(inst.branch_target)
            
            # Next instruction after branch is also an entry point (if not taken)
            if inst.is_branch and i + 1 < len(instructions):
                entry_points.add(instructions[i + 1].address)
        
        entry_points.update(branch_targets)
        
        # Build basic blocks
        blocks = []
        current_block_insts = []
        current_start = instructions[0].address
        
        for i, inst in enumerate(instructions):
            # Check if this is an entry point
            if inst.address in entry_points and current_block_insts:
                # End current block
                block = BasicBlock(current_start, current_block_insts)
                blocks.append(block)
                current_block_insts = [inst]
                current_start = inst.address
            else:
                current_block_insts.append(inst)
            
            # Check if this instruction ends the block
            if inst.is_branch or inst.is_return:
                # End current block
                if current_block_insts:
                    block = BasicBlock(current_start, current_block_insts)
                    blocks.append(block)
                    current_block_insts = []
                    if i + 1 < len(instructions):
                        current_start = instructions[i + 1].address
        
        # Add last block if any
        if current_block_insts:
            block = BasicBlock(current_start, current_block_insts)
            blocks.append(block)
        
        # Build edges
        for block in blocks:
            if not block.instructions:
                continue
            
            last_inst = block.instructions[-1]
            
            if last_inst.is_branch:
                # Add edge to branch target
                if last_inst.branch_target:
                    target_block = self._find_block_at_address(blocks, last_inst.branch_target)
                    if target_block:
                        block.add_successor(target_block)
                        target_block.add_predecessor(block)
                
                # Add fallthrough edge (if not return)
                if not last_inst.is_return and not last_inst.is_call:
                    fallthrough_addr = last_inst.address + last_inst.size
                    fallthrough_block = self._find_block_at_address(blocks, fallthrough_addr)
                    if fallthrough_block:
                        block.add_successor(fallthrough_block)
                        fallthrough_block.add_predecessor(block)
            elif not last_inst.is_return:
                # Fallthrough
                fallthrough_addr = last_inst.address + last_inst.size
                fallthrough_block = self._find_block_at_address(blocks, fallthrough_addr)
                if fallthrough_block:
                    block.add_successor(fallthrough_block)
                    fallthrough_block.add_predecessor(block)
        
        # Create CFG
        if blocks:
            cfg = CFG(blocks[0])
            for block in blocks[1:]:
                cfg.add_block(block)
            return cfg
        else:
            empty_block = BasicBlock(0, [])
            return CFG(empty_block)
    
    def _find_block_at_address(self, blocks: List[BasicBlock], address: int) -> Optional[BasicBlock]:
        """Find block containing address"""
        for block in blocks:
            if block.start_address <= address < block.end_address:
                return block
        return None
    
    def build_cfg_from_assembly_file(self, asm_file_path: str, arch: str = 'x86_64') -> CFG:
        """
        Build CFG from assembly file (requires parsing first).
        
        Note: This is a placeholder - in practice, you'd parse the assembly
        file and convert to binary, or use objdump output.
        
        Args:
            asm_file_path: Path to assembly file
            arch: Architecture name
        
        Returns:
            CFG object
        """
        # This would require parsing assembly and converting to binary
        # For now, return empty CFG
        empty_block = BasicBlock(0, [])
        return CFG(empty_block)


def build_cfg(instructions: List, arch: str = 'x86_64') -> CFG:
    """
    Convenience function to build CFG from instructions.
    
    Args:
        instructions: List of instruction objects (CapstoneInstruction or custom)
        arch: Architecture name
    
    Returns:
        CFG object
    """
    builder = CFGBuilder(arch)
    
    # Convert instructions to CapstoneInstruction if needed
    if instructions and not isinstance(instructions[0], CapstoneInstruction):
        # Assume instructions have address, mnemonic, operands, size
        # Convert to CapstoneInstruction-like objects
        cs_instructions = []
        for inst in instructions:
            # Create minimal CapstoneInstruction wrapper
            class MockCSInst:
                def __init__(self, inst):
                    self.address = getattr(inst, 'address', 0)
                    self.mnemonic = getattr(inst, 'mnemonic', '')
                    self.op_str = ', '.join(getattr(inst, 'operands', []))
                    self.size = getattr(inst, 'size', 4)
                    self.is_branch = self._check_branch(inst)
                    self.is_call = self._check_call(inst)
                    self.is_return = self._check_return(inst)
                    self.branch_target = None
                
                def _check_branch(self, inst):
                    mnemonic = getattr(inst, 'mnemonic', '').lower()
                    return mnemonic.startswith('j') or mnemonic in ['call', 'ret', 'retn']
                
                def _check_call(self, inst):
                    return getattr(inst, 'mnemonic', '').lower() == 'call'
                
                def _check_return(self, inst):
                    mnemonic = getattr(inst, 'mnemonic', '').lower()
                    return mnemonic in ['ret', 'retn']
            
            cs_instructions.append(MockCSInst(inst))
        
        instructions = cs_instructions
    
    return builder.build_cfg_from_instructions(instructions)


if __name__ == "__main__":
    # Test CFG construction
    from capstone_wrapper import create_decoder
    
    decoder = create_decoder('x86_64')
    
    # Simple code: mov eax, 1; test eax, eax; jne label; mov eax, 2; label: ret
    code = b'\xb8\x01\x00\x00\x00\x85\xc0\x75\x02\xb8\x02\x00\x00\x00\xc3'
    
    instructions = decoder.decode_bytes(code, 0x1000)
    print(f"Decoded {len(instructions)} instructions")
    
    builder = CFGBuilder('x86_64')
    cfg = builder.build_cfg_from_instructions(instructions)
    
    print(f"\nCFG has {len(cfg.blocks)} basic blocks:")
    for block in cfg.blocks:
        print(f"  {block}")
        print(f"    Predecessors: {len(block.predecessors)}")
        print(f"    Successors: {len(block.successors)}")


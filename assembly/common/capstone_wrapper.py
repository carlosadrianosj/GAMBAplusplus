#!/usr/bin/env python3
"""
Capstone Wrapper
Provides unified interface for instruction decoding across architectures
"""

try:
    import capstone as cs
    CAPSTONE_AVAILABLE = True
except ImportError:
    CAPSTONE_AVAILABLE = False
    cs = None

from typing import List, Dict, Optional, Tuple
from enum import Enum


class Architecture(Enum):
    """Supported architectures"""
    X86_32 = "x86_32"
    X86_64 = "x86_64"
    ARM_32 = "arm_32"
    ARM_64 = "arm_64"
    ARM_THUMB = "arm_thumb"
    MIPS_32 = "mips_32"
    MIPS_64 = "mips_64"
    POWERPC_32 = "powerpc_32"
    POWERPC_64 = "powerpc_64"
    RISCV_32 = "riscv_32"
    RISCV_64 = "riscv_64"


class CapstoneInstruction:
    """Wrapper for Capstone instruction"""
    
    def __init__(self, cs_insn):
        self.address = cs_insn.address
        self.mnemonic = cs_insn.mnemonic
        self.op_str = cs_insn.op_str
        self.bytes = bytes(cs_insn.bytes)
        self.size = cs_insn.size
        self.operands = self._parse_operands()
        self.is_branch = cs_insn.group(cs.CS_GRP_JUMP) or cs_insn.group(cs.CS_GRP_CALL)
        self.is_call = cs_insn.group(cs.CS_GRP_CALL)
        self.is_return = cs_insn.group(cs.CS_GRP_RET)
        self.branch_target = self._get_branch_target(cs_insn)
    
    def _parse_operands(self) -> List[str]:
        """Parse operands from op_str"""
        if not self.op_str:
            return []
        
        # Split by comma, handle brackets
        operands = []
        current = ""
        bracket_depth = 0
        
        for char in self.op_str:
            if char == '[':
                bracket_depth += 1
                current += char
            elif char == ']':
                bracket_depth -= 1
                current += char
            elif char == ',' and bracket_depth == 0:
                if current.strip():
                    operands.append(current.strip())
                current = ""
            else:
                current += char
        
        if current.strip():
            operands.append(current.strip())
        
        return operands
    
    def _get_branch_target(self, cs_insn) -> Optional[int]:
        """Extract branch target address"""
        if not self.is_branch:
            return None
        
        # Get operands
        if cs_insn.operands:
            for op in cs_insn.operands:
                if op.type == cs.CS_OP_IMM:
                    return op.value.imm
                elif op.type == cs.CS_OP_REG:
                    # Indirect jump - return None
                    return None
        
        return None
    
    def __repr__(self):
        return f"CapstoneInstruction(0x{self.address:X}: {self.mnemonic} {self.op_str})"


class CapstoneDecoder:
    """Capstone-based instruction decoder"""
    
    def __init__(self, arch: Architecture, mode: int = None):
        if not CAPSTONE_AVAILABLE:
            raise ImportError("Capstone is not available. Install with: pip install capstone")
        
        self.arch = arch
        self.cs_arch, self.cs_mode = self._get_capstone_config(arch, mode)
        self.md = cs.Cs(self.cs_arch, self.cs_mode)
        self.md.detail = True  # Enable detailed instruction information
    
    def _get_capstone_config(self, arch: Architecture, mode: Optional[int]) -> Tuple[int, int]:
        """Get Capstone architecture and mode constants"""
        arch_map = {
            Architecture.X86_32: (cs.CS_ARCH_X86, cs.CS_MODE_32),
            Architecture.X86_64: (cs.CS_ARCH_X86, cs.CS_MODE_64),
            Architecture.ARM_32: (cs.CS_ARCH_ARM, cs.CS_MODE_ARM),
            Architecture.ARM_64: (cs.CS_ARCH_ARM64, cs.CS_MODE_ARM),
            Architecture.ARM_THUMB: (cs.CS_ARCH_ARM, cs.CS_MODE_THUMB),
            Architecture.MIPS_32: (cs.CS_ARCH_MIPS, cs.CS_MODE_MIPS32),
            Architecture.MIPS_64: (cs.CS_ARCH_MIPS, cs.CS_MODE_MIPS64),
            Architecture.POWERPC_32: (cs.CS_ARCH_PPC, cs.CS_MODE_32),
            Architecture.POWERPC_64: (cs.CS_ARCH_PPC, cs.CS_MODE_64),
            Architecture.RISCV_32: (cs.CS_ARCH_RISCV, cs.CS_MODE_RISCV32),
            Architecture.RISCV_64: (cs.CS_ARCH_RISCV, cs.CS_MODE_RISCV64),
        }
        
        if arch not in arch_map:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        cs_arch, default_mode = arch_map[arch]
        cs_mode = mode if mode is not None else default_mode
        
        return cs_arch, cs_mode
    
    def decode_bytes(self, code: bytes, base_address: int = 0) -> List[CapstoneInstruction]:
        """
        Decode bytes to instructions.
        
        Args:
            code: Binary code bytes
            base_address: Base address for instructions
        
        Returns:
            List of CapstoneInstruction objects
        """
        instructions = []
        
        for insn in self.md.disasm(code, base_address):
            instructions.append(CapstoneInstruction(insn))
        
        return instructions
    
    def decode_instruction(self, code: bytes, address: int = 0) -> Optional[CapstoneInstruction]:
        """Decode a single instruction"""
        instructions = self.decode_bytes(code, address)
        return instructions[0] if instructions else None


def create_decoder(arch_name: str, mode: Optional[str] = None) -> CapstoneDecoder:
    """
    Create decoder for architecture.
    
    Args:
        arch_name: Architecture name (x86_64, arm64, mips32, etc.)
        mode: Optional mode (thumb, etc.)
    
    Returns:
        CapstoneDecoder instance
    """
    arch_map = {
        'x86_32': Architecture.X86_32,
        'x86_64': Architecture.X86_64,
        'x86': Architecture.X86_64,
        'arm32': Architecture.ARM_32,
        'arm64': Architecture.ARM_64,
        'arm': Architecture.ARM_64,
        'thumb': Architecture.ARM_THUMB,
        'mips32': Architecture.MIPS_32,
        'mips64': Architecture.MIPS_64,
        'mips': Architecture.MIPS_32,
        'powerpc32': Architecture.POWERPC_32,
        'powerpc64': Architecture.POWERPC_64,
        'powerpc': Architecture.POWERPC_64,
        'ppc32': Architecture.POWERPC_32,
        'ppc64': Architecture.POWERPC_64,
        'riscv32': Architecture.RISCV_32,
        'riscv64': Architecture.RISCV_64,
        'riscv': Architecture.RISCV_64,
    }
    
    arch = arch_map.get(arch_name.lower())
    if not arch:
        raise ValueError(f"Unknown architecture: {arch_name}")
    
    return CapstoneDecoder(arch)


if __name__ == "__main__":
    import sys
    
    if not CAPSTONE_AVAILABLE:
        print("Capstone not available. Install with: pip install capstone")
        sys.exit(1)
    
    # Test decoding
    decoder = create_decoder('x86_64')
    
    # x86-64: mov eax, ebx; add eax, 0x10
    code = b'\x89\xd8\x83\xc0\x10'
    
    instructions = decoder.decode_bytes(code, 0x1000)
    print(f"Decoded {len(instructions)} instructions:")
    for insn in instructions:
        print(f"  0x{insn.address:X}: {insn.mnemonic} {insn.op_str}")


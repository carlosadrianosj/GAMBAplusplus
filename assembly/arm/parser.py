#!/usr/bin/env python3
"""
ARM Assembly Parser for GAMBA MBA Automation
Parses ARM32/ARM64 assembly files extracted from IDA into structured instruction data
"""

import re
from pathlib import Path
from typing import List, Dict, Optional


class Instruction:
    """Represents a single ARM assembly instruction"""
    
    def __init__(self, address: int, mnemonic: str, operands: List[str], 
                 disasm: str, bytes_hex: str = "", size: int = 0, arch: str = "arm64"):
        self.address = address
        self.mnemonic = mnemonic.lower()
        self.operands = operands
        self.disasm = disasm
        self.bytes_hex = bytes_hex
        self.size = size
        self.arch = arch  # "arm32" or "arm64"
    
    def __repr__(self):
        return f"Instruction(0x{self.address:X}: {self.mnemonic} {', '.join(self.operands)})"
    
    def __str__(self):
        return f"0x{self.address:X}: {self.disasm}"


def parse_assembly_file(asm_file: Path, arch: str = "arm64") -> List[Instruction]:
    """
    Parse ARM assembly file extracted from IDA.
    
    Expected format:
    0xADDRESS: instruction
    0xADDRESS: instruction
    
    Args:
        asm_file: Path to .asm file
        arch: Architecture type ("arm32" or "arm64")
    
    Returns:
        List of Instruction objects
    """
    instructions = []
    
    if not asm_file.exists():
        raise FileNotFoundError(f"Assembly file not found: {asm_file}")
    
    with open(asm_file, 'r') as f:
        lines = f.readlines()
    
    # Pattern to match: 0xADDRESS: instruction
    # Example: 0x10000: add    w0, w1, w2
    instruction_pattern = re.compile(
        r'^(0x[0-9a-fA-F]+):\s+(.+)$'
    )
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith(';') or line.startswith('#'):
            continue
        
        # Try to match instruction pattern
        match = instruction_pattern.match(line)
        if not match:
            continue
        
        address_str = match.group(1)
        instruction_str = match.group(2).strip()
        
        try:
            address = int(address_str, 16)
        except ValueError:
            continue
        
        # Parse instruction into mnemonic and operands
        parts = instruction_str.split(None, 1)
        if not parts:
            continue
        
        mnemonic = parts[0]
        operands_str = parts[1] if len(parts) > 1 else ""
        
        # Parse operands (split by comma, handle brackets and braces)
        operands = parse_operands(operands_str)
        
        # Create instruction object
        inst = Instruction(
            address=address,
            mnemonic=mnemonic,
            operands=operands,
            disasm=instruction_str,
            bytes_hex="",
            size=4 if arch == "arm32" else 4,  # ARM instructions are 4 bytes
            arch=arch
        )
        
        instructions.append(inst)
    
    return instructions


def parse_operands(operands_str: str) -> List[str]:
    """
    Parse operands string into list of operands.
    
    Handles:
    - Multiple operands separated by commas
    - Memory references with brackets: [x0, #8]
    - Register lists with braces: {x0, x1, x2}
    - Shift operations: lsl #2, lsr #3
    
    Args:
        operands_str: String containing operands
    
    Returns:
        List of operand strings
    """
    if not operands_str:
        return []
    
    operands = []
    current = ""
    bracket_depth = 0
    brace_depth = 0
    
    for char in operands_str:
        if char == '[':
            bracket_depth += 1
            current += char
        elif char == ']':
            bracket_depth -= 1
            current += char
        elif char == '{':
            brace_depth += 1
            current += char
        elif char == '}':
            brace_depth -= 1
            current += char
        elif char == ',' and bracket_depth == 0 and brace_depth == 0:
            if current.strip():
                operands.append(current.strip())
            current = ""
        else:
            current += char
    
    # Add last operand
    if current.strip():
        operands.append(current.strip())
    
    return operands


def get_function_metadata(asm_file: Path) -> Dict:
    """
    Extract function metadata from assembly file header comments.
    
    Args:
        asm_file: Path to .asm file
    
    Returns:
        Dictionary with function metadata
    """
    metadata = {
        "function_name": None,
        "start_address": None,
        "end_address": None,
        "size": None,
        "instruction_count": None
    }
    
    if not asm_file.exists():
        return metadata
    
    with open(asm_file, 'r') as f:
        # Read first few lines for metadata
        for i, line in enumerate(f):
            if i > 10:  # Only check first 10 lines
                break
            
            line = line.strip()
            
            # Parse metadata comments
            if line.startswith('; Function:') or line.startswith('# Function:'):
                metadata["function_name"] = line.split(':', 1)[1].strip()
            elif line.startswith('; Address:') or line.startswith('# Address:'):
                addr_str = line.split(':', 1)[1].strip()
                # Format: 0xSTART - 0xEND
                match = re.search(r'0x([0-9a-fA-F]+)\s*-\s*0x([0-9a-fA-F]+)', addr_str)
                if match:
                    metadata["start_address"] = int(match.group(1), 16)
                    metadata["end_address"] = int(match.group(2), 16)
            elif line.startswith('; Size:') or line.startswith('# Size:'):
                size_str = line.split(':', 1)[1].strip()
                # Extract number
                match = re.search(r'(\d+)', size_str)
                if match:
                    metadata["size"] = int(match.group(1))
            elif line.startswith('; Instructions:') or line.startswith('# Instructions:'):
                count_str = line.split(':', 1)[1].strip()
                match = re.search(r'(\d+)', count_str)
                if match:
                    metadata["instruction_count"] = int(match.group(1))
    
    return metadata


def parse_assembly(asm_file: Path, arch: str = "arm64") -> Dict:
    """
    Parse ARM assembly file and return structured data.
    
    Args:
        asm_file: Path to .asm file
        arch: Architecture type ("arm32" or "arm64")
    
    Returns:
        Dictionary with metadata and instructions
    """
    asm_path = Path(asm_file)
    
    metadata = get_function_metadata(asm_path)
    instructions = parse_assembly_file(asm_path, arch=arch)
    
    return {
        "metadata": metadata,
        "instructions": instructions,
        "instruction_count": len(instructions)
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parser.py <assembly_file.asm> [arm32|arm64]")
        sys.exit(1)
    
    asm_file = Path(sys.argv[1])
    arch = sys.argv[2] if len(sys.argv) > 2 else "arm64"
    
    try:
        result = parse_assembly(asm_file, arch=arch)
        print(f"Parsed {result['instruction_count']} instructions")
        print(f"Architecture: {arch}")
        print(f"Function: {result['metadata'].get('function_name', 'Unknown')}")
        print(f"Address range: {hex(result['metadata'].get('start_address', 0))} - {hex(result['metadata'].get('end_address', 0))}")
        
        # Print first 10 instructions
        print("\nFirst 10 instructions:")
        for inst in result['instructions'][:10]:
            print(f"  {inst}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


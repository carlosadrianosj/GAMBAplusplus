"""
x86/x64 assembly parser, MBA detector, and expression converter.
"""

from .parser import parse_assembly, Instruction
from .detector import detect_mba_blocks, MBABlock
from .converter import convert_mba_block_to_expression, RegisterTracker

__all__ = [
    'parse_assembly',
    'Instruction',
    'detect_mba_blocks',
    'MBABlock',
    'convert_mba_block_to_expression',
    'RegisterTracker'
]


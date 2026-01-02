"""
Assembly processing module for GAMBA++
Supports multiple architectures: x86_64, ARM32/ARM64, MIPS, PowerPC, RISC-V
"""

from typing import List, Optional

# Architecture-specific imports
try:
    from .x86_64 import parse_assembly as parse_x86_64, detect_mba_blocks as detect_mba_x86_64
    from .arm import parse_assembly as parse_arm, detect_mba_blocks as detect_mba_arm
except ImportError:
    pass

# Common utilities
from .common.cfg import build_cfg, CFG, BasicBlock
from .common.mba_function_detector import detect_mba_functions, MBAFunction
from .common.slicing import slice_mba_block, convert_to_mini_ssa
from .common.normalization import normalize_expression, normalize_expression_list

# Unified API
def detect_mba_functions_unified(instructions: List, arch: str = 'x86_64') -> List[MBAFunction]:
    """Detect MBA functions (unified API)"""
    from .common.mba_function_detector import detect_mba_functions
    return detect_mba_functions(instructions, arch=arch)


def build_cfg_unified(instructions: List, arch: str = 'x86_64') -> CFG:
    """Build CFG (unified API)"""
    return build_cfg(instructions, arch=arch)

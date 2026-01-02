#!/usr/bin/env python3
"""
CFG Renderer - Professional Control Flow Graph Visualization
Renders Control Flow Graphs using Graphviz with rectangular blocks and proper alignment
"""

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    graphviz = None

from typing import List, Optional, Dict, Set
from pathlib import Path
from assembly.common.cfg import CFG, BasicBlock
from assembly.x86_64.detector import MBABlock


def format_instruction(inst) -> str:
    """Format instruction for display"""
    mnemonic = getattr(inst, 'mnemonic', '')
    operands = getattr(inst, 'operands', [])
    op_str = getattr(inst, 'op_str', '')
    
    # Special formatting for simplified instructions
    if mnemonic == 'simplified':
        return f"→ {op_str}"
    elif mnemonic == 'mba_attempted':
        return f"⚠ {op_str}"
    
    if op_str:
        return f"{mnemonic} {op_str}"
    elif operands:
        return f"{mnemonic} {', '.join(operands[:3])}"
    else:
        return mnemonic


def build_block_label(block: BasicBlock, max_instructions: int = None, show_address: bool = True, 
                      compact: bool = False) -> str:
    """
    Build formatted label for a basic block.
    Shows all instructions to allow size comparison.
    """
    label_parts = []
    
    if show_address:
        label_parts.append(f"📍 0x{block.start_address:X}")
    
    if block.instructions:
        inst_count = len(block.instructions)
        
        # Show instruction count as header
        label_parts.append(f"📊 {inst_count} instruction{'s' if inst_count != 1 else ''}")
        
        # Show all instructions (or up to max if specified)
        if max_instructions is None:
            # Show all instructions for size comparison
            instructions_to_show = block.instructions
        else:
            instructions_to_show = block.instructions[:max_instructions]
        
        for inst in instructions_to_show:
            inst_str = format_instruction(inst)
            # Truncate very long instructions but keep them visible
            if len(inst_str) > 50:
                inst_str = inst_str[:47] + "..."
            label_parts.append(f"  • {inst_str}")
        
        # If we truncated, show count
        if max_instructions is not None and inst_count > max_instructions:
            remaining = inst_count - max_instructions
            label_parts.append(f"  ... ({remaining} more)")
    else:
        label_parts.append("(empty block)")
    
    return "\\n".join(label_parts)


def get_mba_overlap(block: BasicBlock, mba_blocks: Optional[List[MBABlock]]) -> Dict:
    """Check if block overlaps with MBA blocks and return details"""
    if not mba_blocks:
        return {'has_mba': False, 'mba_count': 0, 'patterns': []}
    
    overlapping = []
    for mba_block in mba_blocks:
        mba_start = getattr(mba_block, 'start_address', 0)
        mba_end = getattr(mba_block, 'end_address', 0)
        
        # Check overlap
        if (block.start_address <= mba_start < block.end_address or
            block.start_address < mba_end <= block.end_address or
            (mba_start <= block.start_address and mba_end >= block.end_address)):
            pattern = getattr(mba_block, 'pattern_type', 'unknown')
            overlapping.append({
                'start': mba_start,
                'end': mba_end,
                'pattern': pattern
            })
    
    return {
        'has_mba': len(overlapping) > 0,
        'mba_count': len(overlapping),
        'patterns': [m['pattern'] for m in overlapping],
        'overlapping': overlapping
    }


def render_cfg(cfg: CFG, mba_blocks: Optional[List[MBABlock]] = None,
              output_path: Optional[Path] = None, format: str = 'png',
              title: str = "Control Flow Graph"):
    """
    Render CFG with MBA blocks highlighted using rectangular blocks.
    
    Args:
        cfg: Control Flow Graph
        mba_blocks: Optional list of MBA blocks to highlight
        output_path: Output file path (required)
        format: Output format (png, svg, pdf)
        title: Graph title
    """
    if output_path is None:
        raise ValueError("output_path is required")
    
    if not GRAPHVIZ_AVAILABLE:
        raise ImportError("Graphviz not available. Install with: pip install graphviz")
    
    # Create graph with professional styling
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir='TB')  # Top to bottom
    dot.attr('graph', 
             fontname='Arial',
             fontsize='12',
             label=title,
             labelloc='top',
             labeljust='left',
             bgcolor='white',
             pad='0.5',
             nodesep='0.5',
             ranksep='0.8')
    
    # Node styling - rectangular blocks
    # Size will be determined by content (fixedsize=False)
    dot.attr('node',
             shape='box',  # Rectangular, not rounded
             style='filled,rounded',  # Filled with rounded corners (subtle)
             fontname='Courier New',
             fontsize='9',
             margin='0.15,0.1',
             fixedsize='false')  # Let size be determined by content
    
    # Edge styling
    dot.attr('edge',
             fontname='Arial',
             fontsize='9',
             arrowsize='0.8',
             color='#333333')
    
    # Add nodes (basic blocks)
    # Sort by address for consistent ordering
    sorted_blocks = sorted(cfg.blocks, key=lambda b: b.start_address)
    
    for block in sorted_blocks:
        node_id = f"block_{block.start_address:X}"
        
        # Build label - show all instructions for size comparison
        label = build_block_label(block, max_instructions=None, show_address=True)
        
        # Check MBA overlap
        mba_info = get_mba_overlap(block, mba_blocks)
        
        # Color scheme based on MBA presence
        if mba_info['has_mba']:
            # MBA blocks - red/orange gradient
            fillcolor = '#ff6b6b'  # Red
            fontcolor = 'white'
            border_color = '#c92a2a'  # Dark red border
            # Add MBA indicator to label
            mba_label = f"⚠️ MBA ({mba_info['mba_count']})"
            if mba_info['patterns']:
                mba_label += f"\\nPatterns: {', '.join(set(mba_info['patterns']))}"
            label = f"{mba_label}\\n{label}"
        else:
            # Normal blocks - blue gradient
            fillcolor = '#4dabf7'  # Blue
            fontcolor = 'white'
            border_color = '#1971c2'  # Dark blue border
        
        # Create node with proper styling
        dot.node(node_id, label,
                fillcolor=fillcolor,
                fontcolor=fontcolor,
                color=border_color,
                penwidth='2.0')
    
    # Add edges with labels for conditional branches
    for block in sorted_blocks:
        node_id = f"block_{block.start_address:X}"
        
        if not block.instructions:
            continue
        
        last_inst = block.instructions[-1]
        is_conditional = False
        edge_label = ""
        
        # Check if last instruction is a conditional branch
        mnemonic = getattr(last_inst, 'mnemonic', '').lower()
        if mnemonic.startswith('j') and mnemonic not in ['jmp', 'jmpq']:
            is_conditional = True
            edge_label = "T"  # Taken
        elif mnemonic in ['call', 'callq']:
            edge_label = "call"
        
        for i, succ in enumerate(block.successors):
            succ_id = f"block_{succ.start_address:X}"
            
            # For conditional branches, label the first edge as "taken"
            if is_conditional and i == 0:
                dot.edge(node_id, succ_id, label=edge_label, style='solid', color='#2b8a3e')
            elif is_conditional and i == 1:
                # Second edge is fallthrough (not taken)
                dot.edge(node_id, succ_id, label="F", style='dashed', color='#868e96')
            else:
                dot.edge(node_id, succ_id, style='solid')
    
    # Render
    output_path = Path(output_path)
    dot.render(output_path.with_suffix(''), format=format, cleanup=True)
    print(f"CFG rendered to: {output_path}")


def render_cfg_comparison(cfg_before: CFG, cfg_after: CFG,
                         mba_before: Optional[List[MBABlock]] = None,
                         mba_after: Optional[List[MBABlock]] = None,
                         output_path: Optional[Path] = None, format: str = 'png',
                         title: str = "CFG Comparison: Before vs After Simplification"):
    """
    Render side-by-side comparison of CFG before and after simplification.
    Ensures proper alignment and matching of blocks.
    
    Args:
        cfg_before: CFG before simplification
        cfg_after: CFG after simplification
        mba_before: MBA blocks before
        mba_after: MBA blocks after
        output_path: Output file path (required)
        format: Output format
        title: Graph title
    """
    if output_path is None:
        raise ValueError("output_path is required")
    
    if not GRAPHVIZ_AVAILABLE:
        raise ImportError("Graphviz not available. Install with: pip install graphviz")
    
    # Create main graph
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir='LR')  # Left to right for side-by-side
    dot.attr('graph',
             fontname='Arial',
             fontsize='14',
             label=title,
             labelloc='top',
             labeljust='center',
             bgcolor='#f8f9fa',
             pad='1.0',
             nodesep='0.6',
             ranksep='1.2',
             splines='ortho')  # Orthogonal edges
    
    # Node styling - size determined by content for visual comparison
    dot.attr('node',
             shape='box',
             style='filled,rounded',
             fontname='Courier New',
             fontsize='8',
             margin='0.1,0.08',
             fixedsize='false')  # Size based on content
    
    # Edge styling
    dot.attr('edge',
             fontname='Arial',
             fontsize='8',
             arrowsize='0.7',
             color='#495057')
    
    # Create mapping of addresses to align blocks
    before_blocks_by_addr = {block.start_address: block for block in cfg_before.blocks}
    after_blocks_by_addr = {block.start_address: block for block in cfg_after.blocks}
    
    # Find common addresses for alignment
    common_addresses = set(before_blocks_by_addr.keys()) & set(after_blocks_by_addr.keys())
    
    # Sort blocks by address for consistent ordering
    before_blocks_sorted = sorted(cfg_before.blocks, key=lambda b: b.start_address)
    after_blocks_sorted = sorted(cfg_after.blocks, key=lambda b: b.start_address)
    
    # Subgraph for BEFORE
    with dot.subgraph(name='cluster_before') as before:
        before.attr(label='BEFORE Simplification',
                   fontname='Arial',
                   fontsize='12',
                   fontcolor='#c92a2a',
                   style='filled',
                   fillcolor='#fff5f5',
                   color='#c92a2a',
                   penwidth='2')
        
        before.attr('node', fillcolor='#4dabf7', fontcolor='white', color='#1971c2')
        before.attr('graph', rankdir='TB', nodesep='0.5', ranksep='0.8')
        
        # Add all before blocks in sorted order
        for block in before_blocks_sorted:
            node_id = f"before_{block.start_address:X}"
            # Show all instructions for size comparison
            label = build_block_label(block, max_instructions=None, show_address=True)
            
            # Check MBA
            mba_info = get_mba_overlap(block, mba_before)
            if mba_info['has_mba']:
                mba_label = f"⚠️ MBA ({mba_info['mba_count']})\\n"
                label = f"{mba_label}{label}"
                before.node(node_id, label,
                           fillcolor='#ff6b6b',
                           fontcolor='white',
                           color='#c92a2a',
                           penwidth='2.5')
            else:
                before.node(node_id, label)
        
        # Add edges for before
        for block in cfg_before.blocks:
            node_id = f"before_{block.start_address:X}"
            for succ in block.successors:
                succ_id = f"before_{succ.start_address:X}"
                before.edge(node_id, succ_id, color='#1971c2')
    
    # Subgraph for AFTER
    with dot.subgraph(name='cluster_after') as after:
        after.attr(label='AFTER Simplification',
                  fontname='Arial',
                  fontsize='12',
                  fontcolor='#2b8a3e',
                  style='filled',
                  fillcolor='#f0fdf4',
                  color='#2b8a3e',
                  penwidth='2')
        
        after.attr('node', fillcolor='#51cf66', fontcolor='white', color='#2b8a3e')
        after.attr('graph', rankdir='TB', nodesep='0.5', ranksep='0.8')
        
        # Add all after blocks in sorted order
        for block in after_blocks_sorted:
            node_id = f"after_{block.start_address:X}"
            # Show all instructions for size comparison
            label = build_block_label(block, max_instructions=None, show_address=True)
            
            # Check if block was simplified or MBA was attempted
            is_simplified = getattr(block, 'simplified', False)
            simplified_info = getattr(block, 'simplified_info', None)
            mba_attempted = getattr(block, 'mba_attempted', False)
            mba_info = getattr(block, 'mba_info', None)
            
            if is_simplified and simplified_info:
                # Block was simplified - show with green color and simplified indicator
                reduction = simplified_info.get('complexity_reduction', 0)
                simplified_label = f"✅ SIMPLIFIED (reduced {reduction} chars)\\n"
                label = f"{simplified_label}{label}"
                after.node(node_id, label,
                          fillcolor='#51cf66',
                          fontcolor='white',
                          color='#2b8a3e',
                          penwidth='2.5')
            elif mba_attempted and mba_info:
                # MBA was attempted but simplification failed - show in yellow/orange
                attempted_label = f"⚠️ MBA ATTEMPTED (failed)\\n"
                label = f"{attempted_label}{label}"
                after.node(node_id, label,
                          fillcolor='#ffd43b',
                          fontcolor='black',
                          color='#fab005',
                          penwidth='2.0')
            else:
                # Check MBA (should be none or fewer)
                mba_info_check = get_mba_overlap(block, mba_after)
                if mba_info_check['has_mba']:
                    mba_label = f"⚠️ MBA ({mba_info_check['mba_count']})\\n"
                    label = f"{mba_label}{label}"
                    after.node(node_id, label,
                              fillcolor='#ffd43b',
                              fontcolor='black',
                              color='#fab005',
                              penwidth='2.0')
                else:
                    after.node(node_id, label)
        
        # Add edges for after
        for block in cfg_after.blocks:
            node_id = f"after_{block.start_address:X}"
            for succ in block.successors:
                succ_id = f"after_{succ.start_address:X}"
                after.edge(node_id, succ_id, color='#2b8a3e')
    
    # Add alignment edges between matching blocks (invisible, for layout)
    # This ensures blocks with same addresses are aligned horizontally
    for addr in sorted(common_addresses):
        before_id = f"before_{addr:X}"
        after_id = f"after_{addr:X}"
        # Invisible edge to align blocks horizontally
        dot.edge(before_id, after_id,
                style='invis',
                constraint='true',
                weight='100')  # High weight for strong alignment
    
    # Also add rank constraints for better alignment
    # Group blocks by their rank (position in execution flow)
    # Match blocks by address for proper alignment
    for before_block in before_blocks_sorted:
        before_addr = before_block.start_address
        if before_addr in after_blocks_by_addr:
            after_block = after_blocks_by_addr[before_addr]
            before_id = f"before_{before_addr:X}"
            after_id = f"after_{before_addr:X}"
            # Add to same rank for alignment
            dot.edge(before_id, after_id, style='invis', constraint='true', weight='100')
    
    # Add legend
    with dot.subgraph(name='cluster_legend') as legend:
        legend.attr(label='Legend',
                   fontname='Arial',
                   fontsize='10',
                   style='filled',
                   fillcolor='#ffffff',
                   color='#dee2e6',
                   margin='10')
        
        legend.attr('node',
                   shape='box',
                   style='filled',
                   fontname='Arial',
                   fontsize='9',
                   width='1.5',
                   height='0.4',
                   margin='0.1')
        
        legend.node('legend_mba', '⚠️ MBA Block', fillcolor='#ff6b6b', fontcolor='white')
        legend.node('legend_normal', 'Normal Block', fillcolor='#4dabf7', fontcolor='white')
        legend.node('legend_simplified', 'Simplified Block', fillcolor='#51cf66', fontcolor='white')
        legend.node('legend_edge_t', 'Edge: T (Taken)', fillcolor='white', fontcolor='#2b8a3e')
        legend.node('legend_edge_f', 'Edge: F (Not Taken)', fillcolor='white', fontcolor='#868e96')
    
    # Render
    output_path = Path(output_path)
    dot.render(output_path.with_suffix(''), format=format, cleanup=True)
    print(f"CFG comparison rendered to: {output_path}")


def render_cfg_with_analysis(cfg: CFG, mba_blocks: Optional[List[MBABlock]] = None,
                            analysis_data: Optional[Dict] = None,
                            output_path: Optional[Path] = None, format: str = 'png'):
    """
    Render CFG with detailed analysis information.
    
    Args:
        cfg: Control Flow Graph
        mba_blocks: MBA blocks
        analysis_data: Additional analysis data (metrics, etc.)
        output_path: Output file path
        format: Output format
    """
    if output_path is None:
        raise ValueError("output_path is required")
    
    # Build title with analysis
    title = "Control Flow Graph"
    if analysis_data:
        summary = analysis_data.get('summary', {})
        inst_count = summary.get('instruction_count', 0)
        mba_count = summary.get('mba_block_count', 0)
        bb_count = summary.get('basic_block_count', 0)
        title = f"CFG Analysis | {inst_count} insts | {mba_count} MBA blocks | {bb_count} basic blocks"
    
    render_cfg(cfg, mba_blocks, output_path, format, title)


if __name__ == "__main__":
    # Test rendering
    from assembly.common.cfg import build_cfg
    
    # Create simple test CFG
    class TestInst:
        def __init__(self, address, mnemonic, operands, size=4):
            self.address = address
            self.mnemonic = mnemonic
            self.operands = operands
            self.size = size
            self.op_str = ', '.join(operands)
            self.is_branch = mnemonic.startswith('j') or mnemonic in ['call', 'ret']
            self.is_call = mnemonic == 'call'
            self.is_return = mnemonic in ['ret', 'retn']
            self.branch_target = None
    
    test_instructions = [
        TestInst(0x1000, 'mov', ['eax', '1']),
        TestInst(0x1004, 'test', ['eax', 'eax']),
        TestInst(0x1008, 'jne', ['0x1010']),
        TestInst(0x100C, 'mov', ['eax', '2']),
        TestInst(0x1010, 'ret', []),
    ]
    
    cfg = build_cfg(test_instructions, arch='x86_64')
    print(f"CFG has {len(cfg.blocks)} blocks")
    
    if GRAPHVIZ_AVAILABLE:
        render_cfg(cfg, None, Path('test_cfg.png'))
        print("CFG rendered to test_cfg.png")
    else:
        print("Graphviz not available, skipping render")

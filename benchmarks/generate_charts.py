#!/usr/bin/env python3
"""
Generate benchmark comparison charts for GAMBA++

Creates visualizations comparing GAMBA against other MBA simplification tools
using published benchmark results.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_benchmark_data(json_path: Path) -> Dict:
    """Load benchmark data from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_bar_chart(data: Dict, output_path: Path):
    """Create bar chart comparing success rates across all tools"""
    tools = []
    avg_success = []
    colors = []
    
    tool_order = ['GAMBA++', 'GAMBA', 'SiMBA', 'NeuReduce', 'Syntia', 'QSynth']
    color_map = {
        'GAMBA++': '#1abc9c',    # Teal (highlighted)
        'GAMBA': '#2ecc71',      # Green
        'SiMBA': '#3498db',      # Blue
        'NeuReduce': '#9b59b6',  # Purple
        'Syntia': '#e74c3c',     # Red
        'QSynth': '#f39c12',     # Orange
    }
    
    for tool in tool_order:
        success_rates = []
        for dataset_name, dataset_data in data['datasets'].items():
            if tool in dataset_data and dataset_data[tool] is not None:
                success_rates.append(dataset_data[tool])
        
        if success_rates:
            avg = np.mean(success_rates)
            tools.append(tool)
            avg_success.append(avg)
            colors.append(color_map.get(tool, '#95a5a6'))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(tools, avg_success, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_success):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Average Success Rate (%)', fontweight='bold')
    ax.set_xlabel('Tool', fontweight='bold')
    ax.set_title('MBA Simplification Tools: Average Success Rate Comparison', fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created bar chart: {output_path}")


def create_grouped_bar_chart(data: Dict, output_path: Path):
    """Create grouped bar chart comparing tools per dataset"""
    datasets = list(data['datasets'].keys())
    tools = ['GAMBA++', 'GAMBA', 'SiMBA', 'NeuReduce', 'Syntia', 'QSynth']
    
    x = np.arange(len(datasets))
    width = 0.13
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {
        'GAMBA++': '#1abc9c',
        'GAMBA': '#2ecc71',
        'SiMBA': '#3498db',
        'NeuReduce': '#9b59b6',
        'Syntia': '#e74c3c',
        'QSynth': '#f39c12',
    }
    
    for i, tool in enumerate(tools):
        values = []
        for dataset_name in datasets:
            dataset_data = data['datasets'][dataset_name]
            if tool in dataset_data and dataset_data[tool] is not None:
                values.append(dataset_data[tool])
            else:
                values.append(0)
        
        offset = (i - len(tools)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=tool, color=colors.get(tool, '#95a5a6'),
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}%' if val >= 10 else '',
                        ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_title('Tool Comparison: Success Rate per Dataset', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created grouped bar chart: {output_path}")


def create_radar_chart(data: Dict, output_path: Path):
    """Create radar chart showing tool capabilities"""
    datasets = list(data['datasets'].keys())
    tools = ['GAMBA++', 'GAMBA', 'SiMBA', 'NeuReduce', 'Syntia', 'QSynth']
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = {
        'GAMBA++': '#1abc9c',
        'GAMBA': '#2ecc71',
        'SiMBA': '#3498db',
        'NeuReduce': '#9b59b6',
        'Syntia': '#e74c3c',
        'QSynth': '#f39c12',
    }
    
    for tool in tools:
        values = []
        for dataset_name in datasets:
            dataset_data = data['datasets'][dataset_name]
            if tool in dataset_data and dataset_data[tool] is not None:
                values.append(dataset_data[tool])
            else:
                values.append(0)
        
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=tool, color=colors.get(tool, '#95a5a6'))
        ax.fill(angles, values, alpha=0.15, color=colors.get(tool, '#95a5a6'))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.set_title('Tool Capabilities: Success Rate Across Datasets', fontweight='bold', pad=20, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created radar chart: {output_path}")


def create_heatmap(data: Dict, output_path: Path):
    """Create heatmap showing tool vs dataset success matrix"""
    datasets = list(data['datasets'].keys())
    tools = ['GAMBA++', 'GAMBA', 'SiMBA', 'NeuReduce', 'Syntia', 'QSynth']
    
    # Build matrix
    matrix = []
    for tool in tools:
        row = []
        for dataset_name in datasets:
            dataset_data = data['datasets'][dataset_name]
            if tool in dataset_data and dataset_data[tool] is not None:
                row.append(dataset_data[tool])
            else:
                row.append(np.nan)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(tools)))
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_yticklabels(tools)
    
    # Add text annotations
    for i in range(len(tools)):
        for j in range(len(datasets)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontweight='bold', fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Success Rate (%)', rotation=270, labelpad=20, fontweight='bold')
    
    ax.set_title('Tool vs Dataset Success Rate Heatmap', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created heatmap: {output_path}")


def create_performance_chart(data: Dict, output_path: Path):
    """Create performance comparison chart showing processing speed"""
    if 'performance' not in data:
        print("No performance data available, skipping performance chart")
        return
    
    perf_data = data['performance']
    
    # Extract GAMBA++ variants
    tools = []
    times = []
    colors_list = []
    
    color_map = {
        'GAMBA++ Optimized': '#1abc9c',    # Teal
        'GAMBA++ Parallel': '#16a085',     # Darker teal
        'GAMBA++ Sequential': '#2ecc71',   # Green
        'GAMBA': '#27ae60',                # Darker green
    }
    
    # Add GAMBA++ variants (best to worst)
    if 'GAMBA++ Optimized' in perf_data and perf_data['GAMBA++ Optimized'] is not None:
        tools.append('GAMBA++ Optimized')
        times.append(perf_data['GAMBA++ Optimized'])
        colors_list.append(color_map['GAMBA++ Optimized'])
    
    if 'GAMBA++ Parallel' in perf_data and perf_data['GAMBA++ Parallel'] is not None:
        tools.append('GAMBA++ Parallel')
        times.append(perf_data['GAMBA++ Parallel'])
        colors_list.append(color_map['GAMBA++ Parallel'])
    
    if 'GAMBA++ Sequential' in perf_data and perf_data['GAMBA++ Sequential'] is not None:
        tools.append('GAMBA++ Sequential')
        times.append(perf_data['GAMBA++ Sequential'])
        colors_list.append(color_map['GAMBA++ Sequential'])
    
    if 'GAMBA' in perf_data and perf_data['GAMBA'] is not None:
        tools.append('GAMBA (Original)')
        times.append(perf_data['GAMBA'])
        colors_list.append(color_map['GAMBA'])
    
    if not tools:
        print("No performance data available, skipping performance chart")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(tools, times, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Calculate and show speedup
    if len(times) > 1:
        baseline = times[-1]  # GAMBA (original) as baseline
        for i, (bar, val) in enumerate(zip(bars[:-1], times[:-1])):
            speedup = baseline / val if val > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                    f'{speedup:.2f}x faster',
                    ha='center', va='center', fontweight='bold', fontsize=10,
                    color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_ylabel('Average Time per Expression (seconds)', fontweight='bold')
    ax.set_xlabel('Tool', fontweight='bold')
    ax.set_title('Performance Comparison: Processing Speed', fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created performance chart: {output_path}")


def main():
    """Generate all benchmark charts"""
    script_dir = Path(__file__).parent
    json_path = script_dir / 'published_results' / 'tool_comparison.json'
    charts_dir = script_dir / 'charts'
    
    charts_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading benchmark data...")
    data = load_benchmark_data(json_path)
    
    # Generate charts
    print("\nGenerating charts...")
    create_bar_chart(data, charts_dir / 'bar_chart_comparison.png')
    create_grouped_bar_chart(data, charts_dir / 'grouped_bar_chart.png')
    create_radar_chart(data, charts_dir / 'radar_chart.png')
    create_heatmap(data, charts_dir / 'heatmap.png')
    create_performance_chart(data, charts_dir / 'performance_comparison.png')
    
    print("\nAll charts generated successfully!")
    print(f"Charts saved to: {charts_dir}")


if __name__ == "__main__":
    main()


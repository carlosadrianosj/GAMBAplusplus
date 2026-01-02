# How to Use This Framework

This comprehensive guide provides detailed examples and diagrams for using GAMBA++ in various scenarios.

## Table of Contents

1. [Docker Setup](#0-docker-setup)
2. [Direct Expression Input](#1-direct-expression-input)
3. [File-Based Input](#2-file-based-input)
4. [Assembly Interpretation](#3-assembly-interpretation)
5. [Advanced Usage Patterns](#4-advanced-usage-patterns)
6. [Performance Optimization](#5-performance-optimization)

---

## 0. Docker Setup

GAMBA++ can be run using Docker for easy deployment and scalability. This is the recommended approach for production environments.

### 0.1 Building the Docker Image

```bash
# Build the Docker image
docker build -t gamba-plusplus:latest .
```

### 0.2 Running with Docker Compose (Recommended)

Docker Compose provides an easy way to run and scale GAMBA++:

```bash
# Start the services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale workers (for parallel processing)
docker-compose up -d --scale gamba-worker=4

# Stop services
docker-compose down
```

### 0.3 Running a Single Container

```bash
# Run interactively
docker run -it --rm \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/benchmarks/datasets:/app/benchmarks/datasets:ro \
  gamba-plusplus:latest \
  python -c "from optimization.batch_advanced import process_expressions_batch_advanced; print('GAMBA++ ready!')"

# Run with Python script
docker run --rm \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/your_script.py:/app/script.py \
  gamba-plusplus:latest \
  python script.py
```

### 0.4 Using GAMBA++ Inside Docker

Once inside a container, you can use GAMBA++ exactly as you would locally:

```python
# Inside Docker container
from optimization.batch_advanced import process_expressions_batch_advanced

expressions = [
    "(x ^ y) + 2*(x & y)",
    "(x | y) - (x & y)",
]

results = process_expressions_batch_advanced(
    expressions=expressions,
    bitcount=32,
    max_workers=8,
    use_cache=True,
    show_progress=True
)

for result in results:
    if result["success"]:
        print(f"{result['original']} -> {result['simplified']}")
```

### 0.5 Volume Mounts

The Docker setup uses the following volume mounts:

- `/app/cache` - Cache directory for GAMBA results (persistent)
- `/app/output` - Output directory for results
- `/app/benchmarks/datasets` - Read-only access to benchmark datasets

### 0.6 Environment Variables

You can customize the Docker container behavior with environment variables:

```bash
# Set maximum workers
docker run -e MAX_WORKERS=16 gamba-plusplus:latest

# Enable worker mode
docker run -e WORKER_MODE=true gamba-plusplus:latest
```

### 0.7 Scaling for Production

For production deployments, use Docker Compose with multiple workers:

```yaml
# docker-compose.yml
services:
  gamba-worker:
    deploy:
      replicas: 4  # Run 4 worker instances
```

This allows GAMBA++ to process expressions in parallel across multiple containers, significantly improving throughput for large workloads.

---

## 1. Direct Expression Input

The simplest way to use GAMBA++ is to directly provide MBA expressions as Python strings.

### 1.1 Basic Simplification

```python
from gamba.simplify_general import GeneralSimplifier

# Create a simplifier instance
simplifier = GeneralSimplifier(bitcount=32, modRed=False, verifBitCount=None)

# Simplify a single expression
expression = "(x ^ y) + 2*(x & y)"
result = simplifier.simplify(expression, useZ3=False)

print(f"Original:   {expression}")
print(f"Simplified: {result}")  # Output: x + y
```

**Workflow Diagram:**

```
┌─────────────────┐
│ Input Expression│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│GeneralSimplifier │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GAMBA Algorithm │
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│Simplified Expression│
└────────┬─────────┘
         │
         ▼
┌──────────────┐
│ Output Result│
└──────────────┘
```

### 1.2 Multiple Expressions (Sequential)

```python
from gamba.simplify_general import GeneralSimplifier

expressions = [
    "(x ^ y) + 2*(x & y)",
    "(x | y) - (x & y)",
    "~x + ~y + 1",
]

simplifier = GeneralSimplifier(32, False, None)

results = []
for expr in expressions:
    simplified = simplifier.simplify(expr, useZ3=False)
    results.append({
        "original": expr,
        "simplified": simplified
    })
    print(f"{expr} -> {simplified}")
```

### 1.3 Multiple Expressions (Optimized - Recommended)

For better performance, use the optimized batch processing:

```python
from optimization.batch_advanced import process_expressions_batch_advanced

expressions = [
    "(x ^ y) + 2*(x & y)",
    "(x | y) - (x & y)",
    "~x + ~y + 1",
    "((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))",
]

# Process with all optimizations (8 cores, cache, batch)
results = process_expressions_batch_advanced(
    expressions=expressions,
    bitcount=32,
    max_workers=8,
    batch_size=10,
    use_cache=True,
    show_progress=True
)

# Process results
for result in results:
    if result["success"]:
        print(f"{result['original']} -> {result['simplified']}")
    else:
        print(f"Error: {result['error']}")
```

**Optimized Processing Flow:**

```
                    ┌──────────────────┐
                    │ Input Expressions│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Check Cache   │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
            [Hit]│                         │[Miss]
                │                         │
                ▼                         ▼
        ┌──────────────┐        ┌──────────────────┐
        │Return Cached │        │Quick Simplify    │
        └──────┬───────┘        │     Check        │
               │                └────────┬─────────┘
               │                         │
               │            ┌────────────┴────────────┐
               │            │                         │
               │        [Simple]│                 │[Complex]
               │            │                         │
               │            ▼                         ▼
               │    ┌──────────────┐        ┌──────────────────┐
               │    │Early         │        │Batch Processing  │
               │    │Termination   │        └────────┬──────────┘
               │    └──────┬───────┘                 │
               │           │                         ▼
               │           │            ┌──────────────────────────┐
               │           │            │ProcessPoolExecutor      │
               │           │            │    (8 Workers)          │
               │           │            └────────┬─────────────────┘
               │           │                     │
               │           │                     ▼
               │           │            ┌──────────────────┐
               │           │            │GAMBA             │
               │           │            │Simplification    │
               │           │            └────────┬─────────┘
               │           │                     │
               │           │                     ▼
               │           │            ┌──────────────────┐
               │           │            │  Update Cache    │
               │           │            └────────┬─────────┘
               │           │                     │
               └───────────┴─────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ Return Results │
                  └────────────────┘
```

---

## 2. File-Based Input

GAMBA++ can process expressions from files, which is useful for batch processing large datasets.

### 2.1 Reading Expressions from a Text File

**File Format:** One expression per line

```
(x ^ y) + 2*(x & y)
(x | y) - (x & y)
~x + ~y + 1
((x & y) | (x ^ y)) + ((~x & y) | (x & ~y))
```

**Code:**

```python
from pathlib import Path
from optimization.batch_advanced import process_expressions_batch_advanced

# Read expressions from file
input_file = Path("expressions.txt")
expressions = []

with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):  # Skip empty lines and comments
            expressions.append(line)

print(f"Loaded {len(expressions)} expressions from {input_file}")

# Process all expressions
results = process_expressions_batch_advanced(
    expressions=expressions,
    bitcount=32,
    max_workers=8,
    use_cache=True,
    show_progress=True
)

# Write results to output file
output_file = Path("results.txt")
with open(output_file, 'w') as f:
    for result in results:
        if result["success"]:
            f.write(f"{result['original']} -> {result['simplified']}\n")
        else:
            f.write(f"{result['original']} -> ERROR: {result['error']}\n")

print(f"Results written to {output_file}")
```

### 2.2 Processing Benchmark Datasets

GAMBA++ includes benchmark datasets in the `benchmarks/datasets/` directory:

```python
from pathlib import Path
from optimization.batch_advanced import process_expressions_batch_advanced

# Load a benchmark dataset
dataset_file = Path("benchmarks/datasets/neureduce.txt")
expressions = []

with open(dataset_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            # Some datasets have format: "original -> expected"
            # Extract just the original expression
            if " -> " in line:
                expr = line.split(" -> ")[0]
            else:
                expr = line
            expressions.append(expr)

print(f"Processing {len(expressions)} expressions from {dataset_file.name}")

# Process with optimizations
results = process_expressions_batch_advanced(
    expressions=expressions,
    bitcount=32,
    max_workers=8,
    batch_size=20,
    use_cache=True,
    show_progress=True
)

# Calculate success rate
successful = sum(1 for r in results if r["success"])
print(f"Success rate: {successful}/{len(expressions)} ({100*successful/len(expressions):.1f}%)")
```

**File Processing Workflow:**

```
┌───────────┐      ┌───────────┐      ┌─────────────────┐
│Input File │ ───► │Read Lines │ ───► │Parse Expressions│
└───────────┘      └───────────┘      └────────┬────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │Batch Processing │
                                    └────────┬────────┘
                                             │
                                             ▼
                                    ┌─────────────┐
                                    │  GAMBA++   │
                                    └──────┬─────┘
                                           │
                                           ▼
                                    ┌──────────┐
                                    │ Results  │
                                    └────┬─────┘
                                         │
                                         ▼
                                    ┌───────────┐
                                    │Output File│
                                    └───────────┘
```

### 2.3 CSV/JSON Input

For structured data, you can read from CSV or JSON:

```python
import json
import csv
from pathlib import Path
from optimization.batch_advanced import process_expressions_batch_advanced

# Option 1: JSON format
def load_from_json(json_file: Path):
    with open(json_file, 'r') as f:
        data = json.load(f)
        return [item["expression"] for item in data]

# Option 2: CSV format
def load_from_csv(csv_file: Path):
    expressions = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            expressions.append(row["expression"])
    return expressions

# Usage
expressions = load_from_json(Path("expressions.json"))
# or
expressions = load_from_csv(Path("expressions.csv"))

# Process
results = process_expressions_batch_advanced(
    expressions=expressions,
    bitcount=32,
    max_workers=8,
    use_cache=True
)
```

---

## 3. Assembly Interpretation

GAMBA++ can interpret assembly code from **9 different architectures**, automatically detecting MBA patterns and converting them to expressions. This section provides comprehensive examples for each supported architecture.

### 3.1 Overview

**Supported Architectures:**
- **x86_64**: Intel/AMD 64-bit
- **ARM32**: ARM 32-bit (ARMv7)
- **ARM64**: ARM 64-bit (AArch64)
- **MIPS32**: MIPS 32-bit
- **MIPS64**: MIPS 64-bit
- **PowerPC32**: PowerPC 32-bit
- **PowerPC64**: PowerPC 64-bit
- **RISC-V32**: RISC-V 32-bit
- **RISC-V64**: RISC-V 64-bit

**Assembly Processing Pipeline:**

```
                    ┌───────────────┐
                    │ Assembly File │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Parse Assembly│
                    └───────┬───────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │Instructions List │
                    └───────┬─────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │Detect MBA Blocks│
                    └───────┬─────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │MBA Block Found?  │
                    └───────┬──────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
         [Yes]                           [No]
            │                               │
            ▼                               ▼
    ┌───────────────┐              ┌──────────────┐
    │Convert to     │              │    Skip      │
    │Expression     │              └──────┬───────┘
    └───────┬───────┘                     │
            │                             │
            ▼                             │
    ┌──────────────────┐                 │
    │GAMBA             │                 │
    │Simplification    │                 │
    └───────┬──────────┘                 │
            │                             │
            ▼                             │
    ┌──────────────────┐                 │
    │Simplified        │                 │
    │Expression        │                 │
    └───────┬──────────┘                 │
            │                             │
            └─────────────┬───────────────┘
                          │
                          ▼
                    ┌───────────┐
                    │ Continue  │
                    └───────────┘
```

### 3.2 x86/x64 Assembly Processing

**Architecture Details:**
- **Registers**: 64-bit (rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8-r15)
- **32-bit subregisters**: eax, ebx, ecx, edx, esi, edi, ebp, esp
- **Bit width**: 64-bit (or 32-bit for x86)
- **Addressing modes**: Complex (e.g., `[rbp-0x10]`, `[rax+rbx*4]`)

**Complete Workflow:**

```python
from pathlib import Path
from assembly.x86_64.parser import parse_assembly
from assembly.x86_64.detector import detect_mba_blocks
from assembly.x86_64.converter import convert_mba_block_to_expression
from optimization.batch_advanced import process_expressions_batch_advanced

# Step 1: Parse assembly file
asm_file = Path("function_x86_64.asm")
result = parse_assembly(asm_file)
instructions = result["instructions"]

print(f"✓ Parsed {len(instructions)} x86_64 instructions")

# Step 2: Detect MBA blocks
mba_blocks = detect_mba_blocks(instructions, min_boolean_chain=5)
print(f"✓ Found {len(mba_blocks)} MBA blocks")

# Step 3: Convert MBA blocks to expressions
expressions = []
for block in mba_blocks:
    expr_data = convert_mba_block_to_expression(block)
    if expr_data and expr_data.get("gamba_expression"):
        expressions.append(expr_data["gamba_expression"])
        print(f"  Block at 0x{block.start_address:X}: {expr_data['gamba_expression']}")

# Step 4: Simplify expressions
if expressions:
    results = process_expressions_batch_advanced(
        expressions=expressions,
        bitcount=64,  # x64 uses 64-bit
        max_workers=8,
        use_cache=True,
        show_progress=True
    )
    
    # Display results
    for i, result in enumerate(results):
        if result["success"]:
            print(f"\nBlock {i+1}:")
            print(f"  Original:   {result['original']}")
            print(f"  Simplified: {result['simplified']}")
```

**Example x86_64 Assembly File:**

```
test_mba.o:     file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <mba_simple>:
   0:	48 89 d8             	mov    rax,rbx
   3:	48 31 c8             	xor    rax,rcx
   6:	48 21 d0             	and    rax,rdx
   9:	48 01 c8             	add    rax,rcx
   c:	c3                   	ret

0000000000000010 <mba_boolean_chain>:
  10:	48 21 d0             	and    rax,rdx
  13:	48 09 c8             	or     rax,rcx
  16:	48 31 d0             	xor    rax,rdx
  19:	48 21 c8             	and    rax,rcx
  1c:	48 09 d0             	or     rax,rdx
  1f:	c3                   	ret
```

**Processing Example:**

```python
# Parse the assembly
parsed = parse_assembly(Path("test_mba.asm"))
instructions = parsed["instructions"]

# Detect MBA patterns
mba_blocks = detect_mba_blocks(instructions)

# Convert and simplify
for block in mba_blocks:
    expr_data = convert_mba_block_to_expression(block)
    if expr_data:
        print(f"MBA Expression: {expr_data['gamba_expression']}")
        # Example output: "((x & y) | z) ^ (x & z)"
```

### 3.3 ARM32/ARM64 Assembly Processing

**Architecture Details:**
- **ARM32 Registers**: r0-r15 (32-bit)
- **ARM64 Registers**: x0-x30 (64-bit), w0-w30 (32-bit subregisters)
- **Bit width**: 32-bit (ARM32) or 64-bit (ARM64)
- **Special features**: Embedded shifts, bitfield instructions, conditional execution

#### ARM64 Example

```python
from pathlib import Path
from assembly.arm.parser import parse_assembly
from assembly.arm.detector import detect_mba_blocks
from assembly.arm.converter import convert_mba_block_to_expression
from optimization.batch_advanced import process_expressions_batch_advanced

# Parse ARM64 assembly
asm_file = Path("function_arm64.asm")
result = parse_assembly(asm_file, arch="arm64")
instructions = result["instructions"]

print(f"✓ Parsed {len(instructions)} ARM64 instructions")

# Detect MBA blocks
mba_blocks = detect_mba_blocks(instructions, min_boolean_chain=5)
print(f"✓ Found {len(mba_blocks)} MBA blocks")

# Convert and simplify
expressions = []
for block in mba_blocks:
    expr_data = convert_mba_block_to_expression(block)
    if expr_data and expr_data.get("gamba_expression"):
        expressions.append(expr_data["gamba_expression"])

if expressions:
    results = process_expressions_batch_advanced(
        expressions=expressions,
        bitcount=64,  # ARM64 uses 64-bit
        max_workers=8,
        use_cache=True
    )
```

**Example ARM64 Assembly File:**

```
test_mba.o:     file format elf64-littleaarch64

Disassembly of section .text:

0000000000000000 <mba_simple>:
   0:	0a010002 	and	w2, w0, w1
   4:	4a010000 	eor	w0, w0, w1
   8:	0b020400 	add	w0, w0, w2, lsl #1
   c:	d65f03c0 	ret

0000000000000010 <mba_embedded_shift>:
  10:	0b010c00 	add	w0, w0, w1, lsl #3
  14:	0b020000 	add	w0, w0, w2
  18:	d65f03c0 	ret

0000000000000020 <mba_bitfield>:
  20:	d3483c00 	ubfx	x0, x0, #8, #8
  24:	0b414000 	add	w0, w0, w1, lsr #16
  28:	d65f03c0 	ret
```

**ARM-Specific Features:**

```python
# ARM64 supports embedded shifts in operands
# Example: add w0, w0, w1, lsl #3
# This is converted to: w0 + (w1 << 3)

# ARM64 supports bitfield instructions
# Example: ubfx x0, x0, #8, #8
# This extracts bits 8-15 from x0

# ARM64 supports conditional selection
# Example: csel w0, w0, w1, gt
# This selects w0 if condition is greater than, else w1
```

#### ARM32 Example

```python
# Parse ARM32 assembly
asm_file = Path("function_arm32.asm")
result = parse_assembly(asm_file, arch="arm32")
instructions = result["instructions"]

print(f"✓ Parsed {len(instructions)} ARM32 instructions")

# Process similarly to ARM64
mba_blocks = detect_mba_blocks(instructions)
# ... rest of the workflow is the same
```

### 3.4 MIPS32/MIPS64 Assembly Processing

**Architecture Details:**
- **Registers**: $0-$31 (32-bit or 64-bit)
- **Bit width**: 32-bit (MIPS32) or 64-bit (MIPS64)
- **Special features**: Delay slots for branch instructions

**Note**: Currently, MIPS assembly is processed using the x86 parser as a template. Full MIPS-specific parsing is planned for future releases.

**Example Workflow:**

```python
from pathlib import Path
from assembly.x86_64.parser import parse_assembly  # Using x86 parser as template
from assembly.x86_64.detector import detect_mba_blocks
from assembly.x86_64.converter import convert_mba_block_to_expression
from optimization.batch_advanced import process_expressions_batch_advanced

# Parse MIPS64 assembly
asm_file = Path("function_mips64.asm")
result = parse_assembly(asm_file)  # Architecture auto-detected
instructions = result["instructions"]

print(f"✓ Parsed {len(instructions)} MIPS64 instructions")

# Detect MBA blocks
mba_blocks = detect_mba_blocks(instructions)
print(f"✓ Found {len(mba_blocks)} MBA blocks")

# Convert and simplify
expressions = []
for block in mba_blocks:
    expr_data = convert_mba_block_to_expression(block)
    if expr_data and expr_data.get("gamba_expression"):
        expressions.append(expr_data["gamba_expression"])

if expressions:
    results = process_expressions_batch_advanced(
        expressions=expressions,
        bitcount=64,  # MIPS64 uses 64-bit
        max_workers=8,
        use_cache=True
    )
```

**Example MIPS64 Assembly File:**

```
test_mba.o:     file format elf64-tradlittlemips

Disassembly of section .text:

0000000000000000 <mba_simple>:
   0:	00851024 	and	v0,a0,a1
   4:	00851026 	xor	v0,a0,a1
   8:	00451020 	add	v0,v0,a1
   c:	03e00008 	jr	ra
  10:	00000000 	nop
```

### 3.5 PowerPC32/PowerPC64 Assembly Processing

**Architecture Details:**
- **Registers**: r0-r31 (32-bit or 64-bit)
- **Bit width**: 32-bit (PowerPC32) or 64-bit (PowerPC64)
- **Special features**: Condition register operations, complex addressing modes

**Note**: Currently, PowerPC assembly is processed using the x86 parser as a template. Full PowerPC-specific parsing is planned for future releases.

**Example Workflow:**

```python
from pathlib import Path
from assembly.x86_64.parser import parse_assembly  # Using x86 parser as template
from assembly.x86_64.detector import detect_mba_blocks
from assembly.x86_64.converter import convert_mba_block_to_expression
from optimization.batch_advanced import process_expressions_batch_advanced

# Parse PowerPC64 assembly
asm_file = Path("function_powerpc64.asm")
result = parse_assembly(asm_file)  # Architecture auto-detected
instructions = result["instructions"]

print(f"✓ Parsed {len(instructions)} PowerPC64 instructions")

# Process similarly to other architectures
mba_blocks = detect_mba_blocks(instructions)
expressions = []
for block in mba_blocks:
    expr_data = convert_mba_block_to_expression(block)
    if expr_data and expr_data.get("gamba_expression"):
        expressions.append(expr_data["gamba_expression"])

if expressions:
    results = process_expressions_batch_advanced(
        expressions=expressions,
        bitcount=64,  # PowerPC64 uses 64-bit
        max_workers=8,
        use_cache=True
    )
```

**Example PowerPC64 Assembly File:**

```
test_mba.o:     file format elf64-powerpc

Disassembly of section .text:

0000000000000000 <mba_simple>:
   0:	7c 04 18 38 	and	r0,r4,r3
   4:	7c 04 18 78 	xor	r0,r4,r3
   8:	7c 00 18 14 	add	r0,r0,r3
   c:	4e 80 00 20 	blr
```

### 3.6 RISC-V32/RISC-V64 Assembly Processing

**Architecture Details:**
- **Registers**: x0-x31 (32-bit or 64-bit)
- **Bit width**: 32-bit (RISC-V32) or 64-bit (RISC-V64)
- **Special features**: Compressed instruction format (C extension)

**Note**: Currently, RISC-V assembly is processed using the x86 parser as a template. Full RISC-V-specific parsing is planned for future releases.

**Example Workflow:**

```python
from pathlib import Path
from assembly.x86_64.parser import parse_assembly  # Using x86 parser as template
from assembly.x86_64.detector import detect_mba_blocks
from assembly.x86_64.converter import convert_mba_block_to_expression
from optimization.batch_advanced import process_expressions_batch_advanced

# Parse RISC-V64 assembly
asm_file = Path("function_riscv64.asm")
result = parse_assembly(asm_file)  # Architecture auto-detected
instructions = result["instructions"]

print(f"✓ Parsed {len(instructions)} RISC-V64 instructions")

# Process similarly to other architectures
mba_blocks = detect_mba_blocks(instructions)
expressions = []
for block in mba_blocks:
    expr_data = convert_mba_block_to_expression(block)
    if expr_data and expr_data.get("gamba_expression"):
        expressions.append(expr_data["gamba_expression"])

if expressions:
    results = process_expressions_batch_advanced(
        expressions=expressions,
        bitcount=64,  # RISC-V64 uses 64-bit
        max_workers=8,
        use_cache=True
    )
```

**Example RISC-V64 Assembly File:**

```
test_mba.o:     file format elf64-littleriscv

Disassembly of section .text:

0000000000000000 <mba_simple>:
   0:	00b57633          	and	a2,a0,a1
   4:	00b54533          	xor	a0,a0,a1
   8:	00c50533          	add	a0,a0,a2
   c:	8082                	ret
```

### 3.7 Multi-Architecture Analysis

Analyze the same code compiled for different architectures:

```python
from pathlib import Path
from assembly.x86_64.parser import parse_assembly as parse_x86
from assembly.arm.parser import parse_assembly as parse_arm
from assembly.x86_64.detector import detect_mba_blocks as detect_mba_x86
from assembly.arm.detector import detect_mba_blocks as detect_mba_arm
from assembly.x86_64.converter import convert_mba_block_to_expression as convert_x86
from assembly.arm.converter import convert_mba_block_to_expression as convert_arm
from optimization.batch_advanced import process_expressions_batch_advanced

def analyze_architecture(arch_name: str, asm_file: Path, bitcount: int):
    """Analyze MBA for a specific architecture"""
    
    # Select parser based on architecture
    if 'arm' in arch_name.lower():
        parse_fn = parse_arm
        detect_fn = detect_mba_arm
        convert_fn = convert_arm
        arch_type = 'arm64' if '64' in arch_name else 'arm32'
    else:
        parse_fn = parse_x86
        detect_fn = detect_mba_x86
        convert_fn = convert_x86
        arch_type = 'x86_64'
    
    # Parse assembly
    if arch_type.startswith('arm'):
        parsed = parse_fn(asm_file, arch=arch_type)
    else:
        parsed = parse_fn(asm_file)
    
    instructions = parsed["instructions"]
    print(f"\n[{arch_name}] Parsed {len(instructions)} instructions")
    
    # Detect MBA blocks
    mba_blocks = detect_fn(instructions)
    print(f"[{arch_name}] Found {len(mba_blocks)} MBA blocks")
    
    # Convert to expressions
    expressions = []
    for block in mba_blocks:
        expr_data = convert_fn(block)
        if expr_data and expr_data.get("gamba_expression"):
            expressions.append(expr_data["gamba_expression"])
    
    # Simplify
    if expressions:
        results = process_expressions_batch_advanced(
            expressions=expressions,
            bitcount=bitcount,
            max_workers=8,
            use_cache=True
        )
        
        success_count = sum(1 for r in results if r["success"])
        print(f"[{arch_name}] Simplified {success_count}/{len(expressions)} expressions")
        
        return {
            'architecture': arch_name,
            'instructions': len(instructions),
            'mba_blocks': len(mba_blocks),
            'expressions': len(expressions),
            'simplified': success_count
        }
    
    return None

# Analyze multiple architectures
architectures = {
    'x86_64': (Path("tests/x86_64/test_mba.asm"), 64),
    'arm64': (Path("tests/arm64/test_mba.asm"), 64),
    'mips64': (Path("tests/mips64/test_mba.asm"), 64),
}

results = {}
for arch_name, (asm_file, bitcount) in architectures.items():
    if asm_file.exists():
        result = analyze_architecture(arch_name, asm_file, bitcount)
        if result:
            results[arch_name] = result

# Compare results
print("\n" + "="*70)
print("Cross-Architecture Comparison")
print("="*70)
for arch, data in results.items():
    print(f"{arch:12} | Instructions: {data['instructions']:3} | "
          f"MBA Blocks: {data['mba_blocks']:2} | "
          f"Simplified: {data['simplified']}/{data['expressions']}")
```

### 3.8 Assembly File Format

GAMBA++ expects assembly files in **objdump format** with addresses and instruction bytes:

**x86/x64 Format:**
```
0x1000: 48 89 d8             	mov    rax,rbx
0x1003: 48 31 c8             	xor    rax,rcx
0x1006: 48 21 d0             	and    rax,rdx
```

**ARM64 Format:**
```
   0:	0a010002 	and	w2, w0, w1
   4:	4a010000 	eor	w0, w0, w1
   8:	0b020400 	add	w0, w0, w2, lsl #1
```

**Generating Assembly Files:**

```bash
# x86_64
objdump -d binary.o > output.asm

# ARM64
aarch64-linux-gnu-objdump -d binary.o > output.asm

# MIPS64
mips64-linux-gnu-objdump -d binary.o > output.asm

# PowerPC64
powerpc64-linux-gnu-objdump -d binary.o > output.asm

# RISC-V64
riscv64-linux-gnu-objdump -d binary.o > output.asm
```

### 3.9 MBA Pattern Detection

GAMBA++ automatically detects three types of MBA patterns across all architectures:

**1. Boolean Chain Pattern:**
Long sequences of boolean operations (and, or, xor, etc.)

```
┌─────┐      ┌─────┐      ┌─────┐      ┌─────┐      ┌──────────┐
│ and │ ───► │ xor │ ───► │ or  │ ───► │ and │ ───► │MBA Block │
└─────┘      └─────┘      └─────┘      └─────┘      └──────────┘
```

**2. Arithmetic-Boolean Pattern:**
Mixed arithmetic and boolean operations

```
┌──────────────┐      ┌──────────────┐      ┌──────────┐
│add/lea/imul  │ ───► │Boolean Chain │ ───► │MBA Block │
└──────────────┘      └──────────────┘      └──────────┘
```

**3. Comparison Chain Pattern:**
Complex comparisons using boolean operations

```
┌─────────────┐      ┌─────────────┐      ┌──────────┐
│ Comparison  │ ───► │Boolean Ops  │ ───► │MBA Block │
└─────────────┘      └─────────────┘      └──────────┘
```

**Detection Example:**

```python
from assembly.x86_64 import detect_mba_blocks

# Detect with custom parameters
mba_blocks = detect_mba_blocks(
    instructions,
    min_boolean_chain=5,      # Minimum boolean operations
    max_block_size=50,        # Maximum instructions per block
    pattern_types=["boolean", "arithmetic_boolean", "comparison"]
)

for block in mba_blocks:
    print(f"Pattern: {block.pattern_type}")
    print(f"Address: 0x{block.start_address:X} - 0x{block.end_address:X}")
    print(f"Instructions: {len(block.instructions)}")
```

**Architecture-Specific Pattern Examples:**

- **x86/x64**: `lea`, `imul` combined with boolean operations
- **ARM**: Embedded shifts (`lsl`, `lsr`, `asr`) in arithmetic operations
- **ARM**: Bitfield instructions (`ubfx`, `sbfx`, `bfi`) as MBA patterns
- **All**: Boolean chains of 5+ consecutive boolean operations

---

## 4. Advanced Usage Patterns

### 4.1 Using Cache Effectively

The cache system stores simplification results to avoid reprocessing:

```python
from optimization.cache import GAMBACache
from optimization.batch_advanced import process_expressions_batch_advanced
from pathlib import Path

# Create or load cache
cache = GAMBACache(Path(".my_cache.json"))

# First run: processes and caches
expressions = ["(x ^ y) + 2*(x & y)"] * 10  # 10 duplicates
results1 = process_expressions_batch_advanced(
    expressions=expressions,
    bitcount=32,
    use_cache=True,
    cache=cache
)

# Second run: uses cache (much faster)
results2 = process_expressions_batch_advanced(
    expressions=expressions,
    bitcount=32,
    use_cache=True,
    cache=cache
)

# Check cache statistics
stats = cache.get_stats()
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate_percent']:.1f}%")
```

**Cache Workflow:**

```
                    ┌─────────────┐
                    │  Expression │
                    └──────┬──────┘
                           │
                           ▼
                    ┌──────────────┐
                    │Cache Exists? │
                    └──────┬───────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
         [Yes]                         [No]
            │                             │
            ▼                             ▼
    ┌──────────────┐            ┌─────────────────┐
    │Return Cached │            │Process with GAMBA│
    └──────┬───────┘            └────────┬────────┘
           │                              │
           │                              ▼
           │                     ┌─────────────────┐
           │                     │ Store in Cache  │
           │                     └────────┬────────┘
           │                              │
           └──────────────┬───────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │Return Result │
                   └──────────────┘
```

### 4.2 Custom Bit Width

Different architectures use different bit widths:

```python
from optimization.batch_advanced import process_expressions_batch_advanced

# 32-bit expressions (x86, ARM32)
results_32 = process_expressions_batch_advanced(
    expressions=expressions,
    bitcount=32,
    max_workers=8
)

# 64-bit expressions (x64, ARM64)
results_64 = process_expressions_batch_advanced(
    expressions=expressions,
    bitcount=64,
    max_workers=8
)
```

### 4.3 Error Handling

```python
from optimization.batch_advanced import process_expressions_batch_advanced

expressions = [
    "(x ^ y) + 2*(x & y)",  # Valid
    "invalid expression!!!",  # Invalid
    "(x | y) - (x & y)",     # Valid
]

results = process_expressions_batch_advanced(
    expressions=expressions,
    bitcount=32,
    max_workers=8,
    use_cache=True
)

# Handle results
for i, result in enumerate(results):
    if result["success"]:
        print(f"✓ Expression {i+1}: {result['simplified']}")
    else:
        print(f"✗ Expression {i+1} failed: {result['error']}")
        print(f"  Original: {result['original']}")
```

---

## 5. CFG Visualization and MBA Function Detection

GAMBA++ provides comprehensive Control Flow Graph (CFG) generation and visualization capabilities, supporting multiple architectures and automatic MBA detection.

### 5.1 Multi-Architecture CFG Generation

GAMBA++ supports CFG generation for 9 different architectures:

- **x86_64**: Intel/AMD 64-bit
- **ARM32**: ARM 32-bit (ARMv7)
- **ARM64**: ARM 64-bit (AArch64)
- **MIPS32**: MIPS 32-bit
- **MIPS64**: MIPS 64-bit
- **PowerPC32**: PowerPC 32-bit
- **PowerPC64**: PowerPC 64-bit
- **RISC-V32**: RISC-V 32-bit
- **RISC-V64**: RISC-V 64-bit

#### Automatic Architecture Detection

The framework automatically detects the architecture from assembly files:

```python
from tests.generate_cfgs import generate_cfgs_for_architecture
from pathlib import Path

# Generate CFGs for a specific architecture
test_dir = Path("tests/x86_64")
generate_cfgs_for_architecture("x86_64", test_dir)

# Output files will be saved to:
# - tests/x86_64/output/cfg_before.png
# - tests/x86_64/output/cfg_after.png
# - tests/x86_64/output/cfg_comparison.png
# - tests/x86_64/output/cfg_analysis.json
```

#### Architecture-Specific Parsing

For manual control, you can specify the architecture:

```python
from assembly.x86_64.parser import parse_assembly as parse_x86
from assembly.arm.parser import parse_assembly as parse_arm
from assembly.common.cfg import build_cfg

# x86_64
parsed_x86 = parse_x86("function_x86.asm")
instructions_x86 = parsed_x86["instructions"]
cfg_x86 = build_cfg(instructions_x86, arch='x86_64')

# ARM64
parsed_arm = parse_arm("function_arm.asm", arch='arm64')
instructions_arm = parsed_arm["instructions"]
cfg_arm = build_cfg(instructions_arm, arch='arm64')
```

### 5.2 Detecting MBA Functions

GAMBA++ can detect functions containing MBA patterns across all supported architectures:

```python
from assembly.common.mba_function_detector import extract_function_boundaries_from_symbols
from assembly.x86_64.parser import parse_assembly
from assembly.x86_64.detector import detect_mba_blocks

# Parse assembly file
parsed = parse_assembly("function.asm")
instructions = parsed["instructions"]

# Extract function boundaries from symbols file
function_boundaries = extract_function_boundaries_from_symbols("function.symbols")

# Detect MBA blocks
mba_blocks = detect_mba_blocks(instructions)

# Match MBA blocks to functions
for func in function_boundaries:
    func_mba_blocks = [b for b in mba_blocks 
                       if func['start'] <= b.start_address < func['end']]
    if func_mba_blocks:
        print(f"Function: {func['name']}")
        print(f"  MBA blocks: {len(func_mba_blocks)}")
        print(f"  Instructions: {func['end'] - func['start']}")
```

### 5.3 Building Control Flow Graphs

Build CFG from instructions for any supported architecture:

```python
from assembly.common.cfg import build_cfg
from assembly.common.capstone_wrapper import disassemble_with_capstone

# Build CFG (architecture is auto-detected or specified)
cfg = build_cfg(instructions, arch='x86_64')

print(f"CFG has {len(cfg.blocks)} basic blocks")
for block in cfg.blocks:
    print(f"  Block 0x{block.start_address:X}: {len(block.instructions)} instructions")
    print(f"    Predecessors: {len(block.predecessors)}")
    print(f"    Successors: {len(block.successors)}")
```

### 5.4 Visualizing CFG

Render CFG with MBA blocks highlighted using Graphviz:

```python
from visualization.cfg_renderer import render_cfg, render_cfg_comparison
from pathlib import Path

# Render CFG before simplification
output_path = Path("output/cfg_before.png")
render_cfg(
    cfg=cfg,
    mba_blocks=mba_blocks,
    output_path=output_path
)

# Render CFG after simplification (with simplified blocks)
render_cfg(
    cfg=cfg_after,
    mba_blocks=[],  # Empty - blocks are already simplified
    simplified_blocks=simplified_blocks,  # Show simplified expressions
    output_path=Path("output/cfg_after.png")
)

# Render side-by-side comparison
render_cfg_comparison(
    cfg_before=cfg,
    cfg_after=cfg_after,
    mba_blocks_before=mba_blocks,
    mba_blocks_after=[],  # Empty after simplification
    simplified_blocks=simplified_blocks,
    mba_attempted_blocks=mba_attempted_blocks,  # Blocks where simplification failed
    output_path=Path("output/cfg_comparison.png")
)
```

**CFG Visualization Features:**
- **Rectangular nodes** with rounded corners for better readability
- **Color coding**: 
  - Red border: MBA blocks before simplification
  - Green fill: Successfully simplified blocks
  - Yellow fill: Blocks where simplification was attempted but failed
  - Blue border: Regular basic blocks
- **Instruction count** displayed at the top of each block
- **Dynamic sizing**: Blocks automatically resize based on content
- **Side-by-side comparison**: Before and after CFGs aligned horizontally

### 5.5 Complete Workflow: Multi-Architecture Analysis

Complete example for analyzing and visualizing MBA across different architectures:

```python
from pathlib import Path
from assembly.x86_64.parser import parse_assembly as parse_x86
from assembly.arm.parser import parse_assembly as parse_arm
from assembly.x86_64.detector import detect_mba_blocks as detect_mba_x86
from assembly.arm.detector import detect_mba_blocks as detect_mba_arm
from assembly.x86_64.converter import convert_mba_block_to_expression as convert_x86
from assembly.arm.converter import convert_mba_block_to_expression as convert_arm
from assembly.common.cfg import build_cfg
from assembly.common.mba_function_detector import extract_function_boundaries_from_symbols
from visualization.cfg_renderer import render_cfg, render_cfg_comparison
from optimization.batch_advanced import process_expressions_batch_advanced

def analyze_architecture(arch_name: str, asm_file: Path, symbols_file: Path):
    """Analyze MBA for a specific architecture"""
    
    # Select parser and detector based on architecture
    if 'arm' in arch_name.lower():
        parse_assembly = parse_arm
        detect_mba = detect_mba_arm
        convert_mba = convert_arm
        arch_type = 'arm64' if '64' in arch_name else 'arm32'
    else:
        parse_assembly = parse_x86
        detect_mba = detect_mba_x86
        convert_mba = convert_x86
        arch_type = 'x86_64'
    
    # 1. Parse assembly
    if arch_type.startswith('arm'):
        parsed = parse_assembly(asm_file, arch=arch_type)
    else:
        parsed = parse_assembly(asm_file)
    instructions = parsed["instructions"]
    print(f"✓ Parsed {len(instructions)} instructions for {arch_name}")
    
    # 2. Extract function boundaries
    function_boundaries = []
    if symbols_file.exists():
        function_boundaries = extract_function_boundaries_from_symbols(str(symbols_file))
        print(f"✓ Found {len(function_boundaries)} functions")
    
    # 3. Detect MBA blocks
    mba_blocks = detect_mba(instructions)
    print(f"✓ Detected {len(mba_blocks)} MBA blocks")
    
    # 4. Build CFG before simplification
    cfg_before = build_cfg(instructions, arch=arch_type)
    print(f"✓ CFG has {len(cfg_before.blocks)} basic blocks")
    
    # 5. Render CFG before
    output_dir = Path(f"output/{arch_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    render_cfg(cfg_before, mba_blocks, output_dir / "cfg_before.png")
    
    # 6. Convert MBA blocks to expressions
    expressions = []
    for block in mba_blocks:
        expr_data = convert_mba(block)
        if expr_data and expr_data.get('gamba_expression'):
            expressions.append(expr_data['gamba_expression'])
    
    # 7. Simplify expressions
    simplified_blocks = []
    if expressions:
        results = process_expressions_batch_advanced(
            expressions=expressions,
            bitcount=64 if '64' in arch_name else 32,
            max_workers=8,
            use_cache=True
        )
        
        # Create simplified blocks mapping
        for i, result in enumerate(results):
            if result["success"] and i < len(mba_blocks):
                simplified_blocks.append({
                    'start': mba_blocks[i].start_address,
                    'end': mba_blocks[i].end_address,
                    'original_expr': expressions[i],
                    'simplified_expr': result['simplified']
                })
    
    # 8. Build CFG after (create simplified representation)
    cfg_after = cfg_before  # In practice, rebuild from simplified code
    
    # 9. Render CFG after
    render_cfg(cfg_after, [], simplified_blocks=simplified_blocks,
               output_path=output_dir / "cfg_after.png")
    
    # 10. Render comparison
    render_cfg_comparison(
        cfg_before, cfg_after, mba_blocks, [],
        simplified_blocks=simplified_blocks,
        output_path=output_dir / "cfg_comparison.png"
    )
    
    print(f"✓ Analysis complete for {arch_name}")
    print(f"  Output: {output_dir}")

# Analyze multiple architectures
architectures = ['x86_64', 'arm64', 'mips64']
for arch in architectures:
    asm_file = Path(f"tests/{arch}/test_mba.asm")
    symbols_file = Path(f"tests/{arch}/test_mba.symbols")
    if asm_file.exists():
        analyze_architecture(arch, asm_file, symbols_file)
```

### 5.6 Automated CFG Generation for All Architectures

Use the provided script to automatically generate CFGs for all test architectures:

```bash
# Build all test binaries and generate CFGs
python tests/build_all_and_generate_cfgs.py
```

This script will:
1. Build test binaries for all 9 architectures
2. Generate assembly files using objdump
3. Parse assembly and detect MBA blocks
4. Build CFG before simplification
5. Simplify MBA expressions
6. Build CFG after simplification
7. Generate visualization images (before, after, comparison)
8. Save analysis results to JSON

**Output Structure:**
```
tests/
├── x86_64/
│   └── output/
│       ├── cfg_before.png
│       ├── cfg_after.png
│       ├── cfg_comparison.png
│       └── cfg_analysis.json
├── arm64/
│   └── output/
│       └── ...
└── ...
```

**CFG Analysis JSON Format:**
```json
{
  "architecture": "x86_64",
  "summary": {
    "instruction_count": 77,
    "mba_block_count": 3,
    "expression_count": 3,
    "simplified_count": 3,
    "basic_block_count": 13,
    "function_count": 8
  },
  "mba_blocks": [...],
  "expressions": [...],
  "simplified_blocks": [...],
  "functions": [...]
}
```

### 5.7 Working with Different Architectures

#### Architecture-Specific Considerations

**x86/x64:**
- Uses 32-bit or 64-bit registers (eax/rax, ebx/rbx, etc.)
- Supports complex addressing modes (e.g., `[rbp-0x10]`)
- Handles instruction prefixes and modifiers

**ARM32/ARM64:**
- ARM32 uses 32-bit registers (r0-r15)
- ARM64 uses 64-bit registers (x0-x30) and 32-bit subregisters (w0-w30)
- Supports embedded shifts in operands (e.g., `add x0, x1, x2, lsl #3`)
- Handles conditional execution and bitfield instructions

**MIPS32/MIPS64:**
- Uses 32-bit or 64-bit general-purpose registers
- Handles delay slots for branch instructions
- Supports coprocessor instructions

**PowerPC32/PowerPC64:**
- Uses 32-bit or 64-bit general-purpose registers
- Handles condition register operations
- Supports complex addressing modes

**RISC-V32/RISC-V64:**
- Uses 32-bit or 64-bit general-purpose registers
- Supports compressed instruction format (C extension)
- Handles standard RISC-V instruction set

#### Example: Cross-Architecture Comparison

```python
# Compare MBA patterns across architectures
architectures = {
    'x86_64': ('tests/x86_64/test_mba.asm', 'x86_64'),
    'arm64': ('tests/arm64/test_mba.asm', 'arm64'),
    'mips64': ('tests/mips64/test_mba.asm', 'mips64'),
}

results = {}
for arch_name, (asm_file, arch_type) in architectures.items():
    # Parse and analyze
    # ... (use architecture-specific parsers)
    
    # Store results
    results[arch_name] = {
        'mba_blocks': len(mba_blocks),
        'expressions': len(expressions),
        'simplified': len(simplified_blocks)
    }

# Compare results
for arch, data in results.items():
    print(f"{arch}: {data['mba_blocks']} MBA blocks, "
          f"{data['simplified']}/{data['expressions']} simplified")
```

## 6. Performance Optimization

### 5.1 Choosing the Right Method

**For Small Batches (< 10 expressions):**
```python
from optimization.parallel_advanced import process_expressions_parallel_advanced

results = process_expressions_parallel_advanced(
    expressions=expressions,
    max_workers=8,
    use_cache=True
)
```

**For Large Batches (> 10 expressions):**
```python
from optimization.batch_advanced import process_expressions_batch_advanced

results = process_expressions_batch_advanced(
    expressions=expressions,
    max_workers=8,
    batch_size=10,  # Optimal batch size
    use_cache=True
)
```

### 5.2 Performance Comparison

```
┌────────────┐
│ Sequential │ ────── 2.06x ──────► ┌──────────┐
│  (1.00x)   │                      │ Parallel │
└────────────┘                      │  (2.06x) │
       │                             └────┬─────┘
       │                                  │
       │                                  │ 1.73x
       │                                  │
       │                                  ▼
       │                            ┌────────────┐
       │                            │ Optimized  │
       │                            │  (3.32x)   │
       │                            └────────────┘
       │                                  ▲
       │                                  │
       └────────── 3.32x ─────────────────┘
```

**Performance Tips:**

1. **Enable Cache**: Always use `use_cache=True` for repeated expressions
2. **Use Batch Processing**: For >10 expressions, use `batch_advanced`
3. **Adjust Workers**: Match `max_workers` to your CPU cores
4. **Batch Size**: Optimal batch size is 10-20 expressions

### 5.3 Monitoring Performance

```python
import time
from optimization.batch_advanced import process_expressions_batch_advanced

expressions = [...]  # Your expressions

start = time.time()
results = process_expressions_batch_advanced(
    expressions=expressions,
    bitcount=32,
    max_workers=8,
    use_cache=True,
    show_progress=True
)
elapsed = time.time() - start

print(f"Processed {len(expressions)} expressions in {elapsed:.2f}s")
print(f"Average: {elapsed/len(expressions)*1000:.2f}ms per expression")
```

---

## Complete Example: End-to-End Workflow

Here's a complete example combining all features:

```python
from pathlib import Path
from assembly.x86_64 import parse_assembly, detect_mba_blocks, convert_mba_block_to_expression
from optimization.batch_advanced import process_expressions_batch_advanced
from optimization.cache import GAMBACache

# 1. Parse assembly
asm_file = Path("obfuscated_function.asm")
result = parse_assembly(asm_file)
instructions = result["instructions"]
print(f"✓ Parsed {len(instructions)} instructions")

# 2. Detect MBA blocks
mba_blocks = detect_mba_blocks(instructions, min_boolean_chain=5)
print(f"✓ Found {len(mba_blocks)} MBA blocks")

# 3. Convert to expressions
expressions = []
block_info = []
for block in mba_blocks:
    expr_data = convert_mba_block_to_expression(block)
    if expr_data and expr_data.get("gamba_expression"):
        expressions.append(expr_data["gamba_expression"])
        block_info.append({
            "address": block.start_address,
            "pattern": block.pattern_type
        })

print(f"✓ Converted {len(expressions)} expressions")

# 4. Simplify with optimizations
cache = GAMBACache(Path(".deobfuscation_cache.json"))
results = process_expressions_batch_advanced(
    expressions=expressions,
    bitcount=64,
    max_workers=8,
    batch_size=10,
    use_cache=True,
    cache=cache,
    show_progress=True
)

# 5. Display results
print("\n" + "="*70)
print("Deobfuscation Results")
print("="*70)
for i, (result, info) in enumerate(zip(results, block_info)):
    if result["success"]:
        print(f"\nBlock {i+1} @ 0x{info['address']:X} ({info['pattern']}):")
        print(f"  Original:   {result['original']}")
        print(f"  Simplified: {result['simplified']}")
    else:
        print(f"\nBlock {i+1} @ 0x{info['address']:X}: FAILED")
        print(f"  Error: {result['error']}")

# 6. Cache statistics
stats = cache.get_stats()
print(f"\n✓ Cache: {stats['hits']} hits, {stats['hit_rate_percent']:.1f}% hit rate")
```

**Complete Workflow Diagram:**

```
                    ┌──────────────┐
                    │Assembly File │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │Parse Assembly│
                    └──────┬───────┘
                           │
                           ▼
                    ┌─────────────────┐
                    │Detect MBA Blocks│
                    └──────┬──────────┘
                           │
                           ▼
                    ┌─────────────────────┐
                    │Convert to Expressions│
                    └──────┬───────────────┘
                           │
                           ▼
                    ┌─────────────────┐
                    │Batch Processing  │
                    └──────┬──────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Cache Check   │
                    └──────┬───────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
         [Hit]                         [Miss]
            │                             │
            ▼                             ▼
    ┌──────────────┐            ┌─────────────────┐
    │Return Cached │            │ GAMBA Simplify  │
    └──────┬───────┘            └────────┬────────┘
           │                              │
           │                              ▼
           │                     ┌─────────────────┐
           │                     │  Update Cache   │
           │                     └────────┬────────┘
           │                              │
           └──────────────┬───────────────┘
                          │
                          ▼
                    ┌──────────┐
                    │ Results  │
                    └────┬─────┘
                         │
                         ▼
                ┌──────────────────┐
                │Deobfuscated Code  │
                └──────────────────┘
```

---

## Summary

GAMBA++ provides multiple ways to simplify MBA expressions:

1. **Direct Input**: Pass expressions as Python strings
2. **File Input**: Read from text files or datasets
3. **Assembly Processing**: Automatically detect and convert assembly to expressions

All methods support advanced optimizations:
- Parallel processing (8 cores)
- Caching for repeated expressions
- Batch processing for efficiency
- Early termination for simple expressions

For best performance, use `process_expressions_batch_advanced` with `use_cache=True` and `max_workers=8`.


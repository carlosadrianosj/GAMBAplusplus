
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
  1c:	d503201f 	nop

0000000000000020 <mba_bitfield>:
  20:	d3483c00 	ubfx	x0, x0, #8, #8
  24:	0b414000 	add	w0, w0, w1, lsr #16
  28:	d65f03c0 	ret
  2c:	d503201f 	nop

0000000000000030 <mba_conditional>:
  30:	4b010002 	sub	w2, w0, w1
  34:	7100005f 	cmp	w2, #0x0
  38:	1a81c000 	csel	w0, w0, w1, gt
  3c:	d65f03c0 	ret

0000000000000040 <mba_boolean_chain>:
  40:	0a010001 	and	w1, w0, w1
  44:	2a020021 	orr	w1, w1, w2
  48:	4a000020 	eor	w0, w1, w0
  4c:	d65f03c0 	ret

0000000000000050 <mba_arithmetic_boolean>:
  50:	0b010002 	add	w2, w0, w1
  54:	0a010000 	and	w0, w0, w1
  58:	4a000040 	eor	w0, w2, w0
  5c:	531f7800 	lsl	w0, w0, #1
  60:	d65f03c0 	ret

Disassembly of section .text.startup:

0000000000000000 <main>:
   0:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
   4:	52800041 	mov	w1, #0x2                   	// #2
   8:	52800020 	mov	w0, #0x1                   	// #1
   c:	910003fd 	mov	x29, sp
  10:	94000000 	bl	0 <main>
  14:	52800081 	mov	w1, #0x4                   	// #4
  18:	528000a2 	mov	w2, #0x5                   	// #5
  1c:	2a0003e3 	mov	w3, w0
  20:	52800060 	mov	w0, #0x3                   	// #3
  24:	94000000 	bl	10 <main+0x10>
  28:	528000e1 	mov	w1, #0x7                   	// #7
  2c:	0b000063 	add	w3, w3, w0
  30:	528000c0 	mov	w0, #0x6                   	// #6
  34:	94000000 	bl	20 <main+0x20>
  38:	52800121 	mov	w1, #0x9                   	// #9
  3c:	0b000063 	add	w3, w3, w0
  40:	52800100 	mov	w0, #0x8                   	// #8
  44:	94000000 	bl	30 <main+0x30>
  48:	52800182 	mov	w2, #0xc                   	// #12
  4c:	0b000063 	add	w3, w3, w0
  50:	52800161 	mov	w1, #0xb                   	// #11
  54:	52800140 	mov	w0, #0xa                   	// #10
  58:	94000000 	bl	40 <main+0x40>
  5c:	0b000063 	add	w3, w3, w0
  60:	528001c1 	mov	w1, #0xe                   	// #14
  64:	528001a0 	mov	w0, #0xd                   	// #13
  68:	94000000 	bl	50 <main+0x50>
  6c:	0b000060 	add	w0, w3, w0
  70:	a8c17bfd 	ldp	x29, x30, [sp], #16
  74:	d65f03c0 	ret

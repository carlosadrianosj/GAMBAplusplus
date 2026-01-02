
test_mba.o:     file format elf64-littleaarch64


Disassembly of section .text:

0000000000000000 <mba_simple>:
   0:	0a010002 	and	w2, w0, w1
   4:	4a010000 	eor	w0, w0, w1
   8:	0b020400 	add	w0, w0, w2, lsl #1
   c:	d65f03c0 	ret

0000000000000010 <mba_shift>:
  10:	4a000422 	eor	w2, w1, w0, lsl #1
  14:	0a800420 	and	w0, w1, w0, asr #1
  18:	0b000040 	add	w0, w2, w0
  1c:	d65f03c0 	ret

0000000000000020 <mba_rotate>:
  20:	131d7c03 	asr	w3, w0, #29
  24:	53051022 	lsl	w2, w1, #27
  28:	2a000c60 	orr	w0, w3, w0, lsl #3
  2c:	2a811441 	orr	w1, w2, w1, asr #5
  30:	0a010002 	and	w2, w0, w1
  34:	4a010000 	eor	w0, w0, w1
  38:	0b020400 	add	w0, w0, w2, lsl #1
  3c:	d65f03c0 	ret

0000000000000040 <mba_boolean_chain>:
  40:	0a010004 	and	w4, w0, w1
  44:	2a020083 	orr	w3, w4, w2
  48:	4a000063 	eor	w3, w3, w0
  4c:	0a010061 	and	w1, w3, w1
  50:	2a020022 	orr	w2, w1, w2
  54:	4a040040 	eor	w0, w2, w4
  58:	d65f03c0 	ret
  5c:	d503201f 	nop

0000000000000060 <mba_comparison>:
  60:	0a010002 	and	w2, w0, w1
  64:	4a010001 	eor	w1, w0, w1
  68:	52800020 	mov	w0, #0x1                   	// #1
  6c:	2b020421 	adds	w1, w1, w2, lsl #1
  70:	5a805400 	cneg	w0, w0, mi  // mi = first
  74:	d65f03c0 	ret
  78:	d503201f 	nop
  7c:	d503201f 	nop

0000000000000080 <mba_arithmetic_boolean>:
  80:	0b010002 	add	w2, w0, w1
  84:	0a010003 	and	w3, w0, w1
  88:	4a030042 	eor	w2, w2, w3
  8c:	2a010000 	orr	w0, w0, w1
  90:	0b020400 	add	w0, w0, w2, lsl #1
  94:	d65f03c0 	ret
  98:	d503201f 	nop
  9c:	d503201f 	nop

00000000000000a0 <mba_stack>:
  a0:	0a010002 	and	w2, w0, w1
  a4:	4a010000 	eor	w0, w0, w1
  a8:	0b020400 	add	w0, w0, w2, lsl #1
  ac:	d65f03c0 	ret

Disassembly of section .text.startup:

0000000000000000 <main>:
   0:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
   4:	52800041 	mov	w1, #0x2                   	// #2
   8:	52800020 	mov	w0, #0x1                   	// #1
   c:	910003fd 	mov	x29, sp
  10:	94000000 	bl	0 <main>
  14:	52800081 	mov	w1, #0x4                   	// #4
  18:	2a0003e5 	mov	w5, w0
  1c:	52800060 	mov	w0, #0x3                   	// #3
  20:	94000000 	bl	10 <main+0x10>
  24:	528000c1 	mov	w1, #0x6                   	// #6
  28:	0b0000a5 	add	w5, w5, w0
  2c:	528000a0 	mov	w0, #0x5                   	// #5
  30:	94000000 	bl	20 <main+0x20>
  34:	0b0000a5 	add	w5, w5, w0
  38:	52800122 	mov	w2, #0x9                   	// #9
  3c:	52800101 	mov	w1, #0x8                   	// #8
  40:	528000e0 	mov	w0, #0x7                   	// #7
  44:	94000000 	bl	40 <main+0x40>
  48:	52800161 	mov	w1, #0xb                   	// #11
  4c:	0b0000a5 	add	w5, w5, w0
  50:	52800140 	mov	w0, #0xa                   	// #10
  54:	94000000 	bl	60 <main+0x60>
  58:	528001a1 	mov	w1, #0xd                   	// #13
  5c:	0b0000a5 	add	w5, w5, w0
  60:	52800180 	mov	w0, #0xc                   	// #12
  64:	94000000 	bl	80 <main+0x80>
  68:	528001e1 	mov	w1, #0xf                   	// #15
  6c:	0b0000a5 	add	w5, w5, w0
  70:	528001c0 	mov	w0, #0xe                   	// #14
  74:	94000000 	bl	a0 <mba_stack>
  78:	a8c17bfd 	ldp	x29, x30, [sp], #16
  7c:	0b0000a0 	add	w0, w5, w0
  80:	d65f03c0 	ret

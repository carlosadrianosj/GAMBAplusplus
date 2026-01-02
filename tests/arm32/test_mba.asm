
test_mba.o:     file format elf32-littlearm


Disassembly of section .text:

00000000 <mba_simple>:
   0:	ea00 0301 	and.w	r3, r0, r1
   4:	4048      	eors	r0, r1
   6:	eb00 0043 	add.w	r0, r0, r3, lsl #1
   a:	4770      	bx	lr

0000000c <mba_boolean_chain>:
   c:	4001      	ands	r1, r0
   e:	4311      	orrs	r1, r2
  10:	4048      	eors	r0, r1
  12:	4770      	bx	lr

Disassembly of section .text.startup:

00000000 <main>:
   0:	b508      	push	{r3, lr}
   2:	2102      	movs	r1, #2
   4:	2001      	movs	r0, #1
   6:	f7ff fffe 	bl	0 <main>
   a:	2205      	movs	r2, #5
   c:	4603      	mov	r3, r0
   e:	2104      	movs	r1, #4
  10:	2003      	movs	r0, #3
  12:	f7ff fffe 	bl	c <main+0xc>
  16:	4418      	add	r0, r3
  18:	bd08      	pop	{r3, pc}
  1a:	bf00      	nop

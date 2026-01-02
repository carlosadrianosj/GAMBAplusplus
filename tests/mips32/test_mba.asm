
test_mba.o:     file format elf32-tradbigmips


Disassembly of section .text:

00000000 <mba_simple>:
   0:	00851024 	and	v0,a0,a1
   4:	00021040 	sll	v0,v0,0x1
   8:	00852026 	xor	a0,a0,a1
   c:	03e00008 	jr	ra
  10:	00441021 	addu	v0,v0,a0

00000014 <mba_boolean_chain>:
  14:	00852824 	and	a1,a0,a1
  18:	00a62825 	or	a1,a1,a2
  1c:	03e00008 	jr	ra
  20:	00a41026 	xor	v0,a1,a0
	...

Disassembly of section .text.startup:

00000000 <main>:
   0:	3c1c0000 	lui	gp,0x0
   4:	279c0000 	addiu	gp,gp,0
   8:	0399e021 	addu	gp,gp,t9
   c:	27bdffe0 	addiu	sp,sp,-32
  10:	8f990000 	lw	t9,0(gp)
  14:	24050002 	li	a1,2
  18:	24040001 	li	a0,1
  1c:	afbf001c 	sw	ra,28(sp)
  20:	afbc0010 	sw	gp,16(sp)
  24:	0320f809 	jalr	t9
  28:	24060005 	li	a2,5
  2c:	24050004 	li	a1,4
  30:	8fbc0010 	lw	gp,16(sp)
  34:	24040003 	li	a0,3
  38:	8f990000 	lw	t9,0(gp)
  3c:	0320f809 	jalr	t9
  40:	00401825 	move	v1,v0
  44:	8fbf001c 	lw	ra,28(sp)
  48:	00621021 	addu	v0,v1,v0
  4c:	03e00008 	jr	ra
  50:	27bd0020 	addiu	sp,sp,32

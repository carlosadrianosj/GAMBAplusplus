
test_mba.o:     file format elf64-tradbigmips


Disassembly of section .text:

0000000000000000 <mba_simple>:
   0:	00851024 	and	v0,a0,a1
   4:	00021040 	sll	v0,v0,0x1
   8:	00852026 	xor	a0,a0,a1
   c:	03e00008 	jr	ra
  10:	00441021 	addu	v0,v0,a0
	...

Disassembly of section .text.startup:

0000000000000000 <main>:
   0:	3c030000 	lui	v1,0x0
   4:	0079182d 	daddu	v1,v1,t9
   8:	64630000 	daddiu	v1,v1,0
   c:	dc790000 	ld	t9,0(v1)
  10:	24050002 	li	a1,2
  14:	03200008 	jr	t9
  18:	24040001 	li	a0,1
  1c:	00000000 	nop

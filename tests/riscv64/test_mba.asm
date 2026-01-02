
test_mba.o:     file format elf64-littleriscv


Disassembly of section .text:

0000000000000000 <mba_simple>:
   0:	00b577b3          	and	a5,a0,a1
   4:	0017979b          	slliw	a5,a5,0x1
   8:	00b54533          	xor	a0,a0,a1
   c:	00a7853b          	addw	a0,a5,a0
  10:	00008067          	ret

Disassembly of section .text.startup:

0000000000000000 <main>:
   0:	00200593          	li	a1,2
   4:	00100513          	li	a0,1
   8:	00000317          	auipc	t1,0x0
   c:	00030067          	jr	t1 # 8 <main+0x8>

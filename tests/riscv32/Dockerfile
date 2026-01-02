FROM debian:bullseye

RUN apt-get update && apt-get install -y \
    gcc-riscv64-linux-gnu \
    binutils-riscv64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY test_mba.c .

RUN riscv64-linux-gnu-gcc -march=rv32im -mabi=ilp32 -O2 -c test_mba.c -o test_mba.o && \
    riscv64-linux-gnu-objdump -d test_mba.o > test_mba.asm && \
    riscv64-linux-gnu-nm test_mba.o > test_mba.symbols

CMD ["cat", "test_mba.asm"]


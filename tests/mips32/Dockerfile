FROM debian:bullseye

RUN apt-get update && apt-get install -y \
    gcc-mips-linux-gnu \
    binutils-mips-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY test_mba.c .

RUN mips-linux-gnu-gcc -O2 -c test_mba.c -o test_mba.o && \
    mips-linux-gnu-objdump -d test_mba.o > test_mba.asm && \
    mips-linux-gnu-nm test_mba.o > test_mba.symbols

CMD ["cat", "test_mba.asm"]


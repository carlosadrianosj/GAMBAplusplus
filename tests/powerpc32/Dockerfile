FROM debian:bullseye

RUN apt-get update && apt-get install -y \
    gcc-powerpc-linux-gnu \
    binutils-powerpc-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY test_mba.c .

RUN powerpc-linux-gnu-gcc -O2 -c test_mba.c -o test_mba.o && \
    powerpc-linux-gnu-objdump -d test_mba.o > test_mba.asm && \
    powerpc-linux-gnu-nm test_mba.o > test_mba.symbols

CMD ["cat", "test_mba.asm"]


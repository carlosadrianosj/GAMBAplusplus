FROM debian:bullseye

RUN apt-get update && apt-get install -y \
    gcc-powerpc64-linux-gnu \
    binutils-powerpc64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY test_mba.c .

RUN powerpc64-linux-gnu-gcc -O2 -c test_mba.c -o test_mba.o && \
    powerpc64-linux-gnu-objdump -d test_mba.o > test_mba.asm && \
    powerpc64-linux-gnu-nm test_mba.o > test_mba.symbols

CMD ["cat", "test_mba.asm"]


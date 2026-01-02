FROM debian:bullseye

RUN apt-get update && apt-get install -y \
    gcc-mips64-linux-gnuabi64 \
    binutils-mips64-linux-gnuabi64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY test_mba.c .

RUN mips64-linux-gnuabi64-gcc -O2 -c test_mba.c -o test_mba.o && \
    mips64-linux-gnuabi64-objdump -d test_mba.o > test_mba.asm && \
    mips64-linux-gnuabi64-nm test_mba.o > test_mba.symbols

CMD ["cat", "test_mba.asm"]


FROM debian:bullseye

RUN apt-get update && apt-get install -y \
    gcc-aarch64-linux-gnu \
    binutils-aarch64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY test_mba.c .

RUN aarch64-linux-gnu-gcc -O2 -c test_mba.c -o test_mba.o && \
    aarch64-linux-gnu-objdump -d test_mba.o > test_mba.asm && \
    aarch64-linux-gnu-nm test_mba.o > test_mba.symbols

CMD ["cat", "test_mba.asm"]


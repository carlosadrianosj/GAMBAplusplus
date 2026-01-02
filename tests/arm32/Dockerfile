FROM debian:bullseye

RUN apt-get update && apt-get install -y \
    gcc-arm-linux-gnueabihf \
    binutils-arm-linux-gnueabihf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY test_mba.c .

RUN arm-linux-gnueabihf-gcc -O2 -c test_mba.c -o test_mba.o && \
    arm-linux-gnueabihf-objdump -d test_mba.o > test_mba.asm && \
    arm-linux-gnueabihf-nm test_mba.o > test_mba.symbols

CMD ["cat", "test_mba.asm"]


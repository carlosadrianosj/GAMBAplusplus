FROM debian:bullseye

RUN apt-get update && apt-get install -y \
    gcc \
    binutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY test_mba.c .

RUN gcc -O2 -c test_mba.c -o test_mba.o && \
    objdump -d test_mba.o > test_mba.asm && \
    nm test_mba.o > test_mba.symbols

CMD ["cat", "test_mba.asm"]


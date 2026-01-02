// MBA Test Functions for RISC-V 64-bit
__attribute__((noinline))
int mba_simple(int x, int y) {
    return (x ^ y) + 2 * (x & y);
}

int main() {
    return mba_simple(1, 2);
}


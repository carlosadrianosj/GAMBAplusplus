// MBA Test Functions for ARM64
// Compile with: aarch64-linux-gnu-gcc -O2 -c test_mba.c -o test_mba.o

// Simple MBA: (x ^ y) + 2*(x & y) = x + y
__attribute__((noinline))
int mba_simple(int x, int y) {
    return (x ^ y) + 2 * (x & y);
}

// MBA with embedded shifts: add x0, x1, x2, lsl #3
__attribute__((noinline))
int mba_embedded_shift(int x, int y, int z) {
    return x + (y << 3) + z;
}

// MBA with bitfield operations
__attribute__((noinline))
int mba_bitfield(int x, int y) {
    int a = (x >> 8) & 0xFF;  // ubfx pattern
    int b = (y >> 16) & 0xFFFF;
    return a + b;
}

// Conditional selection MBA
__attribute__((noinline))
int mba_conditional(int x, int y) {
    int cmp = x - y;
    return (cmp > 0) ? x : y;  // csel pattern
}

// Boolean chain with ARM instructions
__attribute__((noinline))
int mba_boolean_chain(int x, int y, int z) {
    int t1 = x & y;
    int t2 = t1 | z;
    int t3 = t2 ^ x;
    return t3;
}

// Arithmetic-Boolean mix
__attribute__((noinline))
int mba_arithmetic_boolean(int x, int y) {
    int a = x + y;
    int b = x & y;
    int c = a ^ b;
    return c * 2;
}

int main() {
    return mba_simple(1, 2) + 
           mba_embedded_shift(3, 4, 5) + 
           mba_bitfield(6, 7) +
           mba_conditional(8, 9) +
           mba_boolean_chain(10, 11, 12) +
           mba_arithmetic_boolean(13, 14);
}


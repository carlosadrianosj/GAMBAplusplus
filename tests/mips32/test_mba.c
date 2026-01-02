// MBA Test Functions for MIPS32
// Compile with: mips-linux-gnu-gcc -O2 -c test_mba.c -o test_mba.o

__attribute__((noinline))
int mba_simple(int x, int y) {
    return (x ^ y) + 2 * (x & y);
}

__attribute__((noinline))
int mba_boolean_chain(int x, int y, int z) {
    int t1 = x & y;
    int t2 = t1 | z;
    int t3 = t2 ^ x;
    return t3;
}

int main() {
    return mba_simple(1, 2) + mba_boolean_chain(3, 4, 5);
}


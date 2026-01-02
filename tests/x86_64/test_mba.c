// MBA Test Functions for x86_64
// Compile with: gcc -O2 -c test_mba.c -o test_mba.o

// Simple MBA: (x ^ y) + 2*(x & y) = x + y
__attribute__((noinline))
int mba_simple(int x, int y) {
    return (x ^ y) + 2 * (x & y);
}

// Complex MBA with shifts
__attribute__((noinline))
int mba_shift(int x, int y) {
    return ((x << 1) ^ y) + ((x >> 1) & y);
}

// MBA with rotates
__attribute__((noinline))
int mba_rotate(int x, int y) {
    int a = (x << 3) | (x >> 29);  // rol x, 3
    int b = (y >> 5) | (y << 27);  // ror y, 5
    return (a ^ b) + 2 * (a & b);
}

// Boolean chain MBA
__attribute__((noinline))
int mba_boolean_chain(int x, int y, int z) {
    int t1 = x & y;
    int t2 = t1 | z;
    int t3 = t2 ^ x;
    int t4 = t3 & y;
    int t5 = t4 | z;
    return t5 ^ (x & y);
}

// Comparison chain MBA
__attribute__((noinline))
int mba_comparison(int x, int y) {
    int cmp = (x ^ y) + 2 * (x & y);
    int result = 0;
    if (cmp == 0) result = 1;
    else if (cmp < 0) result = -1;
    else result = 1;
    return result;
}

// Arithmetic-Boolean mix
__attribute__((noinline))
int mba_arithmetic_boolean(int x, int y) {
    int a = x + y;
    int b = x & y;
    int c = a ^ b;
    int d = c * 2;
    return d + (x | y);
}

// Stack variable MBA
__attribute__((noinline))
int mba_stack(int x, int y) {
    int local1 = x ^ y;
    int local2 = x & y;
    int local3 = local1 + 2 * local2;
    return local3;
}

// Main function to prevent optimization
int main() {
    return mba_simple(1, 2) + 
           mba_shift(3, 4) + 
           mba_rotate(5, 6) +
           mba_boolean_chain(7, 8, 9) +
           mba_comparison(10, 11) +
           mba_arithmetic_boolean(12, 13) +
           mba_stack(14, 15);
}


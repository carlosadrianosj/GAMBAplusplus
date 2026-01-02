// MBA Test Functions for PowerPC32
__attribute__((noinline))
int mba_simple(int x, int y) {
    return (x ^ y) + 2 * (x & y);
}

int main() {
    return mba_simple(1, 2);
}


#ifndef RANDINT_CPU_H
#define RANDINT_CPU_H

#define MASK 0xffffffff

inline uint64_t umult64hi(uint64_t a, uint64_t b){
    uint64_t alpha = a >> 32;
    uint64_t beta  = a & MASK;
    uint64_t gamma = b >> 32;
    uint64_t delta = b & MASK;

    uint64_t low  = (beta * delta) >> 32;
    uint64_t mid1 = beta * gamma;
    uint64_t mid2 = alpha * delta;

    uint64_t mid1h = mid1 >> 32;
    uint64_t mid1l = mid1 & MASK;
    uint64_t mid2h = mid2 >> 32;
    uint64_t mid2l = mid2 & MASK;

    uint64_t midl = mid1l + mid2l + low;
    uint64_t carry = midl >> 32;

    return alpha * gamma + mid1h + mid2h + carry;
}

#endif

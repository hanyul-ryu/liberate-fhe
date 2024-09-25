#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

#include "../utils/threadpool/greedy_threadpool.h"
using namespace juwhan;
using namespace std;

// Pointwise functions.
template<typename scalar_t> __inline__ scalar_t
mont_mult_scalar_cpu_kernel(
    const scalar_t a, const scalar_t b,
    const scalar_t ql, const scalar_t qh,
    const scalar_t kl, const scalar_t kh) {
    
    // Masks.
    constexpr scalar_t one = 1;
    constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
    constexpr scalar_t half_nbits =  sizeof(scalar_t) * 4 - 1;
    constexpr scalar_t fb_mask = ((one << nbits) - one);
    constexpr scalar_t lb_mask = (one << half_nbits) - one;
    
    const scalar_t al = a & lb_mask;
    const scalar_t ah = a >> half_nbits;
    const scalar_t bl = b & lb_mask;
    const scalar_t bh = b >> half_nbits;

    const scalar_t alpha = ah * bh;
    const scalar_t beta  = ah * bl + al * bh;
    const scalar_t gamma = al * bl;

    // s = xk mod R
    const scalar_t gammal = gamma & lb_mask;
    const scalar_t gammah = gamma >> half_nbits;
    const scalar_t betal  = beta & lb_mask;
    const scalar_t betah  = beta >> half_nbits;

    scalar_t upper = gammal * kh;
    upper = upper + (gammah + betal) * kl;
    upper = upper << half_nbits;
    scalar_t s = upper + gammal * kl;
    s = upper + gammal * kl;
    s = s & fb_mask;

    // t = x + sq
    // u = t/R
    const scalar_t sl   = s & lb_mask;
    const scalar_t sh   = s >> half_nbits;
    const scalar_t sqb  = sh * ql + sl * qh;
    const scalar_t sqbl = sqb & lb_mask;
    const scalar_t sqbh = sqb >> half_nbits;

    scalar_t carry = (gamma + sl * ql) >> half_nbits;
    carry = (carry + betal + sqbl) >> half_nbits;
    
    return alpha + betah + sqbh + carry + sh * qh;
}

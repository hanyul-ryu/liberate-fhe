#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include "randint_cpu.h"

#include "../utils/threadpool/greedy_threadpool.h"

using namespace juwhan;
using namespace std;

#define MASK 0xffffffff

#define ROLL16(x) x = (((x) << 16) | ((x) >> 16)) & MASK
#define ROLL12(x) x = (((x) << 12) | ((x) >> 20)) & MASK
#define ROLL8(x) x = (((x) << 8) | ((x) >> 24)) & MASK
#define ROLL7(x) x = (((x) << 7) | ((x) >> 25)) & MASK

#define QR(x, a, b, c, d)\
    x[a] += x[b];\
    x[a] &= MASK;\
    x[d] ^= x[a];\
    ROLL16(x[d]);\
    x[c] += x[d];\
    x[c] &= MASK;\
    x[b] ^= x[c];\
    ROLL12(x[b]);\
    x[a] += x[b];\
    x[a] &= MASK;\
    x[d] ^= x[a];\
    ROLL8(x[d]);\
    x[c] += x[d];\
    x[c] &= MASK;\
    x[b] ^= x[c];\
    ROLL7(x[b])
    
#define ONE_ROUND(x)\
    QR(x, 0, 4,  8, 12);\
    QR(x, 1, 5,  9, 13);\
    QR(x, 2, 6, 10, 14);\
    QR(x, 3, 7, 11, 15);\
    QR(x, 0, 5, 10, 15);\
    QR(x, 1, 6, 11, 12);\
    QR(x, 2, 7,  8, 13);\
    QR(x, 3, 4,  9, 14)

#define COMBINE_TWO(high, low)\
    ((static_cast<uint64_t>(high) << 32) | static_cast<uint64_t>(low))

// Check types.
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) TORCH_CHECK(x.dtype() == torch::kInt64, #x, " must be a kInt64 tensor")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x); CHECK_LONG(x)



void randint_cpu_kernel(
    long long sptr[], 
    long long optr[], 
    size_t inc, 
    uint64_t amax[],
    int64_t shift,
    size_t idx,
    size_t C,
    size_t N,
    size_t chunk){

    // x holds random bytes.
    uint64_t x[16];
    
    for (size_t j=0; j<chunk; j++){
        auto my_idx = idx + j;
        auto my_c   = my_idx / N;
        auto my_n   = my_idx - my_c*N;

        // What is my p?
        uint64_t p = amax[my_c];

        auto my_optr = optr + (my_c*N*4 + my_n*4);
        auto my_sptr = sptr + j*16;

        // Copy state to x.
        for(int i=0; i<16; ++i){
            x[i] = my_sptr[i];
        }

        // Repeat 10 times for chacha20.
        for(int i=0; i<10; ++i){
            ONE_ROUND(x);
        }
        
        for(int i=0; i<16; ++i){
            x[i] = (x[i] + my_sptr[i]) & MASK;
        }
        
        // Step the state.
        my_sptr[12] += inc;
        my_sptr[13] += (my_sptr[12] >> 32);
        my_sptr[12] &= MASK;

        for(int i=0; i<16; i+=4){
            // Compose into 2 uint64 values, the 4 32 bit values stored in
            // the 4 int64 storage.
            uint64_t x_low = COMBINE_TWO(x[i], x[i+1]);
    
            // Use CUDA integer intrinsics to calculate
            // (x_low * p) >> 64.
            // Refer to https://github.com/apple/swift/pull/39143
            auto alpha = umult64hi(p, x_low);
    
            // We need to calculate carry.
            auto pl = p & MASK;          // 1-32
            auto ph = p >> 32;           // 33-64
            //---------------------------------------
            auto xhh = x[i+2];
            auto xhl = x[i+3];
            //---------------------------------------
            auto plxhl = pl * xhl;       // 65-128
            auto plxhh = pl * xhh;       // 97-160
            auto phxhl = ph * xhl;       // 97-160
            auto phxhh = ph * xhh;       // 129-192
            //---------------------------------------
            auto carry = ((plxhl & MASK) + (alpha & MASK)) >> 32;
            carry = (carry +
                    (plxhl >> 32) +
                    (alpha >> 32) +
                    (phxhl & MASK) +
                    (plxhh & MASK)) >> 32;
            auto sample = (carry +
                         (phxhl >> 32) +
                         (plxhh >> 32) + phxhh);
    
            // Store the result.
            // Don't forget the shift!!!
            const int reduced_idx = i/4;
            my_optr[reduced_idx] = sample + shift;
        }
    }


    
}

torch::Tensor randint_cpu(torch::Tensor state, size_t amax,
    int64_t shift, size_t inc, size_t chunk, size_t tp_ptr) {
    /*
    The state tensor must be a contiguous long tensor of size C x N x 16.  
    At the end of the calculation, counters in the state are implemented by the inc.
    */

    // Convert a python pointer to C pointer.
    uint64_t *amax_ptr = reinterpret_cast<uint64_t*>(amax);

    CHECK_INPUT(state);

    // Find out C and N.
    size_t C = state.size(0);
    size_t N = state.size(1);
    size_t CN = C * N;

    auto output = state.new_empty({C, N * 4});

    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();
    
    // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r;
    
    long long* state_ptr = (long long*)state.data_ptr();
    long long* output_ptr = (long long*)output.data_ptr();

    for(size_t i=0; i < CN; i+=chunk){    
        auto my_state_ptr = state_ptr + i*16;

        r.push_back(tp->submit(randint_cpu_kernel, 
                             my_state_ptr, 
                             output_ptr, 
                             inc,
                             amax_ptr,
                             shift,
                             i,
                             C,
                             N,
                             chunk));
    }

    
    // Join.
    for(auto receipt : r) {
        receipt.wait();
    }

    // Stop the greedy threadpool.
    tp->stop();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("randint_cpu", &randint_cpu, "RANDINT (CPU)");
}

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

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


// Check types.
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) TORCH_CHECK(x.dtype() == torch::kInt64, #x, " must be a kInt64 tensor")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x); CHECK_LONG(x)


void chacha20_cpu_kernel(long long sptr[], long long optr[], size_t inc, size_t chunk){
    
    for (size_t j=0; j<chunk; j++) {
        auto my_optr = optr + j*16;
        auto my_sptr = sptr + j*16;

        // Repeat 10 times for chacha20.
        for(int i=0; i<10; ++i){
            ONE_ROUND(my_optr);
        }
        
        for(int i=0; i<16; ++i){
            my_optr[i] = (my_optr[i] + my_sptr[i]) & MASK;
        }
        
        // Step the state.
        my_sptr[12] += inc;
        my_sptr[13] += (my_sptr[12] >> 32);
        my_sptr[12] &= MASK;
    }
    
}

torch::Tensor chacha20_cpu(torch::Tensor state, size_t inc, size_t chunk, size_t tp_ptr) {
    /*
    The state tensor must be a contiguous long tensor of size N x 16.  
    At the end of the calculation, counters in the state are implemented by the inc.
    */

    CHECK_INPUT(state);
    auto output = state.clone();

    // Find out N.
    size_t N = state.size(0);

    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();

    // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r;
    
    long long* state_ptr = (long long*)state.data_ptr();
    long long* output_ptr = (long long*)output.data_ptr();

    for(size_t i=0; i < N; i+=chunk){    
        auto my_state_ptr = state_ptr + i*16;
        auto my_output_ptr = output_ptr + i*16;

        r.push_back(tp->submit(chacha20_cpu_kernel, my_state_ptr, my_output_ptr, inc, chunk));
       
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
  m.def("chacha20_cpu", &chacha20_cpu, "CHACHA20 (CPU)");
}
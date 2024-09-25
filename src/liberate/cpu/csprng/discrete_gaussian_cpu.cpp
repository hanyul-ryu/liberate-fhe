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


#define GE(x_high, x_low, y_high, y_low)\
    (((x_high) > (y_high)) | (((x_high) == (y_high)) & ((x_low) >= (y_low))))
    
#define COMBINE_TWO(high, low)\
    ((static_cast<uint64_t>(high) << 32) | static_cast<uint64_t>(low))


// Check types.
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) TORCH_CHECK(x.dtype() == torch::kInt64, #x, " must be a kInt64 tensor")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x); CHECK_LONG(x)


void dg_cpu_kernel(long long sptr[],
            long long optr[],
            size_t inc,
            uint64_t *btree,
            int btree_size,
            int depth,
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

            // Traverse the tree in the LUT (btree).
            // Note that, out of the 16 32-bit randon numbers,
            // we generate 4 discrete gaussian samples.
            int jump = 1;
            int current = 0;
            int counter = 0;
    
            // Compose into 2 uint64 values, the 4 32 bit values stored in
            // the 4 int64 storage.
            uint64_t x_low = COMBINE_TWO(x[i], x[i+1]);
            uint64_t x_high = COMBINE_TWO(x[i+2], x[i+3]);
    
            // Reserve a sign bit.
            // Since we are dealing with the half plane,
            // The CDT values in the LUT are at most 0.5, which means
            // that the values are 127 bits.
            // Also, rigorously speaking, we need to take out the MSB from the x_high
            // value to take the sign, but every bit has probability of occurrence=0.5.
            // Hence, it doesn't matter where we take the bit.
            // For convenience, take the LSB of x_high.
            int64_t sign_bit = x_high & 1;
            x_high >>= 1;
    
            // Traverse the binary search tree.
            for(int k=0; k<depth; k++){
                int ge_flag = GE(x_high, x_low, 
                              btree[counter+current+btree_size],
                              btree[counter+current]);
    
                // Update the current location.
                current = 2 * current + ge_flag;
    
                // Update the counter.
                counter += jump;
    
                // Update the jump
                jump *= 2;    
            }
            int64_t sample = (sign_bit * 2 - 1) * static_cast<int64_t>(current);
    
            // Store the result.
            const int reduced_idx = i/4;
            my_optr[reduced_idx] = sample;
        }
    }
    
}

torch::Tensor dg_cpu(torch::Tensor state,
                    size_t inc,
                    size_t btree_ptr,
                    int btree_size,
                    int depth,
                    size_t chunk,
                    size_t tp_ptr) {
    /*
    The state tensor must be a contiguous long tensor of size N x 16.  
    At the end of the calculation, counters in the state are implemented by the inc.
    */

    // Find out C and N.
    size_t C = state.size(0);
    size_t N = state.size(1);
    size_t CN = C * N;
    
    CHECK_INPUT(state);
    auto output = state.new_empty({C, N * 4});

    // reinterpret pointers from numpy.
    uint64_t *btree = reinterpret_cast<uint64_t*>(btree_ptr);

    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();

    // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r;
    
    long long* state_ptr = (long long*)state.data_ptr();
    long long* output_ptr = (long long*)output.data_ptr();

    for(size_t i=0; i < CN; i+=chunk){    
        auto my_state_ptr = state_ptr + i*16;

        r.push_back(tp->submit(
            dg_cpu_kernel,
            my_state_ptr,
            output_ptr,
            inc,
            btree,
            btree_size,
            depth,
            i, C, N,
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
  m.def("discrete_gaussian_cpu", &dg_cpu, "DISCRETE GAUSSIAN (CPU)");
}

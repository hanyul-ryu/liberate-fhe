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


void randround_cpu_kernel(
            long long sptr[],
            double iptr[],
            long long optr[],
            size_t inc,
            size_t chunk) {

    // x holds random bytes.
    uint64_t x[16];
    
    for (size_t j=0; j<chunk; j++){

        auto my_sptr = sptr + j*16;
        auto my_iptr = iptr + j*16;
        auto my_optr = optr + j*16;
        
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

        for(int i=0; i<16; i++){

            auto coef = my_iptr[i];

            // std::signbit gives ture for negative and false for otherwise
            auto sign_bit = std::signbit(coef);
            auto abs_coef = std::fabs(coef);

            // Separate out the integral and the fractional parts.
            auto integ = std::floor(abs_coef);
            auto frac = abs_coef - integ;
            int64_t intinteg = static_cast<int64_t>(integ);
            
            // Use c++ std::round().
            constexpr double rounder = static_cast<double>(0x100000000);
            int64_t ifrac = std::round(frac * rounder);
            
            // Random round.
            // The bool value must be 1 for True.
            int64_t round = x[i] < ifrac;
            
            // Round and recover sign.
            int64_t sign = (sign_bit)? -1 : 1;
            int64_t rounded = sign * (intinteg + round);
            
            // Put back.
            my_optr[i] = rounded;
        }
    }
    
}

torch::Tensor randround_cpu(torch::Tensor state,
                    torch::Tensor input,
                    size_t inc,
                    size_t chunk,
                    size_t tp_ptr
                    ) {
    /*
    The state tensor must be a contiguous long tensor of size M x 16 = (poly length / 16) * 16.
    The input and output tensors must be of size poly length.
    The input has dtype double, while the output must be long long.
    At the end of the calculation, counters in the state are implemented by the inc.
    */

    // Find out C and N.
    // M = N / 4
    size_t M = state.size(0);
    
    CHECK_INPUT(state);

    // Create the output tensor
    auto output = state.new_empty({M * 16});

    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();

    // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r;
    
    long long* state_ptr = (long long*)state.data_ptr();
    double* input_ptr = (double*)input.data_ptr();
    long long* output_ptr = (long long*)output.data_ptr();

    for(size_t i=0; i < M; i+=chunk){    
        auto my_state_ptr = state_ptr + i*16;
        auto my_input_ptr = input_ptr + i*16;
        auto my_output_ptr = output_ptr + i*16;

        r.push_back(tp->submit(
            randround_cpu_kernel,
            my_state_ptr,
            my_input_ptr,
            my_output_ptr,
            inc,
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
  m.def("randround_cpu", &randround_cpu, "RANDOM ROUNDING (CPU)");
}

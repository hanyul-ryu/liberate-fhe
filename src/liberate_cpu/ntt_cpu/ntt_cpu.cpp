#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

#include "ntt_cpu.h"

//------------------------------------------------------------------
// mont_mult
//------------------------------------------------------------------

template<typename scalar_t>
void mont_mult_cpu_kernel(
    const scalar_t a[],
    const scalar_t b[],
    scalar_t c[],
    const scalar_t ql[],
    const scalar_t qh[],
    const scalar_t kl[],
    const scalar_t kh[],
    size_t N,
    size_t idx,
    size_t chunk){

    for(auto i=0; i<chunk; i++){
        auto my_loc = idx + i;
        auto my_c   = my_loc / N;

        auto my_ql  = ql[my_c];
        auto my_qh  = qh[my_c];
        auto my_kl  = kl[my_c];
        auto my_kh  = kh[my_c];

        c[my_loc] = mont_mult_scalar_cpu_kernel(
		a[my_loc],
		b[my_loc],
		my_ql,
		my_qh,
		my_kl,
		my_kh
	);
    }
}

template<typename scalar_t>
void mont_mult_cpu_typed(
    const std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> b,
    std::vector<torch::Tensor> c,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh,
    size_t tp_ptr,
    size_t chunk){

    
    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();
    
    const auto num_a = a.size();
    auto C = a[0].size(0);
    auto N = a[0].size(1);
    auto CN = C * N;
    
    // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r;
    
    for (int k=0; k<num_a; ++k){
        for(auto i=0; i<CN; i+=chunk){
            r.push_back(tp->submit(
            mont_mult_cpu_kernel<scalar_t>,
            (scalar_t*)a[k].data_ptr(),
            (scalar_t*)b[k].data_ptr(),
            (scalar_t*)c[k].data_ptr(),
            (scalar_t*)ql.data_ptr(),
            (scalar_t*)qh.data_ptr(),
            (scalar_t*)kl.data_ptr(),
            (scalar_t*)kh.data_ptr(),
            N,
            i,
            chunk));
        }
    }
    // Join.
    for(auto receipt : r) {
        receipt.wait();
    }

    // Stop the greedy threadpool.
    tp->stop();
}

std::vector<torch::Tensor> mont_mult_cpu(
    const std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> b,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh,
    size_t tp_ptr,
    size_t chunk) {
        
    // Prepare the output.
    std::vector<torch::Tensor> c;
    size_t size_a = a.size();
    for(int i=0; i<size_a; i++){
        c.push_back(torch::empty_like(a[i]));
    }
    // torch::Tensor c = torch::empty_like(a[0]);
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(
        a[0].type(), 
        "typed_mont_mult_cpu", 
        ([&] {mont_mult_cpu_typed<scalar_t>(a, b, c, ql[0], qh[0], kl[0], kh[0], tp_ptr, chunk);}));
    
    return c;
}

//------------------------------------------------------------------
// mont_enter
//------------------------------------------------------------------

template<typename scalar_t>
void mont_enter_cpu_kernel(
    scalar_t a[],
    const scalar_t Rs[],
    const scalar_t ql[],
    const scalar_t qh[],
    const scalar_t kl[],
    const scalar_t kh[],
    size_t N,
    size_t idx,
    size_t chunk){

    for(auto i=0; i<chunk; i++){
        auto my_loc = idx + i;
        auto my_c   = my_loc / N;
	
        auto my_ql  = ql[my_c];
        auto my_qh  = qh[my_c];
        auto my_kl  = kl[my_c];
        auto my_kh  = kh[my_c];

        a[my_loc] = mont_mult_scalar_cpu_kernel(
			a[my_loc], 
			Rs[my_c], 
			my_ql, 
			my_qh, 
			my_kl, 
			my_kh
        );
    }
}

template<typename scalar_t>
void mont_enter_cpu_typed(
    std::vector<torch::Tensor> a,
    const torch::Tensor Rs,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh,
    size_t tp_ptr,
    size_t chunk){

    const auto num_a = a.size();

    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();

    auto C = a[0].size(0);
    auto N = a[0].size(1);
    auto CN = C * N;

    // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r;
    for (int k=0; k<num_a; ++k){
        for(auto i=0; i<CN; i+=chunk){
            r.push_back(tp->submit(
            mont_enter_cpu_kernel<scalar_t>,
            (scalar_t*)a[k].data_ptr(),
            (scalar_t*)Rs.data_ptr(),
            (scalar_t*)ql.data_ptr(),
            (scalar_t*)qh.data_ptr(),
            (scalar_t*)kl.data_ptr(),
            (scalar_t*)kh.data_ptr(),
            N,
            i,
            chunk));
        }    
    }

    // Join.
    for(auto receipt : r) {
        receipt.wait();
    }
    // Stop the greedy threadpool.
    tp->stop();
}

void mont_enter_cpu(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> Rs,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh,
    size_t tp_ptr,
    size_t chunk) {
        
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(
        a[0].type(), 
        "typed_mont_enter_cpu", 
        ([&] {mont_enter_cpu_typed<scalar_t>(a, Rs[0], ql[0], qh[0], kl[0], kh[0], tp_ptr, chunk);}));
}

//------------------------------------------------------------------
// ntt
//------------------------------------------------------------------

template<typename scalar_t>
void ntt_cpu_kernel(
    scalar_t a[],
    const int even[],
    const int odd[],
    const scalar_t psi[],
    const scalar_t _2q[],
    const scalar_t ql[],
    const scalar_t qh[],
    const scalar_t kl[],
    const scalar_t kh[],
    const size_t level,
    const size_t logN,
    const size_t N,
    size_t idx,
    size_t chunk){

    for(size_t i=0; i<chunk; ++i){
        // Where am I?
    	auto my_loc = idx+i;
    	auto my_c = my_loc/N;
        auto CN = my_c*(N*2);
    	auto my_N = my_loc - my_c*N;
    	
    	// Montgomery inputs.
    	auto my__2q = _2q[my_c];
    	auto my_ql  = ql[my_c];
    	auto my_qh  = qh[my_c];
    	auto my_kl  = kl[my_c];
    	auto my_kh  = kh[my_c];
    
    	// Butterfly.
    	const int even_j = even[level*N + my_N];
    	const int odd_j = odd[level*N + my_N];
        
    	const scalar_t U = a[CN + even_j];
    	const scalar_t S = psi[my_c*logN*N + level*N + my_N];
    	const scalar_t O = a[CN + odd_j];
        const scalar_t V = mont_mult_scalar_cpu_kernel(S, O, my_ql, my_qh, my_kl, my_kh);
    	
    	// Store back.
    	const scalar_t UplusV = U + V;
    	const scalar_t UminusV = U + my__2q - V;
        
    	a[CN + even_j] = (UplusV < my__2q)? UplusV : UplusV - my__2q;
    	a[CN + odd_j]  = (UminusV < my__2q)? UminusV : UminusV - my__2q;
    }
}

template<typename scalar_t>
void ntt_cpu_typed(
    torch::Tensor a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh,
    size_t tp_ptr,
    size_t chunk) {
    
    // The problem dimension.
    const auto C = ql.size(0);
    const auto logN = even.size(0);
    const auto N_half = even.size(1);
    const auto CN_half = C*N_half;

    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();
    
    // Run the cpu kernel.
    for(size_t i=0; i<logN; ++i){
        // Prepare receipts array.
        vector<typename greedy_threadpool::template receipt_type<void>> r;
        
        for(size_t j=0; j<CN_half; j+=chunk){
            r.push_back(tp->submit(
            ntt_cpu_kernel<scalar_t>,
            (scalar_t*)a.data_ptr(),
            (int*)even.data_ptr(),
            (int*)odd.data_ptr(),
            (scalar_t*)psi.data_ptr(),
            (scalar_t*)_2q.data_ptr(),
            (scalar_t*)ql.data_ptr(),
            (scalar_t*)qh.data_ptr(),
            (scalar_t*)kl.data_ptr(),
            (scalar_t*)kh.data_ptr(),
            i,
            logN,
            N_half,
            j,
            chunk));
        }
        
        // Join.
        for(auto receipt : r) {
            receipt.wait();
        }

        r.clear();
    }

    // Stop the greedy threadpool.
    tp->stop();
}

void ntt_cpu(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> even,
    const std::vector<torch::Tensor> odd,
    const std::vector<torch::Tensor> psi,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh,
    size_t tp_ptr,
    size_t chunk) {
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(a[0].type(), "typed_ntt_cpu", ([&] {
    ntt_cpu_typed<scalar_t>(a[0], even[0], odd[0], psi[0], _2q[0], ql[0], qh[0], kl[0], kh[0], tp_ptr, chunk);
    }));
}

//------------------------------------------------------------------
// enter_ntt
//------------------------------------------------------------------

template<typename scalar_t>
void enter_ntt_cpu_typed(
    std::vector<torch::Tensor> a,
    const torch::Tensor Rs,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh,
    size_t tp_ptr,
    size_t chunk) {

    const auto num_a = a.size(); 

    // The problem dimension.
    // Be careful. even and odd has half the length of the a.
    const auto C = ql.size(0);
    const auto logN = even.size(0);
    const auto N_half = even.size(1);
    const auto N = a[0].size(1);
    const auto CN = C*N;
    const auto CN_half = C*N_half;

    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();

    // // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r1;
    
    for (int k=0; k<num_a; ++k){
        // enter.
        for(auto i=0; i<CN; i+=chunk){
            r1.push_back(tp->submit(
            mont_enter_cpu_kernel<scalar_t>,
            (scalar_t*)a[k].data_ptr(),
            (scalar_t*)Rs.data_ptr(),
            (scalar_t*)ql.data_ptr(),
            (scalar_t*)qh.data_ptr(),
            (scalar_t*)kl.data_ptr(),
            (scalar_t*)kh.data_ptr(),
            N,
            i,
            chunk));
        }
    }
    
    // Join.
    for(auto receipt : r1) {
        receipt.wait();
    }
    
    // ntt.
    for(size_t i=0; i<logN; ++i){ 
        vector<typename greedy_threadpool::template receipt_type<void>> r2;
        
        for (int k=0; k<num_a; ++k){
            for(size_t j=0; j<CN_half; j+=(chunk/2)){
                r2.push_back(tp->submit(
                ntt_cpu_kernel<scalar_t>,
                (scalar_t*)a[k].data_ptr(),
                (int*)even.data_ptr(),
                (int*)odd.data_ptr(),
                (scalar_t*)psi.data_ptr(),
                (scalar_t*)_2q.data_ptr(),
                (scalar_t*)ql.data_ptr(),
                (scalar_t*)qh.data_ptr(),
                (scalar_t*)kl.data_ptr(),
                (scalar_t*)kh.data_ptr(),
                i,
                logN,
                N_half,
                j,
                (chunk/2)));
            }
        }
        // Join.
        for(auto receipt : r2) {
            receipt.wait();
        }
    }
    
    // Stop the greedy threadpool.
    tp->stop();
    
}

void enter_ntt_cpu(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> Rs,
    const std::vector<torch::Tensor> even,
    const std::vector<torch::Tensor> odd,
    const std::vector<torch::Tensor> psi,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh,
    size_t tp_ptr,
    size_t chunk) {
    
    // Dispatch to the correct data type.
    // AT_DISPATCH_INTEGRAL_TYPES(a[0].type(), "typed_enter_ntt_cpu", ([&] {
    // enter_ntt_cpu_typed<scalar_t>(a[0], Rs[0], even[0], odd[0], psi[0], _2q[0], ql[0], qh[0], kl[0], kh[0], tp_ptr, chunk);
    // }));
    AT_DISPATCH_INTEGRAL_TYPES(a[0].type(), "typed_enter_ntt_cpu", ([&] {
    enter_ntt_cpu_typed<scalar_t>(a, Rs[0], even[0], odd[0], psi[0], _2q[0], ql[0], qh[0], kl[0], kh[0], tp_ptr, chunk);
    }));
}

//------------------------------------------------------------------
// intt
//------------------------------------------------------------------

template<typename scalar_t>
void intt_cpu_kernel(
    scalar_t a[],
    const int even[],
    const int odd[],
    const scalar_t psi[],
    const scalar_t _2q[],
    const scalar_t ql[],
    const scalar_t qh[],
    const scalar_t kl[],
    const scalar_t kh[],
    const size_t level,
    const size_t logN,
    const size_t N,
    size_t idx,
    size_t chunk){

	for(size_t i=0; i<chunk; ++i){
        // Where am I?
		auto my_loc = idx + i;
		auto my_c = my_loc / N;
		auto my_N = my_loc - my_c*N;
        auto CN   = my_c*(N*2);

		// Montgomery inputs.
		auto my__2q = _2q[my_c];
		auto my_ql  = ql[my_c];
		auto my_qh  = qh[my_c];
		auto my_kl  = kl[my_c];
		auto my_kh  = kh[my_c];

		// Butterfly.
		const int even_j = even[level*N + my_N];
		const int odd_j = odd[level*N + my_N];
        
		const scalar_t U = a[CN + even_j];
		const scalar_t S = psi[my_c*logN*N + level*N + my_N];
		const scalar_t V = a[CN + odd_j];

		const scalar_t UminusV = U + my__2q - V;
		const scalar_t O = (UminusV < my__2q)? UminusV : UminusV - my__2q;
        
		const scalar_t W = mont_mult_scalar_cpu_kernel(S, O, my_ql, my_qh, my_kl, my_kh);
		a[CN + odd_j]  = W;

		const scalar_t UplusV = U + V;
		a[CN + even_j] = (UplusV < my__2q)? UplusV : UplusV - my__2q;
	}
}

template<typename scalar_t>
void intt_cpu_typed(
    torch::Tensor a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor Ninv,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh,
    size_t tp_ptr,
    size_t chunk){
    
    const auto C = ql.size(0);
    const auto logN = even.size(0);
    const auto N_half = even.size(1);
    const auto N = a.size(1);
    const auto CN = C*N;
    const auto CN_half = C*N_half;

    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();

    // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r2;

    for(size_t i=0; i<logN; ++i){
        vector<typename greedy_threadpool::template receipt_type<void>> r1;
        
        for(size_t j=0; j<CN_half; j+=(chunk/2)){
            r1.push_back(tp->submit(
            intt_cpu_kernel<scalar_t>,
            (scalar_t*)a.data_ptr(),
            (int*)even.data_ptr(),
            (int*)odd.data_ptr(),
            (scalar_t*)psi.data_ptr(),
            (scalar_t*)_2q.data_ptr(),
            (scalar_t*)ql.data_ptr(),
            (scalar_t*)qh.data_ptr(),
            (scalar_t*)kl.data_ptr(),
            (scalar_t*)kh.data_ptr(),
            i,
            logN,
            N_half,
            j,
            (chunk/2)));
        }

        // Join.
        for(auto receipt : r1) {
            receipt.wait();
        }
    }

    for(auto i=0; i<CN; i+=chunk){
        r2.push_back(tp->submit(
        mont_enter_cpu_kernel<scalar_t>,
        (scalar_t*)a.data_ptr(),
        (scalar_t*)Ninv.data_ptr(),
        (scalar_t*)ql.data_ptr(),
        (scalar_t*)qh.data_ptr(),
        (scalar_t*)kl.data_ptr(),
        (scalar_t*)kh.data_ptr(),
        N,
        i,
        chunk));
    }

    // Join.
    for(auto receipt : r2) {
        receipt.wait();
    }
    
    // Stop the greedy threadpool.
    tp->stop();
}

void intt_cpu(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> even,
    const std::vector<torch::Tensor> odd,
    const std::vector<torch::Tensor> psi,
    const std::vector<torch::Tensor> Ninv,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh,
    size_t tp_ptr,
    size_t chunk){
        // Dispatch to the correct data type.
        AT_DISPATCH_INTEGRAL_TYPES(a[0].type(), "typed_intt_cpu", ([&] {
        intt_cpu_typed<scalar_t>(a[0], even[0], odd[0], psi[0], Ninv[0], _2q[0], ql[0], qh[0], kl[0], kh[0], tp_ptr, chunk);
        }));
}

//------------------------------------------------------------------
// mont_redc
//------------------------------------------------------------------

template<typename scalar_t>
void mont_redc_cpu_kernel(
    scalar_t a[],
    const scalar_t my_ql[],
    const scalar_t my_qh[],
    const scalar_t my_kl[],
    const scalar_t my_kh[],
    size_t N,
    size_t idx,
    size_t chunk){
    
    for(auto i=0; i<chunk; i++){
        auto my_loc = idx + i;
        auto my_c   = my_loc / N;
        
        // Masks.
        constexpr scalar_t one = 1;
        constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
        constexpr scalar_t half_nbits =  sizeof(scalar_t) * 4 - 1;
        constexpr scalar_t fb_mask = ((one << nbits) - one);
        constexpr scalar_t lb_mask = (one << half_nbits) - one;
        
        // Inputs.
        const scalar_t x = a[my_loc];
        const scalar_t ql = my_ql[my_c];
        const scalar_t qh = my_qh[my_c];
        const scalar_t kl = my_kl[my_c];
        const scalar_t kh = my_kh[my_c];
        
        // Implementation.
        // s= xk mod R
        const scalar_t xl = x & lb_mask;
        const scalar_t xh = x >> half_nbits;
        const scalar_t xkb = xh * kl + xl * kh;
        scalar_t s = (xkb << half_nbits) + xl * kl;
        s = s & fb_mask;
    
        // t = x + sq
        // u = t/R
        // Note that x gets erased in t/R operation if x < R.
        const scalar_t sl = s & lb_mask;
        const scalar_t sh = s >> half_nbits;
        const scalar_t sqb = sh * ql + sl * qh;
        const scalar_t sqbl = sqb & lb_mask;
        const scalar_t sqbh = sqb >> half_nbits;
        scalar_t carry = (x + sl * ql) >> half_nbits;
        carry = (carry + sqbl) >> half_nbits;
        
        // Assume we have satisfied the condition 4*q < R.
        // Return the calculated value directly without conditional subtraction.
        a[my_loc] = sqbh + carry + sh * qh;
    }
}

template<typename scalar_t>
void mont_redc_cpu_typed(
    torch::Tensor a,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh,
    size_t tp_ptr,
    size_t chunk) {
    
    // The problem dimension.
    auto C = a.size(0);
    auto N = a.size(1);
    auto CN = C*N;

    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();

    // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r;
    
    // Run the cpu kernel.
    for(auto i=0; i<CN; i+=chunk){
        r.push_back(tp->submit(
        mont_redc_cpu_kernel<scalar_t>,
            (scalar_t*)a.data_ptr(),
            (scalar_t*)ql.data_ptr(),
            (scalar_t*)qh.data_ptr(),
            (scalar_t*)kl.data_ptr(),
            (scalar_t*)kh.data_ptr(),
            N,
            i,
            chunk));
    }

    
    // Join.
    for(auto receipt : r) {
        receipt.wait();
    }

    // Stop the greedy threadpool.
    tp->stop();
}

void mont_redc_cpu(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh,
    size_t tp_ptr,
    size_t chunk) {
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(a[0].type(), "typed_mont_redc_cpu", ([&] {
    mont_redc_cpu_typed<scalar_t>(a[0], ql[0], qh[0], kl[0], kh[0], tp_ptr, chunk);
    }));
}

//------------------------------------------------------------------
// Chained intt series.
//------------------------------------------------------------------

/**************************************************************/
/* CPU kernels                                                */
/**************************************************************/

template<typename scalar_t>
void reduce_cpu_kernel(
    scalar_t a[],
    const scalar_t _2q[],
    size_t N,
    size_t idx,
    size_t chunk){

    for(auto i=0; i<chunk; i++){
        auto my_loc = idx + i;
        auto my_c   = my_loc / N;

        // Inputs.
        constexpr scalar_t one = 1;
        const scalar_t my_a = a[my_loc];
        const scalar_t q = _2q[my_c] >> one;

        // Reduce.
        a[my_loc] = (my_a < q)? my_a : my_a - q; 
    }
}

template<typename scalar_t>
void make_signed_cpu_kernel(
    scalar_t a[],
    const scalar_t _2q[],
    size_t N,
    size_t idx,
    size_t chunk){

    for(auto i=0; i<chunk; i++){
        auto my_loc = idx + i;
        auto my_c   = my_loc / N;

        // Inputs.
        constexpr scalar_t one = 1;
        const scalar_t my_a = a[my_loc];
        const scalar_t q = _2q[my_c] >> one;
        const scalar_t q_half = q >> one;

        // Make signed.
        a[my_loc] = (my_a <= q_half)? my_a : my_a - q; 
    }
}

/**************************************************************/
/* Typed functions                                            */
/**************************************************************/

///////////////////////////////////////////////////////////////
// intt exit

template<typename scalar_t>
void intt_exit_cpu_typed(
    torch::Tensor a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor Ninv,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh,
    size_t tp_ptr,
    size_t chunk) {
    
    // The problem dimension.
    // Be careful. even and odd has half the length of the a.
    const auto C = ql.size(0);
    const auto logN = even.size(0);
    const auto N_half = even.size(1);
    const auto N = a.size(1);
    const auto CN = C*N;
    const auto CN_half = C*N_half;

    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();

    // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r2;
    vector<typename greedy_threadpool::template receipt_type<void>> r3;
    
    // Run the cpu kernel.
    for(int i=0; i<logN; ++i){
        vector<typename greedy_threadpool::template receipt_type<void>> r1;
        for(size_t j=0; j<CN_half; j+=chunk/2){
            r1.push_back(tp->submit(
            intt_cpu_kernel<scalar_t>,
            (scalar_t*)a.data_ptr(),
            (int*)even.data_ptr(),
            (int*)odd.data_ptr(),
            (scalar_t*)psi.data_ptr(),
            (scalar_t*)_2q.data_ptr(),
            (scalar_t*)ql.data_ptr(),
            (scalar_t*)qh.data_ptr(),
            (scalar_t*)kl.data_ptr(),
            (scalar_t*)kh.data_ptr(),
            i,
            logN,
            N_half,
            j,
            chunk/2));
        }
        // Join.
        for(auto receipt : r1) {
            receipt.wait();
        }
    }
    
    // Normalize.
    for(auto i=0; i<CN; i+=chunk){
        r2.push_back(tp->submit(
        mont_enter_cpu_kernel<scalar_t>,
        (scalar_t*)a.data_ptr(),
        (scalar_t*)Ninv.data_ptr(),
        (scalar_t*)ql.data_ptr(),
        (scalar_t*)qh.data_ptr(),
        (scalar_t*)kl.data_ptr(),
        (scalar_t*)kh.data_ptr(),
        N,
        i,
        chunk));
    }

    // Join.
    for(auto receipt : r2) {
        receipt.wait();
    }
    
    // Exit.
    for(auto i=0; i<CN; i+=chunk){
        r3.push_back(tp->submit(
        mont_redc_cpu_kernel<scalar_t>,
            (scalar_t*)a.data_ptr(),
            (scalar_t*)ql.data_ptr(),
            (scalar_t*)qh.data_ptr(),
            (scalar_t*)kl.data_ptr(),
            (scalar_t*)kh.data_ptr(),
            N,
            i,
            chunk));
    }

    // Join.
    for(auto receipt : r3) {
        receipt.wait();
    }

    // Stop the greedy threadpool.
    tp->stop();
}

///////////////////////////////////////////////////////////////
// intt exit reduce

template<typename scalar_t>
void intt_exit_reduce_cpu_typed(
    // torch::Tensor a,
    std::vector<torch::Tensor> a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor Ninv,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh,
    size_t tp_ptr,
    size_t chunk) {

    const auto num_a = a.size();

    for (int k=0; k<num_a; ++k){    
        // The problem dimension.
        // Be careful. even and odd has half the length of the a.
        const auto C = ql.size(0);
        const auto logN = even.size(0);
        const auto N_half = even.size(1);
        const auto N = a[k].size(1);
        const auto CN = C*N;
        const auto CN_half = C*N_half;
    
        // Start the greedy threadpool.
        auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
        tp->go();
    
        // Prepare receipts array.
        vector<typename greedy_threadpool::template receipt_type<void>> r2;
        vector<typename greedy_threadpool::template receipt_type<void>> r3;
        vector<typename greedy_threadpool::template receipt_type<void>> r4;
        
        // Run the cpu kernel.
        for(int i=0; i<logN; ++i){
            vector<typename greedy_threadpool::template receipt_type<void>> r1;
            for(size_t j=0; j<CN_half; j+=chunk/2){
                r1.push_back(tp->submit(
                intt_cpu_kernel<scalar_t>,
                (scalar_t*)a[k].data_ptr(),
                (int*)even.data_ptr(),
                (int*)odd.data_ptr(),
                (scalar_t*)psi.data_ptr(),
                (scalar_t*)_2q.data_ptr(),
                (scalar_t*)ql.data_ptr(),
                (scalar_t*)qh.data_ptr(),
                (scalar_t*)kl.data_ptr(),
                (scalar_t*)kh.data_ptr(),
                i,
                logN,
                N_half,
                j,
                chunk/2));
            }
    
            // Join.
            for(auto receipt : r1) {
                receipt.wait();
            }
        }
    
        // Normalize.
        for(auto i=0; i<CN; i+=chunk){
            r2.push_back(tp->submit(
            mont_enter_cpu_kernel<scalar_t>,
            (scalar_t*)a[k].data_ptr(),
            (scalar_t*)Ninv.data_ptr(),
            (scalar_t*)ql.data_ptr(),
            (scalar_t*)qh.data_ptr(),
            (scalar_t*)kl.data_ptr(),
            (scalar_t*)kh.data_ptr(),
            N,
            i,
            chunk));
        }
    
        // Join.
        for(auto receipt : r2) {
            receipt.wait();
        }
    
        // Exit.
        for(auto i=0; i<CN; i+=chunk){
            r3.push_back(tp->submit(
            mont_redc_cpu_kernel<scalar_t>,
                (scalar_t*)a[k].data_ptr(),
                (scalar_t*)ql.data_ptr(),
                (scalar_t*)qh.data_ptr(),
                (scalar_t*)kl.data_ptr(),
                (scalar_t*)kh.data_ptr(),
                N,
                i,
                chunk));
        }
    
        // Join.
        for(auto receipt : r3) {
            receipt.wait();
        }
    
        // Reduce.
        for(auto i=0; i<CN; i+=chunk){
            r4.push_back(tp->submit(
            reduce_cpu_kernel<scalar_t>,
                (scalar_t*)a[k].data_ptr(),
                (scalar_t*)_2q.data_ptr(),
                N,
                i,
                chunk));
        }
    
        // Join.
        for(auto receipt : r4) {
            receipt.wait();
        }
    }
}

///////////////////////////////////////////////////////////////
// intt exit reduce signed

template<typename scalar_t>
void intt_exit_reduce_signed_cpu_typed(
    std::vector<torch::Tensor> a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor Ninv,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh,
    size_t tp_ptr,
    size_t chunk) {

    const auto num_a = a.size();

    for (int k=0; k<num_a; ++k){
    
        // The problem dimension.
        // Be careful. even and odd has half the length of the a.
        const auto C = ql.size(0);
        const auto logN = even.size(0);
        const auto N_half = even.size(1);
        const auto N = a[k].size(1);
        const auto CN = C*N;
        const auto CN_half = C*N_half;
    
        // Start the greedy threadpool.
        auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
        tp->go();
    
        // Prepare receipts array.
        vector<typename greedy_threadpool::template receipt_type<void>> r2;
        vector<typename greedy_threadpool::template receipt_type<void>> r3;
        vector<typename greedy_threadpool::template receipt_type<void>> r4;
        vector<typename greedy_threadpool::template receipt_type<void>> r5;
        
        // Run the cpu kernel.
        for(int i=0; i<logN; ++i){
            vector<typename greedy_threadpool::template receipt_type<void>> r1;
            for(size_t j=0; j<CN_half; j+=chunk/2){
                r1.push_back(tp->submit(
                intt_cpu_kernel<scalar_t>,
                (scalar_t*)a[k].data_ptr(),
                (int*)even.data_ptr(),
                (int*)odd.data_ptr(),
                (scalar_t*)psi.data_ptr(),
                (scalar_t*)_2q.data_ptr(),
                (scalar_t*)ql.data_ptr(),
                (scalar_t*)qh.data_ptr(),
                (scalar_t*)kl.data_ptr(),
                (scalar_t*)kh.data_ptr(),
                i,
                logN,
                N_half,
                j,
                chunk/2));
            }
            
            // Join.
            for(auto receipt : r1) {
                receipt.wait();
            }
        }
    
        // Normalize.
        for(auto i=0; i<CN; i+=chunk){
            r2.push_back(tp->submit(
            mont_enter_cpu_kernel<scalar_t>,
            (scalar_t*)a[k].data_ptr(),
            (scalar_t*)Ninv.data_ptr(),
            (scalar_t*)ql.data_ptr(),
            (scalar_t*)qh.data_ptr(),
            (scalar_t*)kl.data_ptr(),
            (scalar_t*)kh.data_ptr(),
            N,
            i,
            chunk));
        }
    
        // Join.
        for(auto receipt : r2) {
            receipt.wait();
        }
    
        // Exit.
        for(auto i=0; i<CN; i+=chunk){
            r3.push_back(tp->submit(
            mont_redc_cpu_kernel<scalar_t>,
                (scalar_t*)a[k].data_ptr(),
                (scalar_t*)ql.data_ptr(),
                (scalar_t*)qh.data_ptr(),
                (scalar_t*)kl.data_ptr(),
                (scalar_t*)kh.data_ptr(),
                N,
                i,
                chunk));
        }
    
        // Join.
        for(auto receipt : r3) {
            receipt.wait();
        }
    
        // Reduce.
        for(auto i=0; i<CN; i+=chunk){
            r4.push_back(tp->submit(
            reduce_cpu_kernel<scalar_t>,
                (scalar_t*)a[k].data_ptr(),
                (scalar_t*)_2q.data_ptr(),
                N,
                i,
                chunk));
        }
    
        // Join.
        for(auto receipt : r4) {
            receipt.wait();
        }
    
        // Make signed.
        for(auto i=0; i<CN; i+=chunk){
            r5.push_back(tp->submit(
            make_signed_cpu_kernel<scalar_t>,
                (scalar_t*)a[k].data_ptr(),
                (scalar_t*)_2q.data_ptr(),
                N,
                i,
                chunk));
        }
    
        // Join.
        for(auto receipt : r5) {
            receipt.wait();
        }
    }
}

/**************************************************************/
/* Connectors                                                 */
/**************************************************************/

///////////////////////////////////////////////////////////////
// intt exit

void intt_exit_cpu(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> even,
    const std::vector<torch::Tensor> odd,
    const std::vector<torch::Tensor> psi,
    const std::vector<torch::Tensor> Ninv,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh,
    size_t tp_ptr,
    size_t chunk) {
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(a[0].type(), "typed_intt_exit_cpu", ([&] {
    intt_exit_cpu_typed<scalar_t>(a[0], even[0], odd[0], psi[0], Ninv[0], _2q[0], ql[0], qh[0], kl[0], kh[0], tp_ptr, chunk);
    }));
}

///////////////////////////////////////////////////////////////
// intt exit reduce

void intt_exit_reduce_cpu(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> even,
    const std::vector<torch::Tensor> odd,
    const std::vector<torch::Tensor> psi,
    const std::vector<torch::Tensor> Ninv,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh,
    size_t tp_ptr,
    size_t chunk) {
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(a[0].type(), "typed_intt_exit_reduce_cpu", ([&] {
    // intt_exit_reduce_cpu_typed<scalar_t>(a[0], even[0], odd[0], psi[0], Ninv[0], _2q[0], ql[0], qh[0], kl[0], kh[0], tp_ptr, chunk);
    intt_exit_reduce_cpu_typed<scalar_t>(a, even[0], odd[0], psi[0], Ninv[0], _2q[0], ql[0], qh[0], kl[0], kh[0], tp_ptr, chunk);
    }));
}

///////////////////////////////////////////////////////////////
// intt exit reduce signed

void intt_exit_reduce_signed_cpu(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> even,
    const std::vector<torch::Tensor> odd,
    const std::vector<torch::Tensor> psi,
    const std::vector<torch::Tensor> Ninv,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh,
    size_t tp_ptr,
    size_t chunk) {
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(a[0].type(), "typed_intt_exit_reduce_signed_cpu", ([&] {
    intt_exit_reduce_signed_cpu_typed<scalar_t>(a, even[0], odd[0], psi[0], Ninv[0], _2q[0], ql[0], qh[0], kl[0], kh[0], tp_ptr, chunk);
    }));
}

//------------------------------------------------------------------
// Misc
//------------------------------------------------------------------

template<typename scalar_t>
void make_unsigned_cpu_kernel(
    scalar_t a[],
    const scalar_t _2q[],
    size_t N,
    size_t idx,
    size_t chunk){

    for(auto i=0; i<chunk; i++){
        auto my_loc = idx + i;
        auto my_c   = my_loc / N;

        // Inputs.
        constexpr scalar_t one = 1;
        const scalar_t q = _2q[my_c] >> one;

        // Make signed.
        a[my_loc] += q;
    }
}

template<typename scalar_t>
void tile_unsigned_cpu_kernel(
    const scalar_t a[],
    scalar_t dst[],
    const scalar_t _2q[],
    size_t N,
    size_t idx,
    size_t chunk){

    for(auto i=0; i<chunk; i++){
        auto my_loc = idx + i;
        auto my_c   = my_loc / N;
        auto my_n   = my_loc - my_c*N;
        
        // Inputs.
        constexpr scalar_t one = 1;
        const scalar_t q = _2q[my_c] >> one;
        const scalar_t my_a = a[my_n];

        // Make unsigned.
        dst[my_loc] = my_a + q;
    }
}



template<typename scalar_t>
void mont_add_cpu_kernel(
    const scalar_t my_a[],
    const scalar_t my_b[],
    scalar_t c[],
    const scalar_t my__2q[],
    size_t N,
    size_t idx,
    size_t chunk){
    for(auto i=0; i<chunk; i++){
        auto my_loc = idx + i;
        auto my_c   = my_loc / N;

	// Inputs.
	const scalar_t a = my_a[my_loc];
	const scalar_t b = my_b[my_loc];
	const scalar_t _2q = my__2q[my_c];
        
        // Add.
        const scalar_t aplusb = a + b;
        c[my_loc] = (aplusb < _2q)? aplusb : aplusb - _2q;
    }
}

template<typename scalar_t>
void mont_sub_cpu_kernel(
    const scalar_t my_a[],
    const scalar_t my_b[],
    scalar_t c[],
    const scalar_t my__2q[],
    size_t N,
    size_t idx,
    size_t chunk){
    for(auto i=0; i<chunk; i++){
        auto my_loc = idx + i;
        auto my_c   = my_loc / N;

        // Inputs.
        constexpr scalar_t one = 1;
        const scalar_t a = my_a[my_loc];
        const scalar_t b = my_b[my_loc];
        const scalar_t _2q = my__2q[my_c];

        // Sub.
        const scalar_t aminusb = a + _2q - b;
        c[my_loc] = (aminusb < _2q)? aminusb : aminusb - _2q;
    }
}

template<typename scalar_t>
void reduce_2q_cpu_typed(
    std::vector<torch::Tensor> a, 
    const torch::Tensor _2q,
    size_t tp_ptr,
    size_t chunk) {
    
    const auto num_a = a.size(); 

    for (int k=0; k<num_a; ++k){
        const auto C = a[k].size(0);
        const auto N = a[k].size(1);
        const auto CN = C*N;
    
        // Start the greedy threadpool.
        auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
        tp->go();
    
         // Prepare receipts array.
        vector<typename greedy_threadpool::template receipt_type<void>> r;
        
        for(auto i=0; i<CN; i+=chunk){
            r.push_back(tp->submit(
            reduce_cpu_kernel<scalar_t>,
                (scalar_t*)a[k].data_ptr(),
                (scalar_t*)_2q.data_ptr(),
                N,
                i,
                chunk));
        }
    
        // Join.
        for(auto receipt : r) {
            receipt.wait();
        }
    
        // Stop the greedy threadpool.
        tp->stop();
    }
}

template<typename scalar_t>
void make_signed_cpu_typed(
    torch::Tensor a, 
    const torch::Tensor _2q,
    size_t tp_ptr,
    size_t chunk) {
    
    const auto C = a.size(0);
    const auto N = a.size(1);
    const auto CN = C*N;
    
    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();

     // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r;
    
    for(auto i=0; i<CN; i+=chunk){
        r.push_back(tp->submit(
        make_signed_cpu_kernel<scalar_t>,
            (scalar_t*)a.data_ptr(),
            (scalar_t*)_2q.data_ptr(),
            N,
            i,
            chunk));
    }

    // Join.
    for(auto receipt : r) {
        receipt.wait();
    }

    // Stop the greedy threadpool.
    tp->stop();
}

template<typename scalar_t>
void tile_unsigned_cpu_typed(
    const torch::Tensor a,
    torch::Tensor dst, 
    const torch::Tensor _2q,
    size_t tp_ptr,
    size_t chunk) {
    
    const auto C = _2q.size(0);
    const auto N = a.size(0);
    const auto CN = C*N;
    
    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();

     // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r;

    // Run the cpu kernel.
    for(auto i=0; i<CN; i+=chunk){
        r.push_back(tp->submit(
        tile_unsigned_cpu_kernel<scalar_t>,
            (scalar_t*)a.data_ptr(),
            (scalar_t*)dst.data_ptr(),
            (scalar_t*)_2q.data_ptr(),
            N,
            i,
            chunk));
    }

    // Join.
    for(auto receipt : r) {
        receipt.wait();
    }

    // Stop the greedy threadpool.
    tp->stop();
}

template<typename scalar_t>
void make_unsigned_cpu_typed(
    torch::Tensor a, 
    const torch::Tensor _2q,
    size_t tp_ptr,
    size_t chunk) {
    
    const auto C = a.size(0);
    const auto N = a.size(1);
    const auto CN = C*N;

    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();

     // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r;

    // Run the cpu kernel.
    for(auto i=0; i<CN; i+=chunk){
        r.push_back(tp->submit(
        make_unsigned_cpu_kernel<scalar_t>,
            (scalar_t*)a.data_ptr(),
            (scalar_t*)_2q.data_ptr(),
            N,
            i,
            chunk));
    }

    // Join.
    for(auto receipt : r) {
        receipt.wait();
    }

    // Stop the greedy threadpool.
    tp->stop();
}

template<typename scalar_t>
void mont_add_cpu_typed(
    const torch::Tensor a,
    const torch::Tensor b,
    torch::Tensor c,
    const torch::Tensor _2q,
    size_t tp_ptr,
    size_t chunk) {
    
    auto C = a.size(0);
    auto N = a.size(1);
    const auto CN = C*N;
    
    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();

     // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r;

    // Run the cpu kernel.
    for(auto i=0; i<CN; i+=chunk){
        r.push_back(tp->submit(
        mont_add_cpu_kernel<scalar_t>,
            (scalar_t*)a.data_ptr(),
            (scalar_t*)b.data_ptr(),
            (scalar_t*)c.data_ptr(),
            (scalar_t*)_2q.data_ptr(),
            N,
            i,
            chunk));
    }

    // Join.
    for(auto receipt : r) {
        receipt.wait();
    }

    // Stop the greedy threadpool.
    tp->stop();
}

template<typename scalar_t>
void mont_sub_cpu_typed(
    const torch::Tensor a,
    const torch::Tensor b,
    torch::Tensor c,
    const torch::Tensor _2q,
    size_t tp_ptr,
    size_t chunk) {
    
    auto C = a.size(0);
    auto N = a.size(1);
    const auto CN = C*N;
    
    // Start the greedy threadpool.
    auto *tp = reinterpret_cast<greedy_threadpool *>(tp_ptr);
    tp->go();

     // Prepare receipts array.
    vector<typename greedy_threadpool::template receipt_type<void>> r;

    // Run the cpu kernel.
    for(auto i=0; i<CN; i+=chunk){
        r.push_back(tp->submit(
        mont_sub_cpu_kernel<scalar_t>,
            (scalar_t*)a.data_ptr(),
            (scalar_t*)b.data_ptr(),
            (scalar_t*)c.data_ptr(),
            (scalar_t*)_2q.data_ptr(),
            N,
            i,
            chunk));
    }

    // Join.
    for(auto receipt : r) {
        receipt.wait();
    }

    // Stop the greedy threadpool.
    tp->stop();
}

void reduce_2q_cpu(
    std::vector<torch::Tensor> a, 
    const std::vector<torch::Tensor> _2q,
    size_t tp_ptr,
    size_t chunk) {
    AT_DISPATCH_INTEGRAL_TYPES(a[0].type(), "typed_reduce_2q_cpu", ([&] {
    reduce_2q_cpu_typed<scalar_t>(a, _2q[0], tp_ptr, chunk);
    }));
}

void make_signed_cpu(
    std::vector<torch::Tensor> a, 
    const torch::Tensor _2q,
    size_t tp_ptr,
    size_t chunk) {
    AT_DISPATCH_INTEGRAL_TYPES(a[0].type(), "typed_make_signed_cpu", ([&] {
    make_signed_cpu_typed<scalar_t>(a[0], _2q, tp_ptr, chunk);
    }));
}

void make_unsigned_cpu(
    std::vector<torch::Tensor> a, 
    const torch::Tensor _2q,
    size_t tp_ptr,
    size_t chunk) {
    AT_DISPATCH_INTEGRAL_TYPES(a[0].type(), "typed_make_unsigned_cpu", ([&] {
    make_unsigned_cpu_typed<scalar_t>(a[0], _2q, tp_ptr, chunk);
    }));
}

torch::Tensor tile_unsigned_cpu(
    const std::vector<torch::Tensor> a, 
    const torch::Tensor _2q,
    size_t tp_ptr,
    size_t chunk) {
    a[0].squeeze_();
    const auto C = _2q.size(0);
    const auto N = a[0].size(0);
    auto c = a[0].new_empty({C, N});
    AT_DISPATCH_INTEGRAL_TYPES(a[0].type(), "typed_tile_unsigned_cpu", ([&] {
    tile_unsigned_cpu_typed<scalar_t>(a[0], c, _2q, tp_ptr, chunk);
    }));
    return c;
}

torch::Tensor mont_add_cpu(
    const std::vector<torch::Tensor> a, 
    const std::vector<torch::Tensor> b, 
    const std::vector<torch::Tensor> _2q,
    size_t tp_ptr,
    size_t chunk) {
    torch::Tensor c = torch::empty_like(a[0]);
    AT_DISPATCH_INTEGRAL_TYPES(a[0].type(), "typed_mont_add_cpu", ([&] {
    mont_add_cpu_typed<scalar_t>(a[0], b[0], c, _2q[0], tp_ptr, chunk);
    }));
    return c;
}

torch::Tensor mont_sub_cpu(
    const std::vector<torch::Tensor> a, 
    const std::vector<torch::Tensor> b,
    const torch::Tensor _2q,
    size_t tp_ptr,
    size_t chunk) {
    torch::Tensor c = torch::empty_like(a[0]);
    AT_DISPATCH_INTEGRAL_TYPES(a[0].type(), "typed_mont_sub_cpu", ([&] {
    mont_sub_cpu_typed<scalar_t>(a[0], b[0], c, _2q, tp_ptr, chunk);
    }));
    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mont_mult", &mont_mult_cpu, "MONT_MULT (CPU)");
    m.def("mont_enter", &mont_enter_cpu, "MONT_ENTER (CPU)");
    m.def("ntt", &ntt_cpu, "NTT (CPU)");
    m.def("enter_ntt", &enter_ntt_cpu, "ENTER_NTT (CPU)");
    m.def("intt", &intt_cpu, "INTT (CPU)");
    m.def("mont_redc", &mont_redc_cpu, "MONT_REDC (CPU)");
    m.def("intt_exit", &intt_exit_cpu, "INTT_EXIT (CPU)");
    m.def("intt_exit_reduce", &intt_exit_reduce_cpu, "INTT_EXIT_REDUCE (CPU)");
    m.def("intt_exit_reduce_signed", &intt_exit_reduce_signed_cpu, "INTT_EXIT_REDUCE_SIGNED (CPU)");
    m.def("reduce_2q", &reduce_2q_cpu, "REDUCE_2Q (CPU)");
    m.def("make_signed", &make_signed_cpu, "MAKE_SIGNED (CPU)");
    m.def("make_unsigned", &make_unsigned_cpu, "MAKE_UNSIGNED (CPU)");
    m.def("tile_unsigned", &tile_unsigned_cpu, "TILE_UNSIGNED (CPU)");
    m.def("mont_add", &mont_add_cpu, "MONT_ADD (CPU)");
    m.def("mont_sub", &mont_sub_cpu, "MONT_SUB (CPU)");
}

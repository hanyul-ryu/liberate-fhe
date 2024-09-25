#include <torch/extension.h>

#include "greedy_threadpool.h"

using namespace juwhan;
using namespace std;

uint64_t create(size_t num_threads) {
    greedy_threadpool *tp = new greedy_threadpool(num_threads);
    auto ptr = reinterpret_cast<std::uintptr_t>(tp);
    return ptr;
}

void clean(uint64_t ptr) {
    auto *tp = reinterpret_cast<greedy_threadpool *>(ptr);
    delete tp;
}

void increment_one(long long v[], size_t i) {
    v[i]++;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create", &create, "CREATE A THREADPOOL.");
    m.def("clean", &clean, "DESTROY A THREADPOOL.");
}

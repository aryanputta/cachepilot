#include <pybind11/pybind11.h>

extern "C" size_t cachepilot_block_size_bytes();
extern "C" int cachepilot_cuda_runtime_version();
extern "C" float cachepilot_fp8_compression_vs_fp16();
extern "C" int cachepilot_launch_noop();

namespace py = pybind11;

PYBIND11_MODULE(_cuda_kernels, m) {
    m.doc() = "CachePilot CUDA kernels compiled via setup.py + pybind11";
    m.def("block_size_bytes", &cachepilot_block_size_bytes);
    m.def("cuda_runtime_version", &cachepilot_cuda_runtime_version);
    m.def("fp8_compression_vs_fp16", &cachepilot_fp8_compression_vs_fp16);
    m.def("launch_noop", &cachepilot_launch_noop);
}

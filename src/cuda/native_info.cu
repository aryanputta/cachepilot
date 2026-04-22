#include <cuda_runtime.h>
#include <stddef.h>

#define CACHEPILOT_BLOCK_BYTES (16 * 1024 * 1024)

__global__ void cachepilot_noop_kernel() {}

extern "C" {

size_t cachepilot_block_size_bytes() {
    return CACHEPILOT_BLOCK_BYTES;
}

int cachepilot_cuda_runtime_version() {
    int version = 0;
    cudaRuntimeGetVersion(&version);
    return version;
}

float cachepilot_fp8_compression_vs_fp16() {
    return 2.0f;
}

int cachepilot_launch_noop() {
    cachepilot_noop_kernel<<<1, 1>>>();
    return (int)cudaGetLastError();
}

}  // extern "C"

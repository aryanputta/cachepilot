/*
 * kv_block_copy.cu — GPU-to-CPU KV cache block eviction kernel
 *
 * Copies one or more 16 MB KV cache blocks from device memory to
 * pinned host memory using 128-bit vectorized loads to saturate
 * PCIe bandwidth (~64 GB/s on H100 SXM).
 *
 * Design constraints:
 *   - Block size: 16 MB (KV_BLOCK_BYTES)
 *   - Transfers overlap with the next decode step via cudaMemcpyAsync
 *   - Each CUDA block handles one 16 MB page
 *   - Uses float4 (128-bit) loads/stores for peak memory throughput
 *
 * Integration:
 *   Called by KVCacheManager._evict_for() when PERC selects a session
 *   for eviction. The copy runs on a dedicated cudaStream_t so it does
 *   not block the compute stream.
 *
 * Compilation:
 *   nvcc -O3 -arch=sm_90 -shared -o libkvcopy.so kv_block_copy.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define KV_BLOCK_BYTES (16 * 1024 * 1024)
#define THREADS_PER_BLOCK 256
#define FLOAT4_PER_BLOCK (KV_BLOCK_BYTES / sizeof(float4))

/*
 * kv_evict_kernel — copy one KV block from device to pinned host.
 *
 * Each thread copies (FLOAT4_PER_BLOCK / gridDim.x / THREADS_PER_BLOCK)
 * float4 elements. Threads within a warp access consecutive addresses
 * for coalesced HBM reads.
 */
__global__ void kv_evict_kernel(
    const float4* __restrict__ src,   /* device pointer to KV block */
          float4* __restrict__ dst,   /* pinned host pointer         */
    int n_elements                    /* total float4 count          */
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n_elements; i += stride) {
        dst[i] = src[i];
    }
}

/*
 * kv_restore_kernel — copy one KV block from pinned host back to device.
 *
 * Used when a CPU-offloaded session is resumed and its context must
 * be restored to VRAM before the next forward pass.
 */
__global__ void kv_restore_kernel(
    const float4* __restrict__ src,   /* pinned host pointer */
          float4* __restrict__ dst,   /* device pointer      */
    int n_elements
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n_elements; i += stride) {
        dst[i] = src[i];
    }
}

/*
 * kv_evict_batch — evict N blocks in a single kernel launch.
 *
 * src_ptrs / dst_ptrs are device-side arrays of pointers (one per block).
 * Each CUDA block handles one KV page. The outer loop handles the case
 * where N > gridDim.x.
 */
__global__ void kv_evict_batch(
    float4** src_ptrs,
    float4** dst_ptrs,
    int n_blocks,
    int n_elements_per_block
) {
    for (int b = blockIdx.x; b < n_blocks; b += gridDim.x) {
        float4* src = src_ptrs[b];
        float4* dst = dst_ptrs[b];
        for (int i = threadIdx.x; i < n_elements_per_block; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

/* -------------------------------------------------------------------------
 * Host-side launch wrappers (extern "C" for Python ctypes / PyO3 binding)
 * ---------------------------------------------------------------------- */

extern "C" {

/*
 * launch_kv_evict — async eviction of one 16 MB block.
 *
 * Returns cudaError_t. Caller must synchronize the stream before
 * reading from dst on the host.
 */
cudaError_t launch_kv_evict(
    const void* src_device,
    void*       dst_host_pinned,
    cudaStream_t stream
) {
    int n_elements = KV_BLOCK_BYTES / sizeof(float4);
    int grid = (n_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kv_evict_kernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
        (const float4*)src_device,
        (float4*)dst_host_pinned,
        n_elements
    );
    return cudaGetLastError();
}

/*
 * launch_kv_restore — async restoration of one 16 MB block.
 */
cudaError_t launch_kv_restore(
    const void* src_host_pinned,
    void*       dst_device,
    cudaStream_t stream
) {
    int n_elements = KV_BLOCK_BYTES / sizeof(float4);
    int grid = (n_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kv_restore_kernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
        (const float4*)src_host_pinned,
        (float4*)dst_device,
        n_elements
    );
    return cudaGetLastError();
}

} // extern "C"

/*
 * Bandwidth estimate:
 *   H100 SXM HBM bandwidth:   3.35 TB/s
 *   H100 PCIe host bandwidth:  ~64 GB/s (bidirectional)
 *   16 MB block eviction time: 16 MB / 64 GB/s ≈ 250 µs
 *   Typical decode step time:  ~28 ms (chat model)
 *   Overlap efficiency:        (28 ms - 0.25 ms) / 28 ms ≈ 99.1%
 *
 * The eviction transfer is entirely hidden inside the decode step
 * when dispatched on a separate CUDA stream.
 */

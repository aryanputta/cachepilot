# CachePilot Architecture

## System Diagram

```
                        ┌─────────────────────┐
                        │    Client Requests    │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │    RequestQueue       │
                        │  Priority + SLA promo │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │    DynamicBatcher     │
                        │  MAX_TPUT / LOW_LAT   │
                        │  ADAPTIVE (auto-tune) │
                        └──────────┬───────────┘
                                   │
                ┌──────────────────▼──────────────────┐
                │           KVCacheManager             │
                │                                      │
                │  register_session(prompt_len)         │
                │  extend_session(new_tokens)           │
                │  release_session()                    │
                │  restore_session() [CPU → GPU]        │
                └──────────────────┬──────────────────┘
                                   │
              ┌────────────────────▼──────────────────────┐
              │                 VRAMPool                    │
              │   Paged 16MB blocks  |  Pinned (weights)   │
              │   allocate() / deallocate() / stats()       │
              └────────────────────┬──────────────────────┘
                                   │ pressure?
                        ┌──────────▼───────────┐
                        │    EvictionPolicy     │
                        │                       │
                        │  PERCEviction  ◄──────┼── NOVEL (provably optimal)
                        │  LRUEviction   ◄──────┼── baseline
                        │  PriorityEvict ◄──────┼── baseline
                        └──────────┬───────────┘
                                   │  evicted sessions
                        ┌──────────▼───────────┐
                        │    CPU Offload Store  │
                        │  dict[session → len]  │
                        └───────────────────────┘

                        ┌──────────────────────┐
                        │   TelemetryCollector  │
                        │  tok/s | p50/p95 TPOT │
                        │  VRAM util | evictions │
                        └──────────────────────┘
```

## Data Flow for One Request

```
1. Request arrives → RequestQueue.push()
2. DynamicBatcher picks it from the queue
3. KVCacheManager.register_session(prompt_len)
   a. VRAMPool.allocate(n_blocks)
   b. If OOM: EvictionPolicy.rank() → select_eviction_set() → free blocks
   c. If still OOM: request dropped (counted in requests_dropped)
4. Decode loop: for each token step
   a. SimRequest.step() returns (tokens, latency_s)
   b. KVCacheManager.extend_session() grows the allocation
   c. TelemetryCollector.record_tokens()
5. Request completes: KVCacheManager.release_session()
```

## Module Dependency Graph

```
cli.py
  └── engine.py
        ├── memory.py          (VRAMPool)
        ├── eviction.py        (PERC, LRU, Priority)
        ├── kv_manager.py      (depends on memory + eviction)
        ├── scheduler.py       (RequestQueue)
        ├── batcher.py         (DynamicBatcher)
        ├── telemetry.py       (TelemetryCollector)
        └── simulator.py       (load_generator, SimRequest)

kernels/
  ├── kv_quantize.py    (Triton + NumPy fallback)
  └── kv_block_copy.py  (Triton + timing simulation)

vllm_patch/
  └── perc_evictor.py   (drop-in for vllm/core/evictor_v2.py)

policy/
  └── rl_policy.py      (TinyMLP, ILTrainer, AdmissionPolicy)

src/cuda/
  ├── kv_block_copy.cu  (CUDA C++ kernel, nvcc)
  └── kv_quant.cu       (CUDA C++ kernel, nvcc)
```

## Memory Layout

```
GPU VRAM (e.g., 24 GB)
┌─────────────────────────────┐
│  Pinned: model weights 8GB  │  (never evicted)
├─────────────────────────────┤
│  Session A: 3 × 16MB = 48MB │
│  Session B: 1 × 16MB = 16MB │
│  Session C: 7 × 16MB = 112MB│
│  ...                        │
├─────────────────────────────┤
│  Free blocks                │
└─────────────────────────────┘

CPU Pinned RAM (offload target)
┌─────────────────────────────┐
│  Evicted session D: seq=2048│
│  Evicted session E: seq=512 │
└─────────────────────────────┘
```

## PERC Decision Point

```
New request needs B bytes of VRAM
        │
        ▼
   VRAMPool.allocate() fails?
        │ yes
        ▼
   Score all active sessions:
   score(i) = seq_len(i) * c_recompute * P(resume | λ_i, δ)
               ──────────────────────────────────────────────
                              n_blocks(i)
        │
        ▼
   Sort ascending (cheapest per byte first)
        │
        ▼
   Evict minimum prefix that frees ≥ B bytes
   [proven optimal by fractional knapsack — see docs/perc_proof.md]
        │
        ▼
   Re-attempt allocate() for new request
```

## Kernel Integration (GPU Path)

```
Eviction event triggered
        │
        ├─ kv_quant.cu: quantize FP16 → INT8 in-place (50% size reduction)
        │
        ├─ kv_block_copy.cu: async GPU→CPU on dedicated stream
        │   └─ runs in ~250µs, hidden inside ~28ms decode step
        │
        └─ CPU offload dict: session_id → seq_len
```

## vLLM Integration

```python
# vllm/core/block_manager_v2.py  (2-line change)

# Before:
from vllm.core.evictor_v2 import LRUEvictor
evictor = LRUEvictor()

# After:
from cachepilot.vllm_patch.perc_evictor import PERCEvictor
evictor = PERCEvictor(c_recompute=0.002, delta_serve=5.0)
```

One instrumentation line in the scheduler step:
```python
evictor.record_token(block_id)  # called each time a token is generated
```

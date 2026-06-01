# FlashInfer Header Snapshot For MPR

This directory vendors the small FlashInfer header subset needed by the MPR
Quest-style estimate kernel prototype.

Source snapshot:

```text
/home/han/KV_cache_quant/quest/kernels/3rdparty/flashinfer/include/flashinfer
```

Vendored headers:

```text
include/flashinfer/layout.cuh
include/flashinfer/utils.cuh
include/flashinfer/math.cuh
include/flashinfer/cp_async.cuh
include/flashinfer/vec_dtypes.cuh
```

These files retain their original FlashInfer Apache-2.0 copyright and license
headers. The MPR-specific Quest estimate kernel code lives outside this
third-party snapshot under `csrc/mpr`.

// The integer tensor-core product for PTQ1_0 (dense products only), in its own unit so the heavy
// template builds in parallel with the others.
#include "ggml_bridge_mmq.cuh"

NINFER_GGUF_DENSE_MATRIX_INSTANCE(GGML_TYPE_PTQ1_0);

/* common/hadamard.glsl - in-place orthonormal fast Walsh-Hadamard transform of shared-memory rows. Requires: common/nf_common.glsl */
/*
 * GLSL twin of kernels/llm/common/hadamard.h (the semantic reference: ggml_gen_hadamard applied per block as
 * llama_mul_mat_hadamard). GLSL cannot pass a shared array to a function, so NF_FWHT_SHARED_FN(fn, buf) defines
 *     void fn(uint base, uint n, uint block)
 * over the shared float array buf: it transforms every aligned block of `block` values of buf[base, base + n) in
 * place with the butterflies of nf_fwht_local (stage h = 1, 2, 4, ... < block maps pair p < n/2 to
 * i = 2h*(p/h) + p%h and (buf[i], buf[i + h]) -> (sum, difference)), then scales every value by 1/sqrt(block).
 * block is a power of two dividing n (1 is the identity). Every invocation of the workgroup calls it in uniform
 * control flow once the writes to buf are synchronized; it contains barriers and ends with one.
 */
#ifndef NF_HADAMARD_GLSL
#define NF_HADAMARD_GLSL

#ifndef NF_COMMON_GLSL
#error "common/hadamard.glsl requires common/nf_common.glsl"
#endif

#define NF_FWHT_SHARED_FN(fn, buf) \
    void fn(uint base, uint n, uint block) { \
        const uint lid = gl_LocalInvocationID.x; \
        const uint lsize = gl_WorkGroupSize.x; \
        const uint half_n = n >> 1; \
        for (uint h = 1u; h < block; h <<= 1) { \
            for (uint p = lid; p < half_n; p += lsize) { \
                const uint i = base + (((p & ~(h - 1u)) << 1) | (p & (h - 1u))); \
                const float a = buf[i]; \
                const float c = buf[i + h]; \
                buf[i] = a + c; \
                buf[i + h] = a - c; \
            } \
            barrier(); \
        } \
        const float scale = 1.0 / sqrt(float(block)); \
        for (uint k = lid; k < n; k += lsize) { \
            buf[base + k] *= scale; \
        } \
        barrier(); \
    }

#endif /* NF_HADAMARD_GLSL */

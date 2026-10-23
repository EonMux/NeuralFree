/* common/hadamard.h - in-place orthonormal fast Walsh-Hadamard transform of local-memory rows. Requires: common/dialect.h */
/*
 * The Sylvester Hadamard matrix scaled by 1/sqrt(block) of ggml_gen_hadamard (llama.cpp src/llama-kv-cache.cpp:21-56),
 * applied to every aligned block of `block` values as llama_mul_mat_hadamard (llama.cpp src/llama-impl.h:57-75). H*H =
 * I, so the transform is its own inverse. Shared by hadamard.cl (PRISM and quantized-KV row rotation), kv_cache.cl
 * (rotated block-type RoPE shift) and dsv4/compress.cl, dsv4/attn.cl (k_rot of the DeepSeek-V4 caches).
 */
#ifndef NF_HADAMARD_H
#define NF_HADAMARD_H

/*
 * Transforms every aligned block of `block` values of buf[0, n) in place. Butterfly stage h = 1, 2, 4, ... < block
 * maps pair p < n/2 to i = 2h*(p/h) + p%h and (buf[i], buf[i + h]) -> (buf[i] + buf[i + h], buf[i] - buf[i + h]),
 * which leaves the natural (Sylvester) order H_2n = [[H_n, H_n], [H_n, -H_n]]; every value is then scaled by
 * 1/sqrt(block). block is a power of two dividing n (1 is the identity). Every work-item of the group calls it in
 * uniform control flow once the writes to buf are synchronized; it contains barriers and ends with one.
 */
NF_INLINE void nf_fwht_local(NF_LOCAL_PTR float* buf, int n, int block) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int half_n = n >> 1;
    for (int h = 1; h < block; h <<= 1) {
        for (int p = lid; p < half_n; p += lsize) {
            const int i = ((p & ~(h - 1)) << 1) | (p & (h - 1));
            const float a = buf[i];
            const float c = buf[i + h];
            buf[i] = a + c;
            buf[i + h] = a - c;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float scale = 1.0f / sqrt((float)block);
    for (int k = lid; k < n; k += lsize) {
        buf[k] *= scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

#endif /* NF_HADAMARD_H */

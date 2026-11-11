/* common/tts_attend.glsl - the attention of one (query row, head) over a contiguous range of key rows, run by a whole
 * workgroup. GLSL twin of nf_tts_attend in kernels/llm/speech.cl (the semantic reference).
 * Requires: common/nf_common.glsl, and the including shader's buffers q_in, keys_in, values_in (readonly float) and
 * out_data (writeonly float).
 *
 *   out_data[o_off + d] = sum_j exp(s_j - max) * values_in[j*kv_ch + kv_off + d] / sum_j exp(s_j - max),
 *   s_j = (q_in[q_off ..] . keys_in[j*kv_ch + kv_off ..]) / sqrt(d_head),   j_lo <= j <= j_hi
 * An empty range writes zeros. Invocations take keys j_lo + lid, + L, ... for the scores (each dot in ascending d),
 * the maximum is an exact tree reduction, and each chunk's weights are staged in shared memory: invocation 0 sums them
 * in ascending j and invocation t accumulates output dimensions t, t + L, ... in ascending j. Every invocation calls
 * it in uniform control flow; the local size is a power of two <= LLM_DIMX.
 */
#ifndef NF_TTS_ATTEND_GLSL
#define NF_TTS_ATTEND_GLSL

#ifndef NF_COMMON_GLSL
#error "common/tts_attend.glsl requires common/nf_common.glsl"
#endif

#define NF_TTS_CHUNK 256   /* keys whose weights are staged at a time */
#define NF_TTS_ACC 4       /* output dimensions per invocation per pass over the keys */

shared float nf_tts_w[NF_TTS_CHUNK];
shared float nf_tts_red[gl_WorkGroupSize.x];

float nf_tts_dot(const uint q_off, const uint k_off, const int d_head) {
    float dot = 0.0;
    for (int d = 0; d < d_head; ++d) {
        dot += q_in[q_off + uint(d)] * keys_in[k_off + uint(d)];
    }
    return dot;
}

void nf_tts_attend(const uint q_off, const uint kv_ch, const uint kv_off, const int d_head, const int j_lo,
                   const int j_hi, const uint o_off) {
    const int lid = int(gl_LocalInvocationID.x);
    const int lsize = int(gl_WorkGroupSize.x);
    const float scale = 1.0 / sqrt(float(d_head));

    float mx = -3.402823466e+38;
    for (int j = j_lo + lid; j <= j_hi; j += lsize) {
        const float score = nf_tts_dot(q_off, uint(j) * kv_ch + kv_off, d_head) * scale;
        if (score > mx) {
            mx = score;
        }
    }
    nf_tts_red[lid] = mx;
    barrier();
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride && nf_tts_red[lid + stride] > nf_tts_red[lid]) {
            nf_tts_red[lid] = nf_tts_red[lid + stride];
        }
        barrier();
    }
    mx = nf_tts_red[0];
    barrier();

    for (int d0 = 0; d0 < d_head; d0 += lsize * NF_TTS_ACC) {
        float acc[NF_TTS_ACC];
        for (int a = 0; a < NF_TTS_ACC; ++a) {
            acc[a] = 0.0;
        }
        float sum = 0.0;
        for (int c0 = j_lo; c0 <= j_hi; c0 += NF_TTS_CHUNK) {
            const int n = min(NF_TTS_CHUNK, j_hi - c0 + 1);
            for (int t = lid; t < n; t += lsize) {
                const float score = nf_tts_dot(q_off, uint(c0 + t) * kv_ch + kv_off, d_head) * scale;
                nf_tts_w[t] = exp(score - mx);
            }
            barrier();
            if (lid == 0) {
                for (int t = 0; t < n; ++t) {
                    sum += nf_tts_w[t];
                }
            }
            for (int a = 0; a < NF_TTS_ACC; ++a) {
                const int d = d0 + a * lsize + lid;
                if (d < d_head) {
                    for (int t = 0; t < n; ++t) {
                        acc[a] += nf_tts_w[t] * values_in[uint(c0 + t) * kv_ch + kv_off + uint(d)];
                    }
                }
            }
            barrier();
        }
        if (lid == 0) {
            nf_tts_red[0] = sum;
        }
        barrier();
        const float total = nf_tts_red[0];
        const float inv = total > 0.0 ? 1.0 / total : 0.0;
        for (int a = 0; a < NF_TTS_ACC; ++a) {
            const int d = d0 + a * lsize + lid;
            if (d < d_head) {
                out_data[o_off + uint(d)] = acc[a] * inv;
            }
        }
        barrier();
    }
}

#endif

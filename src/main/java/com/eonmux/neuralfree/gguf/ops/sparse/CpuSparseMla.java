package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.ops.sparse.CpuDsaIndexer.RowReader;

import java.util.Arrays;
import java.util.Objects;

/**
 * CPU twin of {@code LLM_MLA_SparseDecode_<KV>} and {@code LLM_MLA_SparsePrefill_<KV>}: absorbed multi-head latent attention
 * restricted to the cells the DSA indexer selected (llama.cpp src/llama-graph.cpp:3402-3478, MQA over the latent cache
 * as in llama.cpp src/models/deepseek32.cpp:368-388).
 *
 * <p>With {@code dK = kvLora + ropeDim}, query row {@code q} ({@code [absorbed nope | rope]}, like the latent cache row
 * {@code [kv_lora | rope]}) and selected rows {@code r_j = sel[j]} ({@code r_j < 0} skipped):
 * <pre>
 *   s_j    = softcap &gt; 0 ? softcap * tanh(q.k_j * (kqScale / softcap)) : q.k_j * kqScale
 *   m      = max(max_j s_j, sink)                 (sink only with sinks)
 *   l      = sum_j exp(s_j - m) + exp(sink - m)
 *   out[d] = (1/l) * sum_j exp(s_j - m) * k_j[d],  d &lt; kvLora   (zeros when l == 0)
 * </pre>
 * The softcap convention is ggml flash attention's ({@code scale /= softcap; s = softcap * tanh(s * scale)}).
 * The selected latent rows of a token are read once and shared by every head.
 */
public final class CpuSparseMla {

    private CpuSparseMla() {
    }

    /**
     * One decode token: head {@code h} reads {@code qAbs[qOff + h*dK ..]} and writes {@code out[outOff + h*kvLora ..]};
     * the selection is {@code sel[selOff .. selOff + nSel)}.
     *
     * @param sinks per-head sink logits at {@code sinks[sinksOff + h]}; ignored (may be null) without sinks
     * @throws IllegalArgumentException for invalid shapes
     */
    public static void decode(int nHead, int kvLora, int ropeDim, int nSel, float kqScale, float softcap, boolean hasSinks,
                              float[] qAbs, int qOff, RowReader latent, int[] sel, int selOff,
                              float[] sinks, int sinksOff, float[] out, int outOff) {
        prefill(1, nHead, kvLora, ropeDim, nSel, kqScale, softcap, hasSinks, qAbs, qOff, 0, latent, sel, selOff,
                sinks, sinksOff, out, outOff);
    }

    /**
     * A batch of tokens: token {@code t}, head {@code h} reads {@code qAbs[qOff + t*qStride + h*dK ..]}, the selection
     * {@code sel[selOff + t*nSel ..]}, and writes {@code out[outOff + (t*nHead + h)*kvLora ..]}.
     *
     * @param sinks per-head sink logits at {@code sinks[sinksOff + h]}; ignored (may be null) without sinks
     * @throws IllegalArgumentException for invalid shapes
     */
    public static void prefill(int nTokens, int nHead, int kvLora, int ropeDim, int nSel, float kqScale, float softcap,
                               boolean hasSinks, float[] qAbs, int qOff, int qStride, RowReader latent,
                               int[] sel, int selOff, float[] sinks, int sinksOff, float[] out, int outOff) {
        if (nTokens < 1 || nHead < 1 || kvLora < 1 || ropeDim < 0 || nSel < 1) {
            throw new IllegalArgumentException("sparse MLA needs nTokens, nHead, kvLora, nSel >= 1 and ropeDim >= 0, got "
                    + nTokens + "/" + nHead + "/" + kvLora + "/" + nSel + "/" + ropeDim);
        }
        Objects.requireNonNull(qAbs, "qAbs");
        Objects.requireNonNull(latent, "latent");
        Objects.requireNonNull(sel, "sel");
        Objects.requireNonNull(out, "out");
        if (hasSinks) {
            Objects.requireNonNull(sinks, "sinks");
        }
        int dK = kvLora + ropeDim;
        float[] rows = new float[nSel * dK];
        for (int t = 0; t < nTokens; t++) {
            int selBase = selOff + t * nSel;
            CpuParallel.parallelFor(nSel, 16, (from, to) -> {
                for (int j = from; j < to; j++) {
                    int r = sel[selBase + j];
                    if (r >= 0) {
                        latent.read(r, rows, j * dK);
                    }
                }
            });
            int qBase = qOff + t * qStride;
            int outBase = outOff + t * nHead * kvLora;
            CpuParallel.parallelFor(nHead, 1, (from, to) -> {
                float[] p = new float[nSel];
                float[] acc = new float[kvLora];
                for (int h = from; h < to; h++) {
                    float sink = hasSinks ? sinks[sinksOff + h] : 0.0f;
                    attendRow(qAbs, qBase + h * dK, rows, sel, selBase, nSel, kvLora, dK, kqScale, softcap,
                            hasSinks, sink, p, acc, out, outBase + h * kvLora);
                }
            });
        }
    }

    /** Attention logit of one selected cell. */
    static float logit(float qk, float kqScale, float softcap) {
        return softcap > 0.0f ? softcap * (float) Math.tanh(qk * (kqScale / softcap)) : qk * kqScale;
    }

    private static void attendRow(float[] q, int qOff, float[] rows, int[] sel, int selBase, int nSel, int kvLora, int dK,
                                  float kqScale, float softcap, boolean hasSinks, float sink, float[] p, float[] acc,
                                  float[] out, int outOff) {
        float m = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < nSel; j++) {
            float s = Float.NEGATIVE_INFINITY;
            if (sel[selBase + j] >= 0) {
                int k = j * dK;
                float qk = 0.0f;
                for (int d = 0; d < dK; d++) {
                    qk += q[qOff + d] * rows[k + d];
                }
                s = logit(qk, kqScale, softcap);
            }
            p[j] = s;
            m = Math.max(m, s);
        }
        if (hasSinks) {
            m = Math.max(m, sink);
        }
        float l = 0.0f;
        for (int j = 0; j < nSel; j++) {
            float pj = p[j] == Float.NEGATIVE_INFINITY ? 0.0f : (float) Math.exp(p[j] - m);
            p[j] = pj;
            l += pj;
        }
        if (hasSinks && m > Float.NEGATIVE_INFINITY) {
            l += (float) Math.exp(sink - m);
        }
        float invL = l > 0.0f ? 1.0f / l : 0.0f;
        Arrays.fill(acc, 0, kvLora, 0.0f);
        for (int j = 0; j < nSel; j++) {
            float pj = p[j];
            if (pj != 0.0f) {
                int v = j * dK;
                for (int d = 0; d < kvLora; d++) {
                    acc[d] += pj * rows[v + d];
                }
            }
        }
        for (int d = 0; d < kvLora; d++) {
            out[outOff + d] = acc[d] * invL;
        }
    }
}

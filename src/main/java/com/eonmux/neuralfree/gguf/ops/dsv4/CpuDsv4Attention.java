package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;

import java.util.Arrays;
import java.util.Objects;

/**
 * CPU twin of {@code kernels/llm/dsv4/attn.cl} ({@code LLM_DSV4_CompressedAttn}): DeepSeek-V4 MQA attention of every
 * query head over the visible raw-window rows and compressed rows, one shared K row serving as K and V, with
 * attention sinks, the cache Hadamard rotation and the inverse RoPE of the output tail (llama.cpp
 * src/models/deepseek4.cpp: 714-860, 1185-1215). Keys are visited in the kernel's order (raw rows, then compressed
 * rows); the softmax is the two-pass form of the kernel's online softmax:
 * <pre>
 *   s_k = (H q . k_k) * kq_scale,  M = max(max_k s_k, sink),  l = sum_k e^(s_k - M) + e^(sink - M)
 *   out = derope_p(H (sum_k e^(s_k - M) k_k / l))
 * </pre>
 * with {@code H} the identity when {@code rotBlock == 0}, no sink term without sinks, and zeros when {@code l == 0}.
 */
public final class CpuDsv4Attention {

    private CpuDsv4Attention() {
    }

    /** Immutable per-layer attention shape. */
    public static final class Layer {

        private final int nHead;
        private final int headDim;
        private final float kqScale;
        private final int nSwa;
        private final Dsv4Kernels.CompMode mode;
        private final int ratio;
        private final int rotBlock;
        private final Dsv4Kernels.RopeParams derope;
        private final float[] sinks;

        private Layer(int nHead, int headDim, float kqScale, int nSwa, Dsv4Kernels.CompMode mode, int ratio,
                      int rotBlock, Dsv4Kernels.RopeParams derope, float[] sinks) {
            this.nHead = nHead;
            this.headDim = headDim;
            this.kqScale = kqScale;
            this.nSwa = nSwa;
            this.mode = mode;
            this.ratio = ratio;
            this.rotBlock = rotBlock;
            this.derope = derope;
            this.sinks = sinks;
        }

        /**
         * @param nSwa     raw window ({@code n_swa}); a raw row at {@code kpos} is visible to {@code p} when
         *                 {@code 0 <= kpos <= p} and {@code p - kpos < nSwa}
         * @param ratio    compress ratio of the layer, 0 for a window-only layer
         * @param rotBlock cache Hadamard block, 0 for none
         * @param derope   the layer's RoPE, undone on the output tail
         * @param sinks    {@code nHead} sink logits, or null without sinks
         * @throws IllegalArgumentException for an unsupported shape
         */
        public static Layer of(int nHead, int headDim, float kqScale, int nSwa, int ratio, int rotBlock,
                               Dsv4Kernels.RopeParams derope, float[] sinks) {
            if (nHead < 1) {
                throw new IllegalArgumentException("attention needs at least one head, got " + nHead);
            }
            Dsv4Kernels.checkRowWidth(headDim);
            if (nSwa < 1) {
                throw new IllegalArgumentException("raw window must be positive, got " + nSwa);
            }
            Dsv4Kernels.CompMode mode = Dsv4Kernels.CompMode.forRatio(ratio);
            Dsv4Kernels.checkRotBlock(rotBlock, headDim);
            Dsv4Kernels.checkRope(derope, headDim);
            if (sinks != null && sinks.length < nHead) {
                throw new IllegalArgumentException("sinks hold " + sinks.length + " < " + nHead + " values");
            }
            return new Layer(nHead, headDim, kqScale, nSwa, mode, ratio, rotBlock, derope,
                    sinks == null ? null : sinks.clone());
        }

        public int nHead() { return nHead; }
        public int headDim() { return headDim; }
        public float kqScale() { return kqScale; }
        public int nSwa() { return nSwa; }
        public Dsv4Kernels.CompMode mode() { return mode; }
        public int ratio() { return ratio; }
        public int rotBlock() { return rotBlock; }
        public Dsv4Kernels.RopeParams derope() { return derope; }
        public boolean hasSinks() { return sinks != null; }

        @Override
        public String toString() {
            return "Dsv4Attention{nHead=" + nHead + ", headDim=" + headDim + ", kqScale=" + kqScale + ", nSwa=" + nSwa
                    + ", mode=" + mode + ", ratio=" + ratio + ", rotBlock=" + rotBlock + ", sinks=" + hasSinks() + "}";
        }
    }

    /** Compressed keys of one step: the cache, its row count and, for {@code SELECTED}, the per-token selection. */
    public static final class Compressed {

        private static final Compressed NONE = new Compressed(null, 0, null, 0, 0);

        private final KvAccessor cache;
        private final int nComp;
        private final int[] sel;
        private final int selOff;
        private final int selStride;

        private Compressed(KvAccessor cache, int nComp, int[] sel, int selOff, int selStride) {
            this.cache = cache;
            this.nComp = nComp;
            this.sel = sel;
            this.selOff = selOff;
            this.selStride = selStride;
        }

        /** No compressed keys (window-only layer). */
        public static Compressed none() {
            return NONE;
        }

        /** Rows {@code [0, nComp)} of {@code cache} (HCA). */
        public static Compressed dense(KvAccessor cache, int nComp) {
            Objects.requireNonNull(cache, "compressed cache");
            if (nComp < 0 || nComp > cache.layout().rows()) {
                throw new IllegalArgumentException("dense compressed rows " + nComp + " outside [0, "
                        + cache.layout().rows() + "]");
            }
            return new Compressed(cache, nComp, null, 0, 0);
        }

        /**
         * Rows {@code sel[selOff + t * selStride + j]}, {@code j < nComp}, of {@code cache} for token {@code t} (CSA);
         * negative entries are padding.
         */
        public static Compressed selected(KvAccessor cache, int nComp, int[] sel, int selOff, int selStride) {
            Objects.requireNonNull(cache, "compressed cache");
            Objects.requireNonNull(sel, "selection");
            if (nComp < 0 || selOff < 0 || selStride < nComp) {
                throw new IllegalArgumentException("selection needs n_comp >= 0, offset >= 0 and stride >= n_comp, got "
                        + nComp + "/" + selOff + "/" + selStride);
            }
            return new Compressed(cache, nComp, sel, selOff, selStride);
        }
    }

    /**
     * Attention of {@code nTokens} query tokens. Query and output rows are {@code [t][h][headDim]} at {@code qOff} /
     * {@code outOff}; {@code qPos[t]} is the token position; raw row {@code r < nRaw} holds position
     * {@code rawPos[r]} ({@code -1} when empty).
     *
     * @throws IllegalArgumentException when the inputs do not match the layer or each other
     */
    public static void attend(Layer layer, int nTokens, float[] q, int qOff, int[] qPos, KvAccessor raw, int nRaw,
                              int[] rawPos, Compressed comp, float[] out, int outOff) {
        checkInputs(layer, nTokens, q, qOff, qPos, raw, nRaw, rawPos, comp, out, outOff);
        CpuParallel.parallelFor(nTokens, 1, (from, to) -> {
            final int hd = layer.headDim;
            float[] keys = new float[(nRaw + comp.nComp) * hd];
            float[] qh = new float[hd];
            float[] o = new float[hd];
            float[] s = new float[nRaw + comp.nComp];
            for (int t = from; t < to; t++) {
                int nKeys = gatherKeys(layer, t, qPos[t], raw, nRaw, rawPos, comp, keys);
                for (int h = 0; h < layer.nHead; h++) {
                    int row = (t * layer.nHead + h) * hd;
                    System.arraycopy(q, qOff + row, qh, 0, hd);
                    attendHead(layer, h, qPos[t], qh, keys, nKeys, s, o);
                    System.arraycopy(o, 0, out, outOff + row, hd);
                }
            }
        });
    }

    /** Dequantized visible key rows of token {@code t} into {@code keys}, raw rows first; returns their count. */
    private static int gatherKeys(Layer layer, int t, int p, KvAccessor raw, int nRaw, int[] rawPos, Compressed comp,
                                  float[] keys) {
        final int hd = layer.headDim;
        int n = 0;
        for (int r = 0; r < nRaw; r++) {
            int kp = rawPos[r];
            if (kp >= 0 && kp <= p && p - kp < layer.nSwa) {
                raw.readK(r, 0, keys, n * hd);
                n++;
            }
        }
        if (layer.mode == Dsv4Kernels.CompMode.NONE) {
            return n;
        }
        int nVis = Dsv4Kernels.visibleRows(p, layer.ratio);
        for (int j = 0; j < comp.nComp; j++) {
            int c = layer.mode == Dsv4Kernels.CompMode.SELECTED ? comp.sel[comp.selOff + t * comp.selStride + j] : j;
            if (c >= 0 && c < nVis) {
                if (c >= comp.cache.layout().rows()) {
                    throw new IllegalArgumentException("compressed row " + c + " exceeds the cache's "
                            + comp.cache.layout().rows() + " rows");
                }
                comp.cache.readK(c, 0, keys, n * hd);
                n++;
            }
        }
        return n;
    }

    /** One (token, head) row: {@code qh} is the head query (rotated in place), {@code o} receives the output. */
    private static void attendHead(Layer layer, int h, int p, float[] qh, float[] keys, int nKeys, float[] s,
                                   float[] o) {
        final int hd = layer.headDim;
        if (layer.rotBlock > 0) {
            CpuDsv4Compressor.hadamard(qh, 0, hd, layer.rotBlock);
        }
        float m = Float.NEGATIVE_INFINITY;
        for (int k = 0; k < nKeys; k++) {
            float dot = 0f;
            int kb = k * hd;
            for (int d = 0; d < hd; d++) {
                dot += qh[d] * keys[kb + d];
            }
            s[k] = dot * layer.kqScale;
            m = Math.max(m, s[k]);
        }
        if (layer.sinks != null) {
            m = Math.max(m, layer.sinks[h]);
        }
        Arrays.fill(o, 0f);
        float l = 0f;
        for (int k = 0; k < nKeys; k++) {
            float e = (float) Math.exp(s[k] - m);
            l += e;
            int kb = k * hd;
            for (int d = 0; d < hd; d++) {
                o[d] += e * keys[kb + d];
            }
        }
        if (layer.sinks != null) {
            l += (float) Math.exp(layer.sinks[h] - m);
        }
        float inv = l > 0f ? 1f / l : 0f;
        for (int d = 0; d < hd; d++) {
            o[d] *= inv;
        }
        if (layer.rotBlock > 0) {
            CpuDsv4Compressor.hadamard(o, 0, hd, layer.rotBlock);
        }
        CpuDsv4Compressor.rope(o, 0, hd, layer.derope, p, true);
    }

    private static void checkInputs(Layer layer, int nTokens, float[] q, int qOff, int[] qPos, KvAccessor raw,
                                    int nRaw, int[] rawPos, Compressed comp, float[] out, int outOff) {
        Objects.requireNonNull(layer, "layer");
        Objects.requireNonNull(q, "q");
        Objects.requireNonNull(qPos, "q_pos");
        Objects.requireNonNull(raw, "raw cache");
        Objects.requireNonNull(rawPos, "raw_pos");
        Objects.requireNonNull(comp, "compressed keys");
        Objects.requireNonNull(out, "out");
        final long rows = (long) nTokens * layer.nHead * layer.headDim;
        if (nTokens < 0 || qPos.length < nTokens) {
            throw new IllegalArgumentException("q_pos holds " + qPos.length + " positions for " + nTokens + " tokens");
        }
        if (qOff < 0 || outOff < 0 || qOff + rows > q.length || outOff + rows > out.length) {
            throw new IllegalArgumentException("q/out need " + rows + " values from offsets " + qOff + "/" + outOff
                    + ", got " + q.length + "/" + out.length);
        }
        CpuDsv4Compressor.checkKeyCache(raw, layer.headDim, "raw window cache");
        if (nRaw < 0 || nRaw > raw.layout().rows() || rawPos.length < nRaw) {
            throw new IllegalArgumentException("raw window cache of " + raw.layout().rows() + " rows with "
                    + rawPos.length + " positions cannot serve " + nRaw + " rows");
        }
        if (layer.mode == Dsv4Kernels.CompMode.NONE) {
            return;
        }
        if (comp.cache == null) {
            throw new IllegalArgumentException("layer " + layer + " needs a compressed cache");
        }
        CpuDsv4Compressor.checkKeyCache(comp.cache, layer.headDim, "compressed cache");
        if ((comp.sel != null) != (layer.mode == Dsv4Kernels.CompMode.SELECTED)) {
            throw new IllegalArgumentException("layer mode " + layer.mode + " does not match the compressed keys");
        }
        if (comp.sel != null && nTokens > 0
                && comp.selOff + (long) (nTokens - 1) * comp.selStride + comp.nComp > comp.sel.length) {
            throw new IllegalArgumentException("selection holds " + comp.sel.length + " entries, " + nTokens
                    + " rows of " + comp.nComp + " at stride " + comp.selStride + " do not fit");
        }
    }
}

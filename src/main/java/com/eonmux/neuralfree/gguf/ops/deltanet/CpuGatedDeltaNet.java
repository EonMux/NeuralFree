package com.eonmux.neuralfree.gguf.ops.deltanet;

import com.eonmux.neuralfree.gguf.arch.GdnHeadMap;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;

import java.util.Objects;

/**
 * CPU reference of {@code LLM_GDN_GatePrep} and {@code LLM_GDN_Scan} (gated DeltaNet and Kimi delta attention),
 * with the kernels' index formulas and state layout ({@code s[h*S_v*S_k + col*S_k + i]}, value column {@code col},
 * key {@code i}). Pure static math over {@code float[]} with explicit offsets and strides; the CPU
 * mixers call it per head range so heads can run in parallel.
 *
 * <p>Recurrence (llama.cpp ggml/src/ggml-cpu/ops.cpp:11214-11360), per value head h and token t:
 * <pre>
 *   S[i][col] *= exp(g)      (g per head, or g[i] per key dim for KDA)
 *   delta     = (v[col] - sum_i S[i][col] k[i]) * beta
 *   S[i][col] += k[i] * delta
 *   o[col]    = scale * sum_i S[i][col] q[i]
 * </pre>
 */
public final class CpuGatedDeltaNet {

    private CpuGatedDeltaNet() {
    }

    /** Token-major rows of an activation buffer: element {@code (t, j)} is {@code data[off + t*stride + j]}. */
    public static final class Rows {
        private final float[] data;
        private final int off;
        private final int stride;

        public Rows(float[] data, int off, int stride) {
            this.data = Objects.requireNonNull(data, "data");
            if (off < 0 || stride < 0) {
                throw new IllegalArgumentException("rows need a non-negative offset and stride: off=" + off + " stride=" + stride);
            }
            this.off = off;
            this.stride = stride;
        }

        public float[] data() { return data; }
        public int off() { return off; }
        public int stride() { return stride; }

        int index(int t, int j) {
            return off + t * stride + j;
        }
    }

    /** Immutable shape of {@code LLM_GDN_Scan}. */
    public static final class ScanShape {
        private final int sK;
        private final int sV;
        private final int nVHeads;
        private final int nKHeads;
        private final GdnHeadMap headMap;
        private final boolean perKeyGate;
        private final float scale;

        /**
         * @param perKeyGate true for KDA ({@code gate_mode 1}: one decay per key dim), false for GDN
         * @param scale      output scale; llama.cpp uses {@code 1/sqrt(S_k)} (delta-net-base.cpp:45)
         * @throws IllegalArgumentException when the kernel cannot run this shape
         */
        public ScanShape(int sK, int sV, int nVHeads, int nKHeads, GdnHeadMap headMap, boolean perKeyGate, float scale) {
            Objects.requireNonNull(headMap, "headMap");
            DeltaNetKernels.validateScan(sK, sV, nVHeads, nKHeads, headMap.code(),
                    perKeyGate ? DeltaNetKernels.GATE_MODE_PER_KEY : DeltaNetKernels.GATE_MODE_PER_HEAD);
            this.sK = sK;
            this.sV = sV;
            this.nVHeads = nVHeads;
            this.nKHeads = nKHeads;
            this.headMap = headMap;
            this.perKeyGate = perKeyGate;
            this.scale = scale;
        }

        /**
         * Scan shape of a GDN or KDA layer: GDN uses the spec's head map and scalar gate; KDA has one key head per
         * value head and a per-key gate. The scale is {@code 1/sqrt(S_k)}.
         *
         * @throws IllegalArgumentException when the spec is neither GDN nor KDA or exceeds the kernel limits
         */
        public static ScanShape of(RecurrentSpec spec) {
            Objects.requireNonNull(spec, "spec");
            if (spec.kind() == MixerKind.GDN) {
                return new ScanShape(spec.headKDim(), spec.headVDim(), spec.nVHeads(), spec.nKHeads(), spec.headMap(),
                        false, defaultScale(spec.headKDim()));
            }
            if (spec.kind() == MixerKind.KDA) {
                return new ScanShape(spec.headDim(), spec.headDim(), spec.nHead(), spec.nHead(), GdnHeadMap.TILED,
                        true, defaultScale(spec.headDim()));
            }
            throw new IllegalArgumentException("delta-rule scan needs a GDN or KDA layer, got " + spec.kind());
        }

        public int sK() { return sK; }
        public int sV() { return sV; }
        public int nVHeads() { return nVHeads; }
        public int nKHeads() { return nKHeads; }
        public GdnHeadMap headMap() { return headMap; }
        public boolean perKeyGate() { return perKeyGate; }
        public float scale() { return scale; }
        /** {@code gate_mode} kernel argument. */
        public int gateMode() { return perKeyGate ? DeltaNetKernels.GATE_MODE_PER_KEY : DeltaNetKernels.GATE_MODE_PER_HEAD; }
        /** Floats of one state row: {@code n_v_heads * S_v * S_k}. */
        public long stateFloats() { return (long) nVHeads * sV * sK; }
        /** Decay values per (token, value head): 1 or S_k. */
        public int gatesPerHead() { return perKeyGate ? sK : 1; }

        /** Key head read by value head {@code h} (tiled {@code h % n_k}, grouped {@code h / (n_v / n_k)}). */
        public int keyHead(int h) {
            return CpuGatedDeltaNet.keyHead(h, nVHeads, nKHeads, headMap);
        }
    }

    /** Immutable shape of {@code LLM_GDN_GatePrep}. */
    public static final class GateShape {
        private final int nVHeads;
        private final int nKHeads;
        private final int layout;
        private final RecurrentSpec.DecayGate decayGate;
        private final int headDim;
        private final float lb;

        /**
         * @param layout  {@link DeltaNetKernels#LAYOUT_SEPARATE} or {@link DeltaNetKernels#LAYOUT_FUSED_BA}
         * @param headDim KDA head dim (gated values per head); ignored for GDN
         * @param lb      KDA gate lower bound; read only by {@code LB_NEG} and {@code LB_POS}, which need it finite
         * @throws IllegalArgumentException when the kernel cannot run this shape
         */
        public GateShape(int nVHeads, int nKHeads, int layout, RecurrentSpec.DecayGate decayGate, int headDim, float lb) {
            Objects.requireNonNull(decayGate, "decayGate");
            DeltaNetKernels.validateGatePrep(nVHeads, nKHeads, layout, decayGate.code(), headDim, lb);
            this.nVHeads = nVHeads;
            this.nKHeads = nKHeads;
            this.layout = layout;
            this.decayGate = decayGate;
            this.headDim = headDim;
            this.lb = lb;
        }

        /**
         * Gate shape of a GDN layer (fused {@code ssm_ba} when the spec says so) or a KDA layer.
         *
         * @throws IllegalArgumentException when the spec is neither GDN nor KDA
         */
        public static GateShape of(RecurrentSpec spec) {
            Objects.requireNonNull(spec, "spec");
            if (spec.kind() == MixerKind.GDN) {
                return new GateShape(spec.nVHeads(), spec.nKHeads(),
                        spec.fusedBetaAlpha() ? DeltaNetKernels.LAYOUT_FUSED_BA : DeltaNetKernels.LAYOUT_SEPARATE,
                        RecurrentSpec.DecayGate.GDN, spec.headVDim(), 0f);
            }
            if (spec.kind() == MixerKind.KDA) {
                RecurrentSpec.DecayGate gate = spec.decayGate();
                float lb = gate == RecurrentSpec.DecayGate.SOFTPLUS ? 0f : spec.gateLowerBound();
                return new GateShape(spec.nHead(), spec.nHead(), DeltaNetKernels.LAYOUT_SEPARATE, gate, spec.headDim(), lb);
            }
            throw new IllegalArgumentException("delta-rule gate prep needs a GDN or KDA layer, got " + spec.kind());
        }

        public int nVHeads() { return nVHeads; }
        public int nKHeads() { return nKHeads; }
        public int layout() { return layout; }
        public RecurrentSpec.DecayGate decayGate() { return decayGate; }
        /** {@code kda_mode} kernel argument. */
        public int kdaMode() { return decayGate.code(); }
        public int headDim() { return headDim; }
        public float lb() { return lb; }
        /** Gate values per (token, value head): 1 for GDN, {@code headDim} for KDA. */
        public int gatesPerHead() { return decayGate == RecurrentSpec.DecayGate.GDN ? 1 : headDim; }
    }

    /** {@code 1/sqrt(S_k)} as llama.cpp computes it in float (delta-net-base.cpp:45, ggml-cpu/ops.cpp:11284). */
    public static float defaultScale(int sK) {
        return 1.0f / (float) Math.sqrt(sK);
    }

    /** Key head of value head {@code h} under {@code map}. */
    public static int keyHead(int h, int nVHeads, int nKHeads, GdnHeadMap map) {
        return map == GdnHeadMap.GROUPED ? h / (nVHeads / nKHeads) : h % nKHeads;
    }

    /**
     * Decay gate of one element, with {@code x = a + dt_bias} and {@code a_h = ssm_a[h]}.
     *
     * @param kdaMode {@link RecurrentSpec.DecayGate#code()}: -1 GDN and 0 SOFTPLUS give {@code softplus(x)*a_h},
     *                1 LB_NEG gives {@code lb*sigmoid(-(a_h*x))}, 2 LB_POS gives {@code lb*sigmoid(a_h*x)}
     */
    public static float gateValue(int kdaMode, float x, float aH, float lb) {
        switch (kdaMode) {
            case 1:
                return lb * ActMath.sigmoid(-(aH * x));
            case 2:
                return lb * ActMath.sigmoid(aH * x);
            case -1:
            case 0:
                return ActMath.softplus(x) * aH;
            default:
                throw new IllegalArgumentException("kda_mode must be -1, 0, 1 or 2, got " + kdaMode);
        }
    }

    /**
     * {@code LLM_GDN_GatePrep} for {@code nTokens} tokens. Outputs are dense:
     * {@code gOut[gOff + (t*n_v + h)*G + d]} with G = {@link GateShape#gatesPerHead()} and
     * {@code betaOut[betaOff + t*n_v + h]}.
     *
     * @param a      alpha (GDN) or {@code f} (KDA) rows; for the fused layout the {@code ssm_ba} rows
     * @param b      beta logits rows; for the fused layout the same rows as {@code a}
     * @param dtBias {@code ssm_dt} bias: n_v floats (GDN) or n_v*headDim floats (KDA)
     * @param ssmA   {@code ssm_a}: n_v floats
     */
    public static void gatePrep(GateShape shape, int nTokens, Rows a, Rows b, float[] dtBias, float[] ssmA,
                                float[] gOut, int gOff, float[] betaOut, int betaOff) {
        final int nV = shape.nVHeads;
        final int perHead = shape.gatesPerHead();
        final int kdaMode = shape.kdaMode();
        final boolean fused = shape.layout == DeltaNetKernels.LAYOUT_FUSED_BA;
        final int ratio = nV / shape.nKHeads;
        for (int t = 0; t < nTokens; t++) {
            for (int h = 0; h < nV; h++) {
                int bIdx;
                int aBase;
                if (fused) {
                    int kh = h / ratio;
                    int r = h - kh * ratio;
                    bIdx = kh * 2 * ratio + r;
                    aBase = bIdx + ratio;
                } else {
                    bIdx = h;
                    aBase = kdaMode < 0 ? h : h * shape.headDim;
                }
                float aH = ssmA[h];
                int dtBase = kdaMode < 0 ? h : h * shape.headDim;
                int gBase = gOff + (t * nV + h) * perHead;
                for (int d = 0; d < perHead; d++) {
                    float x = a.data[a.index(t, aBase + d)] + dtBias[dtBase + d];
                    gOut[gBase + d] = gateValue(kdaMode, x, aH, shape.lb);
                }
                betaOut[betaOff + t * nV + h] = ActMath.sigmoid(b.data[b.index(t, bIdx)]);
            }
        }
    }

    /** {@link #scanHeads} over every value head. */
    public static void scan(ScanShape shape, int nTokens, Rows q, Rows k, Rows v, float[] g, int gOff,
                            float[] beta, int betaOff, float[] state, int stateOff, Rows out) {
        scanHeads(shape, nTokens, q, k, v, g, gOff, beta, betaOff, state, stateOff, out, 0, shape.nVHeads);
    }

    /**
     * {@code LLM_GDN_Scan} for value heads {@code [h0, h1)}.
     * <ul>
     *   <li>q, k: key head kh of token t at {@code index(t, kh*S_k + i)}; v, out: {@code index(t, h*S_v + col)}.</li>
     *   <li>g: {@code g[gOff + t*n_v + h]} (per head) or {@code g[gOff + (t*n_v + h)*S_k + i]} (per key); log space.</li>
     *   <li>beta: {@code beta[betaOff + t*n_v + h]}; state row at {@code stateOff}: {@code s[h*S_v*S_k + col*S_k + i]}.</li>
     * </ul>
     * {@code out} may alias {@code v} with the same offset and stride (each element of v is read before the same
     * element of out is written). Disjoint head ranges touch disjoint state and output elements.
     */
    public static void scanHeads(ScanShape shape, int nTokens, Rows q, Rows k, Rows v, float[] g, int gOff,
                                 float[] beta, int betaOff, float[] state, int stateOff, Rows out, int h0, int h1) {
        if (h0 < 0 || h1 > shape.nVHeads || h0 > h1) {
            throw new IllegalArgumentException("value head range [" + h0 + ", " + h1 + ") outside [0, "
                    + shape.nVHeads + ")");
        }
        final int sK = shape.sK;
        final int sV = shape.sV;
        final int nV = shape.nVHeads;
        final float[] decay = new float[sK];
        for (int h = h0; h < h1; h++) {
            final int kh = shape.keyHead(h);
            final int headBase = stateOff + h * sV * sK;
            for (int t = 0; t < nTokens; t++) {
                final int qBase = q.index(t, kh * sK);
                final int kBase = k.index(t, kh * sK);
                final int gt = t * nV + h;
                if (shape.perKeyGate) {
                    final int gBase = gOff + gt * sK;
                    for (int i = 0; i < sK; i++) {
                        decay[i] = (float) Math.exp(g[gBase + i]);
                    }
                } else {
                    final float e = (float) Math.exp(g[gOff + gt]);
                    for (int i = 0; i < sK; i++) {
                        decay[i] = e;
                    }
                }
                final float b = beta[betaOff + gt];
                for (int col = 0; col < sV; col++) {
                    final int s0 = headBase + col * sK;
                    float kv = 0.0f;
                    for (int i = 0; i < sK; i++) {
                        float s = state[s0 + i] * decay[i];
                        state[s0 + i] = s;
                        kv += s * k.data[kBase + i];
                    }
                    final float delta = (v.data[v.index(t, h * sV + col)] - kv) * b;
                    float o = 0.0f;
                    for (int i = 0; i < sK; i++) {
                        float s = state[s0 + i] + k.data[kBase + i] * delta;
                        state[s0 + i] = s;
                        o += s * q.data[qBase + i];
                    }
                    out.data[out.index(t, h * sV + col)] = o * shape.scale;
                }
            }
        }
    }
}

package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuAttnKernels;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec.BiasMode;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.IntUnaryOperator;

/**
 * CPU attention with the semantics of the flash kernels ({@code kernels/llm/flash_decode.cl},
 * {@code flash_prefill.cl}, {@code flash_combine.cl}), which follow llama.cpp {@code ggml_flash_attn_ext}
 * (ggml-cpu/ops.cpp:8997-9145):
 * <ul>
 *   <li>score {@code s = softcap > 0 ? softcap * tanh(dot * (scale / softcap)) : dot * scale}, plus the position
 *       bias ({@link BiasMode}: ALiBi {@code slope(h) * -|p_q - p_k|}, or the T5 relative-position bias
 *       {@code rel_bias[bucket(p_k, p_q) * n_head + h]});</li>
 *   <li>keys are visible per the causal flag and the window ({@link KvWindow}, llama.cpp {@code is_masked_swa});
 *       an empty cache row (position -1) is never visible;</li>
 *   <li>query head {@code h} reads KV head {@code h / (n_head / n_head_kv)} (GQA);</li>
 *   <li>a scaled FP8 cache ({@link Params}: {@code kScale}/{@code vScale} per KV head) stores its rows divided by
 *       their scale, so each dequantized K row is multiplied by {@code kScale[kvh]} before the dot product and each
 *       V row by {@code vScale[kvh]} before the weighted sum, as vLLM's scaled FP8 attention is;</li>
 *   <li>softmax over the visible keys with {@code m = max s}, {@code p = e^(s - m)}, {@code l = sum p}, then the
 *       head's sink logit joins the denominator ({@code m' = max(m, sink)}, {@code l' = l e^(m-m') + e^(sink-m')});</li>
 *   <li>a row with no visible key and no sink outputs zeros.</li>
 * </ul>
 * Layouts: query {@code q[qOff + h * d_k + d]}, output {@code out[outOff + h * d_v + d]}, multi-row calls add a row
 * stride per token. Keys come from a {@link KvAccessor} (cache rows, positions from a row-position function) or
 * from {@link DenseKv} arrays (current batch, cross-attention, encoders).
 *
 * <p>Like the kernels, the query heads of one KV head are processed together (up to {@code NF_FD_MAX_G} per chunk,
 * each K and V row read once per chunk), and keys are consumed in blocks with an online softmax, so the work memory
 * per thread is independent of the context length.
 *
 * <p>When a multi-row call has fewer (token, KV head, head chunk) units than worker threads and at least
 * {@link #SPLIT_MIN_KEYS} keys (a decode row), the key range is also split across threads, as ggml's CPU flash path
 * does for one query row (ggml-cpu/ops.cpp:9580-9610): each split keeps its unnormalized online-softmax state
 * {@code (m_i, l_i, o_i)}, and the splits combine to {@code m = max m_i}, {@code l = sum l_i e^(m_i - m)},
 * {@code o = sum o_i e^(m_i - m)} before the sink and the normalization ({@code LLM_FlashAttn_Combine}).
 *
 * <p>A multi-row call with enough units attends up to {@link #QUERY_BLOCK} consecutive query rows per unit: each
 * key and value row is read (and dequantized) once for the block, and every (row, head) keeps its own online-softmax
 * state and runs the steps of its single-row call, so blocking changes memory traffic, never a result.</p>
 */
public final class CpuAttentionCore {
    /** Keys scored per block; bounds the score scratch to {@code HEAD_CHUNK * KEY_BLOCK} floats per thread. */
    static final int KEY_BLOCK = 256;
    /** Query heads of one KV head processed together (each K and V row read once per chunk); {@code NF_FD_MAX_G}. */
    static final int HEAD_CHUNK = LlmKernelConstants.FD_MAX_G;
    /** Fewest keys for which a call splits its key range across threads (ggml-cpu/ops.cpp:9580-9610). */
    static final int SPLIT_MIN_KEYS = 512;
    /** Most query rows of a multi-row call one unit attends together, each key and value row read once for all. */
    static final int QUERY_BLOCK = 8;
    /** Units per worker a multi-row call keeps when it blocks its query rows, for balance. */
    static final int UNITS_PER_WORKER = 4;

    private CpuAttentionCore() {
    }

    /** True for the two ALiBi bias modes (linear and square-root distance). */
    static boolean isAlibiMode(AttentionSpec.BiasMode mode) {
        return mode == AttentionSpec.BiasMode.ALIBI || mode == AttentionSpec.BiasMode.ALIBI_SQRT;
    }

    /** Score, bias and mask parameters of one attention layer. Immutable. */
    public static final class Params {
        private final int nHead;
        final float scale;
        final float softcap;
        private final float maxBias;
        final KvWindow window;
        private final BiasMode biasMode;
        private final int nBuckets;
        private final int maxDistance;
        private final int relExtent;
        private final float[] sinks;
        private final int sinkTokens;
        private final float[] sinkK;
        private final float[] sinkV;
        private final float[] relBias;
        /**
         * The Inkling per-step relative-logit table: caller-owned scratch the mixer rewrites before every step, so
         * unlike the configuration arrays it is not copied (the table is per-step data, not a fixed parameter).
         */
        private final float[] inklingTable;
        final float[] kScale;
        final float[] vScale;
        private final float[] slopes;

        private Params(Builder b) {
            this.nHead = b.nHead;
            this.scale = b.scale;
            this.softcap = b.softcap;
            this.maxBias = b.maxBias;
            this.window = b.window;
            this.biasMode = b.biasMode;
            this.nBuckets = b.nBuckets;
            this.maxDistance = b.maxDistance;
            this.relExtent = b.relExtent;
            this.sinks = b.sinks == null ? null : b.sinks.clone();
            this.sinkTokens = b.sinkTokens;
            this.sinkK = b.sinkK == null ? null : b.sinkK.clone();
            this.sinkV = b.sinkV == null ? null : b.sinkV.clone();
            this.relBias = b.relBias == null ? null : b.relBias.clone();
            this.inklingTable = b.inklingTable;
            this.kScale = b.kScale == null ? null : b.kScale.clone();
            this.vScale = b.vScale == null ? null : b.vScale.clone();
            this.slopes = new float[nHead];
            for (int h = 0; h < nHead; h++) {
                slopes[h] = isAlibi() ? alibiSlope(h, nHead, maxBias) : 1.0f;
            }
        }

        public static Builder builder(int nHead) {
            return new Builder(nHead);
        }

        /**
         * Parameters of an attention layer: {@link AttentionSpec#scoreScale()}, softcap, ALiBi max bias, the
         * effective window and causal flag, the bias mode with its T5 bucket parameters.
         *
         * @param sinks   per-head sink logits ({@code attn_sinks}); required when {@link AttentionSpec#sinks()}
         * @param relBias T5 {@code attn_rel_b} values {@code [n_buckets][n_head]}; required for a T5 bias mode
         * @throws NullPointerException     when a required array is missing
         * @throws IllegalArgumentException when a required array is too short
         */
        public static Params of(AttentionSpec spec, float[] sinks, float[] relBias) {
            return of(spec, sinks, relBias, null, null, null, null);
        }

        /**
         * {@link #of(AttentionSpec, float[], float[])} with the per-KV-head FP8 cache scales: a scaled cache stores
         * key and value rows divided by their scale, so the core multiplies each dequantized K row by
         * {@code kScale[kvHead]} and each V row by {@code vScale[kvHead]} (null for all ones, as an unscaled cache).
         *
         * @throws IllegalArgumentException when a scale array holds a non-finite or non-positive value
         */
        public static Params of(AttentionSpec spec, float[] sinks, float[] relBias, float[] kScale, float[] vScale) {
            return of(spec, sinks, relBias, kScale, vScale, null, null);
        }

        /**
         * {@link #of(AttentionSpec, float[], float[], float[], float[])} with the learned static sink tokens of
         * {@link AttentionSpec#sinkTokens()} (null arrays without them). The keys are already normalized and
         * un-rotated; {@code n_sink * n_head_kv} rows of {@code d_k} / {@code d_v} values.
         */
        public static Params of(AttentionSpec spec, float[] sinks, float[] relBias, float[] kScale, float[] vScale,
                                float[] sinkK, float[] sinkV) {
            Objects.requireNonNull(spec, "spec");
            Builder b = builder(spec.nHead())
                    .scale(spec.scoreScale())
                    .softcap(spec.softcap());
            SwaType swa = spec.effectiveSwaType();
            b.window(KvWindow.of(swa, swa == SwaType.NONE ? 0 : spec.nSwa(), spec.causal()));
            if (isAlibiMode(spec.biasMode())) {
                b.alibi(spec.alibiMaxBias(), spec.biasMode() == BiasMode.ALIBI_SQRT);
            } else if (spec.biasMode() == BiasMode.T5_CAUSAL || spec.biasMode() == BiasMode.T5_BIDIR) {
                b.t5(spec.biasMode() == BiasMode.T5_BIDIR, spec.t5Buckets(), spec.t5MaxDistance(), relBias);
            } else if (spec.biasMode() == BiasMode.INKLING) {
                b.inkling(spec.relBiasExtent(), relBias);
            }
            if (spec.sinks()) {
                b.sinks(sinks);
            }
            b.kvScales(kScale, vScale);
            b.sinkKv(sinkK, sinkV, spec.sinkTokens());
            return b.build();
        }

        public int nHead() { return nHead; }
        public float scale() { return scale; }
        public float softcap() { return softcap; }
        public float maxBias() { return maxBias; }
        /** Distances of the {@link BiasMode#INKLING} relative table ({@code rel_extent}); 0 for the other modes. */
        public int relExtent() { return relExtent; }
        public KvWindow window() { return window; }
        public BiasMode biasMode() { return biasMode; }
        public int nBuckets() { return nBuckets; }
        public int maxDistance() { return maxDistance; }
        public boolean hasSinks() { return sinks != null; }
        /** Learned static sink tokens joining every query; 0 when there are none. */
        public int sinkTokens() { return sinkTokens; }
        /** Static sink keys {@code [n_sink][n_head_kv][d_k]}, or null without them. */
        public float[] sinkK() { return sinkK; }
        /** Static sink values {@code [n_sink][n_head_kv][d_v]}, or null without them. */
        public float[] sinkV() { return sinkV; }

        /** Per-KV-head K cache scale, or null when every key row is unscaled. */
        public float[] kScale() { return kScale; }

        /** Per-KV-head V cache scale, or null when every value row is unscaled. */
        public float[] vScale() { return vScale; }

        /**
         * These parameters with the causal mask rule disabled (llama.cpp
         * {@code llama_set_causal_attn(false)}): the window keeps its type and span, keys after the query become
         * visible; every other parameter stays.
         */
        /** These parameters with another visibility window; every other parameter stays. */
        public Params withWindow(KvWindow other) {
            Objects.requireNonNull(other, "other");
            Builder b = builder(nHead)
                    .scale(scale)
                    .softcap(softcap)
                    .window(other)
                    .kvScales(kScale, vScale);
            if (isAlibiMode(biasMode)) {
                b.alibi(maxBias, biasMode == BiasMode.ALIBI_SQRT);
            } else if (biasMode == BiasMode.T5_CAUSAL || biasMode == BiasMode.T5_BIDIR) {
                b.t5(biasMode == BiasMode.T5_BIDIR, nBuckets, maxDistance, relBias);
            }
            if (sinks != null) {
                b.sinks(sinks);
            }
            return b.build();
        }

        public Params nonCausal() {
            Builder b = builder(nHead)
                    .scale(scale)
                    .softcap(softcap)
                    .window(window.nonCausal())
                    .kvScales(kScale, vScale);
            if (isAlibiMode(biasMode)) {
                b.alibi(maxBias, biasMode == BiasMode.ALIBI_SQRT);
            } else if (biasMode == BiasMode.T5_CAUSAL || biasMode == BiasMode.T5_BIDIR) {
                b.t5(biasMode == BiasMode.T5_BIDIR, nBuckets, maxDistance, relBias);
            } else if (biasMode == BiasMode.INKLING) {
                b.inkling(relExtent, inklingTable);
            }
            if (sinks != null) {
                b.sinks(sinks);
            }
            if (sinkTokens > 0) {
                b.sinkKv(sinkK, sinkV, sinkTokens);
            }
            return b.build();
        }

        /** True when this layer adds an ALiBi position bias (linear or square-root distance). */
        private boolean isAlibi() {
            return isAlibiMode(biasMode);
        }

        /** Sink logit of head {@code h}. @throws IllegalStateException without sinks */
        public float sink(int h) {
            if (sinks == null) {
                throw new IllegalStateException("attention has no sinks");
            }
            return sinks[h];
        }

        /** Position bias of head {@code h} for query position {@code pq} and key position {@code pk}. */
        public float bias(int h, int pq, int pk) {
            return bias(h, pq, pk, 0);
        }

        /**
         * Position bias of head {@code h} for the query of table row {@code qRow} at position {@code pq} and the key
         * at position {@code pk}. Row {@code qRow} is a token index of the current step for the relative tables
         * ({@link BiasMode#INKLING}), and ignored by the position-only modes.
         */
        public float bias(int h, int pq, int pk, int qRow) {
            switch (biasMode) {
                case ALIBI:
                    return slopes[h] * -(float) Math.abs(pq - pk);
                case ALIBI_SQRT:
                    // vLLM's use_alibi_sqrt: the past keys get -sqrt(distance), later keys nothing.
                    return pk <= pq ? slopes[h] * -(float) Math.sqrt(pq - pk) : 0.0f;
                case T5_CAUSAL:
                case T5_BIDIR:
                    return relBias[t5Bucket(pk, pq, nBuckets, maxDistance, biasMode == BiasMode.T5_BIDIR) * nHead + h];
                case INKLING: {
                    int d = pq - pk;
                    if (d < 0 || d >= relExtent) {
                        return 0.0f;
                    }
                    return inklingTable[(qRow * nHead + h) * relExtent + d];
                }
                default:
                    return 0.0f;
            }
        }

        @Override
        public String toString() {
            return "Params{nHead=" + nHead + ", scale=" + scale + ", softcap=" + softcap + ", " + window + ", bias="
                    + biasMode + (hasSinks() ? ", sinks" : "") + '}';
        }
    }

    /** Builder of {@link Params}; defaults: scale 1, no softcap, full causal window, no bias, no sinks. */
    public static final class Builder {
        private final int nHead;
        private float scale = 1.0f;
        private float softcap;
        private float maxBias;
        private KvWindow window = KvWindow.FULL_CAUSAL;
        private BiasMode biasMode = BiasMode.NONE;
        private int nBuckets;
        private int maxDistance;
        private int relExtent;
        private float[] sinks;
        private int sinkTokens;
        private float[] sinkK;
        private float[] sinkV;
        private float[] relBias;
        private float[] inklingTable;
        private float[] kScale;
        private float[] vScale;

        private Builder(int nHead) {
            if (nHead <= 0) {
                throw new IllegalArgumentException("attention needs at least one head, got " + nHead);
            }
            this.nHead = nHead;
        }

        public Builder scale(float scale) {
            if (!Float.isFinite(scale)) {
                throw new IllegalArgumentException("attention scale must be finite, got " + scale);
            }
            this.scale = scale;
            return this;
        }

        /** Logit softcap; 0 disables. */
        public Builder softcap(float softcap) {
            if (!(softcap >= 0.0f) || Float.isInfinite(softcap)) {
                throw new IllegalArgumentException("softcap must be finite and non-negative, got " + softcap);
            }
            this.softcap = softcap;
            return this;
        }

        public Builder window(KvWindow window) {
            this.window = Objects.requireNonNull(window, "window");
            return this;
        }

        /** ALiBi bias with {@code f_max_alibi_bias}; a non-positive value gives slope 1, as ggml. */
        public Builder alibi(float maxBias) {
            return alibi(maxBias, false);
        }

        /**
         * ALiBi bias with {@code f_max_alibi_bias}; {@code sqrt} selects the negative square-root distance of
         * vLLM's {@code use_alibi_sqrt} instead of the linear distance.
         */
        public Builder alibi(float maxBias, boolean sqrt) {
            this.biasMode = sqrt ? BiasMode.ALIBI_SQRT : BiasMode.ALIBI;
            this.maxBias = maxBias;
            return this;
        }

        /**
         * T5 relative-position bias.
         *
         * @param relBias {@code attn_rel_b} values {@code [nBuckets][nHead]}
         */
        public Builder t5(boolean bidirectional, int nBuckets, int maxDistance, float[] relBias) {
            if (nBuckets < 2 || maxDistance < 2) {
                throw new IllegalArgumentException("T5 bias needs at least 2 buckets and max distance, got " + nBuckets
                        + "/" + maxDistance);
            }
            Objects.requireNonNull(relBias, "T5 bias needs the attn_rel_b values");
            if (relBias.length < (long) nBuckets * nHead) {
                throw new IllegalArgumentException("attn_rel_b holds " + relBias.length + " values, need " + nBuckets
                        + " x " + nHead);
            }
            this.biasMode = bidirectional ? BiasMode.T5_BIDIR : BiasMode.T5_CAUSAL;
            this.nBuckets = nBuckets;
            this.maxDistance = maxDistance;
            this.relBias = relBias;
            return this;
        }

        /**
         * Inkling relative-logit table: per query row and head, one value per distance {@code [0, relExtent)}
         * ({@code attn_rel_proj} applied to that row's {@code attn_rel} output). The table is per-step scratch the
         * caller rewrites before every step, so the parameters keep the array itself rather than a copy.
         *
         * @param relBias the table, {@code qRows * nHead * relExtent} values
         */
        public Builder inkling(int relExtent, float[] relBias) {
            if (relExtent <= 0) {
                throw new IllegalArgumentException("the Inkling relative table needs a positive extent, got "
                        + relExtent);
            }
            Objects.requireNonNull(relBias, "the Inkling bias mode needs the relative-logit table");
            long row = (long) nHead * relExtent;
            if (relBias.length < row || relBias.length % row != 0) {
                throw new IllegalArgumentException("the relative-logit table holds " + relBias.length
                        + " values, not a positive multiple of n_head x rel_extent = " + row);
            }
            this.biasMode = BiasMode.INKLING;
            this.relExtent = relExtent;
            this.inklingTable = relBias;
            return this;
        }

        /** Per-head sink logits. */
        public Builder sinks(float[] sinks) {
            Objects.requireNonNull(sinks, "attention sinks are enabled but no sink values were given");
            if (sinks.length < nHead) {
                throw new IllegalArgumentException("attn_sinks holds " + sinks.length + " values, need " + nHead);
            }
            this.sinks = sinks;
            return this;
        }

        /**
         * Learned static sink tokens: {@code sinkK} holds {@code [n_sink][n_head_kv][d_k]} and {@code sinkV}
         * {@code [n_sink][n_head_kv][d_v]} row-major, already normalized and without RoPE.
         *
         * @throws IllegalArgumentException when a count or length is inconsistent
         */
        public Builder sinkKv(float[] sinkK, float[] sinkV, int sinkTokens) {
            if (sinkTokens < 0) {
                throw new IllegalArgumentException("sink token count must be >= 0, got " + sinkTokens);
            }
            if (sinkTokens == 0) {
                this.sinkTokens = 0;
                this.sinkK = null;
                this.sinkV = null;
                return this;
            }
            Objects.requireNonNull(sinkK, "static sink tokens need their key rows");
            Objects.requireNonNull(sinkV, "static sink tokens need their value rows");
            this.sinkTokens = sinkTokens;
            this.sinkK = sinkK;
            this.sinkV = sinkV;
            return this;
        }

        /**
         * Per-KV-head FP8 cache scales (both null for an unscaled cache; one array may be null when only the other
         * cache is FP8). A scale array holds one value per KV head, finite and positive.
         *
         * @throws IllegalArgumentException when a non-null array holds a non-finite or non-positive value
         */
        public Builder kvScales(float[] kScale, float[] vScale) {
            checkScale("k_scale", kScale);
            checkScale("v_scale", vScale);
            this.kScale = kScale;
            this.vScale = vScale;
            return this;
        }

        private static void checkScale(String what, float[] scale) {
            if (scale == null) {
                return;
            }
            for (float v : scale) {
                if (!Float.isFinite(v) || v <= 0f) {
                    throw new IllegalArgumentException(what + " holds " + v + "; KV-cache scales must be finite"
                            + " and positive");
                }
            }
        }

        public Params build() {
            return new Params(this);
        }
    }

    /**
     * K/V held in caller arrays: K value {@code d} of row {@code r}, head {@code h} at
     * {@code k[kOff + r * kRowStride + h * kHeadStride + d]} (V likewise); the position of row {@code r} is
     * {@code pos[posOff + r]}. A view: the arrays are not copied. An MLA latent pair passes the same array for k
     * and v.
     */
    public static final class DenseKv {
        private final float[] k;
        private final int kOff;
        private final int kRowStride;
        private final int kHeadStride;
        private final float[] v;
        private final int vOff;
        private final int vRowStride;
        private final int vHeadStride;
        private final int nHeadKv;
        private final int dK;
        private final int dV;
        private final int[] pos;
        private final int posOff;

        private DenseKv(float[] k, int kOff, int kRowStride, int kHeadStride, float[] v, int vOff, int vRowStride,
                        int vHeadStride, int nHeadKv, int dK, int dV, int[] pos, int posOff) {
            this.k = k;
            this.kOff = kOff;
            this.kRowStride = kRowStride;
            this.kHeadStride = kHeadStride;
            this.v = v;
            this.vOff = vOff;
            this.vRowStride = vRowStride;
            this.vHeadStride = vHeadStride;
            this.nHeadKv = nHeadKv;
            this.dK = dK;
            this.dV = dV;
            this.pos = pos;
            this.posOff = posOff;
        }

        /** @throws IllegalArgumentException for non-positive dims or negative offsets and strides */
        public static DenseKv of(float[] k, int kOff, int kRowStride, int kHeadStride,
                                 float[] v, int vOff, int vRowStride, int vHeadStride,
                                 int nHeadKv, int dK, int dV, int[] pos, int posOff) {
            Objects.requireNonNull(k, "k");
            Objects.requireNonNull(v, "v");
            Objects.requireNonNull(pos, "pos");
            if (nHeadKv <= 0 || dK <= 0 || dV <= 0) {
                throw new IllegalArgumentException("dense KV needs positive heads and dims, got " + nHeadKv + "/" + dK
                        + "/" + dV);
            }
            if (kOff < 0 || vOff < 0 || posOff < 0 || kRowStride < 0 || vRowStride < 0 || kHeadStride < 0
                    || vHeadStride < 0) {
                throw new IllegalArgumentException("dense KV offsets and strides must be non-negative");
            }
            return new DenseKv(k, kOff, kRowStride, kHeadStride, v, vOff, vRowStride, vHeadStride, nHeadKv, dK, dV,
                    pos, posOff);
        }
    }

    /** Reusable work arrays of one thread, sized for one key block of one head chunk; grows on demand. Not thread-safe. */
    public static final class Scratch {
        float[] scores = new float[0];
        float[] v = new float[0];
        float[] acc = new float[0];
        float[] max = new float[0];
        float[] sum = new float[0];
        /** The query heads of the unit as lanes ({@link CpuAttentionScores#prepare}). */
        float[] qT = new float[0];
        /** One key group's K rows, and their dots with every lane. */
        float[] keys = new float[0];
        float[] dots = new float[0];
        /** Per key of a group: its index in the block, position and visible query rows (bit per row). */
        final int[] keyAt = new int[CpuAttentionScores.KEY_GROUP];
        final int[] keyPos = new int[CpuAttentionScores.KEY_GROUP];
        final int[] keyVis = new int[CpuAttentionScores.KEY_GROUP];
        /** Per query row of the unit: its position and its mask row. */
        final int[] qPos = new int[QUERY_BLOCK];
        final int[] qRow = new int[QUERY_BLOCK];
        int rowsInUnit;
        int headsInUnit;
        int lanes;
        /** The attention kernels of the unit ({@link CpuAttentionScores#prepare}). */
        CpuAttnKernels kernels;

        void ensure(int dK, int dV) {
            ensure(dK, dV, HEAD_CHUNK);
        }

        /** Room for the online-softmax state of {@code heads} (query row, head) pairs. */
        void ensure(int dK, int dV, int heads) {
            if (scores.length < heads * KEY_BLOCK) {
                scores = new float[heads * KEY_BLOCK];
                max = new float[heads];
                sum = new float[heads];
            }
            if (v.length < dV) {
                v = new float[dV];
            }
            if (acc.length < heads * dV) {
                acc = new float[heads * dV];
            }
        }
    }

    // ------------------------------------------------------------------ public entry points

    /**
     * The tree-attention mask of one verify step ({@code gguf.engine.ForwardBatch#treeMask()} carried through the
     * forward): the last {@code fresh} key rows of a call's key range are the step's fresh rows (a tree verify's N
     * nodes at the cache slots after the committed prefix), and bit {@code j} of {@code mask[t]} says whether query
     * row {@code t} sees fresh row {@code j} — its ancestors and itself, siblings excluded. Within the fresh rows
     * the mask replaces the position visibility rules entirely (siblings share one depth position, which the
     * ordinary rules would wrongly make mutually visible); the committed prefix and the sink rows stay visible by
     * the ordinary rules. At most 64 fresh rows: one {@code long} per query row.
     */
    public static final class TreeMask {

        private final long[] mask; // per query row of the attention call
        private final int fresh;   // fresh key rows at the end of the key range the mask covers

        /**
         * @param mask  one entry per query row of the call
         * @param fresh the fresh key rows at the end of the key range, between one and 64
         * @throws IllegalArgumentException when {@code fresh} leaves {@code [1, 64]} or no mask row matches it
         */
        public TreeMask(long[] mask, int fresh) {
            Objects.requireNonNull(mask, "mask");
            if (fresh < 1 || fresh > 64) {
                throw new IllegalArgumentException("a tree mask covers between one and 64 fresh key rows, got "
                        + fresh);
            }
            if (mask.length < fresh) {
                throw new IllegalArgumentException("a tree mask of " + fresh + " fresh rows needs at least one"
                        + " mask entry per query row, got " + mask.length);
            }
            this.mask = mask.clone();
            this.fresh = fresh;
        }

        /** One ancestor-or-self bitmask per query row of the call. */
        public long[] mask() {
            return mask.clone();
        }

        /** The fresh key rows at the end of the key range the mask covers. */
        public int freshRows() {
            return fresh;
        }
    }

    /**
     * Attention of one query row against cache rows {@code [rowStart, rowEnd)} of {@code kv}, on the calling thread.
     *
     * @param rowPos position stored in a cache row, -1 for an empty row (for example
     *               {@code r -> memory.positionOfRow(kv.group(), r)})
     */
    public static void attend(Params p, float[] q, int qOff, int qPos, KvAccessor kv, IntUnaryOperator rowPos,
                              int rowStart, int rowEnd, float[] out, int outOff, Scratch scratch) {
        attendOne(p, q, qOff, qPos, accessorRows(kv, rowPos), rowStart, rowEnd, out, outOff, scratch, null, 0);
    }

    /** Attention of one query row against rows {@code [rowStart, rowEnd)} of a dense K/V pair, on the calling thread. */
    public static void attend(Params p, float[] q, int qOff, int qPos, DenseKv kv, int rowStart, int rowEnd,
                              float[] out, int outOff, Scratch scratch) {
        attendOne(p, q, qOff, qPos, denseRows(kv), rowStart, rowEnd, out, outOff, scratch, null, 0);
    }

    /**
     * Attention of {@code nTokens} query rows (row {@code t} at {@code qOff + t * qRowStride}, position
     * {@code qPos[qPosOff + t]}, output at {@code outOff + t * outRowStride}) against cache rows
     * {@code [rowStart, rowEnd)}, parallel over (token, KV head, head chunk) units and, when those are fewer than the
     * worker threads, over key splits (see the class contract), so a single decode row also spreads over the worker
     * pool. Results equal {@link #attend} row by row; with key splits up to the rounding of the combine.
     */
    public static void attendRows(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos, int qPosOff,
                                  KvAccessor kv, IntUnaryOperator rowPos, int rowStart, int rowEnd,
                                  float[] out, int outOff, int outRowStride) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, accessorRows(kv, rowPos), rowStart, rowEnd,
                out, outOff, outRowStride);
    }

    /** Multi-row form of {@link #attend(Params, float[], int, int, DenseKv, int, int, float[], int, Scratch)}. */
    public static void attendRows(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos, int qPosOff,
                                  DenseKv kv, int rowStart, int rowEnd, float[] out, int outOff, int outRowStride) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, denseRows(kv), rowStart, rowEnd,
                out, outOff, outRowStride);
    }

    /**
     * Attention of {@code nTokens} query rows (row {@code t} at {@code qOff + t * qRowStride}, position
     * {@code qPos[qPosOff + t]}, output at {@code outOff + t * outRowStride}) against the cache slots
     * {@code [slotStart, slotEnd)} of one paged sequence: slot {@code s} is the physical row
     * {@code PagedStep.cellIdx} convention maps it to, {@code pageTable[s >> pageShift] << pageShift |
     * (s & (pageCount - 1))} of {@code common/attn_common.h} nf_kv_row; a page id below 0 masks the slot's keys.
     * Positions come from {@code rowPos} over those physical rows, as the kernels read
     * {@code kv_pos[nf_kv_row(...)]}.
     */
    public static void attendRows(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos, int qPosOff,
                                  KvAccessor kv, IntUnaryOperator rowPos, int pageShift, int[] pageTable,
                                  int slotStart, int slotEnd, float[] out, int outOff, int outRowStride) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff,
                pagedRows(kv, rowPos, pageShift, pageTable), slotStart, slotEnd, out, outOff, outRowStride);
    }

    /**
     * {@link #attendRows(Params, int, float[], int, int, int[], int, KvAccessor, IntUnaryOperator, int, int,
     * float[], int, int)} under a verify step's tree mask: the last {@code tree.freshRows()} rows of the key range
     * are the step's fresh rows and each query row sees exactly the fresh rows its bitmask names (see
     * {@link TreeMask}).
     */
    public static void attendRows(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos, int qPosOff,
                                  KvAccessor kv, IntUnaryOperator rowPos, int rowStart, int rowEnd,
                                  float[] out, int outOff, int outRowStride, TreeMask tree) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, accessorRows(kv, rowPos), rowStart, rowEnd,
                out, outOff, outRowStride, tree);
    }

    /** Dense form of {@link #attendRows(Params, int, float[], int, int, int[], int, KvAccessor, IntUnaryOperator,
     * int, int, float[], int, int, TreeMask)}. */
    public static void attendRows(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos, int qPosOff,
                                  DenseKv kv, int rowStart, int rowEnd, float[] out, int outOff, int outRowStride,
                                  TreeMask tree) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, denseRows(kv), rowStart, rowEnd,
                out, outOff, outRowStride, tree);
    }

    /** Paged form of {@link #attendRows(Params, int, float[], int, int, int[], int, KvAccessor, IntUnaryOperator,
     * int, int, float[], int, int, TreeMask)}: the fresh rows are the last slots of the span. */
    public static void attendRows(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos, int qPosOff,
                                  KvAccessor kv, IntUnaryOperator rowPos, int pageShift, int[] pageTable,
                                  int slotStart, int slotEnd, float[] out, int outOff, int outRowStride,
                                  TreeMask tree) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff,
                pagedRows(kv, rowPos, pageShift, pageTable), slotStart, slotEnd, out, outOff, outRowStride, tree);
    }

    /**
     * {@link #attendRows(Params, int, float[], int, int, int[], int, KvAccessor, IntUnaryOperator, int, int, float[],
     * int, int)} that also reports each row's log-sum-exp: {@code lse[lseOff + t * lseStride + h]} becomes
     * {@code log(sum_j e^(s_j))}, or {@code -inf} when the row has no visible key and no sink. The outputs are the
     * same normalized rows, so a caller can merge stages with
     * {@code out = sum_s out_s * e^(lse_s - m) / sum_s e^(lse_s - m)}. Dual-chunk attention merges its intra, succ
     * and inter stages that way.
     */
    public static void attendRowsLse(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos,
                                     int qPosOff, KvAccessor kv, IntUnaryOperator rowPos, int rowStart, int rowEnd,
                                     float[] out, int outOff, int outRowStride, float[] lse, int lseOff,
                                     int lseStride) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, accessorRows(kv, rowPos), rowStart, rowEnd,
                out, outOff, outRowStride, lse, lseOff, lseStride);
    }

    /**
     * Dense form of {@link #attendRowsLse(Params, int, float[], int, int, int[], int, KvAccessor, IntUnaryOperator,
     * int, int, float[], int, int, float[], int, int)} over rows {@code [rowStart, rowEnd)}.
     */
    public static void attendRowsLse(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos,
                                     int qPosOff, DenseKv kv, int rowStart, int rowEnd, float[] out, int outOff,
                                     int outRowStride, float[] lse, int lseOff, int lseStride) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, denseRows(kv), rowStart, rowEnd,
                out, outOff, outRowStride, lse, lseOff, lseStride);
    }

    /**
     * Paged form of {@link #attendRowsLse(Params, int, float[], int, int, int[], int, KvAccessor, IntUnaryOperator,
     * int, int, float[], int, int, float[], int, int)} over slots {@code [slotStart, slotEnd)}.
     */
    public static void attendRowsLse(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos,
                                     int qPosOff, KvAccessor kv, IntUnaryOperator rowPos, int pageShift,
                                     int[] pageTable, int slotStart, int slotEnd, float[] out, int outOff,
                                     int outRowStride, float[] lse, int lseOff, int lseStride) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff,
                pagedRows(kv, rowPos, pageShift, pageTable), slotStart, slotEnd, out, outOff, outRowStride,
                lse, lseOff, lseStride);
    }

    /**
     * {@link #attendRows(Params, int, float[], int, int, int[], int, DenseKv, int, int, float[], int, int)} with an
     * explicit key-split count instead of the one derived from the worker pool; {@code splits} is capped at one
     * split per key block.
     *
     * @throws IllegalArgumentException when {@code splits < 1}
     */
    static void attendRowsSplit(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos, int qPosOff,
                                DenseKv kv, int rowStart, int rowEnd, float[] out, int outOff, int outRowStride,
                                int splits) {
        if (splits < 1) {
            throw new IllegalArgumentException("key splits must be positive, got " + splits);
        }
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, denseRows(kv), rowStart, rowEnd, out, outOff,
                outRowStride, splits, null, 0, 0);
    }

    // ------------------------------------------------------------------ shared math (mirrors the kernels)

    /** {@code nf_attn_score}: capped and scaled logit. */
    public static float score(float dot, float scale, float softcap) {
        if (softcap > 0.0f) {
            return softcap * (float) Math.tanh(dot * (scale / softcap));
        }
        return dot * scale;
    }

    /**
     * {@code nf_alibi_slope}: slope of head {@code h} (ggml-cpu/ops.cpp:9002-9005, 9024); 1 when {@code maxBias <= 0}.
     */
    public static float alibiSlope(int h, int nHead, float maxBias) {
        if (maxBias <= 0.0f) {
            return 1.0f;
        }
        int nHeadLog2 = Integer.highestOneBit(nHead);
        float m0 = (float) Math.pow(2.0, -maxBias / (float) nHeadLog2);
        float m1 = (float) Math.pow(2.0, -(maxBias / 2.0f) / (float) nHeadLog2);
        return h < nHeadLog2 ? (float) Math.pow(m0, h + 1) : (float) Math.pow(m1, 2 * (h - nHeadLog2) + 1);
    }

    /**
     * {@code nf_t5_bucket}: llama_relative_position_bucket(x, y, nBuckets, bidirectional)
     * (llama-graph.cpp:4316-4343) in single precision, as the kernels; x is the key position, y the query position.
     */
    public static int t5Bucket(int x, int y, int nBuckets, int maxDistance, boolean bidirectional) {
        int buckets = bidirectional ? nBuckets >> 1 : nBuckets;
        int maxExact = buckets >> 1;
        int rel = x - y;
        int bucket = 0;
        if (bidirectional) {
            bucket += rel > 0 ? buckets : 0;
            rel = Math.abs(rel);
        } else {
            rel = rel < 0 ? -rel : 0;
        }
        if (rel < maxExact) {
            return bucket + rel;
        }
        float scaled = (float) Math.log((float) rel / (float) maxExact) * (float) (buckets - maxExact)
                / (float) Math.log((float) maxDistance / (float) maxExact);
        int large = (int) Math.floor((float) maxExact + scaled);
        return bucket + Math.min(large, buckets - 1);
    }

    /** {@code nf_attn_rescale}: {@code e^(x - m)}, exactly 0 for a masked score. */
    static float rescale(float x, float m) {
        return x == Float.NEGATIVE_INFINITY ? 0.0f : (float) Math.exp(x - m);
    }

    // ------------------------------------------------------------------ implementation

    /** Key rows of one attention call. */
    interface Rows {
        int nHeadKv();

        int dK();

        int dV();

        int pos(int row);

        /** Reads K head {@code head} of {@code row} into {@code dst[off, off + dK())}. */
        void readK(int row, int head, float[] dst, int off);

        void readV(int row, int head, float[] dst);

        /** True for a learned static sink row ({@link #pos(int)} is its sink index). */
        default boolean sink(int row) {
            return false;
        }
    }

    /**
     * The key rows of {@code p}'s static sinks followed by {@code base}'s rows (vLLM
     * {@code StaticSinkAttention} prepends the sink tokens to the KV cache): rows {@code [0, n)} are the sinks,
     * row {@code n + r} is {@code base}'s row {@code r}. The sink keys are already normalized and un-rotated.
     *
     * @throws IllegalArgumentException when the sink arrays do not hold {@code n * n_head_kv} rows
     */
    private static Rows sinkRows(final Rows base, final Params p) {
        final int n = p.sinkTokens();
        final int heads = base.nHeadKv();
        final int dK = base.dK();
        final int dV = base.dV();
        if (p.sinkK.length < (long) n * heads * dK || p.sinkV.length < (long) n * heads * dV) {
            throw new IllegalArgumentException("static sink rows hold " + p.sinkK.length + " key and "
                    + p.sinkV.length + " value values, need " + n + " x " + heads + " x (" + dK + ", " + dV + ")");
        }
        return new Rows() {
            @Override public int nHeadKv() { return heads; }
            @Override public int dK() { return dK; }
            @Override public int dV() { return dV; }
            @Override public boolean sink(int row) { return row < n; }
            @Override public int pos(int row) { return row < n ? row : base.pos(row - n); }

            @Override
            public void readK(int row, int head, float[] dst, int off) {
                if (row < n) {
                    System.arraycopy(p.sinkK, (row * heads + head) * dK, dst, off, dK);
                } else {
                    base.readK(row - n, head, dst, off);
                }
            }

            @Override
            public void readV(int row, int head, float[] dst) {
                if (row < n) {
                    System.arraycopy(p.sinkV, (row * heads + head) * dV, dst, 0, dV);
                } else {
                    base.readV(row - n, head, dst);
                }
            }
        };
    }

    /**
     * {@code rows} with the static sink rows of {@code p} prepended, and the key range they add; a caller whose key
     * range does not begin at row 0 cannot place the sinks, so that is refused.
     */
    private static Rows prependSinks(Params p, Rows rows, int rowStart) {
        if (p.sinkTokens() == 0) {
            return rows;
        }
        if (rowStart != 0) {
            throw new UnsupportedOperationException("static sink attention starts its key range at row 0, not "
                    + rowStart);
        }
        return sinkRows(rows, p);
    }

    private static Rows accessorRows(final KvAccessor kv, final IntUnaryOperator rowPos) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(rowPos, "rowPos");
        return new Rows() {
            @Override public int nHeadKv() { return kv.nHeadKv(); }
            @Override public int dK() { return kv.headDimK(); }
            @Override public int dV() { return kv.headDimV(); }
            @Override public int pos(int row) { return rowPos.applyAsInt(row); }
            @Override public void readK(int row, int head, float[] dst, int off) { kv.readK(row, head, dst, off); }
            @Override public void readV(int row, int head, float[] dst) { kv.readV(row, head, dst, 0); }
        };
    }

    private static Rows denseRows(final DenseKv kv) {
        Objects.requireNonNull(kv, "kv");
        return new Rows() {
            @Override public int nHeadKv() { return kv.nHeadKv; }
            @Override public int dK() { return kv.dK; }
            @Override public int dV() { return kv.dV; }
            @Override public int pos(int row) { return kv.pos[kv.posOff + row]; }

            @Override
            public void readK(int row, int head, float[] dst, int off) {
                System.arraycopy(kv.k, kv.kOff + row * kv.kRowStride + head * kv.kHeadStride, dst, off, kv.dK);
            }

            @Override
            public void readV(int row, int head, float[] dst) {
                System.arraycopy(kv.v, kv.vOff + row * kv.vRowStride + head * kv.vHeadStride, dst, 0, kv.dV);
            }
        };
    }

    /**
     * Key rows behind a page table: the row of slot {@code s} is {@code (pageTable[s >> pageShift] << pageShift) |
     * (s & ((1 << pageShift) - 1))} ({@code nf_kv_row} of {@code kernels/llm/common/attn_common.h}); the kernels
     * mask the keys of a page id below 0 and the host mirror reports those rows' positions as
     * -1, so this source reports the mirror's position directly and the window masks the row without reading it.
     */
    private static Rows pagedRows(final KvAccessor kv, final IntUnaryOperator rowPos, final int pageShift,
                                  final int[] pageTable) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(pageTable, "pageTable");
        final int pageMask = (1 << pageShift) - 1;
        return new Rows() {
            @Override public int nHeadKv() { return kv.nHeadKv(); }
            @Override public int dK() { return kv.headDimK(); }
            @Override public int dV() { return kv.headDimV(); }
            @Override public int pos(int slot) { return rowPos.applyAsInt(rowOf(slot)); }
            @Override
            public void readK(int slot, int head, float[] dst, int off) {
                kv.readK(rowOf(slot), head, dst, off);
            }
            @Override public void readV(int slot, int head, float[] dst) { kv.readV(rowOf(slot), head, dst, 0); }

            private int rowOf(int slot) {
                int page = pageTable[slot >> pageShift];
                return (page << pageShift) | (slot & pageMask);
            }
        };
    }

    /** Checks the head mapping and the key range of a call. */
    private static void checkShape(Params p, Rows rows, int rowStart, int rowEnd) {
        Objects.requireNonNull(p, "params");
        if (p.nHead % rows.nHeadKv() != 0) {
            throw new IllegalArgumentException("n_head " + p.nHead + " is not a multiple of n_head_kv " + rows.nHeadKv());
        }
        if (rowStart < 0 || rowEnd < rowStart) {
            throw new IllegalArgumentException("invalid key row range [" + rowStart + ", " + rowEnd + ")");
        }
    }

    /** Checks that the query rows {@code qFirst, qLast} and output rows {@code outFirst, outLast} lie in the arrays. */
    private static void checkRows(Params p, Rows rows, float[] q, long qFirst, long qLast, float[] out, long outFirst,
                                  long outLast) {
        Objects.requireNonNull(q, "q");
        Objects.requireNonNull(out, "out");
        long qLen = (long) p.nHead * rows.dK();
        long outLen = (long) p.nHead * rows.dV();
        if (qFirst < 0 || qLast + qLen > q.length || outFirst < 0 || outLast + outLen > out.length) {
            throw new IllegalArgumentException("query or output rows out of bounds: q[" + qFirst + ".." + qLast + " + "
                    + qLen + "] of " + q.length + ", out[" + outFirst + ".." + outLast + " + " + outLen + "] of "
                    + out.length);
        }
    }

    private static void attendOne(Params p, float[] q, int qOff, int qPos, Rows rows, int rowStart, int rowEnd,
                                  float[] out, int outOff, Scratch s, float[] lse, int lseOff) {
        checkShape(p, rows, rowStart, rowEnd);
        rows = prependSinks(p, rows, rowStart);
        rowEnd += p.sinkTokens();
        checkRows(p, rows, q, qOff, qOff, out, outOff, outOff);
        Objects.requireNonNull(s, "scratch");
        final int g0 = p.nHead / rows.nHeadKv();
        for (int kvh = 0; kvh < rows.nHeadKv(); kvh++) {
            for (int g = 0; g < g0; g += HEAD_CHUNK) {
                attendChunk(p, q, qOff, qPos, 0, rows, rowStart, rowEnd, out, outOff, s, kvh * g0 + g,
                        Math.min(HEAD_CHUNK, g0 - g), kvh, lse, lseOff, null, Integer.MAX_VALUE);
            }
        }
    }

    private static void attendMany(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos,
                                   int qPosOff, Rows rows, int rowStart, int rowEnd, float[] out, int outOff,
                                   int outRowStride) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, rows, rowStart, rowEnd, out, outOff, outRowStride,
                0, null, 0, 0);
    }

    private static void attendMany(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos,
                                   int qPosOff, Rows rows, int rowStart, int rowEnd, float[] out, int outOff,
                                   int outRowStride, float[] lse, int lseOff, int lseStride) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, rows, rowStart, rowEnd, out, outOff, outRowStride,
                0, lse, lseOff, lseStride);
    }

    /** {@link #attendMany(Params, int, float[], int, int, int[], int, Rows, int, int, float[], int, int, int,
     * float[], int, int)} under a verify step's tree mask. */
    private static void attendMany(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos,
                                   int qPosOff, Rows rows, int rowStart, int rowEnd, float[] out, int outOff,
                                   int outRowStride, TreeMask tree) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, rows, rowStart, rowEnd, out, outOff, outRowStride,
                0, null, 0, 0, tree);
    }

    /**
     * Key splits of a call: 1 unless the units leave workers idle and the range holds at least
     * {@link #SPLIT_MIN_KEYS} keys; then enough splits to occupy the pool, at most one per key block.
     */
    static int splitCount(long units, int keys, int workers) {
        if (keys < SPLIT_MIN_KEYS || units >= workers) {
            return 1;
        }
        long wanted = (workers + units - 1) / units;
        return (int) Math.min(wanted, (keys + KEY_BLOCK - 1) / KEY_BLOCK);
    }

    /** {@code forcedSplits} 0 derives the key splits from the worker pool ({@link #splitCount}). */
    private static void attendMany(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos,
                                   int qPosOff, Rows rows, int rowStart, int rowEnd, float[] out, int outOff,
                                   int outRowStride, int forcedSplits, float[] lse, int lseOff, int lseStride) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, rows, rowStart, rowEnd, out, outOff, outRowStride,
                forcedSplits, lse, lseOff, lseStride, null);
    }

    /** {@link #attendMany(Params, int, float[], int, int, int[], int, Rows, int, int, float[], int, int, int,
     * float[], int, int)} with the verify step's tree mask; {@code tree} null runs the ordinary rules. */
    private static void attendMany(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos,
                                   int qPosOff, Rows rows, int rowStart, int rowEnd, float[] out, int outOff,
                                   int outRowStride, int forcedSplits, float[] lse, int lseOff, int lseStride,
                                   TreeMask tree) {
        Objects.requireNonNull(qPos, "qPos");
        checkShape(p, rows, rowStart, rowEnd);
        final Rows view = prependSinks(p, rows, rowStart);
        final int end = rowEnd + p.sinkTokens();
        if (nTokens < 0 || qRowStride < 0 || outRowStride < 0 || qPosOff < 0 || qPosOff + (long) nTokens > qPos.length) {
            throw new IllegalArgumentException("invalid query rows: n=" + nTokens + ", strides " + qRowStride + "/"
                    + outRowStride + ", positions " + qPosOff + ".." + (qPosOff + (long) nTokens) + " of " + qPos.length);
        }
        if (nTokens == 0) {
            return;
        }
        final long[] treeBits = tree == null ? null : tree.mask();
        final int freshStart = tree == null ? Integer.MAX_VALUE : end - tree.freshRows();
        if (tree != null && freshStart < rowStart) {
            throw new IllegalArgumentException("a tree mask covers " + tree.freshRows() + " fresh key rows, the"
                    + " key range holds " + (end - rowStart));
        }
        long last = nTokens - 1L;
        checkRows(p, view, q, qOff, qOff + last * qRowStride, out, outOff, outOff + last * outRowStride);
        final int nHeadKv = view.nHeadKv();
        final int g0 = p.nHead / nHeadKv;
        final int chunks = (g0 + HEAD_CHUNK - 1) / HEAD_CHUNK;
        final int unitsPerToken = nHeadKv * chunks;
        long units = (long) nTokens * unitsPerToken;
        if (units > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(nTokens + " query rows x " + unitsPerToken + " head chunks exceed "
                    + Integer.MAX_VALUE + " work units");
        }
        int keys = end - rowStart;
        /* Batch invariance sizes the key splits from one query row's units, so another row in the call cannot
         * change how a row's keys are reduced (a single-row call uses exactly this count). */
        long splitUnits = BatchInvariance.enabled() ? unitsPerToken : units;
        int splits = forcedSplits > 0 ? Math.min(forcedSplits, Math.max(1, (keys + KEY_BLOCK - 1) / KEY_BLOCK))
                : splitCount(splitUnits, keys, CpuParallel.parallelism());
        if (splits > 1) {
            attendSplit(p, (int) units, unitsPerToken, chunks, g0, splits, q, qOff, qRowStride, qPos, qPosOff, view,
                    rowStart, end, out, outOff, outRowStride, lse, lseOff, lseStride, treeBits, freshStart);
            return;
        }
        final int qb = queryBlock(units, CpuParallel.parallelism());
        final int blocks = (nTokens + qb - 1) / qb;
        CpuParallel.parallelFor(blocks * unitsPerToken, 1, (from, to) -> {
            Scratch scratch = new Scratch();
            for (int u = from; u < to; u++) {
                final int tb = u / unitsPerToken;
                final int rest = u - tb * unitsPerToken;
                final int kvh = rest / chunks;
                final int g = (rest - kvh * chunks) * HEAD_CHUNK;
                final int t = tb * qb;
                final int gc = Math.min(HEAD_CHUNK, g0 - g);
                final int lo = lse == null ? 0 : lseOff + t * lseStride;
                if (Math.min(qb, nTokens - t) == 1) {
                    attendChunk(p, q, qOff + t * qRowStride, qPos[qPosOff + t], qPosOff + t, view, rowStart, end,
                            out, outOff + t * outRowStride, scratch, kvh * g0 + g, gc, kvh, lse, lo, treeBits,
                            freshStart);
                } else {
                    attendBlock(p, Math.min(qb, nTokens - t), q, qOff + t * qRowStride, qRowStride, qPos,
                            qPosOff + t, view, rowStart, end, out, outOff + t * outRowStride, outRowStride, scratch,
                            kvh * g0 + g, gc, kvh, lse, lo, lseStride, treeBits, freshStart);
                }
            }
        });
    }

    /**
     * Query rows one unit of a multi-row call attends together: up to {@link #QUERY_BLOCK}, fewer when the
     * blocks would leave fewer than {@link #UNITS_PER_WORKER} units per worker. Every row's result is the same
     * whatever the block, so the block changes only how often a key or value row is read.
     */
    static int queryBlock(long units, int workers) {
        long fit = units / ((long) UNITS_PER_WORKER * workers);
        return (int) Math.max(1L, Math.min(QUERY_BLOCK, fit));
    }

    /**
     * {@link #attendChunk} of {@code nt} consecutive query rows (row {@code t} at {@code q[qOff + t * qStride]},
     * position {@code qPos[qRow0 + t]}, mask row {@code qRow0 + t}) against heads {@code [h0, h0 + gc)}: each key
     * and value row is read and scaled once for every row of the block, and the state of (row {@code t}, head
     * {@code g}) sits at index {@code t * gc + g}, so each pair runs exactly the scoring, rescale and accumulation
     * steps of its single-row call.
     */
    private static void attendBlock(Params p, int nt, float[] q, int qOff, int qStride, int[] qPos, int qRow0,
                                    Rows rows, int rowStart, int rowEnd, float[] out, int outOff, int outStride,
                                    Scratch s, int h0, int gc, int kvh, float[] lse, int lseOff, int lseStride,
                                    long[] treeBits, int freshStart) {
        final int dV = rows.dV();
        final int heads = nt * gc;
        s.ensure(rows.dK(), dV, heads);
        Arrays.fill(s.max, 0, heads, Float.NEGATIVE_INFINITY);
        Arrays.fill(s.sum, 0, heads, 0.0f);
        Arrays.fill(s.acc, 0, heads * dV, 0.0f);
        CpuAttentionScores.prepare(s, q, qOff, qStride, nt, qPos, qRow0, h0, gc, rows.dK());
        for (int b = rowStart; b < rowEnd; b += KEY_BLOCK) {
            final int nb = Math.min(KEY_BLOCK, rowEnd - b);
            CpuAttentionScores.score(p, rows, b, nb, s, h0, kvh, treeBits, freshStart);
            rescaleBlock(s, nb, heads, dV);
            CpuAttentionScores.addValues(p, rows, b, nb, s, heads, kvh, dV);
        }
        for (int t = 0; t < nt; t++) {
            finish(p, s.max, t * gc, s.sum, t * gc, s.acc, t * gc * dV, dV, h0, gc, out, outOff + t * outStride, lse,
                    lse == null ? 0 : lseOff + t * lseStride);
        }
    }

    /**
     * Query heads {@code [h0, h0 + gc)} (all reading KV head {@code kvh}) of one query row. Keys are processed in
     * blocks of {@link #KEY_BLOCK} with the online softmax of the tiled kernel: per head, {@code m' = max(m, block max)},
     * {@code alpha = e^(m - m')}, {@code p = e^(s - m')}, {@code l = l * alpha + sum p}, {@code o = o * alpha + sum p V};
     * masked keys score -INF and contribute exactly 0. The sink and the normalization follow the last block.
     */
    private static void attendChunk(Params p, float[] q, int qOff, int qPos, int qRow, Rows rows, int rowStart,
                                    int rowEnd, float[] out, int outOff, Scratch s, int h0, int gc, int kvh,
                                    float[] lse, int lseOff, long[] treeBits, int freshStart) {
        accumulate(p, q, qOff, qPos, qRow, rows, rowStart, rowEnd, s, h0, gc, kvh, treeBits, freshStart);
        finish(p, s.max, 0, s.sum, 0, s.acc, 0, rows.dV(), h0, gc, out, outOff, lse, lseOff);
    }

    /**
     * The online-softmax state {@code (m, l, o)} of heads {@code [h0, h0 + gc)} over keys {@code [rowStart, rowEnd)},
     * left unnormalized in {@code s.max}, {@code s.sum} and {@code s.acc}. {@code treeBits} (with its
     * {@code freshStart}) carries the verify step's tree mask; the fresh key rows at or after {@code freshStart}
     * score only for the query rows whose bitmask names them.
     */
    private static void accumulate(Params p, float[] q, int qOff, int qPos, int qRow, Rows rows, int rowStart,
                                   int rowEnd, Scratch s, int h0, int gc, int kvh, long[] treeBits, int freshStart) {
        final int dV = rows.dV();
        s.ensure(rows.dK(), dV);
        Arrays.fill(s.max, 0, gc, Float.NEGATIVE_INFINITY);
        Arrays.fill(s.sum, 0, gc, 0.0f);
        Arrays.fill(s.acc, 0, gc * dV, 0.0f);
        CpuAttentionScores.prepareRow(s, q, qOff, qPos, qRow, h0, gc, rows.dK());
        for (int b = rowStart; b < rowEnd; b += KEY_BLOCK) {
            final int nb = Math.min(KEY_BLOCK, rowEnd - b);
            CpuAttentionScores.score(p, rows, b, nb, s, h0, kvh, treeBits, freshStart);
            rescaleBlock(s, nb, gc, dV);
            CpuAttentionScores.addValues(p, rows, b, nb, s, gc, kvh, dV);
        }
    }

    /**
     * Online-softmax step of one key block per head: {@code m' = max(m, block max)}, scores become
     * {@code p = e^(s - m')}, {@code l = l * e^(m - m') + sum p} and the accumulator is scaled by {@code e^(m - m')}.
     */
    private static void rescaleBlock(Scratch s, int nb, int gc, int dV) {
        final float[] sc = s.scores;
        final float[] acc = s.acc;
        final float[] mx = s.max;
        final float[] sum = s.sum;
        for (int g = 0; g < gc; g++) {
            final int sb = g * KEY_BLOCK;
            float m = mx[g];
            for (int r = 0; r < nb; r++) {
                m = Math.max(m, sc[sb + r]);
            }
            float l = 0.0f;
            for (int r = 0; r < nb; r++) {
                final float e = rescale(sc[sb + r], m);
                sc[sb + r] = e;
                l += e;
            }
            final float alpha = rescale(mx[g], m);
            if (alpha != 1.0f) {
                final int ab = g * dV;
                for (int d = 0; d < dV; d++) {
                    acc[ab + d] *= alpha;
                }
            }
            sum[g] = sum[g] * alpha + l;
            mx[g] = m;
        }
    }

    /**
     * Sink and normalization of heads {@code [h0, h0 + gc)} from their state {@code m = mx[mOff + g]},
     * {@code l = sum[mOff + g]}, {@code o = acc[accOff + g * dV + d]}: with a sink {@code m' = max(m, sink)},
     * {@code out = o e^(m - m') / (l e^(m - m') + e^(sink - m'))}, else {@code out = o / l}; zeros when the
     * denominator is 0.
     */
    private static void finish(Params p, float[] mx, int mOff, float[] sum, int lOff, float[] acc, int accOff, int dV,
                               int h0, int gc, float[] out, int outOff, float[] lse, int lseOff) {
        for (int g = 0; g < gc; g++) {
            final int h = h0 + g;
            final float m = mx[mOff + g];
            final float l = sum[lOff + g];
            float factor = 1.0f;
            float denom = l;
            float top = m;
            if (p.hasSinks()) {
                final float sink = p.sinks[h];
                final float m2 = Math.max(m, sink);
                factor = rescale(m, m2);
                denom = l * factor + (float) Math.exp(sink - m2);
                top = m2;
            }
            final int ob = outOff + h * dV;
            final int ab = accOff + g * dV;
            for (int d = 0; d < dV; d++) {
                out[ob + d] = denom > 0.0f ? acc[ab + d] * factor / denom : 0.0f;
            }
            if (lse != null) {
                lse[lseOff + h] = denom > 0.0f ? top + (float) Math.log(denom) : Float.NEGATIVE_INFINITY;
            }
        }
    }

    /**
     * Split-K form of the multi-row call: task {@code (unit, split)} accumulates the unit's heads over the split's
     * keys (whole {@link #KEY_BLOCK}s, so every split reads the blocks the unsplit pass would), then each unit merges
     * its splits in split order and finishes ({@link #finish}).
     */
    private static void attendSplit(Params p, int units, int unitsPerToken, int chunks, int g0, int splits, float[] q,
                                    int qOff, int qRowStride, int[] qPos, int qPosOff, Rows rows, int rowStart,
                                    int rowEnd, float[] out, int outOff, int outRowStride, float[] lse, int lseOff,
                                    int lseStride, long[] treeBits, int freshStart) {
        final int dV = rows.dV();
        final int keys = rowEnd - rowStart;
        final int blocks = (keys + KEY_BLOCK - 1) / KEY_BLOCK;
        final int splitLen = ((blocks + splits - 1) / splits) * KEY_BLOCK;
        final int nSplit = (keys + splitLen - 1) / splitLen;
        long states = (long) units * nSplit * HEAD_CHUNK;
        if (states * dV > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(units + " units x " + nSplit + " key splits of " + HEAD_CHUNK + " x "
                    + dV + " partial values exceed one array");
        }
        final float[] pm = new float[(int) states];
        final float[] pl = new float[(int) states];
        final float[] po = new float[(int) (states * dV)];
        CpuParallel.parallelFor(units * nSplit, 1, (from, to) -> {
            Scratch scratch = new Scratch();
            for (int i = from; i < to; i++) {
                final int u = i / nSplit;
                final int sp = i - u * nSplit;
                final int t = u / unitsPerToken;
                final int rest = u - t * unitsPerToken;
                final int kvh = rest / chunks;
                final int g = (rest - kvh * chunks) * HEAD_CHUNK;
                final int gc = Math.min(HEAD_CHUNK, g0 - g);
                final int k0 = rowStart + sp * splitLen;
                accumulate(p, q, qOff + t * qRowStride, qPos[qPosOff + t], qPosOff + t, rows, k0,
                        Math.min(rowEnd, k0 + splitLen), scratch, kvh * g0 + g, gc, kvh, treeBits, freshStart);
                final int base = i * HEAD_CHUNK;
                System.arraycopy(scratch.max, 0, pm, base, gc);
                System.arraycopy(scratch.sum, 0, pl, base, gc);
                System.arraycopy(scratch.acc, 0, po, base * dV, gc * dV);
            }
        });
        CpuParallel.parallelFor(units, 1, (from, to) -> {
            final float[] m = new float[HEAD_CHUNK];
            final float[] l = new float[HEAD_CHUNK];
            final float[] o = new float[HEAD_CHUNK * dV];
            for (int u = from; u < to; u++) {
                final int t = u / unitsPerToken;
                final int rest = u - t * unitsPerToken;
                final int kvh = rest / chunks;
                final int g = (rest - kvh * chunks) * HEAD_CHUNK;
                final int gc = Math.min(HEAD_CHUNK, g0 - g);
                combine(pm, pl, po, u * nSplit, nSplit, gc, dV, m, l, o);
                finish(p, m, 0, l, 0, o, 0, dV, kvh * g0 + g, gc, out, outOff + t * outRowStride, lse,
                        lse == null ? 0 : lseOff + t * lseStride);
            }
        });
    }

    /**
     * Merges the states of splits {@code [first, first + nSplit)} into {@code m}, {@code l}, {@code o}:
     * {@code m = max m_i}, {@code l = sum l_i e^(m_i - m)}, {@code o = sum o_i e^(m_i - m)} (flash_combine.cl);
     * a split without a visible key ({@code m_i = -INF}) contributes exactly 0.
     */
    private static void combine(float[] pm, float[] pl, float[] po, int first, int nSplit, int gc, int dV, float[] m,
                                float[] l, float[] o) {
        for (int g = 0; g < gc; g++) {
            float mm = Float.NEGATIVE_INFINITY;
            for (int sp = 0; sp < nSplit; sp++) {
                mm = Math.max(mm, pm[(first + sp) * HEAD_CHUNK + g]);
            }
            float ll = 0.0f;
            final int ob = g * dV;
            Arrays.fill(o, ob, ob + dV, 0.0f);
            for (int sp = 0; sp < nSplit; sp++) {
                final int st = (first + sp) * HEAD_CHUNK + g;
                final float w = rescale(pm[st], mm);
                if (w == 0.0f) {
                    continue;
                }
                ll += pl[st] * w;
                CpuAttentionScores.merge(po, st * dV, w, o, ob, dV);
            }
            m[g] = mm;
            l[g] = ll;
        }
    }
}

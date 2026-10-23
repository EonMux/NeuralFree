package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
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
 */
public final class CpuAttentionCore {

    /** Keys scored per block; bounds the score scratch to {@code HEAD_CHUNK * KEY_BLOCK} floats per thread. */
    static final int KEY_BLOCK = 256;
    /** Query heads of one KV head processed together (each K and V row read once per chunk); {@code NF_FD_MAX_G}. */
    static final int HEAD_CHUNK = LlmKernelConstants.FD_MAX_G;
    /** Fewest keys for which a call splits its key range across threads (ggml-cpu/ops.cpp:9580-9610). */
    static final int SPLIT_MIN_KEYS = 512;

    private CpuAttentionCore() {
    }

    /** Score, bias and mask parameters of one attention layer. Immutable. */
    public static final class Params {
        private final int nHead;
        private final float scale;
        private final float softcap;
        private final float maxBias;
        private final KvWindow window;
        private final BiasMode biasMode;
        private final int nBuckets;
        private final int maxDistance;
        private final float[] sinks;
        private final float[] relBias;
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
            this.sinks = b.sinks == null ? null : b.sinks.clone();
            this.relBias = b.relBias == null ? null : b.relBias.clone();
            this.slopes = new float[nHead];
            for (int h = 0; h < nHead; h++) {
                slopes[h] = biasMode == BiasMode.ALIBI ? alibiSlope(h, nHead, maxBias) : 1.0f;
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
            Objects.requireNonNull(spec, "spec");
            SwaType swa = spec.effectiveSwaType();
            Builder b = builder(spec.nHead())
                    .scale(spec.scoreScale())
                    .softcap(spec.softcap())
                    .window(KvWindow.of(swa, swa == SwaType.NONE ? 0 : spec.nSwa(), spec.causal()));
            if (spec.biasMode() == BiasMode.ALIBI) {
                b.alibi(spec.alibiMaxBias());
            } else if (spec.biasMode() == BiasMode.T5_CAUSAL || spec.biasMode() == BiasMode.T5_BIDIR) {
                b.t5(spec.biasMode() == BiasMode.T5_BIDIR, spec.t5Buckets(), spec.t5MaxDistance(), relBias);
            }
            if (spec.sinks()) {
                b.sinks(sinks);
            }
            return b.build();
        }

        public int nHead() { return nHead; }
        public float scale() { return scale; }
        public float softcap() { return softcap; }
        public float maxBias() { return maxBias; }
        public KvWindow window() { return window; }
        public BiasMode biasMode() { return biasMode; }
        public int nBuckets() { return nBuckets; }
        public int maxDistance() { return maxDistance; }
        public boolean hasSinks() { return sinks != null; }

        /** Sink logit of head {@code h}. @throws IllegalStateException without sinks */
        public float sink(int h) {
            if (sinks == null) {
                throw new IllegalStateException("attention has no sinks");
            }
            return sinks[h];
        }

        /** Position bias of head {@code h} for query position {@code pq} and key position {@code pk}. */
        public float bias(int h, int pq, int pk) {
            switch (biasMode) {
                case ALIBI:
                    return slopes[h] * -(float) Math.abs(pq - pk);
                case T5_CAUSAL:
                case T5_BIDIR:
                    return relBias[t5Bucket(pk, pq, nBuckets, maxDistance, biasMode == BiasMode.T5_BIDIR) * nHead + h];
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
        private float[] sinks;
        private float[] relBias;

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
            this.biasMode = BiasMode.ALIBI;
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

        /** Per-head sink logits. */
        public Builder sinks(float[] sinks) {
            Objects.requireNonNull(sinks, "attention sinks are enabled but no sink values were given");
            if (sinks.length < nHead) {
                throw new IllegalArgumentException("attn_sinks holds " + sinks.length + " values, need " + nHead);
            }
            this.sinks = sinks;
            return this;
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
        private float[] scores = new float[0];
        private float[] k = new float[0];
        private float[] v = new float[0];
        private float[] acc = new float[0];
        private float[] max = new float[0];
        private float[] sum = new float[0];

        void ensure(int dK, int dV) {
            if (scores.length < HEAD_CHUNK * KEY_BLOCK) {
                scores = new float[HEAD_CHUNK * KEY_BLOCK];
                max = new float[HEAD_CHUNK];
                sum = new float[HEAD_CHUNK];
            }
            if (k.length < dK) {
                k = new float[dK];
            }
            if (v.length < dV) {
                v = new float[dV];
            }
            if (acc.length < HEAD_CHUNK * dV) {
                acc = new float[HEAD_CHUNK * dV];
            }
        }
    }

    // ------------------------------------------------------------------ public entry points

    /**
     * Attention of one query row against cache rows {@code [rowStart, rowEnd)} of {@code kv}, on the calling thread.
     *
     * @param rowPos position stored in a cache row, -1 for an empty row (for example
     *               {@code r -> memory.positionOfRow(kv.group(), r)})
     */
    public static void attend(Params p, float[] q, int qOff, int qPos, KvAccessor kv, IntUnaryOperator rowPos,
                              int rowStart, int rowEnd, float[] out, int outOff, Scratch scratch) {
        attendOne(p, q, qOff, qPos, accessorRows(kv, rowPos), rowStart, rowEnd, out, outOff, scratch);
    }

    /** Attention of one query row against rows {@code [rowStart, rowEnd)} of a dense K/V pair, on the calling thread. */
    public static void attend(Params p, float[] q, int qOff, int qPos, DenseKv kv, int rowStart, int rowEnd,
                              float[] out, int outOff, Scratch scratch) {
        attendOne(p, q, qOff, qPos, denseRows(kv), rowStart, rowEnd, out, outOff, scratch);
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
                outRowStride, splits);
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
    private interface Rows {
        int nHeadKv();

        int dK();

        int dV();

        int pos(int row);

        void readK(int row, int head, float[] dst);

        void readV(int row, int head, float[] dst);
    }

    private static Rows accessorRows(final KvAccessor kv, final IntUnaryOperator rowPos) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(rowPos, "rowPos");
        return new Rows() {
            @Override public int nHeadKv() { return kv.nHeadKv(); }
            @Override public int dK() { return kv.headDimK(); }
            @Override public int dV() { return kv.headDimV(); }
            @Override public int pos(int row) { return rowPos.applyAsInt(row); }
            @Override public void readK(int row, int head, float[] dst) { kv.readK(row, head, dst, 0); }
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
            public void readK(int row, int head, float[] dst) {
                System.arraycopy(kv.k, kv.kOff + row * kv.kRowStride + head * kv.kHeadStride, dst, 0, kv.dK);
            }

            @Override
            public void readV(int row, int head, float[] dst) {
                System.arraycopy(kv.v, kv.vOff + row * kv.vRowStride + head * kv.vHeadStride, dst, 0, kv.dV);
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
                                  float[] out, int outOff, Scratch s) {
        checkShape(p, rows, rowStart, rowEnd);
        checkRows(p, rows, q, qOff, qOff, out, outOff, outOff);
        Objects.requireNonNull(s, "scratch");
        final int g0 = p.nHead / rows.nHeadKv();
        for (int kvh = 0; kvh < rows.nHeadKv(); kvh++) {
            for (int g = 0; g < g0; g += HEAD_CHUNK) {
                attendChunk(p, q, qOff, qPos, rows, rowStart, rowEnd, out, outOff, s, kvh * g0 + g,
                        Math.min(HEAD_CHUNK, g0 - g), kvh);
            }
        }
    }

    private static void attendMany(Params p, int nTokens, float[] q, int qOff, int qRowStride, int[] qPos,
                                   int qPosOff, Rows rows, int rowStart, int rowEnd, float[] out, int outOff,
                                   int outRowStride) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, rows, rowStart, rowEnd, out, outOff, outRowStride,
                0);
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
                                   int outRowStride, int forcedSplits) {
        Objects.requireNonNull(qPos, "qPos");
        checkShape(p, rows, rowStart, rowEnd);
        if (nTokens < 0 || qRowStride < 0 || outRowStride < 0 || qPosOff < 0 || qPosOff + (long) nTokens > qPos.length) {
            throw new IllegalArgumentException("invalid query rows: n=" + nTokens + ", strides " + qRowStride + "/"
                    + outRowStride + ", positions " + qPosOff + ".." + (qPosOff + (long) nTokens) + " of " + qPos.length);
        }
        if (nTokens == 0) {
            return;
        }
        long last = nTokens - 1L;
        checkRows(p, rows, q, qOff, qOff + last * qRowStride, out, outOff, outOff + last * outRowStride);
        final int nHeadKv = rows.nHeadKv();
        final int g0 = p.nHead / nHeadKv;
        final int chunks = (g0 + HEAD_CHUNK - 1) / HEAD_CHUNK;
        final int unitsPerToken = nHeadKv * chunks;
        long units = (long) nTokens * unitsPerToken;
        if (units > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(nTokens + " query rows x " + unitsPerToken + " head chunks exceed "
                    + Integer.MAX_VALUE + " work units");
        }
        int keys = rowEnd - rowStart;
        int splits = forcedSplits > 0 ? Math.min(forcedSplits, Math.max(1, (keys + KEY_BLOCK - 1) / KEY_BLOCK))
                : splitCount(units, keys, CpuParallel.parallelism());
        if (splits > 1) {
            attendSplit(p, (int) units, unitsPerToken, chunks, g0, splits, q, qOff, qRowStride, qPos, qPosOff, rows,
                    rowStart, rowEnd, out, outOff, outRowStride);
            return;
        }
        CpuParallel.parallelFor((int) units, 1, (from, to) -> {
            Scratch scratch = new Scratch();
            for (int u = from; u < to; u++) {
                final int t = u / unitsPerToken;
                final int rest = u - t * unitsPerToken;
                final int kvh = rest / chunks;
                final int g = (rest - kvh * chunks) * HEAD_CHUNK;
                attendChunk(p, q, qOff + t * qRowStride, qPos[qPosOff + t], rows, rowStart, rowEnd,
                        out, outOff + t * outRowStride, scratch, kvh * g0 + g, Math.min(HEAD_CHUNK, g0 - g), kvh);
            }
        });
    }

    /**
     * Query heads {@code [h0, h0 + gc)} (all reading KV head {@code kvh}) of one query row. Keys are processed in
     * blocks of {@link #KEY_BLOCK} with the online softmax of the tiled kernel: per head, {@code m' = max(m, block max)},
     * {@code alpha = e^(m - m')}, {@code p = e^(s - m')}, {@code l = l * alpha + sum p}, {@code o = o * alpha + sum p V};
     * masked keys score -INF and contribute exactly 0. The sink and the normalization follow the last block.
     */
    private static void attendChunk(Params p, float[] q, int qOff, int qPos, Rows rows, int rowStart, int rowEnd,
                                    float[] out, int outOff, Scratch s, int h0, int gc, int kvh) {
        accumulate(p, q, qOff, qPos, rows, rowStart, rowEnd, s, h0, gc, kvh);
        finish(p, s.max, 0, s.sum, 0, s.acc, 0, rows.dV(), h0, gc, out, outOff);
    }

    /**
     * The online-softmax state {@code (m, l, o)} of heads {@code [h0, h0 + gc)} over keys {@code [rowStart, rowEnd)},
     * left unnormalized in {@code s.max}, {@code s.sum} and {@code s.acc}.
     */
    private static void accumulate(Params p, float[] q, int qOff, int qPos, Rows rows, int rowStart, int rowEnd,
                                   Scratch s, int h0, int gc, int kvh) {
        final int dV = rows.dV();
        s.ensure(rows.dK(), dV);
        Arrays.fill(s.max, 0, gc, Float.NEGATIVE_INFINITY);
        Arrays.fill(s.sum, 0, gc, 0.0f);
        Arrays.fill(s.acc, 0, gc * dV, 0.0f);
        for (int b = rowStart; b < rowEnd; b += KEY_BLOCK) {
            final int nb = Math.min(KEY_BLOCK, rowEnd - b);
            scoreBlock(p, q, qOff, qPos, rows, b, nb, s, h0, gc, kvh);
            rescaleBlock(s, nb, gc, dV);
            addValues(rows, b, nb, s, gc, kvh, dV);
        }
    }

    /**
     * Scores {@code s.scores[g * KEY_BLOCK + r]} of heads {@code [h0, h0 + gc)} against keys {@code [b, b + nb)};
     * keys outside the query's window score -INF.
     */
    private static void scoreBlock(Params p, float[] q, int qOff, int qPos, Rows rows, int b, int nb, Scratch s,
                                   int h0, int gc, int kvh) {
        final int dK = rows.dK();
        final float[] sc = s.scores;
        final float[] kbuf = s.k;
        final KvWindow window = p.window;
        for (int r = 0; r < nb; r++) {
            final int row = b + r;
            final int pk = rows.pos(row);
            if (!window.visible(qPos, pk)) {
                for (int g = 0; g < gc; g++) {
                    sc[g * KEY_BLOCK + r] = Float.NEGATIVE_INFINITY;
                }
                continue;
            }
            rows.readK(row, kvh, kbuf);
            for (int g = 0; g < gc; g++) {
                final int h = h0 + g;
                final int qb = qOff + h * dK;
                float dot = 0.0f;
                for (int d = 0; d < dK; d++) {
                    dot += q[qb + d] * kbuf[d];
                }
                sc[g * KEY_BLOCK + r] = score(dot, p.scale, p.softcap) + p.bias(h, qPos, pk);
            }
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

    /** {@code o += p V} over keys {@code [b, b + nb)}; a value row no head weights is not read. */
    private static void addValues(Rows rows, int b, int nb, Scratch s, int gc, int kvh, int dV) {
        final float[] sc = s.scores;
        final float[] vbuf = s.v;
        final float[] acc = s.acc;
        for (int r = 0; r < nb; r++) {
            boolean used = false;
            for (int g = 0; g < gc && !used; g++) {
                used = sc[g * KEY_BLOCK + r] != 0.0f;
            }
            if (!used) {
                continue;
            }
            rows.readV(b + r, kvh, vbuf);
            for (int g = 0; g < gc; g++) {
                final float pr = sc[g * KEY_BLOCK + r];
                if (pr != 0.0f) {
                    final int ab = g * dV;
                    for (int d = 0; d < dV; d++) {
                        acc[ab + d] += pr * vbuf[d];
                    }
                }
            }
        }
    }

    /**
     * Sink and normalization of heads {@code [h0, h0 + gc)} from their state {@code m = mx[mOff + g]},
     * {@code l = sum[mOff + g]}, {@code o = acc[accOff + g * dV + d]}: with a sink {@code m' = max(m, sink)},
     * {@code out = o e^(m - m') / (l e^(m - m') + e^(sink - m'))}, else {@code out = o / l}; zeros when the
     * denominator is 0.
     */
    private static void finish(Params p, float[] mx, int mOff, float[] sum, int lOff, float[] acc, int accOff, int dV,
                               int h0, int gc, float[] out, int outOff) {
        for (int g = 0; g < gc; g++) {
            final int h = h0 + g;
            final float m = mx[mOff + g];
            final float l = sum[lOff + g];
            float factor = 1.0f;
            float denom = l;
            if (p.hasSinks()) {
                final float sink = p.sinks[h];
                final float m2 = Math.max(m, sink);
                factor = rescale(m, m2);
                denom = l * factor + (float) Math.exp(sink - m2);
            }
            final int ob = outOff + h * dV;
            final int ab = accOff + g * dV;
            for (int d = 0; d < dV; d++) {
                out[ob + d] = denom > 0.0f ? acc[ab + d] * factor / denom : 0.0f;
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
                                    int rowEnd, float[] out, int outOff, int outRowStride) {
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
                accumulate(p, q, qOff + t * qRowStride, qPos[qPosOff + t], rows, k0, Math.min(rowEnd, k0 + splitLen),
                        scratch, kvh * g0 + g, gc, kvh);
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
                finish(p, m, 0, l, 0, o, 0, dV, kvh * g0 + g, gc, out, outOff + t * outRowStride);
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
                final int pb = st * dV;
                for (int d = 0; d < dV; d++) {
                    o[ob + d] += po[pb + d] * w;
                }
            }
            m[g] = mm;
            l[g] = ll;
        }
    }
}

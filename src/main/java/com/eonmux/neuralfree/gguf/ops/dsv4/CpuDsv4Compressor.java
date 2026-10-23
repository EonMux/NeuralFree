package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;

import java.util.Objects;

/**
 * CPU twin of {@code kernels/llm/dsv4/compress.cl}: the DeepSeek-V4 KV compressor (CSA ratio 4 with overlapped
 * windows, HCA ratio 128, and the ratio-4 lightning-indexer compressor), the compressor-state persist and the raw
 * window store, with the kernels' formulas and evaluation order (llama.cpp src/models/deepseek4.cpp:470-585, 940-1180;
 * src/llama-kv-cache-dsv4.cpp:431-740). Also hosts the row primitives {@link CpuDsv4Attention} shares: RMS norm,
 * RoPE and the Walsh-Hadamard transform.
 *
 * <p>Compressor state: F32 planes {@code kv} and {@code score} of {@link Dsv4Kernels#stateRows(int)} x
 * {@link Dsv4Kernels#stateWidth(int, int)}; row {@code p % stateRows} holds position {@code p} with the score including
 * {@code ape[p % ratio]}. The current ubatch rows ({@code cur_kv}, {@code cur_score} without ape) hold token {@code t}
 * at {@code off + t * stateWidth}. A ubatch is the contiguous positions {@code [pos0, pos0 + nTokens)} of one sequence
 * whose earlier positions were all persisted with {@link #stateUpdate}. The live plane starts each state array;
 * rollback snapshot planes follow it ({@link #stateSnapshot}).
 */
public final class CpuDsv4Compressor {

    private CpuDsv4Compressor() {
    }

    /** Immutable shape of one compressor: ratio, compressed row width, RMS epsilon, row RoPE and Hadamard block. */
    public static final class Config {

        private final int ratio;
        private final int width;
        private final float eps;
        private final Dsv4Kernels.RopeParams rope;
        private final int rotBlock;

        private Config(int ratio, int width, float eps, Dsv4Kernels.RopeParams rope, int rotBlock) {
            this.ratio = ratio;
            this.width = width;
            this.eps = eps;
            this.rope = rope;
            this.rotBlock = rotBlock;
        }

        /**
         * @param ratio    {@link Dsv4Spec#RATIO_CSA} or {@link Dsv4Spec#RATIO_HCA}
         * @param width    compressed row width (head dim, or the indexer head size)
         * @param eps      RMS norm epsilon ({@code f_norm_rms_eps})
         * @param rope     rotation of the row tail at position {@code b * ratio} (compress base)
         * @param rotBlock cache Walsh-Hadamard block, 0 for none
         * @throws IllegalArgumentException for an unsupported ratio, width, RoPE or block
         */
        public static Config of(int ratio, int width, float eps, Dsv4Kernels.RopeParams rope, int rotBlock) {
            Dsv4Kernels.checkRatio(ratio);
            Dsv4Kernels.checkRowWidth(width);
            if (!(eps >= 0f)) {
                throw new IllegalArgumentException("RMS epsilon must be non-negative, got " + eps);
            }
            Dsv4Kernels.checkRope(rope, width);
            Dsv4Kernels.checkRotBlock(rotBlock, width);
            return new Config(ratio, width, eps, rope, rotBlock);
        }

        public int ratio() { return ratio; }
        public int width() { return width; }
        public float eps() { return eps; }
        public Dsv4Kernels.RopeParams rope() { return rope; }
        public int rotBlock() { return rotBlock; }
        public boolean overlapped() { return Dsv4Kernels.overlapped(ratio); }
        public int stateRows() { return Dsv4Kernels.stateRows(ratio); }
        public int stateWidth() { return Dsv4Kernels.stateWidth(ratio, width); }

        @Override
        public String toString() {
            return "Dsv4Compressor{ratio=" + ratio + ", width=" + width + ", eps=" + eps + ", rope=" + rope
                    + ", rotBlock=" + rotBlock + "}";
        }
    }

    /** Current ubatch projections and the compressor weights: inputs shared by every compressor operation. */
    public static final class Step {

        private final int pos0;
        private final int nTokens;
        private final float[] curKv;
        private final int curKvOff;
        private final float[] curScore;
        private final int curScoreOff;
        private final float[] ape;

        /**
         * @param ape {@code ratio x stateWidth} absolute position embedding ({@code comp_ape}, F32)
         */
        public Step(int pos0, int nTokens, float[] curKv, int curKvOff, float[] curScore, int curScoreOff,
                    float[] ape) {
            if (pos0 < 0 || nTokens < 0) {
                throw new IllegalArgumentException("compressor step needs pos0 >= 0 and n_tokens >= 0, got " + pos0
                        + "/" + nTokens);
            }
            this.pos0 = pos0;
            this.nTokens = nTokens;
            this.curKv = Objects.requireNonNull(curKv, "cur_kv");
            this.curKvOff = curKvOff;
            this.curScore = Objects.requireNonNull(curScore, "cur_score");
            this.curScoreOff = curScoreOff;
            this.ape = Objects.requireNonNull(ape, "ape");
        }

        public int pos0() { return pos0; }
        public int nTokens() { return nTokens; }
    }

    /**
     * {@code LLM_DSV4_Compress}: writes compressed row {@code b} of {@code cache} for every block the ubatch completes.
     *
     * @param stateKv     plane {@code kv} as it was before this ubatch
     * @param stateScore  plane {@code score} as it was before this ubatch
     * @param normW       {@code width} RMS norm weights ({@code comp_norm})
     * @param cache       compressed cache with rows of {@code width} values
     */
    public static void compress(Config cfg, Step step, float[] stateKv, float[] stateScore, float[] normW,
                                KvAccessor cache) {
        checkStep(cfg, step, stateKv, stateScore);
        Objects.requireNonNull(normW, "norm_w");
        if (normW.length < cfg.width) {
            throw new IllegalArgumentException("norm weights hold " + normW.length + " < " + cfg.width + " values");
        }
        checkKeyCache(cache, cfg.width, "compressed cache");
        final int b0 = Dsv4Kernels.firstBlock(step.pos0, cfg.ratio);
        final int nBlocks = Dsv4Kernels.blockCount(step.pos0, step.nTokens, cfg.ratio);
        if (nBlocks > 0 && b0 + nBlocks > cache.layout().rows()) {
            throw new IllegalArgumentException("compressed row " + (b0 + nBlocks - 1) + " exceeds the cache's "
                    + cache.layout().rows() + " rows");
        }
        CpuParallel.parallelFor(nBlocks, 1, (from, to) -> {
            float[] row = new float[cfg.width];
            for (int g = from; g < to; g++) {
                int b = b0 + g;
                compressRow(cfg, step, b, stateKv, stateScore, normW, row);
                if (cfg.rotBlock > 0) {
                    hadamard(row, 0, cfg.width, cfg.rotBlock);
                }
                cache.writeK(b, row, 0);
            }
        });
    }

    /**
     * {@code LLM_DSV4_LidUpdate_<KV>}: the lightning-indexer cache update, {@link #compress} with ratio 4 into an
     * always Hadamard-rotated cache of the context's K type (llama-kv-cache-dsv4.cpp:1308-1312).
     *
     * @throws IllegalArgumentException when the configuration is not ratio 4 with a Hadamard block or the cache type
     *                                  cannot store rows of the configured width
     */
    public static void lidUpdate(Config cfg, Step step, float[] stateKv, float[] stateScore, float[] normW,
                                 KvAccessor cache) {
        if (cfg.ratio != Dsv4Spec.RATIO_CSA) {
            throw new IllegalArgumentException("the indexer compressor has ratio " + Dsv4Spec.RATIO_CSA + ", got "
                    + cfg.ratio);
        }
        Objects.requireNonNull(cache, "cache");
        Dsv4Kernels.checkLidCache(cache.layout().dtypeK(), cfg.rotBlock, cfg.width);
        compress(cfg, step, stateKv, stateScore, normW, cache);
    }

    /**
     * {@code LLM_DSV4_StateUpdate}: plane row {@code r} receives the latest ubatch token {@code t} with
     * {@code (pos0 + t) % stateRows == r} (kv as is, score plus {@code ape[(pos0 + t) % ratio]}); other rows keep
     * their contents (llama-kv-cache-dsv4.cpp:506-520, deepseek4.cpp:1025-1031). Runs after {@link #compress} of the
     * same planes.
     */
    public static void stateUpdate(Config cfg, Step step, float[] stateKv, float[] stateScore) {
        checkStep(cfg, step, stateKv, stateScore);
        final int rows = cfg.stateRows();
        final int sw = cfg.stateWidth();
        for (int r = 0; r < rows; r++) {
            int first = Math.floorMod(r - step.pos0 % rows, rows);
            if (first >= step.nTokens) {
                continue;
            }
            int t = first + ((step.nTokens - 1 - first) / rows) * rows;
            int apeRow = ((step.pos0 + t) % cfg.ratio) * sw;
            for (int col = 0; col < sw; col++) {
                int src = t * sw + col;
                stateKv[r * sw + col] = step.curKv[step.curKvOff + src];
                stateScore[r * sw + col] = step.curScore[step.curScoreOff + src] + step.ape[apeRow + col];
            }
        }
    }

    /**
     * {@code LLM_DSV4_StateSnapshot}: the rollback snapshots of the planes (llama.cpp {@code state_snapshot_src/dst_idxs},
     * src/llama-kv-cache-dsv4.cpp:684-713, deepseek4.cpp:245-262). Plane {@code d} ({@code 1..snapshots}, at
     * {@code d * stateRows * stateWidth} in the state arrays) receives the live plane as {@link #stateUpdate} would
     * leave it after only the first {@code nTokens - d} tokens of the ubatch: row {@code r} takes the latest such token
     * {@code t} with {@code (pos0 + t) % stateRows == r} (kv as is, score plus {@code ape[(pos0 + t) % ratio]}), and the
     * live row otherwise. Runs after {@link #compress} / {@link #lidUpdate} and before {@link #stateUpdate} of the same
     * planes, so the live plane is still the state before the ubatch.
     *
     * @param snapshots snapshot planes after the live one ({@code n_rs_seq}), at least 1
     * @throws IllegalArgumentException when the state arrays hold fewer than {@code 1 + snapshots} planes
     */
    public static void stateSnapshot(Config cfg, Step step, float[] stateKv, float[] stateScore, int snapshots) {
        checkStep(cfg, step, stateKv, stateScore);
        Dsv4Kernels.checkSnapshots(snapshots);
        final int rows = cfg.stateRows();
        final int sw = cfg.stateWidth();
        final int plane = rows * sw;
        long need = (long) (1 + snapshots) * plane;
        if (stateKv.length < need || stateScore.length < need) {
            throw new IllegalArgumentException("compressor state with " + snapshots + " snapshot planes needs " + need
                    + " values, got " + stateKv.length + "/" + stateScore.length);
        }
        for (int d = 1; d <= snapshots; d++) {
            final int prefix = step.nTokens - d;
            final int base = d * plane;
            for (int r = 0; r < rows; r++) {
                int first = Math.floorMod(r - step.pos0 % rows, rows);
                if (first >= prefix) {
                    System.arraycopy(stateKv, r * sw, stateKv, base + r * sw, sw);
                    System.arraycopy(stateScore, r * sw, stateScore, base + r * sw, sw);
                    continue;
                }
                int t = first + ((prefix - 1 - first) / rows) * rows;
                int apeRow = ((step.pos0 + t) % cfg.ratio) * sw;
                for (int col = 0; col < sw; col++) {
                    int src = t * sw + col;
                    stateKv[base + r * sw + col] = step.curKv[step.curKvOff + src];
                    stateScore[base + r * sw + col] = step.curScore[step.curScoreOff + src] + step.ape[apeRow + col];
                }
            }
        }
    }

    /**
     * {@code LLM_DSV4_CacheStore}: stores token {@code t} ({@code src[srcOff + t * srcStride, + width)}) as cache row
     * {@code cellIdx[t]}, Hadamard-rotated first when {@code rotBlock > 0}; negative cells are skipped.
     */
    public static void cacheStore(int nTokens, int width, int rotBlock, float[] src, int srcOff, int srcStride,
                                  int[] cellIdx, KvAccessor cache) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(cellIdx, "cell_idx");
        Dsv4Kernels.checkRotBlock(rotBlock, width);
        checkKeyCache(cache, width, "window cache");
        if (srcOff < 0 || srcStride < width) {
            throw new IllegalArgumentException("src rows need offset >= 0 and stride >= " + width + ", got " + srcOff
                    + "/" + srcStride);
        }
        if (nTokens < 0 || cellIdx.length < nTokens) {
            throw new IllegalArgumentException("cell_idx holds " + cellIdx.length + " entries for " + nTokens + " tokens");
        }
        if (nTokens > 0 && (long) srcOff + (long) (nTokens - 1) * srcStride + width > src.length) {
            throw new IllegalArgumentException("src holds " + src.length + " values, " + nTokens + " rows of " + width
                    + " at offset " + srcOff + " stride " + srcStride + " do not fit");
        }
        float[] row = new float[width];
        for (int t = 0; t < nTokens; t++) {
            int r = cellIdx[t];
            if (r < 0) {
                continue;
            }
            System.arraycopy(src, srcOff + t * srcStride, row, 0, width);
            if (rotBlock > 0) {
                hadamard(row, 0, width, rotBlock);
            }
            cache.writeK(r, row, 0);
        }
    }

    /**
     * Pooled, RMS-normalized and rotated row of block {@code b} (before the Hadamard step) into {@code row}: per
     * channel a softmax over the window scores and the weighted kv sum, then RMS norm times {@code normW} and RoPE at
     * {@code b * ratio} (deepseek4.cpp:499-511, 572-585).
     */
    static void compressRow(Config cfg, Step step, int b, float[] stateKv, float[] stateScore, float[] normW,
                            float[] row) {
        final int ratio = cfg.ratio;
        final int width = cfg.width;
        final boolean overlap = cfg.overlapped();
        final int nWin = overlap ? 2 * ratio : ratio;
        final int start = b * ratio;
        float[] kv = new float[nWin];
        float[] sc = new float[nWin];
        for (int c = 0; c < width; c++) {
            float m = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < nWin; j++) {
                boolean hi = overlap && j >= ratio;
                int p = overlap ? (hi ? start + j - ratio : start - ratio + j) : start + j;
                source(cfg, step, p, hi ? width + c : c, stateKv, stateScore, kv, sc, j);
                m = Math.max(m, sc[j]);
            }
            float l = 0f;
            float acc = 0f;
            for (int j = 0; j < nWin; j++) {
                float e = sc[j] == Float.NEGATIVE_INFINITY ? 0f : (float) Math.exp(sc[j] - m);
                l += e;
                acc += e * kv[j];
            }
            row[c] = acc * (1f / l);
        }
        rmsNorm(row, 0, width, normW, cfg.eps);
        rope(row, 0, width, cfg.rope, start, false);
    }

    /** Window source of position {@code p}, state column {@code col}, into {@code kv[j]}, {@code sc[j]}. */
    private static void source(Config cfg, Step step, int p, int col, float[] stateKv, float[] stateScore,
                               float[] kv, float[] sc, int j) {
        if (p < 0) {
            kv[j] = 0f;
            sc[j] = Float.NEGATIVE_INFINITY;
            return;
        }
        final int sw = cfg.stateWidth();
        if (p >= step.pos0) {
            int t = (p - step.pos0) * sw + col;
            kv[j] = step.curKv[step.curKvOff + t];
            sc[j] = step.curScore[step.curScoreOff + t] + step.ape[(p % cfg.ratio) * sw + col];
            return;
        }
        int r = (p % cfg.stateRows()) * sw + col;
        kv[j] = stateKv[r];
        sc[j] = stateScore[r];
    }

    /**
     * Checks that {@code cache} is a single-head key cache whose rows hold {@code width} values, the row shape every
     * DeepSeek-V4 cache shares (one K row per cell serves as K and V).
     */
    static void checkKeyCache(KvAccessor cache, int width, String what) {
        Objects.requireNonNull(cache, what);
        if (cache.nHeadKv() != 1 || cache.headDimK() != width) {
            throw new IllegalArgumentException(what + " rows hold " + cache.nHeadKv() + " head(s) of " + cache.headDimK()
                    + " values, DeepSeek-V4 needs one head of " + width);
        }
    }

    private static void checkStep(Config cfg, Step step, float[] stateKv, float[] stateScore) {
        Objects.requireNonNull(cfg, "config");
        Objects.requireNonNull(step, "step");
        Objects.requireNonNull(stateKv, "state kv");
        Objects.requireNonNull(stateScore, "state score");
        final long plane = (long) cfg.stateRows() * cfg.stateWidth();
        if (stateKv.length < plane || stateScore.length < plane) {
            throw new IllegalArgumentException("compressor planes need " + plane + " values, got " + stateKv.length
                    + "/" + stateScore.length);
        }
        final long cur = (long) step.nTokens * cfg.stateWidth();
        if (step.curKvOff < 0 || step.curScoreOff < 0 || step.curKvOff + cur > step.curKv.length
                || step.curScoreOff + cur > step.curScore.length) {
            throw new IllegalArgumentException("current ubatch rows need " + cur + " values from offsets "
                    + step.curKvOff + "/" + step.curScoreOff + ", got " + step.curKv.length + "/"
                    + step.curScore.length);
        }
        if (step.ape.length < (long) cfg.ratio * cfg.stateWidth()) {
            throw new IllegalArgumentException("ape needs " + (long) cfg.ratio * cfg.stateWidth() + " values, got "
                    + step.ape.length);
        }
    }

    /**
     * In-place RMS norm of {@code x[off, off + n)} times {@code w[0, n)}: {@code x * (1 / sqrt(mean(x^2) + eps)) * w}
     * (llama.cpp ggml-cpu/ops.cpp:4224-4231).
     */
    public static void rmsNorm(float[] x, int off, int n, float[] w, float eps) {
        float ss = 0f;
        for (int i = 0; i < n; i++) {
            ss += x[off + i] * x[off + i];
        }
        float scale = 1f / (float) Math.sqrt(ss / n + eps);
        for (int i = 0; i < n; i++) {
            x[off + i] = x[off + i] * scale * w[i];
        }
    }

    /**
     * Rotates the last {@code rope.nRot()} values of the row {@code x[off, off + width)} at position {@code pos}
     * (ggml rotate_pairs with {@code n_offs = width - n_rot}, llama.cpp ggml-cpu/ops.cpp:6375-6391); {@code inverse}
     * negates the sine as {@code ggml_rope_ext_back} (ops.cpp:6463-6466). A rotation of zero dims leaves the row
     * unchanged.
     */
    public static void rope(float[] x, int off, int width, Dsv4Kernels.RopeParams rope, int pos, boolean inverse) {
        final int nRot = rope.nRot();
        if (nRot == 0) {
            return;
        }
        final int half = nRot >> 1;
        final int base = off + width - nRot;
        final boolean neox = rope.mode() == Dsv4Kernels.ROPE_NEOX;
        float[] cs = new float[2];
        for (int i = 0; i < half; i++) {
            cosSin(rope, pos, i, cs);
            float sn = inverse ? -cs[1] : cs[1];
            int i0 = base + (neox ? i : (i << 1));
            int i1 = i0 + (neox ? half : 1);
            float x0 = x[i0];
            float x1 = x[i1];
            x[i0] = x0 * cs[0] - x1 * sn;
            x[i1] = x0 * sn + x1 * cs[0];
        }
    }

    /**
     * {@code (cos, sin) * mscale} of pair {@code i} at position {@code pos} into {@code cs[0..1]}: rope_yarn with
     * {@code theta_extrap = pos * theta_scale^i} (llama.cpp ggml-cpu/ops.cpp:6262-6285).
     */
    public static void cosSin(Dsv4Kernels.RopeParams rope, int pos, int i, float[] cs) {
        float thetaExtrap = pos * (float) Math.pow(rope.thetaScale(), i);
        float thetaInterp = rope.freqScale() * thetaExtrap;
        float theta = thetaInterp;
        float mscale = rope.attnFactor();
        if (rope.extFactor() != 0f) {
            float y = (i - rope.corrLo()) / Math.max(0.001f, rope.corrHi() - rope.corrLo());
            float rampMix = (1f - Math.min(1f, Math.max(0f, y))) * rope.extFactor();
            theta = thetaInterp * (1f - rampMix) + thetaExtrap * rampMix;
            mscale *= 1f + 0.1f * (float) Math.log(1f / rope.freqScale());
        }
        cs[0] = (float) Math.cos(theta) * mscale;
        cs[1] = (float) Math.sin(theta) * mscale;
    }

    /**
     * In-place orthonormal Walsh-Hadamard transform of every aligned block of {@code block} values of
     * {@code x[off, off + n)}: the normalized Sylvester matrix of {@code ggml_gen_hadamard} (llama.cpp
     * llama-kv-cache.cpp:20-56) applied per block (llama-impl.h:57-75). Its own inverse.
     *
     * @throws IllegalArgumentException when {@code block} is not a power of two {@code >= 2} dividing {@code n}
     */
    public static void hadamard(float[] x, int off, int n, int block) {
        if (block < 2 || Integer.bitCount(block) != 1 || n % block != 0) {
            throw new IllegalArgumentException("Hadamard block must be a power of two >= 2 dividing " + n + ", got "
                    + block);
        }
        final int halfN = n >> 1;
        for (int len = 1; len < block; len <<= 1) {
            for (int k = 0; k < halfN; k++) {
                int i = off + (k / len) * (len << 1) + (k % len);
                float a = x[i];
                float b = x[i + len];
                x[i] = a + b;
                x[i + len] = a - b;
            }
        }
        float s = 1f / (float) Math.sqrt(block);
        for (int k = 0; k < n; k++) {
            x[off + k] *= s;
        }
    }
}

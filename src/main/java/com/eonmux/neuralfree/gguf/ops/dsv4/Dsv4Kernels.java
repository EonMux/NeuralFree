package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Kernel names, argument indices, launch geometry and preconditions of the DeepSeek-V4 compressor and compressed
 * attention kernels ({@code kernels/llm/dsv4/compress.cl}, {@code dsv4/attn.cl} and their GLSL twins
 * {@code dsv4_compress.comp}, {@code dsv4_attn.comp}).
 *
 * <p>Argument lists ({@code (dyn)} marks the per-step arguments with an {@code ARG_*} constant):
 * <pre>
 * LLM_DSV4_Compress      (i pos0(dyn), i n_tokens(dyn), i ratio, i width, i n_rot, i rope_mode, f eps,
 *                         f theta_scale, f freq_scale, f ext_factor, f attn_factor, f corr_lo, f corr_hi,
 *                         i rot_block, i kv_type, i cur_kv_off(dyn), i cur_score_off(dyn),
 *                         R state_kv, R state_score, R cur_kv, R cur_score, R ape, R norm_w, W&lt;KV&gt; cache)
 * LLM_DSV4_LidUpdate_&lt;KV&gt; (i pos0(dyn), i n_tokens(dyn), i width, i n_rot, i rope_mode, f eps, f theta_scale,
 *                         f freq_scale, f ext_factor, f attn_factor, f corr_lo, f corr_hi, i rot_block,
 *                         i cur_kv_off(dyn), i cur_score_off(dyn),
 *                         R state_kv, R state_score, R cur_kv, R cur_score, R ape, R norm_w, W&lt;KV&gt; cache)
 * LLM_DSV4_StateUpdate   (i pos0(dyn), i n_tokens(dyn), i ratio, i width, i cur_kv_off(dyn), i cur_score_off(dyn),
 *                         R cur_kv, R cur_score, R ape, W state_kv, W state_score)
 * LLM_DSV4_StateSnapshot (i pos0(dyn), i n_tokens(dyn), i ratio, i width, i n_snap, i cur_kv_off(dyn),
 *                         i cur_score_off(dyn), R cur_kv, R cur_score, R ape, RW state_kv, RW state_score)
 * LLM_DSV4_CacheStore    (i n_tokens(dyn), i width, i kv_type, i rot_block, i src_off(dyn), i src_stride,
 *                         R src, R&lt;int&gt; cell_idx, W&lt;KV&gt; cache)
 * LLM_DSV4_CompressedAttn(i n_tokens(dyn), i n_head, i head_dim, i g_tile, i q_off(dyn), i out_off(dyn),
 *                         i n_raw(dyn), i raw_type, i n_swa, i comp_mode, i n_comp(dyn), i comp_type, i ratio,
 *                         i sel_stride(dyn), i has_sinks, f kq_scale, i rot_block, i n_rot, i rope_mode, f theta_scale,
 *                         f freq_scale, f ext_factor, f attn_factor, f corr_lo, f corr_hi,
 *                         R q, R&lt;KV&gt; raw_k, R&lt;int&gt; raw_pos, R&lt;KV&gt; comp_k, R&lt;int&gt; sel, R sinks,
 *                         R&lt;int&gt; q_pos, W out)
 * </pre>
 * {@code W<KV>}/{@code R<KV>} caches are byte buffers of {@code kv_type} rows ({@link #kvTypeCode(KvDType)}); the
 * indexer cache's element type selects the {@code LLM_DSV4_LidUpdate} variant ({@link #lidUpdate(KvDType)}), as
 * llama.cpp creates it with the context's {@code type_k} (llama-kv-cache-dsv4.cpp:1308-1312); {@code rope_mode} is
 * {@link #ropeModeCode(RopeType)}; the RoPE scalars come from {@link RopeParams}; {@code rot_block} is the cache's
 * Walsh-Hadamard block ({@code RotationPlan.nRotK()}, 0 for none); {@code comp_mode} is {@link CompMode#code()}.
 * Unused buffers ({@code sel} for a dense or window-only layer, {@code comp_k} for a window-only layer, {@code sinks}
 * without sinks) are read-only placeholders. Compressor planes, {@code ape} and
 * {@code norm_w} are F32. The math is reproduced by {@link CpuDsv4Compressor} and {@link CpuDsv4Attention}.
 *
 * <p>The compressor tensors hold {@code 1 + n_snap} consecutive planes of {@link #stateRows} x {@link #stateWidth}:
 * the live plane, which every kernel but the snapshot reads and {@code LLM_DSV4_StateUpdate} writes, then the rollback
 * snapshot planes {@code LLM_DSV4_StateSnapshot} writes (llama.cpp {@code n_rs_seq}, llama-kv-cache-dsv4.cpp:684-713).
 *
 * <p>Order within one layer and step: {@code LLM_DSV4_Compress} / {@code LLM_DSV4_LidUpdate} of a compressor read the
 * live plane, then {@code LLM_DSV4_StateSnapshot} (when the memory keeps snapshots) copies it with the ubatch prefixes
 * into the snapshot planes, then {@code LLM_DSV4_StateUpdate} of the same planes overwrites it;
 * {@code LLM_DSV4_CacheStore} of the raw window and the compressed-row writes precede
 * {@code LLM_DSV4_CompressedAttn}.
 */
public final class Dsv4Kernels {

    public static final String COMPRESS = "LLM_DSV4_Compress";
    /** Name prefix of the {@code LLM_DSV4_LidUpdate_<KV>} variants. */
    public static final String LID_UPDATE_PREFIX = "LLM_DSV4_LidUpdate_";
    public static final String STATE_UPDATE = "LLM_DSV4_StateUpdate";
    public static final String STATE_SNAPSHOT = "LLM_DSV4_StateSnapshot";
    public static final String CACHE_STORE = "LLM_DSV4_CacheStore";
    public static final String COMPRESSED_ATTN = "LLM_DSV4_CompressedAttn";

    public static final List<String> ALL;

    static {
        List<String> all = new ArrayList<>();
        all.add(COMPRESS);
        all.addAll(KvDType.kernelVariants(LID_UPDATE_PREFIX));
        all.add(STATE_UPDATE);
        all.add(STATE_SNAPSHOT);
        all.add(CACHE_STORE);
        all.add(COMPRESSED_ATTN);
        ALL = Collections.unmodifiableList(all);
    }

    public static final int COMPRESS_ARG_COUNT = 24;
    public static final int ARG_COMPRESS_POS0 = 0;
    public static final int ARG_COMPRESS_N_TOKENS = 1;
    public static final int ARG_COMPRESS_CUR_KV_OFF = 15;
    public static final int ARG_COMPRESS_CUR_SCORE_OFF = 16;

    public static final int LID_ARG_COUNT = 22;
    public static final int ARG_LID_POS0 = 0;
    public static final int ARG_LID_N_TOKENS = 1;
    public static final int ARG_LID_CUR_KV_OFF = 13;
    public static final int ARG_LID_CUR_SCORE_OFF = 14;

    public static final int STATE_ARG_COUNT = 11;
    public static final int ARG_STATE_POS0 = 0;
    public static final int ARG_STATE_N_TOKENS = 1;
    public static final int ARG_STATE_CUR_KV_OFF = 4;
    public static final int ARG_STATE_CUR_SCORE_OFF = 5;

    public static final int SNAPSHOT_ARG_COUNT = 12;
    public static final int ARG_SNAPSHOT_POS0 = 0;
    public static final int ARG_SNAPSHOT_N_TOKENS = 1;
    public static final int ARG_SNAPSHOT_N_SNAP = 4;
    public static final int ARG_SNAPSHOT_CUR_KV_OFF = 5;
    public static final int ARG_SNAPSHOT_CUR_SCORE_OFF = 6;

    public static final int STORE_ARG_COUNT = 9;
    public static final int ARG_STORE_N_TOKENS = 0;
    public static final int ARG_STORE_SRC_OFF = 4;

    public static final int ATTN_ARG_COUNT = 33;
    public static final int ARG_ATTN_N_TOKENS = 0;
    public static final int ARG_ATTN_Q_OFF = 4;
    public static final int ARG_ATTN_OUT_OFF = 5;
    public static final int ARG_ATTN_N_RAW = 6;
    public static final int ARG_ATTN_N_COMP = 10;
    public static final int ARG_ATTN_SEL_STRIDE = 13;

    /** Largest compressed row width / head dim a work-group holds in local memory ({@code NF_DSV4_MAX_WIDTH}). */
    public static final int MAX_WIDTH = 1024;
    /** Largest attention head tile ({@code NF_DSV4_ATTN_MAX_G}). */
    public static final int ATTN_MAX_G = 16;
    /** Recommended local size of the work-group kernels. */
    public static final int LOCAL = 128;
    /** Smallest local size of the work-group kernels (one full subgroup). */
    public static final int MIN_LOCAL = 32;

    /** {@code rope_mode} codes ({@code NF_DSV4_ROPE_*}, ggml {@code GGML_ROPE_TYPE_NORMAL} / {@code _NEOX}). */
    public static final int ROPE_NORM = 0;
    public static final int ROPE_NEOX = 2;

    private Dsv4Kernels() {
    }

    /** Which compressed rows {@code LLM_DSV4_CompressedAttn} attends besides the raw window. */
    public enum CompMode {
        /** Window-only layer ({@code compress_ratio == 0}). */
        NONE(0),
        /** HCA layer: every visible compressed row. */
        DENSE(1),
        /** CSA layer: the visible rows the lightning indexer selected. */
        SELECTED(2);

        private final int code;

        CompMode(int code) {
            this.code = code;
        }

        public int code() {
            return code;
        }

        /** Mode of a layer with the given compress ratio (0, {@link Dsv4Spec#RATIO_CSA} or {@link Dsv4Spec#RATIO_HCA}). */
        public static CompMode forRatio(int ratio) {
            if (ratio == 0) {
                return NONE;
            }
            if (ratio == Dsv4Spec.RATIO_CSA) {
                return SELECTED;
            }
            if (ratio == Dsv4Spec.RATIO_HCA) {
                return DENSE;
            }
            throw new IllegalArgumentException("DeepSeek-V4 compress ratio must be 0, " + Dsv4Spec.RATIO_CSA + " or "
                    + Dsv4Spec.RATIO_HCA + ", got " + ratio);
        }
    }

    /**
     * RoPE scalars of one rotation: {@code n_rot} rotated tail dims, ggml mode, {@code theta_scale = base^(-2/n_rot)},
     * {@code freq_scale}, {@code ext_factor}, {@code attn_factor} and the YaRN correction dims
     * (llama.cpp ggml-cpu/ops.cpp:6262-6285, {@code ggml_rope_yarn_corr_dims}).
     */
    public static final class RopeParams {

        private static final RopeParams NONE = new RopeParams(0, ROPE_NORM, 1f, 1f, 0f, 1f, 0f, 0f);

        private final int nRot;
        private final int mode;
        private final float thetaScale;
        private final float freqScale;
        private final float extFactor;
        private final float attnFactor;
        private final float corrLo;
        private final float corrHi;

        private RopeParams(int nRot, int mode, float thetaScale, float freqScale, float extFactor, float attnFactor,
                           float corrLo, float corrHi) {
            this.nRot = nRot;
            this.mode = mode;
            this.thetaScale = thetaScale;
            this.freqScale = freqScale;
            this.extFactor = extFactor;
            this.attnFactor = attnFactor;
            this.corrLo = corrLo;
            this.corrHi = corrHi;
        }

        /** No rotation ({@code n_rot = 0}). */
        public static RopeParams none() {
            return NONE;
        }

        /**
         * @param ropeType   {@link RopeType#NORM} or {@link RopeType#NEOX}
         * @param freqBase   rotation base (the compress base for compressed rows and compressed layers)
         * @throws IllegalArgumentException for an odd or non-positive {@code nRot}, an unsupported type, a non-positive
         *                                  base or frequency scale
         */
        public static RopeParams of(int nRot, RopeType ropeType, float freqBase, float freqScale, float extFactor,
                                    float attnFactor, float corrLo, float corrHi) {
            if (nRot <= 0 || (nRot & 1) != 0) {
                throw new IllegalArgumentException("DeepSeek-V4 RoPE needs a positive even n_rot, got " + nRot);
            }
            if (!(freqBase > 0f) || !(freqScale > 0f)) {
                throw new IllegalArgumentException("DeepSeek-V4 RoPE needs positive freq_base and freq_scale, got "
                        + freqBase + "/" + freqScale);
            }
            float thetaScale = (float) Math.pow(freqBase, -2.0 / nRot);
            return new RopeParams(nRot, ropeModeCode(ropeType), thetaScale, freqScale, extFactor, attnFactor,
                    corrLo, corrHi);
        }

        public int nRot() { return nRot; }
        public int mode() { return mode; }
        public float thetaScale() { return thetaScale; }
        public float freqScale() { return freqScale; }
        public float extFactor() { return extFactor; }
        public float attnFactor() { return attnFactor; }
        public float corrLo() { return corrLo; }
        public float corrHi() { return corrHi; }

        @Override
        public String toString() {
            return "RopeParams{nRot=" + nRot + ", mode=" + mode + ", thetaScale=" + thetaScale + ", freqScale="
                    + freqScale + ", extFactor=" + extFactor + ", attnFactor=" + attnFactor + ", corr=[" + corrLo
                    + ", " + corrHi + "]}";
        }
    }

    /**
     * DeepSeek-V4 attention factor of a rotation: 1 without YaRN extrapolation, else
     * {@code 1 / (1 + 0.1 ln(1 / freq_scale))}, cancelling the YaRN magnitude scale (llama.cpp
     * src/models/deepseek4.cpp:11-17).
     */
    public static float attnFactor(float freqScale, float extFactor) {
        if (extFactor == 0f) {
            return 1f;
        }
        return 1f / (1f + 0.1f * (float) Math.log(1f / freqScale));
    }

    /** {@code rope_mode} code of a rotation type: ggml {@code GGML_ROPE_TYPE_NORMAL} (0) or {@code _NEOX} (2). */
    public static int ropeModeCode(RopeType type) {
        Objects.requireNonNull(type, "rope type");
        if (type != RopeType.NORM && type != RopeType.NEOX) {
            throw new IllegalArgumentException("DeepSeek-V4 kernels rotate NORM or NEOX pairs, got " + type);
        }
        return type.ggmlMode();
    }

    /** {@code LLM_DSV4_LidUpdate} variant storing indexer rows of {@code type}. */
    public static String lidUpdate(KvDType type) {
        return Objects.requireNonNull(type, "indexer cache type").kernel(LID_UPDATE_PREFIX);
    }

    /** {@code kv_type} code of a cache element type ({@link KvDType#code()}). */
    public static int kvTypeCode(KvDType type) {
        return Objects.requireNonNull(type, "cache element type").code();
    }

    /** True for the overlapped (CSA / indexer) compressor of ratio 4. */
    public static boolean overlapped(int ratio) {
        checkRatio(ratio);
        return ratio == Dsv4Spec.RATIO_CSA;
    }

    /** Compressor plane rows: {@code 2 * ratio} overlapped, {@code ratio} otherwise (llama-kv-cache-dsv4.cpp:1319-1335). */
    public static int stateRows(int ratio) {
        return overlapped(ratio) ? 2 * ratio : ratio;
    }

    /** Compressor plane width: {@code 2 * width} overlapped, {@code width} otherwise. */
    public static int stateWidth(int ratio, int width) {
        return overlapped(ratio) ? 2 * width : width;
    }

    /** First compressed row a ubatch starting at {@code pos0} can complete. */
    public static int firstBlock(int pos0, int ratio) {
        checkRatio(ratio);
        checkNonNegative("pos0", pos0);
        return pos0 / ratio;
    }

    /** Compressed rows completed by the ubatch {@code [pos0, pos0 + nTokens)}. */
    public static int blockCount(int pos0, int nTokens, int ratio) {
        checkRatio(ratio);
        checkNonNegative("pos0", pos0);
        checkNonNegative("n_tokens", nTokens);
        return (int) (((long) pos0 + nTokens) / ratio - pos0 / ratio);
    }

    /** Compressed rows visible to position {@code pos}: {@code (pos + 1) / ratio} (llama-graph.cpp:834-870). */
    public static int visibleRows(int pos, int ratio) {
        checkRatio(ratio);
        return pos < 0 ? 0 : (pos + 1) / ratio;
    }

    /**
     * 1-D global size of {@code LLM_DSV4_Compress} / {@code LLM_DSV4_LidUpdate}: one work-group per completed block;
     * zero means no dispatch.
     */
    public static long[] compressGlobal(int pos0, int nTokens, int ratio, int local) {
        checkLocal(local);
        return new long[]{(long) blockCount(pos0, nTokens, ratio) * local};
    }

    /** 1-D global size of {@code LLM_DSV4_StateUpdate}: one work-item per plane element, rounded up to {@code local}. */
    public static long[] stateUpdateGlobal(int ratio, int width, int local) {
        if (local < 1) {
            throw new IllegalArgumentException("state update local size must be positive, got " + local);
        }
        checkPositive("compressor width", width);
        long n = (long) stateRows(ratio) * stateWidth(ratio, width);
        return new long[]{(n + local - 1) / local * local};
    }

    /**
     * 1-D global size of {@code LLM_DSV4_StateSnapshot}: one work-item per element of the {@code snapshots} snapshot
     * planes, rounded up to {@code local}.
     */
    public static long[] stateSnapshotGlobal(int ratio, int width, int snapshots, int local) {
        if (local < 1) {
            throw new IllegalArgumentException("state snapshot local size must be positive, got " + local);
        }
        checkPositive("compressor width", width);
        checkSnapshots(snapshots);
        long n = (long) snapshots * stateRows(ratio) * stateWidth(ratio, width);
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(snapshots + " snapshot planes of the ratio-" + ratio + " compressor of"
                    + " width " + width + " exceed the kernel's int indexing");
        }
        return new long[]{(n + local - 1) / local * local};
    }

    /** 1-D global size of {@code LLM_DSV4_CacheStore}: one work-group per token. */
    public static long[] cacheStoreGlobal(int nTokens, int local) {
        checkLocal(local);
        checkPositive("cache store n_tokens", nTokens);
        return new long[]{(long) nTokens * local};
    }

    /** 1-D global size of {@code LLM_DSV4_CompressedAttn}: one work-group per (token, tile of {@code gTile} heads). */
    public static long[] attnGlobal(int nTokens, int nHead, int gTile, int local) {
        checkLocal(local);
        checkPositive("attention n_tokens", nTokens);
        checkPositive("attention n_head", nHead);
        checkPositive("attention head tile", gTile);
        long tiles = ((long) nHead + gTile - 1) / gTile;
        return new long[]{(long) nTokens * tiles * local};
    }

    /** Local size array {@code {local}}. */
    public static long[] local(int local) {
        return new long[]{local};
    }

    /**
     * Largest head tile of {@code LLM_DSV4_CompressedAttn} that fits {@code smemFloats} local floats:
     * {@code g * (2 * headDim + local) <= smemFloats}, at most {@link #ATTN_MAX_G} and {@code nHead}.
     *
     * @param smemFloats the device's {@code NF_FA_SMEM_FLOATS}
     * @throws IllegalArgumentException when not even one head fits
     */
    public static int attnHeadTile(int headDim, int nHead, int local, int smemFloats) {
        checkLocal(local);
        checkPositive("attention n_head", nHead);
        checkRowWidth(headDim);
        long perHead = 2L * headDim + local;
        long fit = smemFloats / perHead;
        if (fit < 1) {
            throw new IllegalArgumentException("DeepSeek-V4 attention needs " + perHead + " local floats per head (head "
                    + headDim + ", local " + local + "), the device provides " + smemFloats);
        }
        return (int) Math.min(Math.min(fit, ATTN_MAX_G), nHead);
    }

    /**
     * Checks the local size of the work-group kernels: a power of two in [{@link #MIN_LOCAL},
     * {@link LlmKernelConstants#LLM_DIMX}].
     */
    public static void checkLocal(int local) {
        if (local < MIN_LOCAL || local > LlmKernelConstants.LLM_DIMX || Integer.bitCount(local) != 1) {
            throw new IllegalArgumentException("DeepSeek-V4 local size must be a power of two in [" + MIN_LOCAL + ", "
                    + LlmKernelConstants.LLM_DIMX + "], got " + local);
        }
    }

    /** Checks the snapshot plane count of {@code LLM_DSV4_StateSnapshot}: at least one plane is written. */
    public static void checkSnapshots(int snapshots) {
        if (snapshots < 1) {
            throw new IllegalArgumentException("compressor rollback snapshots must be >= 1 where written, got "
                    + snapshots);
        }
    }

    /** Checks a compress ratio: {@link Dsv4Spec#RATIO_CSA} or {@link Dsv4Spec#RATIO_HCA}. */
    public static void checkRatio(int ratio) {
        if (ratio != Dsv4Spec.RATIO_CSA && ratio != Dsv4Spec.RATIO_HCA) {
            throw new IllegalArgumentException("DeepSeek-V4 compress ratio must be " + Dsv4Spec.RATIO_CSA + " or "
                    + Dsv4Spec.RATIO_HCA + ", got " + ratio);
        }
    }

    /**
     * Checks a compressed row width / head dim for a cache of {@code dtype}: a multiple of 4 in [4, {@link #MAX_WIDTH}]
     * that the type can store (a multiple of 32 for a block type).
     */
    public static void checkWidth(int width, KvDType dtype) {
        checkRowWidth(width);
        Objects.requireNonNull(dtype, "cache element type");
        if (!dtype.supportsWidth(width)) {
            throw new IllegalArgumentException("a " + dtype + " cache cannot store rows of " + width + " values");
        }
    }

    /** Checks a Walsh-Hadamard block: 0 (none) or a power of two {@code >= 2} dividing {@code width}. */
    public static void checkRotBlock(int rotBlock, int width) {
        if (rotBlock == 0) {
            return;
        }
        if (rotBlock < 2 || Integer.bitCount(rotBlock) != 1 || width % rotBlock != 0) {
            throw new IllegalArgumentException("Hadamard block must be 0 or a power of two >= 2 dividing " + width
                    + ", got " + rotBlock);
        }
    }

    /** Checks the rotated tail of a row of {@code width}: even and at most {@code width}. */
    public static void checkRope(RopeParams rope, int width) {
        Objects.requireNonNull(rope, "rope params");
        if (rope.nRot() > width) {
            throw new IllegalArgumentException("RoPE rotates " + rope.nRot() + " dims of a " + width + "-value row");
        }
    }

    /**
     * Checks the lightning-indexer cache of {@code LLM_DSV4_LidUpdate_<KV>}: rows of {@code width} values its element
     * type can store (whole 32-value blocks for a block type) and always Hadamard-rotated (llama.cpp
     * src/llama-kv-cache.cpp:331-336 forces k_rot for the deepseek4 indexer).
     */
    public static void checkLidCache(KvDType dtype, int rotBlock, int width) {
        checkWidth(width, dtype);
        if (rotBlock == 0) {
            throw new IllegalArgumentException("the DeepSeek-V4 indexer cache is always Hadamard-rotated");
        }
        checkRotBlock(rotBlock, width);
    }

    /**
     * Checks one {@code LLM_DSV4_Compress} configuration: local size, ratio, a row width the cache type can store, the
     * Hadamard block and the rotated tail.
     */
    public static void checkCompress(int ratio, int width, KvDType dtype, int rotBlock, RopeParams rope, int local) {
        checkLocal(local);
        checkRatio(ratio);
        checkWidth(width, dtype);
        checkRotBlock(rotBlock, width);
        checkRope(rope, width);
    }

    /** Checks one {@code LLM_DSV4_CacheStore} configuration: local size, row width for the cache type, Hadamard block. */
    public static void checkCacheStore(int width, KvDType dtype, int rotBlock, int local) {
        checkLocal(local);
        checkWidth(width, dtype);
        checkRotBlock(rotBlock, width);
    }

    /**
     * Checks one {@code LLM_DSV4_CompressedAttn} launch.
     *
     * @throws IllegalArgumentException for a head tile outside [1, min({@link #ATTN_MAX_G}, nHead)] or one that
     *                                  exceeds {@code smemFloats}, a bad head dim, RoPE or Hadamard block
     */
    public static void checkAttn(int nHead, int headDim, int gTile, int local, int smemFloats, KvDType rawType,
                                 KvDType compType, int rotBlock, RopeParams derope) {
        checkLocal(local);
        checkPositive("attention n_head", nHead);
        checkWidth(headDim, rawType);
        if (compType != null) {
            checkWidth(headDim, compType);
        }
        checkRotBlock(rotBlock, headDim);
        checkRope(derope, headDim);
        if (gTile < 1 || gTile > Math.min(ATTN_MAX_G, nHead)) {
            throw new IllegalArgumentException("attention head tile must be in [1, " + Math.min(ATTN_MAX_G, nHead)
                    + "], got " + gTile);
        }
        if ((long) gTile * (2L * headDim + local) > smemFloats) {
            throw new IllegalArgumentException("attention head tile " + gTile + " needs "
                    + (long) gTile * (2L * headDim + local) + " local floats, the device provides " + smemFloats);
        }
    }

    /** Checks a compressed row width / head dim: a multiple of 4 in [4, {@link #MAX_WIDTH}]. */
    public static void checkRowWidth(int width) {
        if (width < 4 || width > MAX_WIDTH || (width & 3) != 0) {
            throw new IllegalArgumentException("DeepSeek-V4 row width must be a multiple of 4 in [4, " + MAX_WIDTH
                    + "], got " + width);
        }
    }

    private static void checkPositive(String what, int v) {
        if (v < 1) {
            throw new IllegalArgumentException(what + " must be positive, got " + v);
        }
    }

    private static void checkNonNegative(String what, int v) {
        if (v < 0) {
            throw new IllegalArgumentException(what + " must be non-negative, got " + v);
        }
    }
}

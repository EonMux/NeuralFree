package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Element type of a KV cache row: the cache types llama.cpp accepts for {@code -ctk} / {@code -ctv}
 * (llama.cpp common/arg.cpp:304-314), the FP8 KV cache of vLLM ({@code --kv-cache-dtype fp8_e4m3 / fp8_e5m2},
 * vllm/config/cache.py CacheDType), vLLM's per-token-head KV caches ({@code fp8_per_token_head},
 * {@code int8_per_token_head}, {@code int4_per_token_head}) and vLLM's vendor formats (NVFP4 and its 4over6 variant,
 * the TurboQuant key and value streams, and the {@code fp8_ds_mla} / {@code nvfp4_ds_mla} latent rows).
 * F32, F16, BF16 and the FP8 types store one value per element (an FP8 value is one code: E4M3FN, largest finite
 * 448, or E5M2, largest finite 57344, {@code Fp8Codec}); the block types store whole 32-value ggml blocks (Q8_0 34
 * bytes, Q4_0 18, Q4_1 20, IQ4_NL 18, Q5_0 22, Q5_1 24), so a row width of a block type must be a multiple of 32.
 * {@link KvBlockCodec} holds the block layouts and the reference (de)quantizers; {@link KvVendorCodec} holds the
 * vendor formats and their reference codecs.
 *
 * <p>An FP8 cache stores values scaled per KV head: the writer of a row divides by the layer's checkpoint KV scale
 * ({@code blk.N.attn_k.kv_scale} / {@code blk.N.attn_v.kv_scale}, 1.0 by default) before the encoding and attention
 * multiplies the decoded values by it, as vLLM's {@code reshape_and_cache_fp8} and its flash attention do
 * (vllm csrc/cache_kernels.cu, quantization/w8a8/fp8 quant_utils.cuh). The FP8 types are not ggml storage types, so
 * {@link #ggmlType()} is null for them.</p>
 *
 * <p>A per-token-head cache ({@link #perTokenHead()}) stores, beside every row, one scale per (token, KV head):
 * the writer computes {@code absmax / qmax} of the head's values at cache-write time and stores the codes of
 * {@code value / scale} ({@link KvBlockCodec#perTokenHeadCode}); attention multiplies the decoded codes by the row's
 * per-head scale. The scale lives in the row's {@link SidePlane}, which travels with the row through page copies,
 * context shifts and host transfers as the codes do. FP8 per-token-head stores E4M3FN codes exactly like an FP8
 * cache; INT8 stores one signed byte per value (qmax 127); INT4 packs two signed nibbles per byte (qmax 7, first
 * value in the low nibble).</p>
 *
 * <p>A vendor format ({@link #vendorFormat()}) keeps the same row+side-plane form. NVFP4 stores two E2M1 values per
 * byte with one E4M3 scale per 16-value group ({@link #NVFP4_SF_PLANE}); the TurboQuant key stream stores FP8 E4M3
 * codes or 3/4-bit Lloyd-Max centroid indices (with the vector norm in {@link #TQ_NORM_PLANE}), and its value stream
 * stores 3/4-bit uniform codes with the per-head affine scale and zero ({@link #TQ_SCALE_PLANE} /
 * {@link #TQ_ZERO_PLANE}); {@code fp8_ds_mla} and {@code nvfp4_ds_mla} store one DeepSeek MLA latent row of 576
 * values (see {@link #latentFormat()}). NVFP4's checkpoint KV scale follows the FP8 rule
 * ({@link #checkpointScaled()}); the other vendor formats carry their scales inside the row planes.</p>
 */
public enum KvDType {
    F32(GGMLType.F32, 0),
    F16(GGMLType.F16, 1),
    Q8_0(GGMLType.Q8_0, 2),
    BF16(GGMLType.BF16, 3),
    Q4_0(GGMLType.Q4_0, 4),
    Q4_1(GGMLType.Q4_1, 5),
    IQ4_NL(GGMLType.IQ4_NL, 6),
    Q5_0(GGMLType.Q5_0, 7),
    Q5_1(GGMLType.Q5_1, 8),
    FP8_E4M3(9),
    FP8_E5M2(10),
    FP8_PER_TOKEN_HEAD(11, 1, 1, false, true),
    INT8_PER_TOKEN_HEAD(12, 1, 1, false, true),
    INT4_PER_TOKEN_HEAD(13, 2, 1, false, true),
    /** NVFP4 (vLLM {@code --kv-cache-dtype nvfp4}): E2M1 values, one E4M3 scale per 16 values. */
    NVFP4(14, 2, 1, Flags.NVFP4, planes(SidePlane.grouped("sf", FP8_E4M3code(), 16))),
    /** NVFP4 with the per-16-group scale searched between max/6 and max/4 (vLLM {@code nvfp4_4over6}). */
    NVFP4_4OVER6(15, 2, 1, Flags.NVFP4, planes(SidePlane.grouped("sf", FP8_E4M3code(), 16))),
    /** TurboQuant key stream, FP8 E4M3 keys (vLLM {@code turboquant_k8v4}). */
    TQ_K8_KEYS(16, 1, 1, Flags.NONE, planes()),
    /** TurboQuant key stream, 4-bit Lloyd-Max centroids with the vector norm (vLLM {@code turboquant_4bit_nc}). */
    TQ_4BIT_KEYS(17, 2, 1, Flags.TQ_MSE, planes(SidePlane.perHead("norm", F16))),
    /** TurboQuant key stream, 3-bit Lloyd-Max centroids with the vector norm (vLLM {@code turboquant_k3v4_nc}). */
    TQ_3BIT_KEYS(18, 8, 3, Flags.TQ_MSE, planes(SidePlane.perHead("norm", F16))),
    /** TurboQuant value stream, 4-bit uniform codes with the per-head scale and zero. */
    TQ_4BIT_VALUES(19, 2, 1, Flags.TQ_VALUE,
            planes(SidePlane.perHead("scale", F16), SidePlane.perHead("zero", F16))),
    /** TurboQuant value stream, 3-bit uniform codes with the per-head scale and zero. */
    TQ_3BIT_VALUES(20, 8, 3, Flags.TQ_VALUE,
            planes(SidePlane.perHead("scale", F16), SidePlane.perHead("zero", F16))),
    /**
     * DeepSeek MLA latent row in vLLM's {@code fp8_ds_mla} form: 512 E4M3 NoPE values (one power-of-two F32 scale
     * per 128-value tile) followed by 64 BF16 RoPE values.
     */
    FP8_DS_MLA(21, 576, 640, Flags.LATENT, planes(SidePlane.of("tile_scale", F32, 4))),
    /**
     * DeepSeek MLA latent row in vLLM's {@code nvfp4_ds_mla} form: 512 E2M1 NoPE values (one E4M3 scale per
     * 16-value group) followed by 64 unscaled E4M3 RoPE values.
     */
    NVFP4_DS_MLA(22, 576, 320, Flags.LATENT, planes(SidePlane.of("sf", FP8_E4M3code(), 32)));

    /** Name of the per-(token, KV head) scale plane of the per-token-head types. */
    public static final String SCALE_PLANE = "scale";
    /** Name of the per-(token, KV head) vector-norm plane of a TurboQuant key stream. */
    public static final String TQ_NORM_PLANE = "norm";
    /** Name of the per-(token, KV head) affine scale plane of a TurboQuant value stream. */
    public static final String TQ_SCALE_PLANE = "scale";
    /** Name of the per-(token, KV head) affine zero plane of a TurboQuant value stream. */
    public static final String TQ_ZERO_PLANE = "zero";
    /** Name of the per-16-value E4M3 scale plane of NVFP4 and {@code nvfp4_ds_mla}. */
    public static final String NVFP4_SF_PLANE = "sf";
    /** Name of the per-128-value F32 tile-scale plane of {@code fp8_ds_mla}. */
    public static final String DS_MLA_TILE_SCALE_PLANE = "tile_scale";
    /** Values of one DeepSeek MLA latent row: {@code kv_lora_rank = 512} NoPE values then {@code pe_dim = 64}. */
    public static final int DS_MLA_ROW_VALUES = 576;
    /** NoPE values of one DeepSeek MLA latent row ({@code kv_lora_rank}). */
    public static final int DS_MLA_NOPE_VALUES = 512;
    /** RoPE values of one DeepSeek MLA latent row ({@code pe_dim}). */
    public static final int DS_MLA_ROPE_VALUES = 64;

    private static final List<KvDType> ALL;
    private static final List<SidePlane> PER_TOKEN_HEAD_PLANES =
            Collections.singletonList(SidePlane.perHead(SCALE_PLANE, F32));

    static {
        List<KvDType> all = new ArrayList<>();
        for (KvDType t : values()) {
            all.add(t);
        }
        ALL = Collections.unmodifiableList(all);
    }

    /** Flags of a vendor type, packed so the enum constants stay readable. */
    private static final class Flags {
        static final int NONE = 0;
        /** NVFP4: E2M1 values under E4M3 group scales and the checkpoint KV scale. */
        static final int NVFP4 = 1;
        /** TurboQuant key stream: Lloyd-Max centroid indices and a per-head norm. */
        static final int TQ_MSE = 2;
        /** TurboQuant value stream: uniform codes under a per-head affine scale and zero. */
        static final int TQ_VALUE = 4;
        /** A DeepSeek MLA latent row format. */
        static final int LATENT = 8;
        /** Set on every vendor type, so a type whose format needs no other bit still counts as a vendor format. */
        static final int VENDOR = 128;
    }

    private final GGMLType ggmlType;
    private final int code;
    private final int fixedBlockSize;
    private final int fixedBlockBytes;
    private final boolean fp8;
    private final boolean perTokenHead;
    private final int flags;
    private final List<SidePlane> planes;

    KvDType(GGMLType ggmlType, int code) {
        this.ggmlType = ggmlType;
        this.code = code;
        this.fixedBlockSize = 1;
        this.fixedBlockBytes = 1;
        this.fp8 = false;
        this.perTokenHead = false;
        this.flags = Flags.NONE;
        this.planes = Collections.emptyList();
    }

    /** An FP8 type: one OCP code per value. */
    KvDType(int code) {
        this(code, 1, 1, true, false);
    }

    KvDType(int code, int blockSize, int blockBytes, boolean fp8, boolean perTokenHead) {
        this.ggmlType = null;
        this.code = code;
        this.fixedBlockSize = blockSize;
        this.fixedBlockBytes = blockBytes;
        this.fp8 = fp8;
        this.perTokenHead = perTokenHead;
        this.flags = Flags.NONE;
        this.planes = Collections.emptyList();
    }

    /** A vendor type: fixed packed geometry, its flags and its side planes. */
    KvDType(int code, int blockSize, int blockBytes, int flags, List<SidePlane> planes) {
        this.ggmlType = null;
        this.code = code;
        this.fixedBlockSize = blockSize;
        this.fixedBlockBytes = blockBytes;
        this.fp8 = false;
        this.perTokenHead = false;
        this.flags = flags | Flags.VENDOR;
        this.planes = Collections.unmodifiableList(new ArrayList<>(planes));
    }

    private static List<SidePlane> planes(SidePlane... planes) {
        return Arrays.asList(planes);
    }

    /** The F32 KV type, used by a plane declaration before the type is a constant. */
    private static KvDType FP8_E4M3code() {
        return KvDType.FP8_E4M3;
    }

    /**
     * The ggml storage type of the layout, or null for the FP8 and per-token-head types (their geometry is fixed
     * codes per value, not a ggml block).
     */
    public GGMLType ggmlType() {
        return ggmlType;
    }

    /**
     * Element-type code of the KV kernels: the {@code NF_KV_*} values of kernels/llm/common/kv_types.h and
     * shaders/llm/common/kv_types.glsl.
     */
    public int code() {
        return code;
    }

    /**
     * True for the block types ({@code ggml_is_quantized}: Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1); F32, F16, BF16,
     * the FP8 types and the per-token-head types are not quantized.
     */
    public boolean quantized() {
        return ggmlType != null && ggmlType.isQuantized();
    }

    /** True for the FP8 types (one OCP FP8 code per value with a per-head checkpoint scale; {@link #ggmlType()} null). */
    public boolean fp8() {
        return fp8;
    }

    /**
     * True for the per-token-head types: one code per value (two per byte for INT4) plus a {@link #SCALE_PLANE} side
     * plane holding one scale per (token, KV head).
     */
    public boolean perTokenHead() {
        return perTokenHead;
    }

    /** True for a vLLM vendor format ({@link #NVFP4} through {@link #NVFP4_DS_MLA}). */
    public boolean vendorFormat() {
        return (flags & Flags.VENDOR) != 0;
    }

    /** True for NVFP4 and its 4over6 variant: E2M1 values under per-16 E4M3 scales. */
    public boolean nvfp4() {
        return (flags & Flags.NVFP4) != 0;
    }

    /** True for the TurboQuant value streams: uniform codes under a per-head affine scale and zero. */
    public boolean turboQuantValue() {
        return (flags & Flags.TQ_VALUE) != 0;
    }

    /** True for the TurboQuant key streams quantized to Lloyd-Max centroids (a Hadamard rotation of each key). */
    public boolean turboQuantMse() {
        return (flags & Flags.TQ_MSE) != 0;
    }

    /** True for the TurboQuant key streams (FP8 keys or centroid indices). */
    public boolean turboQuantKey() {
        return this == TQ_K8_KEYS || turboQuantMse();
    }

    /** True for the DeepSeek MLA latent row formats ({@code fp8_ds_mla} / {@code nvfp4_ds_mla}). */
    public boolean latentFormat() {
        return (flags & Flags.LATENT) != 0;
    }

    /**
     * True when a row is written and read against the layer's checkpoint KV scale ({@link KvBlockCodec#fp8Code}): the
     * FP8 types and NVFP4.
     */
    public boolean checkpointScaled() {
        return fp8 || nvfp4();
    }

    /**
     * True for the types with a {@code LLM_KV_RopeShift_<type>} context-shift kernel: every type but the vendor
     * formats, whose group scales and centroid codes a rotation would re-derive.
     */
    public boolean hasShiftKernel() {
        return !vendorFormat();
    }

    /**
     * True for the types the sparse attention kernels (DSA, MSA, QSA, DeepSeek-V4) read: every type but the vendor
     * formats, which only the flash attention decodes.
     */
    public boolean hasSparseKernel() {
        return !vendorFormat();
    }

    /** True for the types whose rows are stored as bytes (the block, FP8, per-token-head and vendor types). */
    public boolean byteStored() {
        return ggmlType == null || ggmlType.isQuantized();
    }

    /** Values per storage unit (32 for the block types, 2 for INT4, 1 otherwise). */
    public int blockSize() {
        return ggmlType != null ? ggmlType.getBlockSize() : fixedBlockSize;
    }

    /** Bytes of one storage unit (one value for F32, F16, BF16, FP8 and INT8; two values for INT4). */
    public int blockBytes() {
        return ggmlType != null ? ggmlType.getTypeSize() : fixedBlockBytes;
    }

    /** Suffix of this type's KV kernel variants, e.g. {@code Q8_0} in {@code LLM_KV_Store_Q8_0}. */
    public String kernelSuffix() {
        return name();
    }

    /** The kernel variant of {@code prefix} for this type: {@code prefix + kernelSuffix()}. */
    public String kernel(String prefix) {
        return prefix + kernelSuffix();
    }

    /**
     * Bytes of one row of {@code width} values.
     *
     * @throws IllegalArgumentException for a negative width or a width that does not tile the storage unit
     */
    public long rowBytes(long width) {
        if (width < 0) {
            throw new IllegalArgumentException("KV row width " + width + " is negative");
        }
        if (width % blockSize() != 0) {
            throw new IllegalArgumentException(this + " KV rows need a width that is a multiple of " + blockSize()
                    + ", got " + width);
        }
        return width / blockSize() * (long) blockBytes();
    }

    /** True when rows of this width can be stored in this type. */
    public boolean supportsWidth(long width) {
        return width >= 0 && width % blockSize() == 0;
    }

    /**
     * One per-row plane a dtype stores outside the row itself: an auxiliary row of {@code widthFor(heads)} values of
     * {@code dtype} that travels with every row of the cache when the values are copied, moved, sealed, restored or
     * compared (the row-part rule of the prefix-cache design).
     */
    public static final class SidePlane {

        private final String name;
        private final KvDType dtype;
        private final int width;
        private final boolean perHead;
        private final int group;

        private SidePlane(String name, KvDType dtype, int width, boolean perHead, int group) {
            this.name = name;
            this.dtype = dtype;
            this.width = width;
            this.perHead = perHead;
            this.group = group;
        }

        /** A plane of {@code width} values per row. @throws IllegalArgumentException for a blank name or bad width */
        public static SidePlane of(String name, KvDType dtype, int width) {
            if (width <= 0) {
                throw new IllegalArgumentException("side plane " + name + " needs a positive width, got " + width);
            }
            checkName(name);
            dtype.rowBytes(width);
            return new SidePlane(name.trim(), dtype, width, false, 0);
        }

        /** A plane of one value per KV head: {@link #widthFor(int)} multiplies it by the cache's head count. */
        public static SidePlane perHead(String name, KvDType dtype) {
            checkName(name);
            dtype.rowBytes(1);
            return new SidePlane(name.trim(), dtype, 1, true, 0);
        }

        /**
         * A plane of one value per {@code values} values of a row (one E4M3 scale per 16 values, for example):
         * {@link #widthFor(int, int)} divides the row width by {@code values}.
         *
         * @throws IllegalArgumentException for a non-positive group or a blank name
         */
        public static SidePlane grouped(String name, KvDType dtype, int values) {
            if (values <= 0) {
                throw new IllegalArgumentException("side plane " + name + " needs a positive group, got " + values);
            }
            checkName(name);
            dtype.rowBytes(1);
            return new SidePlane(name.trim(), dtype, 0, false, values);
        }

        private static void checkName(String name) {
            if (name == null || name.trim().isEmpty()) {
                throw new IllegalArgumentException("side plane needs a name");
            }
        }

        public String name() { return name; }
        public KvDType dtype() { return dtype; }
        /** Values of one plane row for a fixed plane, or 1 for a per-head plane. */
        public int width() { return width; }
        /** True when the plane holds one value per KV head. */
        public boolean perHead() { return perHead; }
        /** True when the plane holds one value per {@link #group()} values of a row. */
        public boolean grouped() { return group > 0; }
        /** Values of a row per plane value of a grouped plane; 0 for a fixed or per-head plane. */
        public int group() { return group; }

        /** Values of one plane row of a cache with {@code heads} KV heads. */
        public int widthFor(int heads) {
            if (!perHead) {
                if (grouped()) {
                    throw new IllegalArgumentException("grouped side plane " + name + " needs the row width");
                }
                return width;
            }
            if (heads <= 0) {
                throw new IllegalArgumentException("per-head side plane " + name + " needs a positive head count, got "
                        + heads);
            }
            return width * heads;
        }

        /**
         * Values of one plane row of a cache whose rows hold {@code rowWidth} values across {@code heads} KV heads.
         *
         * @throws IllegalArgumentException for a row width that does not tile a grouped plane
         */
        public int widthFor(int rowWidth, int heads) {
            if (!grouped()) {
                return widthFor(heads);
            }
            if (rowWidth <= 0 || rowWidth % group != 0) {
                throw new IllegalArgumentException("side plane " + name + " holds one value per " + group + " values;"
                        + " the row width " + rowWidth + " is not a multiple of it");
            }
            return rowWidth / group;
        }

        /** Bytes of one plane row of a cache with {@code heads} KV heads. */
        public long rowBytesFor(int heads) {
            return dtype.rowBytes(widthFor(heads));
        }

        /** Bytes of one plane row of a cache of {@code rowWidth} values across {@code heads} KV heads. */
        public long rowBytesFor(int rowWidth, int heads) {
            return dtype.rowBytes(widthFor(rowWidth, heads));
        }

        @Override
        public String toString() {
            return name + "[" + (perHead ? "per head" : grouped() ? "per " + group + " values" : width + " " + dtype)
                    + "]";
        }
    }

    /**
     * The per-row planes this dtype stores outside the row, one instance per plane in declaration order: the
     * {@link #SCALE_PLANE per-head scale} plane of the per-token-head types, the NVFP4 group scales and the
     * TurboQuant / DeepSeek MLA planes of the vendor formats, empty for every other type.
     */
    public List<SidePlane> sidePlanes() {
        return perTokenHead ? PER_TOKEN_HEAD_PLANES : planes;
    }

    /** Every KV type, in declaration order. */
    public static List<KvDType> all() {
        return ALL;
    }

    /**
     * The types that have their own {@code LLM_KV_Store_<type>} / flash kernel variants: every type, the vendor
     * formats included. A kernel family that implements a subset filters on {@link #hasShiftKernel()} /
     * {@link #hasSparseKernel()}.
     */
    public static List<KvDType> kernelTypes() {
        return ALL;
    }

    /** The kernel variants {@code prefix + t.kernelSuffix()} of {@link #kernelTypes()}, in declaration order. */
    public static List<String> kernelVariants(String prefix) {
        return kernelVariants(prefix, t -> true);
    }

    /** The kernel variants of the types {@code types} accepts among {@link #kernelTypes()}, in declaration order. */
    public static List<String> kernelVariants(String prefix, java.util.function.Predicate<KvDType> types) {
        List<String> out = new ArrayList<>();
        for (KvDType t : ALL) {
            if (types.test(t)) {
                out.add(t.kernel(prefix));
            }
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * The accepted user-facing names, lower case as llama.cpp and vLLM spell them ({@code f32, f16, bf16, q8_0, ...,
     * fp8_e4m3, ..., nvfp4, fp8_ds_mla, turboquant_k8v4, ...}). The per-stream TurboQuant constants are internal and
     * are not listed; {@link #parse} accepts them too.
     */
    public static String names() {
        StringBuilder sb = new StringBuilder();
        for (KvDType t : ALL) {
            if (!t.turboQuantKey() && !t.turboQuantValue()) {
                appendName(sb, t);
            }
        }
        appendName(sb, "fp8_inc");
        for (String preset : KvCacheTypes.presets()) {
            appendName(sb, preset);
        }
        return sb.toString();
    }

    private static void appendName(StringBuilder sb, KvDType t) {
        appendName(sb, t.name().toLowerCase(Locale.ROOT));
    }

    private static void appendName(StringBuilder sb, String name) {
        if (sb.length() > 0) {
            sb.append(", ");
        }
        sb.append(name);
    }

    /**
     * Parses a KV type name, case-insensitive: {@code f32}, {@code f16}, {@code bf16}, {@code q8_0}, {@code q4_0},
     * {@code q4_1}, {@code iq4_nl}, {@code q5_0}, {@code q5_1}, {@code fp8_e4m3}, {@code fp8_e5m2},
     * {@code fp8_per_token_head}, {@code int8_per_token_head}, {@code int4_per_token_head} (the ggml type names of
     * llama.cpp common/arg.cpp:304-314 and the {@code CacheDType} names of vLLM vllm/config/cache.py), and the vendor
     * formats {@code fp8_inc}, {@code nvfp4}, {@code nvfp4_4over6}, {@code fp8_ds_mla}, {@code nvfp4_ds_mla},
     * {@code turboquant_k8v4}, {@code turboquant_4bit_nc}, {@code turboquant_k3v4_nc}, {@code turboquant_3bit_nc};
     * {@code q8} is accepted for Q8_0 and {@code fp8} for FP8_E4M3 (the dtype vLLM's {@code fp8} names on a CUDA
     * device). {@code fp8_inc} is Intel Gaudi's name for the E4M3 dtype, so it is the same cache as {@code fp8_e4m3}.
     * The TurboQuant presets name a key and a value stream together and are only accepted where both caches are set
     * at once ({@link KvCacheTypes#parse}).
     *
     * @param property name of the property or flag being parsed, for the error message
     * @throws IllegalArgumentException for any other value, or for a TurboQuant preset name
     */
    public static KvDType parse(String value, String property) {
        if (value == null) {
            throw new IllegalArgumentException(property + " is not set");
        }
        String v = value.trim().toUpperCase(Locale.ROOT);
        if (v.equals("Q8")) {
            return Q8_0;
        }
        if (v.equals("FP8") || v.equals("FP8_INC")) {
            return FP8_E4M3;
        }
        if (v.startsWith("TURBOQUANT_")) {
            throw new IllegalArgumentException(property + "=" + value + " names a TurboQuant key and value stream"
                    + " together; set both caches with " + MemoryConfig.PROP_KV_TYPE + " (or the pair form "
                    + KvCacheTypes.names() + ")");
        }
        for (KvDType t : ALL) {
            if (t.name().equals(v)) {
                return t;
            }
        }
        throw new IllegalArgumentException(property + "=" + value + " is not a supported KV cache type; use one of "
                + names());
    }
}

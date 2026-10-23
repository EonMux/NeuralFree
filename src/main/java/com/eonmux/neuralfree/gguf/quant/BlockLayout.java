package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;

/**
 * Block geometry of every ggml storage type: elements and bytes per block, plus the byte offset of
 * every field inside each block struct of llama.cpp ggml/src/ggml-common.h (lines 178-459). The field
 * offsets are the single source of truth for the CPU decoders and the CPU vec-dot kernels.
 * <p>
 * An instance describes one {@link GGMLType}; instances are immutable and cached.
 */
public final class BlockLayout {

    /** Super-block size of the K, IQ and TQ families ({@code QK_K}, llama.cpp ggml-common.h:89). */
    public static final int QK_K = 256;
    /** Packed 6-bit scale bytes of Q4_K / Q5_K ({@code K_SCALE_SIZE}, llama.cpp ggml-common.h:90). */
    public static final int K_SCALE_SIZE = 12;

    public static final int QK1_0 = 128;
    public static final int QK2_0 = 64;
    public static final int QK4_0 = 32;
    public static final int QK4_1 = 32;
    public static final int QK5_0 = 32;
    public static final int QK5_1 = 32;
    public static final int QK8_0 = 32;
    public static final int QK8_1 = 32;
    public static final int QK4_NL = 32;
    public static final int QK_MXFP4 = 32;
    public static final int QK_NVFP4 = 64;
    /** Elements per NVFP4 UE4M3 scale ({@code QK_NVFP4_SUB}). */
    public static final int QK_NVFP4_SUB = 16;

    // block_q1_0 {half d; u8 qs[16]} (llama.cpp ggml-common.h:180-184)
    public static final int Q1_0_D = 0, Q1_0_QS = 2, Q1_0_BYTES = 18;
    // block_q2_0 {half d; u8 qs[16]} (llama.cpp ggml-common.h:187-191)
    public static final int Q2_0_D = 0, Q2_0_QS = 2, Q2_0_BYTES = 18;
    // block_q4_0 {half d; u8 qs[16]} (llama.cpp ggml-common.h:194-198)
    public static final int Q4_0_D = 0, Q4_0_QS = 2, Q4_0_BYTES = 18;
    // block_q4_1 {half d; half m; u8 qs[16]} (llama.cpp ggml-common.h:201-211)
    public static final int Q4_1_D = 0, Q4_1_M = 2, Q4_1_QS = 4, Q4_1_BYTES = 20;
    // block_mxfp4 {u8 e; u8 qs[16]} (llama.cpp ggml-common.h:214-218)
    public static final int MXFP4_E = 0, MXFP4_QS = 1, MXFP4_BYTES = 17;
    // block_nvfp4 {u8 d[4]; u8 qs[32]} (llama.cpp ggml-common.h:221-226)
    public static final int NVFP4_D = 0, NVFP4_QS = 4, NVFP4_BYTES = 36;
    // block_q5_0 {half d; u8 qh[4]; u8 qs[16]} (llama.cpp ggml-common.h:229-234)
    public static final int Q5_0_D = 0, Q5_0_QH = 2, Q5_0_QS = 6, Q5_0_BYTES = 22;
    // block_q5_1 {half d; half m; u8 qh[4]; u8 qs[16]} (llama.cpp ggml-common.h:237-248)
    public static final int Q5_1_D = 0, Q5_1_M = 2, Q5_1_QH = 4, Q5_1_QS = 8, Q5_1_BYTES = 24;
    // block_q8_0 {half d; i8 qs[32]} (llama.cpp ggml-common.h:251-255)
    public static final int Q8_0_D = 0, Q8_0_QS = 2, Q8_0_BYTES = 34;
    // block_q8_1 {half d; half s; i8 qs[32]} (llama.cpp ggml-common.h:258-268)
    public static final int Q8_1_D = 0, Q8_1_S = 2, Q8_1_QS = 4, Q8_1_BYTES = 36;
    // block_tq1_0 {u8 qs[48]; u8 qh[4]; half d} (llama.cpp ggml-common.h:276-280)
    public static final int TQ1_0_QS = 0, TQ1_0_QS_BYTES = 48, TQ1_0_QH = 48, TQ1_0_QH_BYTES = 4,
            TQ1_0_D = 52, TQ1_0_BYTES = 54;
    // block_tq2_0 {u8 qs[64]; half d} (llama.cpp ggml-common.h:284-287)
    public static final int TQ2_0_QS = 0, TQ2_0_D = 64, TQ2_0_BYTES = 66;
    // block_q2_K {u8 scales[16]; u8 qs[64]; half d; half dmin} (llama.cpp ggml-common.h:298-308)
    public static final int Q2_K_SCALES = 0, Q2_K_QS = 16, Q2_K_D = 80, Q2_K_DMIN = 82, Q2_K_BYTES = 84;
    // block_q3_K {u8 hmask[32]; u8 qs[64]; u8 scales[12]; half d} (llama.cpp ggml-common.h:315-320)
    public static final int Q3_K_HMASK = 0, Q3_K_QS = 32, Q3_K_SCALES = 96, Q3_K_D = 108, Q3_K_BYTES = 110;
    // block_q4_K {half d; half dmin; u8 scales[12]; u8 qs[128]} (llama.cpp ggml-common.h:327-337)
    public static final int Q4_K_D = 0, Q4_K_DMIN = 2, Q4_K_SCALES = 4, Q4_K_QS = 16, Q4_K_BYTES = 144;
    // block_q5_K {half d; half dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]} (llama.cpp ggml-common.h:344-355)
    public static final int Q5_K_D = 0, Q5_K_DMIN = 2, Q5_K_SCALES = 4, Q5_K_QH = 16, Q5_K_QS = 48,
            Q5_K_BYTES = 176;
    // block_q6_K {u8 ql[128]; u8 qh[64]; i8 scales[16]; half d} (llama.cpp ggml-common.h:362-367)
    public static final int Q6_K_QL = 0, Q6_K_QH = 128, Q6_K_SCALES = 192, Q6_K_D = 208, Q6_K_BYTES = 210;
    // block_q8_K {float d; i8 qs[256]; i16 bsums[16]} (llama.cpp ggml-common.h:371-375)
    public static final int Q8_K_D = 0, Q8_K_QS = 4, Q8_K_BSUMS = 260, Q8_K_BYTES = 292;
    // block_iq2_xxs {half d; u16 qs[32]} (llama.cpp ggml-common.h:381-384)
    public static final int IQ2_XXS_D = 0, IQ2_XXS_QS = 2, IQ2_XXS_BYTES = 66;
    // block_iq2_xs {half d; u16 qs[32]; u8 scales[8]} (llama.cpp ggml-common.h:388-392)
    public static final int IQ2_XS_D = 0, IQ2_XS_QS = 2, IQ2_XS_SCALES = 66, IQ2_XS_BYTES = 74;
    // block_iq2_s {half d; u8 qs[64] (grid low bits [32] then signs [32]); u8 qh[8]; u8 scales[8]}
    // (llama.cpp ggml-common.h:396-401)
    public static final int IQ2_S_D = 0, IQ2_S_QS = 2, IQ2_S_SIGNS = 34, IQ2_S_QH = 66, IQ2_S_SCALES = 74,
            IQ2_S_BYTES = 82;
    // block_iq3_xxs {half d; u8 qs[96] (grid indices [64] then scales_and_signs [32])}
    // (llama.cpp ggml-common.h:407-410)
    public static final int IQ3_XXS_D = 0, IQ3_XXS_QS = 2, IQ3_XXS_SAS = 66, IQ3_XXS_BYTES = 98;
    // block_iq3_s {half d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4]} (llama.cpp ggml-common.h:415-421)
    public static final int IQ3_S_D = 0, IQ3_S_QS = 2, IQ3_S_QH = 66, IQ3_S_SIGNS = 74, IQ3_S_SCALES = 106,
            IQ3_S_BYTES = 110;
    // block_iq1_s {half d; u8 qs[32]; u16 qh[8]} (llama.cpp ggml-common.h:425-429)
    public static final int IQ1_S_D = 0, IQ1_S_QS = 2, IQ1_S_QH = 34, IQ1_S_BYTES = 50;
    // block_iq1_m {u8 qs[32]; u8 qh[16]; u8 scales[8]} (llama.cpp ggml-common.h:433-437)
    public static final int IQ1_M_QS = 0, IQ1_M_QH = 32, IQ1_M_SCALES = 48, IQ1_M_BYTES = 56;
    // block_iq4_nl {half d; u8 qs[16]} (llama.cpp ggml-common.h:447-451)
    public static final int IQ4_NL_D = 0, IQ4_NL_QS = 2, IQ4_NL_BYTES = 18;
    // block_iq4_xs {half d; u16 scales_h; u8 scales_l[4]; u8 qs[128]} (llama.cpp ggml-common.h:454-459)
    public static final int IQ4_XS_D = 0, IQ4_XS_SCALES_H = 2, IQ4_XS_SCALES_L = 4, IQ4_XS_QS = 8,
            IQ4_XS_BYTES = 136;

    private static final Map<GGMLType, BlockLayout> LAYOUTS = buildLayouts();

    private final GGMLType type;
    private final int blockElements;
    private final int blockBytes;

    private BlockLayout(GGMLType type) {
        this.type = type;
        this.blockElements = type.getBlockSize();
        this.blockBytes = type.getTypeSize();
    }

    private static Map<GGMLType, BlockLayout> buildLayouts() {
        Map<GGMLType, BlockLayout> m = new EnumMap<>(GGMLType.class);
        for (GGMLType t : GGMLType.values()) {
            if (t.getBlockSize() > 0 && t.getTypeSize() > 0) {
                m.put(t, new BlockLayout(t));
            }
        }
        return Collections.unmodifiableMap(m);
    }

    /** True when {@code type} is a real storage type with a defined block geometry. */
    public static boolean isDefined(GGMLType type) {
        return type != null && LAYOUTS.containsKey(type);
    }

    /**
     * Returns the layout of {@code type}.
     *
     * @throws IllegalArgumentException for {@code UNKNOWN} and the removed types Q4_2 / Q4_3
     */
    public static BlockLayout of(GGMLType type) {
        if (type == null) {
            throw new IllegalArgumentException("ggml type is null");
        }
        BlockLayout layout = LAYOUTS.get(type);
        if (layout == null) {
            throw new IllegalArgumentException("ggml type " + type + " (id " + type.getId()
                    + ") has no storage layout; it is unknown or was removed from ggml");
        }
        return layout;
    }

    public GGMLType type() {
        return type;
    }

    /** Elements per block; 1 for the plain element types (F32, F16, BF16, F64, I8..I64). */
    public int blockElements() {
        return blockElements;
    }

    /** Bytes per block ({@code sizeof(block_x)}, or the element size for plain types). */
    public int blockBytes() {
        return blockBytes;
    }

    /**
     * Number of blocks covering exactly {@code nElements}.
     *
     * @throws IllegalArgumentException when {@code nElements} is negative or not a whole number of blocks
     */
    public long blocksForElements(long nElements) {
        if (nElements < 0) {
            throw new IllegalArgumentException("negative element count " + nElements + " for " + type);
        }
        if (nElements % blockElements != 0) {
            throw new IllegalArgumentException(nElements + " elements is not a multiple of the " + type
                    + " block size " + blockElements);
        }
        return nElements / blockElements;
    }

    /** Bytes holding exactly {@code nElements}; same validation as {@link #blocksForElements(long)}. */
    public long bytesForElements(long nElements) {
        return blocksForElements(nElements) * blockBytes;
    }

    /** Packed bytes of one row of {@code cols} elements; {@code cols} must be a whole number of blocks. */
    public long rowBytes(long cols) {
        return bytesForElements(cols);
    }

    /**
     * Validates that {@code [byteOff, byteOff + bytes)} lies inside {@code src}'s limit and below 2^31,
     * and returns {@code byteOff} as an absolute buffer index.
     *
     * @throws IndexOutOfBoundsException naming the range and the buffer limit
     */
    public static int checkedIndex(ByteBuffer src, long byteOff, long bytes) {
        if (src == null) {
            throw new IllegalArgumentException("source buffer is null");
        }
        if (byteOff < 0 || bytes < 0 || bytes > src.limit() || byteOff > src.limit() - bytes) {
            throw new IndexOutOfBoundsException("byte range [" + byteOff + ", " + (byteOff + bytes)
                    + ") is outside the source buffer limit " + src.limit());
        }
        return (int) byteOff;
    }

    /** Returns {@code src} when it is little-endian, otherwise a little-endian duplicate sharing its content. */
    public static ByteBuffer littleEndian(ByteBuffer src) {
        return src.order() == ByteOrder.LITTLE_ENDIAN ? src : src.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public String toString() {
        return "BlockLayout{" + type + ", " + blockElements + " elements, " + blockBytes + " bytes}";
    }
}

package com.eonmux.neuralfree.gguf;

/**
 * ggml storage types. Ids mirror {@code enum ggml_type} (llama.cpp ggml/include/ggml.h:389-434); {@code typeSize}
 * is the byte size of one block ({@code sizeof(block_x)}, llama.cpp ggml/src/ggml-common.h) and {@code blockSize}
 * the elements per block. Q4_2 and Q4_3 were removed from ggml and keep zero sizes, as does UNKNOWN.
 * <p>
 * Ids 1000-1099 are the NeuralFree extension range ({@link #isExtension()}): block types that carry quantized
 * checkpoint formats exactly, with their scales inside each block. Ids 1000-1005 are assigned below; new extension
 * types take ids from 1006 on. Their block layouts (little-endian) are:
 * <ul>
 *   <li>{@link #F8_E4M3}: {@code u8 qs[32]}, E4M3FN codes; value {@code e4m3(qs[j])}.</li>
 *   <li>{@link #F8_E4M3_B128}: {@code f32 d; u8 qs[128]}; value {@code d * e4m3(qs[j])}.</li>
 *   <li>{@link #MXFP8}: {@code u8 e; u8 qs[32]}; value {@code 2^(e-127) * e4m3(qs[j])} (E8M0 scale).</li>
 *   <li>{@link #I8_B128}: {@code f32 d; i8 qs[128]}; value {@code d * qs[j]}.</li>
 *   <li>{@link #NF4} and {@link #FP4_BNB}: {@code f32 absmax; u8 qs[32]}, byte {@code j} holding element {@code j}
 *       in its low nibble and element {@code j + 32} in its high nibble; value {@code code[q] * absmax} over the
 *       bitsandbytes NF4 or FP4 codebook.</li>
 * </ul>
 */
public enum GGMLType {
    F32(0, 4, 1),
    F16(1, 2, 1),
    Q4_0(2, 18, 32),
    Q4_1(3, 20, 32),
    Q4_2(4, 0, 0),
    Q4_3(5, 0, 0),
    Q5_0(6, 22, 32),
    Q5_1(7, 24, 32),
    Q8_0(8, 34, 32),
    Q8_1(9, 36, 32),
    Q2_K(10, 84, 256),
    Q3_K(11, 110, 256),
    Q4_K(12, 144, 256),
    Q5_K(13, 176, 256),
    Q6_K(14, 210, 256),
    Q8_K(15, 292, 256),
    IQ2_XXS(16, 66, 256),
    IQ2_XS(17, 74, 256),
    IQ3_XXS(18, 98, 256),
    IQ1_S(19, 50, 256),
    IQ4_NL(20, 18, 32),
    IQ3_S(21, 110, 256),
    IQ2_S(22, 82, 256),
    IQ4_XS(23, 136, 256),
    I8(24, 1, 1),
    I16(25, 2, 1),
    I32(26, 4, 1),
    I64(27, 8, 1),
    F64(28, 8, 1),
    IQ1_M(29, 56, 256),
    BF16(30, 2, 1),
    TQ1_0(34, 54, 256),
    TQ2_0(35, 66, 256),
    MXFP4(39, 17, 32),
    NVFP4(40, 36, 64),
    Q1_0(41, 18, 128),
    Q2_0(42, 18, 64),
    F8_E4M3(1000, 32, 32),
    F8_E4M3_B128(1001, 132, 128),
    MXFP8(1002, 33, 32),
    I8_B128(1003, 132, 128),
    NF4(1004, 36, 64),
    FP4_BNB(1005, 36, 64),
    UNKNOWN(-1, 0, 0);

    /** First id of the NeuralFree extension range. */
    public static final int EXTENSION_ID_MIN = 1000;
    /** Last id of the NeuralFree extension range. */
    public static final int EXTENSION_ID_MAX = 1099;

    private final int id;
    private final int blockSize;
    private final int typeSize;

    GGMLType(int id, int typeSize, int blockSize) {
        this.id = id;
        this.typeSize = typeSize;
        this.blockSize = blockSize;
    }

    public int getId() {
        return id;
    }

    public int getBlockSize() {
        return blockSize;
    }

    public int getTypeSize() {
        return typeSize;
    }

    public boolean isQuantized() {
        return blockSize > 1;
    }

    /**
     * True for the NeuralFree extension types (ids 1000-1099), which ggml does not define. A GGUF file may hold
     * tensors of these types only when it carries the BOOL key {@code neuralfree.extension_types = true}.
     */
    public boolean isExtension() {
        return id >= EXTENSION_ID_MIN && id <= EXTENSION_ID_MAX;
    }

    public long getBytesForElements(long numElements) {
        if (blockSize <= 1) {
            return numElements * typeSize;
        }
        long numBlocks = (numElements + blockSize - 1) / blockSize;
        return numBlocks * typeSize;
    }

    public static GGMLType fromId(int id) {
        for (GGMLType type : values()) {
            if (type.id == id) {
                return type;
            }
        }
        return UNKNOWN;
    }
}

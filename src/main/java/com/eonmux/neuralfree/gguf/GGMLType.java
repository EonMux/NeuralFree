package com.eonmux.neuralfree.gguf;

/**
 * ggml storage types. Ids mirror {@code enum ggml_type} (llama.cpp ggml/include/ggml.h:389-434); {@code typeSize}
 * is the byte size of one block ({@code sizeof(block_x)}, llama.cpp ggml/src/ggml-common.h) and {@code blockSize}
 * the elements per block. Q4_2 and Q4_3 were removed from ggml and keep zero sizes, as does UNKNOWN.
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
    UNKNOWN(-1, 0, 0);

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

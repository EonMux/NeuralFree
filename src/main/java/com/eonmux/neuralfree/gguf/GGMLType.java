package com.eonmux.neuralfree.gguf;

/**
 * ggml storage types. Ids mirror {@code enum ggml_type} (llama.cpp ggml/include/ggml.h:389-436); {@code typeSize}
 * is the byte size of one block ({@code sizeof(block_x)}, llama.cpp ggml/src/ggml-common.h) and {@code blockSize}
 * the elements per block. Q4_2 and Q4_3 were removed from ggml and keep zero sizes, as does UNKNOWN.
 * <p>
 * IQ3_K and IQ3_KS use the ids 43 / 44 of the ik_llama.cpp family types (Avalon ggml/include/ggml.h:433-434).
 * Id 43 is shared with {@link #F8_E4M3FN}: mainline llama.cpp assigns it to its F8_E4M3, the per-element format
 * NF reads as F8_E4M3FN, so {@link #fromId} resolves 43 to F8_E4M3FN and {@code GGUFFileParser} re-reads a
 * tensor of id 43 as IQ3_K when its data section slot holds the 110-bytes-per-256-elements block format
 * instead of 1 byte per element. Id 44 is unassigned in mainline ggml (its COUNT is 44); a tensor of id 44
 * laid out per element is rejected by name. NF itself never reads F8_E4M3 tensors from GGUF files (its F8_E4M3
 * is the extension type below).
 * <p>
 * Ids 1000-1099 are the NeuralFree extension range ({@link #isExtension()}): block types that carry quantized
 * checkpoint formats exactly, with their scales inside each block. Ids 1000-1010 are assigned below; new extension
 * types take ids from 1011 on. Their block layouts (little-endian) are:
 * <ul>
 *   <li>{@link #F8_E4M3}: {@code u8 qs[32]}, E4M3FN codes; value {@code e4m3(qs[j])}. The per-element
 *       counterpart ggml itself defines is {@link #F8_E4M3FN}, which stores one code per element.</li>
 *   <li>{@link #F8_E4M3_B128}: {@code f32 d; u8 qs[128]}; value {@code d * e4m3(qs[j])}.</li>
 *   <li>{@link #F8_E4M3_B32}: {@code f32 d; u8 qs[32]}; value {@code d * e4m3(qs[j])}: the 32-element form of
 *       F8_E4M3_B128, for FP8 rows that are not a whole number of 128-element blocks (each run carries the scale
 *       of the 128-wide scale block it lies in).</li>
 *   <li>{@link #MXFP8}: {@code u8 e; u8 qs[32]}; value {@code 2^(e-127) * e4m3(qs[j])} (E8M0 scale).</li>
 *   <li>{@link #I8_B128}: {@code f32 d; i8 qs[128]}; value {@code d * qs[j]}.</li>
 *   <li>{@link #NF4} and {@link #FP4_BNB}: {@code f32 absmax; u8 qs[32]}, byte {@code j} holding element {@code j}
 *       in its low nibble and element {@code j + 32} in its high nibble; value {@code code[q] * absmax} over the
 *       bitsandbytes NF4 or FP4 codebook.</li>
 *   <li>{@link #F8_E5M2}: {@code u8 qs[32]}, E5M2 codes; value {@code e5m2(qs[j])} (IEEE half high bytes, with
 *       the IEEE infinities and NaNs), scaled by the tensor's {@code .scale} sibling like {@link #F8_E4M3}.</li>
 *   <li>{@link #MXFP6_E3M2} and {@link #MXFP6_E2M3}: {@code u8 e (E8M0); u8 qs[24]}, 32 packed 6-bit codes
 *       (element {@code j} = code {@code j}, four codes per three bytes, little-endian bit order); value
 *       {@code 2^(e-127) * e3m2(qs[j])} or {@code 2^(e-127) * e2m3(qs[j])} (Quark OCP-MX FP6, 6.25 bits per
 *       weight). {@code e == 0xFF} is the NaN E8M0 reservation and is rejected at load.</li>
 *   <li>{@link #W12}: the lossless 12-bit streaming form of bfloat16 target weights, one 96-byte plane
 *       footprint per 64 elements in a per-row record ({@code W12Codec}; a device-resident form only, never
 *       written to GGUF files). Each row holds the Q region (one byte per element, bits 7..4 the 4-bit offset
 *       {@code c} into the row's 16-binade exponent window {@code [E-15, E]}, bit 3 the sign, bits 2..0
 *       mantissa bits 6..4), the B region ({@code u8 b[32]}, byte {@code j} holding mantissa bits 3..0 of
 *       element {@code 2j} in its low nibble and of element {@code 2j + 1} in its high nibble), and 12 bytes of
 *       metadata ({@code f32 scale} = {@code 2^(E-127)}, then the row's exact and Q escape counts). After the
 *       row region follow two per-shard escape tables: 12-byte exact entries {@code [col][bf16 bits][pad][val]}
 *       and 8-byte Q entries {@code [col][val]}, with {@code val} the scaled true weight
 *       {@code w * 2^(127-E)}, for the elements the 12-bit region cannot represent (exponent outside the
 *       window, zero, subnormal, infinity or NaN). The row kernels decode the planes with the exponent field
 *       pinned at {@code 112 + c}, cancel the {@code 2^-15} placeholder of an escaped element with its CSR
 *       correction and scale the row sum by {@code 2^(E-127)} once, so every weight's contribution — and every
 *       escape-free row's whole matvec — is bit-exact at BF16 precision. The Q bytes alone are also the nested
 *       8-bit view of the weights: reading mantissa bits 3..0 as the fixed midpoint nibble {@code 0x8} rounds
 *       every in-band element to its window binade's sign/exponent/top-3-mantissa code, which the W12 Q-view
 *       kernels decode in registers.</li>
 * </ul>
 * <p>
 * {@link #F8_E4M3FN} is ggml's {@code GGML_TYPE_F8_E4M3} (id 43, type name {@code f8_e4m3}, llama.cpp
 * ggml/src/ggml.c:776-783): one E4M3FN code per element, no block scale; the weight's {@code .scale} sibling is the
 * output multiplier, applied after the matmul like {@link #F8_E4M3}'s. Rows of any length store because the block is
 * one element.
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
    F8_E4M3FN(43, 1, 1),
    // IQ3_K shares F8_E4M3FN's id 43 (Avalon vs mainline numbering); fromId resolves 43 to F8_E4M3FN, the
    // reading NF writes itself, and GGUFFileParser re-reads id 43 as IQ3_K from the block layout.
    IQ3_K(43, 110, 256),
    IQ3_KS(44, 104, 256),
    F8_E4M3(1000, 32, 32),
    F8_E4M3_B128(1001, 132, 128),
    MXFP8(1002, 33, 32),
    I8_B128(1003, 132, 128),
    NF4(1004, 36, 64),
    FP4_BNB(1005, 36, 64),
    F8_E5M2(1006, 32, 32),
    MXFP6_E3M2(1007, 25, 32),
    MXFP6_E2M3(1008, 25, 32),
    F8_E4M3_B32(1009, 36, 32),
    W12(1010, 96, 64),
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

    /**
     * Whether the type is a device-resident form only ({@link #W12}): weights take it on the device, and no GGUF file
     * stores it, so the header reader and the writer refuse a tensor of it by name.
     */
    public boolean isDeviceOnly() {
        return this == W12;
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

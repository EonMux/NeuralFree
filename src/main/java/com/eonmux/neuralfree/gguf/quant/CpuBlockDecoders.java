package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Set;

/**
 * The single CPU decoding entry point for every ggml storage type: bit-exact ports of the
 * {@code dequantize_row_*} reference functions of llama.cpp ggml/src/ggml-quants.c and the element
 * conversions of llama.cpp ggml/src/ggml.c. Each family lives in its own class
 * ({@link LegacyBlockDecoders}, {@link KQuantBlockDecoders}, {@link IQuantBlockDecoders},
 * {@link TernaryBlockDecoders}, {@link FpBlockDecoders}); this class validates ranges, normalizes byte
 * order and dispatches.
 * <p>
 * Thread-safe and stateless. Source buffers are read with absolute indexing only; their position,
 * limit and byte order are never modified.
 */
public final class CpuBlockDecoders {

    /** Decodes consecutive blocks of one storage type to F32. */
    @FunctionalInterface
    public interface Decoder {
        /**
         * Decodes {@code nBlocks} blocks starting at absolute byte {@code byteOff} of {@code src} into
         * {@code dst[dstOff .. dstOff + nBlocks * blockElements)}.
         *
         * @throws IndexOutOfBoundsException when the source range exceeds {@code src.limit()} or the
         *                                   destination range exceeds {@code dst.length}
         * @throws IllegalArgumentException  when {@code nBlocks} is negative or an argument is null
         */
        void decode(ByteBuffer src, long byteOff, float[] dst, int dstOff, int nBlocks);
    }

    /** A family kernel: little-endian buffer, validated int offset. */
    @FunctionalInterface
    private interface Kernel {
        void run(ByteBuffer le, int off, float[] dst, int dstOff, int nBlocks);
    }

    private static final Map<GGMLType, Decoder> DECODERS = buildDecoders();

    private CpuBlockDecoders() {
    }

    private static Map<GGMLType, Decoder> buildDecoders() {
        Map<GGMLType, Decoder> m = new EnumMap<>(GGMLType.class);
        put(m, GGMLType.F32, FpBlockDecoders::decodeF32);
        put(m, GGMLType.F16, FpBlockDecoders::decodeF16);
        put(m, GGMLType.BF16, FpBlockDecoders::decodeBf16);
        put(m, GGMLType.F64, FpBlockDecoders::decodeF64);
        put(m, GGMLType.I8, FpBlockDecoders::decodeI8);
        put(m, GGMLType.I16, FpBlockDecoders::decodeI16);
        put(m, GGMLType.I32, FpBlockDecoders::decodeI32);
        put(m, GGMLType.I64, FpBlockDecoders::decodeI64);
        put(m, GGMLType.MXFP4, FpBlockDecoders::decodeMxfp4);
        put(m, GGMLType.NVFP4, FpBlockDecoders::decodeNvfp4);
        put(m, GGMLType.Q1_0, FpBlockDecoders::decodeQ1_0);
        put(m, GGMLType.Q2_0, FpBlockDecoders::decodeQ2_0);

        put(m, GGMLType.Q4_0, LegacyBlockDecoders::decodeQ4_0);
        put(m, GGMLType.Q4_1, LegacyBlockDecoders::decodeQ4_1);
        put(m, GGMLType.Q5_0, LegacyBlockDecoders::decodeQ5_0);
        put(m, GGMLType.Q5_1, LegacyBlockDecoders::decodeQ5_1);
        put(m, GGMLType.Q8_0, LegacyBlockDecoders::decodeQ8_0);
        put(m, GGMLType.Q8_1, LegacyBlockDecoders::decodeQ8_1);

        put(m, GGMLType.Q2_K, KQuantBlockDecoders::decodeQ2_K);
        put(m, GGMLType.Q3_K, KQuantBlockDecoders::decodeQ3_K);
        put(m, GGMLType.Q4_K, KQuantBlockDecoders::decodeQ4_K);
        put(m, GGMLType.Q5_K, KQuantBlockDecoders::decodeQ5_K);
        put(m, GGMLType.Q6_K, KQuantBlockDecoders::decodeQ6_K);
        put(m, GGMLType.Q8_K, KQuantBlockDecoders::decodeQ8_K);

        put(m, GGMLType.IQ2_XXS, IQuantBlockDecoders::decodeIq2Xxs);
        put(m, GGMLType.IQ2_XS, IQuantBlockDecoders::decodeIq2Xs);
        put(m, GGMLType.IQ2_S, IQuantBlockDecoders::decodeIq2S);
        put(m, GGMLType.IQ3_XXS, IQuantBlockDecoders::decodeIq3Xxs);
        put(m, GGMLType.IQ3_S, IQuantBlockDecoders::decodeIq3S);
        put(m, GGMLType.IQ1_S, IQuantBlockDecoders::decodeIq1S);
        put(m, GGMLType.IQ1_M, IQuantBlockDecoders::decodeIq1M);
        put(m, GGMLType.IQ4_NL, IQuantBlockDecoders::decodeIq4Nl);
        put(m, GGMLType.IQ4_XS, IQuantBlockDecoders::decodeIq4Xs);

        put(m, GGMLType.TQ1_0, TernaryBlockDecoders::decodeTq1_0);
        put(m, GGMLType.TQ2_0, TernaryBlockDecoders::decodeTq2_0);
        return Collections.unmodifiableMap(m);
    }

    private static void put(Map<GGMLType, Decoder> m, GGMLType type, Kernel kernel) {
        m.put(type, new CheckedDecoder(BlockLayout.of(type), kernel));
    }

    /** True when a CPU decoder exists for {@code type}; false only for UNKNOWN and the removed Q4_2 / Q4_3. */
    public static boolean supports(GGMLType type) {
        return type != null && DECODERS.containsKey(type);
    }

    /** All storage types with a CPU decoder. */
    public static Set<GGMLType> supportedTypes() {
        return DECODERS.keySet();
    }

    /**
     * Returns the decoder of {@code type}.
     *
     * @throws UnsupportedOperationException naming the type when no decoder exists
     */
    public static Decoder forType(GGMLType type) {
        Decoder d = type == null ? null : DECODERS.get(type);
        if (d == null) {
            throw new UnsupportedOperationException("no CPU decoder for ggml type " + type
                    + (type == null ? "" : " (id " + type.getId() + ")")
                    + ": the type is unknown or was removed from ggml");
        }
        return d;
    }

    /** Decodes {@code nBlocks} blocks of {@code type}; see {@link Decoder#decode}. */
    public static void decode(GGMLType type, ByteBuffer src, long byteOff, float[] dst, int dstOff, int nBlocks) {
        forType(type).decode(src, byteOff, dst, dstOff, nBlocks);
    }

    /**
     * Decodes {@code nElements} elements of {@code type} starting at byte {@code byteOff}.
     *
     * @throws IllegalArgumentException when {@code nElements} is not a whole number of blocks
     */
    public static void decodeElements(GGMLType type, ByteBuffer src, long byteOff, float[] dst, int dstOff,
                                      long nElements) {
        Decoder decoder = forType(type);
        long blocks = BlockLayout.of(type).blocksForElements(nElements);
        if (blocks > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(nElements + " " + type + " elements exceed one decode call");
        }
        decoder.decode(src, byteOff, dst, dstOff, (int) blocks);
    }

    /**
     * Decodes a whole tensor held in a heap array into a new float array.
     *
     * @param raw       tensor bytes, starting at index 0
     * @param nElements element count; a whole number of blocks and at most {@code Integer.MAX_VALUE - 8}
     */
    public static float[] dequantize(byte[] raw, GGMLType type, long nElements) {
        if (raw == null) {
            throw new IllegalArgumentException("raw tensor data is null");
        }
        Decoder decoder = forType(type);
        long blocks = BlockLayout.of(type).blocksForElements(nElements);
        if (nElements > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException(nElements + " " + type
                    + " elements do not fit one float array; decode the tensor in row ranges");
        }
        float[] out = new float[(int) nElements];
        decoder.decode(ByteBuffer.wrap(raw), 0L, out, 0, (int) blocks);
        return out;
    }

    /** Range checks, byte-order normalization, then the family kernel. */
    private static final class CheckedDecoder implements Decoder {
        private final BlockLayout layout;
        private final Kernel kernel;

        CheckedDecoder(BlockLayout layout, Kernel kernel) {
            this.layout = layout;
            this.kernel = kernel;
        }

        @Override
        public void decode(ByteBuffer src, long byteOff, float[] dst, int dstOff, int nBlocks) {
            if (nBlocks < 0) {
                throw new IllegalArgumentException("negative block count " + nBlocks + " for " + layout.type());
            }
            if (dst == null) {
                throw new IllegalArgumentException("destination array is null");
            }
            long outElements = (long) nBlocks * layout.blockElements();
            if (dstOff < 0 || dstOff + outElements > dst.length) {
                throw new IndexOutOfBoundsException("destination range [" + dstOff + ", " + (dstOff + outElements)
                        + ") exceeds array length " + dst.length + " decoding " + nBlocks + " " + layout.type()
                        + " blocks");
            }
            int off = BlockLayout.checkedIndex(src, byteOff, (long) nBlocks * layout.blockBytes());
            if (nBlocks == 0) {
                return;
            }
            kernel.run(BlockLayout.littleEndian(src), off, dst, dstOff, nBlocks);
        }
    }
}

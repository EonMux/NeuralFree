package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;

/**
 * The single row-quantization entry point of the offline encoders, the encode-side twin of
 * {@link CpuBlockDecoders}: dispatches one float row to the block quantizer of its type, the way
 * {@code ggml_quantize_chunk} dispatches one row to the {@code quantize_*} entry points of llama.cpp
 * ggml-quants.c. The reference path of a type runs without importance weights; the weighted path runs
 * with them (K-quants, the legacy blocks, the IQ4 searches and the grid-search IQ1/IQ2/IQ3 types; the
 * types whose entry points ignore the matrix take it and discard it, like llama.cpp does). IQ2_XXS,
 * IQ2_XS and IQ1_S cannot run without the matrix ({@link #requiresWeights}). {@code supports} names the
 * types an offline quantization can produce; every other type fails with a named error instead of a
 * silent copy.
 * <p>
 * Thread-safe and stateless.
 */
public final class RowQuantizers {

    private RowQuantizers() {
    }

    /** True when an encoder exists for {@code type}. */
    public static boolean supports(GGMLType type) {
        switch (type) {
            case F32:
            case F16:
            case BF16:
            case Q1_0:
            case Q2_0:
            case Q4_0:
            case Q4_1:
            case Q5_0:
            case Q5_1:
            case Q8_0:
            case Q8_K:
            case Q2_K:
            case Q3_K:
            case Q4_K:
            case Q5_K:
            case Q6_K:
            case IQ4_NL:
            case IQ4_XS:
            case TQ1_0:
            case TQ2_0:
            case MXFP4:
            case IQ2_XXS:
            case IQ2_XS:
            case IQ2_S:
            case IQ3_XXS:
            case IQ3_S:
            case IQ1_S:
            case IQ1_M:
                return true;
            default:
                return false;
        }
    }

    /**
     * True when the encoder of {@code type} needs importance weights for every row: IQ2_XXS, IQ2_XS and
     * IQ1_S ({@code ggml_quantize_requires_imatrix}).
     */
    public static boolean requiresWeights(GGMLType type) {
        return IqGridEncoders.requiresWeights(type) || Iq1Encoders.requiresWeights(type);
    }

    /**
     * The encoded bytes of one row of {@code nPerRow} values of {@code type}.
     *
     * @throws IllegalArgumentException when {@code type} has no encoder or the row is not a whole number
     *                                  of the type's blocks
     */
    public static int bytesPerRow(GGMLType type, int nPerRow) {
        requireSupported(type);
        requireWholeBlocks(type, nPerRow);
        return type.getTypeSize() * (nPerRow / type.getBlockSize());
    }

    /**
     * Quantizes one row of {@code nPerRow} values at {@code xOff} of {@code x} into {@code dst} at
     * {@code dstOff}, {@link #bytesPerRow} bytes; {@code weights} is the row's importance slice (one
     * value per input value) or null.
     *
     * @throws IllegalArgumentException when {@code type} has no encoder, the row is not a whole number of
     *                                  the type's blocks, or {@code weights} is null for a type that
     *                                  {@link #requiresWeights}
     */
    public static void quantizeRow(GGMLType type, float[] x, int xOff, byte[] dst, int dstOff, int nPerRow,
                                   float[] weights) {
        quantizeRow(type, x, xOff, dst, dstOff, nPerRow, weights, 0);
    }

    /**
     * {@link #quantizeRow} of a row whose importance slice starts at {@code weightsOff} of the shared
     * {@code weights} array (one expert matrix after the other, as the tool passes them).
     */
    public static void quantizeRow(GGMLType type, float[] x, int xOff, byte[] dst, int dstOff, int nPerRow,
                                   float[] weights, int weightsOff) {
        requireSupported(type);
        requireWholeBlocks(type, nPerRow);
        switch (type) {
            case F32:
            case F16:
            case BF16:
            case Q1_0:
            case Q2_0:
            case MXFP4:
            case TQ1_0:
            case TQ2_0:
            case Q8_K:
                SimpleQuantEncoders.quantizeRow(type, x, xOff, dst, dstOff, nPerRow, weights);
                return;
            case Q2_K:
            case Q3_K:
            case Q4_K:
            case Q5_K:
            case Q6_K:
                KQuantEncoders.quantizeRow(type, x, xOff, dst, dstOff, nPerRow, weights, weightsOff);
                return;
            case Q4_0:
            case Q4_1:
            case Q5_0:
            case Q5_1:
            case IQ4_NL:
            case IQ4_XS:
                LegacyQuantEncoders.quantizeRow(type, x, xOff, dst, dstOff, nPerRow, weights, weightsOff);
                return;
            case IQ2_XXS:
            case IQ2_XS:
            case IQ2_S:
            case IQ3_XXS:
            case IQ3_S:
                IqGridEncoders.quantizeRow(type, x, xOff, dst, dstOff, nPerRow, weights, weightsOff);
                return;
            case IQ1_S:
            case IQ1_M:
                Iq1Encoders.quantizeRow(type, x, xOff, dst, dstOff, nPerRow, weights, weightsOff);
                return;
            case Q8_0:
                quantizeQ8_0(x, xOff, dst, dstOff, nPerRow);
                return;
            default:
                throw new IllegalArgumentException(type + " has no row encoder");
        }
    }

    /** The Q8_0 reference quantizer, one 32-value block at a time ({@code quantize_row_q8_0_ref}). */
    private static void quantizeQ8_0(float[] x, int xOff, byte[] dst, int dstOff, int nPerRow) {
        float[] block = new float[QK8_0_BLOCK];
        for (int b = 0; b < nPerRow; b += QK8_0_BLOCK) {
            System.arraycopy(x, xOff + b, block, 0, QK8_0_BLOCK);
            KvBlockCodec.quantize(KvDType.Q8_0, block, 0, dst, dstOff + b / QK8_0_BLOCK * KvDType.Q8_0.blockBytes());
        }
    }

    private static final int QK8_0_BLOCK = 32;

    private static void requireSupported(GGMLType type) {
        if (!supports(type)) {
            throw new IllegalArgumentException(type + " has no row encoder: the quantize tool produces "
                    + "F32/F16/BF16, the legacy blocks, the K-quants Q2_K..Q6_K, IQ1_S, IQ1_M, IQ2_XXS, "
                    + "IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, MXFP4 and Q8_K");
        }
    }

    /** The row quantizers read whole blocks; the type's block size already names the constraint. */
    private static void requireWholeBlocks(GGMLType type, int nPerRow) {
        int block = type.getBlockSize();
        if (nPerRow < 0 || nPerRow % block != 0) {
            throw new IllegalArgumentException("a row of " + nPerRow + " values is not a whole number of "
                    + block + "-value " + type + " blocks");
        }
    }
}

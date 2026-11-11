package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

/**
 * Dot product of one weight row, read in place from a {@link CpuTensorRef}, with an activation vector that
 * {@link CpuActivationQuantizer} prepared for the weight's {@code vec_dot_type}. Implementations mirror the
 * {@code ggml_vec_dot_<w>_<a>} kernels of llama.cpp ggml/src/ggml-cpu/quants.c, are stateless and thread-safe,
 * and allocate nothing per call.
 */
public interface CpuVecDot {

    /**
     * Returns {@code sum_j w[row][j] * x[j]} over the {@code w.cols()} elements of the row.
     *
     * @throws IllegalArgumentException when {@code x} has the wrong activation type or length for {@code w}
     */
    float dotRow(CpuTensorRef w, long row, QuantizedActivations x);

    /**
     * Returns {@code sum_j row[off + j] * x[j]} over the {@code x.length()} elements of one weight row held
     * packed in a byte array, as {@link CpuVecDotRows} hands it to the row loops. Implementations perform the
     * shape checks of {@link #dotRow(CpuTensorRef, long, QuantizedActivations)} in their row loops and not here.
     */
    float dotRowBytes(byte[] row, int off, QuantizedActivations x);

    /**
     * Writes {@code dotRow(w, rowStart + r, x)} to {@code y[yOff + r]} for {@code r} in {@code [0, rowCount)}.
     * Implementations may override this to hoist per-call work out of the row loop.
     */
    default void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        for (int r = 0; r < rowCount; r++) {
            y[yOff + r] = dotRow(w, rowStart + r, x);
        }
    }

    /**
     * Writes {@code y[yOff + t*yStride + r] = dotRow(w, rowStart + r, x[t])} for every row {@code r} of the
     * range and every token {@code t}, the batched product of a prefill: each weight row is read from the
     * tensor once and dotted with every token's activations, instead of once per token. An implementation
     * that overrides {@link #dotRowBytes} shares the row the same way.
     */
    default void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations[] x, int nTokens,
                         float[] y, int yOff, int yStride) {
        if (nTokens == 0) {
            return;
        }
        checkCompatible(w, x[0]);
        for (int t = 1; t < nTokens; t++) {
            checkCompatible(w, x[t]);
        }
        CpuVecDotRows.forEachRowTokens(w, rowStart, rowCount, this, x, nTokens, y, yOff, yStride);
    }

    /**
     * The prefill product of {@code rowCount} weight rows with {@code nTokens} activation rows, writing the
     * same values as {@link #dotRows(CpuTensorRef, long, int, QuantizedActivations[], int, float[], int, int)}.
     * <p>
     * Implementations that can reuse work across rows override this to multiply an RM x RN register tile of
     * weight rows and tokens: a decoded weight block then serves RN tokens and a widened activation block
     * serves RM rows, where the plain row loop decodes the weight block once per RN tokens and reloads the
     * activation lanes for every row. The tile size is the implementation's own kernel-shape decision; the
     * default keeps the row loop.
     */
    default void dotTiles(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations[] x, int nTokens,
                          float[] y, int yOff, int yStride) {
        dotRows(w, rowStart, rowCount, x, nTokens, y, yOff, yStride);
    }

    /**
     * The activations of one batched product laid out once for a kernel's tile loop, built by
     * {@link #prepareTiles} before the product's row tasks start and then read by all of them. Implementations are
     * immutable once built and safe to share between threads.
     */
    interface TileActivations {

        /** Tokens the layout holds. */
        int tokens();
    }

    /**
     * Lays out {@code x[0, nTokens)} for {@link #dotTiles(CpuTensorRef, long, int, TileActivations,
     * QuantizedActivations[], int, int, float[], int, int)}, or returns null when the kernel reads the activation
     * rows directly (the default). A caller that splits one product into row and token tasks calls it once per
     * product, so a kernel whose tiles interleave or widen the activations does that work once instead of once per
     * task.
     *
     * @throws IllegalArgumentException when a token is incompatible with the kernel
     */
    default TileActivations prepareTiles(QuantizedActivations[] x, int nTokens) {
        return null;
    }

    /**
     * Token alignment of a prepared layout: the {@code tokenStart} of a prepared
     * {@link #dotTiles(CpuTensorRef, long, int, TileActivations, QuantizedActivations[], int, int, float[], int, int)}
     * call must be a multiple of it. 1 for kernels without a layout.
     */
    default int tileTokens() {
        return 1;
    }

    /**
     * Weight rows of the register tile a prepared {@link #dotTiles(CpuTensorRef, long, int, TileActivations,
     * QuantizedActivations[], int, int, float[], int, int)} multiplies: the row chunks of a batched product are whole
     * multiples of it. 2 by default, the tile of the int16 kernels.
     */
    default int tileRows() {
        return 2;
    }

    /**
     * Tokens of the register tile a prepared product multiplies, a multiple of {@link #tileTokens()}: the token
     * panels of a batched product are whole multiples of it. 8 by default, one tile of the int16 kernels.
     */
    default int tilePanelTokens() {
        return 8;
    }

    /**
     * Bytes one activation element takes in the layout {@link #prepareTiles} builds: a batched product sizes its
     * token slabs by it, so the layout memory stays bounded. 2 by default, the int16 lanes of the widened layouts.
     */
    default int tileLaneBytes() {
        return 2;
    }

    /**
     * The product of rows {@code [rowStart, rowStart + rowCount)} with tokens {@code [tokenStart, tokenStart +
     * nTokens)} of a batched product, writing token {@code tokenStart + t}'s output row at {@code y[yOff +
     * t*yStride]}: the same values as {@link #dotTiles(CpuTensorRef, long, int, QuantizedActivations[], int,
     * float[], int, int)} over those tokens. {@code tiles} is this kernel's {@link #prepareTiles} result for
     * {@code x} (null when it returned null); the default runs the unprepared form over the token range.
     *
     * @throws IllegalArgumentException  as the unprepared form, or for a token start that is not a multiple of
     *                                   {@link #tileTokens()}
     * @throws IndexOutOfBoundsException when the token range exceeds {@code x}
     */
    default void dotTiles(CpuTensorRef w, long rowStart, int rowCount, TileActivations tiles,
                          QuantizedActivations[] x, int tokenStart, int nTokens, float[] y, int yOff, int yStride) {
        if (tokenStart < 0 || nTokens < 0 || tokenStart > x.length - nTokens) {
            throw new IndexOutOfBoundsException("tokens [" + tokenStart + ", " + ((long) tokenStart + nTokens)
                    + ") exceed the " + x.length + "-token activations");
        }
        QuantizedActivations[] part = tokenStart == 0 ? x
                : java.util.Arrays.copyOfRange(x, tokenStart, tokenStart + nTokens);
        dotTiles(w, rowStart, rowCount, part, nTokens, y, yOff, yStride);
    }

    /**
     * Validates the arguments of {@link #dotTiles}: the activation array and its tokens, every token's
     * compatibility with {@code w} and output range, and the row stride.
     *
     * @throws IllegalArgumentException  when the activation array, a token or {@code y} is null, a token has
     *                                   the wrong type or length, or the stride is too small
     * @throws IndexOutOfBoundsException when {@code nTokens} exceeds the array or a range is out of bounds
     */
    default void checkTiles(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations[] x, int nTokens,
                            float[] y, int yOff, int yStride) {
        CpuVecDotRows.checkTokens(w, rowStart, rowCount, x, nTokens, y, yOff, yStride);
        for (int t = 0; t < nTokens; t++) {
            checkCompatible(w, x[t]);
        }
    }

    /**
     * Validates that {@code x} may feed this kernel over {@code w}: the activation type must be the one
     * {@link CpuActivationQuantizer#activationType} pairs with {@code w}'s type, and hold {@code w.cols()}
     * values.
     *
     * @throws IllegalArgumentException naming the expected and actual type and length
     */
    default void checkCompatible(CpuTensorRef w, QuantizedActivations x) {
        x.requireCompatible(w, CpuActivationQuantizer.activationType(w.type()));
    }

    /**
     * One activation vector in the block format a weight type's vec-dot consumes: Q8_0, Q8_1 or Q8_K blocks,
     * or plain F32 for the float weight types. Instances are created only by {@link CpuActivationQuantizer}
     * from freshly allocated arrays that nothing else references, and are never modified afterwards.
     * <p>
     * The arrays are package-private so the vec-dot kernels of this package read them without call
     * overhead; code outside the package uses the accessors.
     */
    final class QuantizedActivations {
        /** Q8_0, Q8_1, Q8_K, or F32. */
        final GGMLType type;
        /** Element count; a whole number of {@link #blockSize()} blocks. */
        final int n;
        /** Quantized values, one per element (null for F32). */
        final byte[] qs;
        /**
         * Per-block scale as the vec-dot reads it: the half-rounded {@code d} for Q8_0 / Q8_1, the float
         * {@code d} for Q8_K (null for F32).
         */
        final float[] d;
        /** Q8_1 only: half-rounded {@code d * sum(qs)} per block ({@code block_q8_1.s}); null otherwise. */
        final float[] s;
        /** Q8_K only: sums of each 16 consecutive quants ({@code block_q8_K.bsums}); null otherwise. */
        final short[] bsums;
        /** F32 only: the activation values; null otherwise. */
        final float[] f32;

        QuantizedActivations(GGMLType type, int n, byte[] qs, float[] d, float[] s, short[] bsums, float[] f32) {
            this.type = type;
            this.n = n;
            this.qs = qs;
            this.d = d;
            this.s = s;
            this.bsums = bsums;
            this.f32 = f32;
        }

        public GGMLType type() {
            return type;
        }

        public int length() {
            return n;
        }

        /** Elements per activation block: 32 for Q8_0 / Q8_1, 256 for Q8_K, 1 for F32. */
        public int blockSize() {
            return type == GGMLType.F32 ? 1 : type.getBlockSize();
        }

        public int blockCount() {
            return n / blockSize();
        }

        /** Quantized value {@code i}. */
        public int quant(int i) {
            return qs[i];
        }

        /** Scale of block {@code b} as the vec-dot reads it. */
        public float scale(int b) {
            return d[b];
        }

        /** Q8_1 block sum term {@code s} of block {@code b}. */
        public float blockSum(int b) {
            return s[b];
        }

        /** Q8_K sum of quants {@code [16g, 16g + 16)}. */
        public int groupSum(int g) {
            return bsums[g];
        }

        /** F32 activation value {@code i}. */
        public float value(int i) {
            return f32[i];
        }

        /**
         * Validates that this vector feeds a vec-dot over {@code w}.
         *
         * @throws IllegalArgumentException naming the expected and actual type and length
         */
        public void requireCompatible(CpuTensorRef w, GGMLType expectedType) {
            if (type != expectedType) {
                throw new IllegalArgumentException(w.type() + " vec-dot needs " + expectedType
                        + " activations, got " + type);
            }
            if (n != w.cols()) {
                throw new IllegalArgumentException(w.type() + " row of " + w.cols()
                        + " elements cannot be dotted with " + n + " activations");
            }
        }

        @Override
        public String toString() {
            return "QuantizedActivations{" + type + ", n=" + n + "}";
        }
    }
}

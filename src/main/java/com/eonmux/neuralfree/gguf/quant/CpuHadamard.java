package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;

import java.util.Objects;

/**
 * CPU twin of the {@value HadamardSpec#KERNEL} kernel: the transform of {@link HadamardSpec} applied to
 * row-major activation rows, with the kernel's index math (gather, signs, butterfly order, final scale).
 */
public final class CpuHadamard {

    /** Rows per parallel task; one row costs {@code width * log2(block)} additions. */
    private static final int ROW_GRAIN = 8;

    private CpuHadamard() {
    }

    /**
     * Transforms {@code nRows} rows of {@code width} elements: row {@code r} is read from
     * {@code x[xOff + r*width ..]} and written to {@code y[yOff + r*width ..]}. Each row is read completely
     * before it is written, so {@code y} may be the same region as {@code x} (same array and offset) even with
     * a permutation; other overlaps are not allowed.
     *
     * @throws IllegalArgumentException when the width does not fit the spec or a range is out of bounds
     */
    public static void apply(HadamardSpec spec, float[] x, int xOff, float[] y, int yOff, int nRows, int width) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        if (nRows < 0) {
            throw new IllegalArgumentException("row count must be non-negative, got " + nRows);
        }
        spec.checkWidth(width);
        checkRange("x", x.length, xOff, nRows, width);
        checkRange("y", y.length, yOff, nRows, width);
        CpuParallel.parallelFor(nRows, ROW_GRAIN, (from, to) -> {
            float[] row = new float[width];
            for (int r = from; r < to; r++) {
                transformRow(spec, x, xOff + r * width, y, yOff + r * width, width, row);
            }
        });
    }

    /** In-place form of {@link #apply}: {@code v[off + r*width ..]} for {@code r < nRows}. */
    public static void applyInPlace(HadamardSpec spec, float[] v, int off, int nRows, int width) {
        apply(spec, v, off, v, off, nRows, width);
    }

    /**
     * Normalized fast Walsh-Hadamard transform of {@code v[off .. off+n)} in place: butterflies
     * {@code (a, c) -> (a + c, a - c)} for strides {@code h = 1, 2, 4, ...} (natural Sylvester order), then
     * scaling by {@code 1/sqrt(n)}; equal to multiplying by {@link #matrix(int)}.
     *
     * @throws IllegalArgumentException when {@code n} is not a positive power of two
     */
    public static void fwht(float[] v, int off, int n) {
        if (n <= 0 || (n & (n - 1)) != 0) {
            throw new IllegalArgumentException("Hadamard size must be a positive power of two, got " + n);
        }
        if (off < 0 || off > v.length - n) {
            throw new IllegalArgumentException("Hadamard range [" + off + ", " + (off + (long) n) + ") exceeds length "
                    + v.length);
        }
        butterflies(v, off, n);
        final float scale = 1.0f / (float) Math.sqrt(n);
        for (int i = 0; i < n; i++) {
            v[off + i] *= scale;
        }
    }

    /**
     * Dense normalized Sylvester-Hadamard matrix, row-major {@code n x n}: entry {@code (row, col)} is
     * {@code -1/sqrt(n)} when {@code popcount(row & col)} is odd and {@code +1/sqrt(n)} otherwise
     * (llama.cpp src/llama-model.cpp:2300-2312). Reference for tests and diagnostics.
     *
     * @throws IllegalArgumentException when {@code n} is not a positive power of two or the matrix exceeds 2^30
     *                                  entries
     */
    public static float[] matrix(int n) {
        if (n <= 0 || (n & (n - 1)) != 0) {
            throw new IllegalArgumentException("Hadamard size must be a positive power of two, got " + n);
        }
        if ((long) n * n > (1L << 30)) {
            throw new IllegalArgumentException("Hadamard matrix of size " + n + " is too large");
        }
        final float scale = 1.0f / (float) Math.sqrt(n);
        float[] m = new float[n * n];
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                m[row * n + col] = (Integer.bitCount(row & col) & 1) != 0 ? -scale : scale;
            }
        }
        return m;
    }

    /** One row: gather (permutation) and leading signs into {@code row}, then per-block FWHT, trailing signs. */
    private static void transformRow(HadamardSpec spec, float[] x, int xBase, float[] y, int yBase, int width,
                                     float[] row) {
        final boolean before = spec.signOrder() == HadamardSpec.SignOrder.BEFORE_ROTATION;
        final boolean after = spec.signOrder() == HadamardSpec.SignOrder.AFTER_ROTATION;
        for (int o = 0; o < width; o++) {
            float v = x[xBase + spec.sourceIndex(o)];
            row[o] = before ? v * spec.sign(o) : v;
        }
        final int block = spec.block();
        final float scale = 1.0f / (float) Math.sqrt(block);
        for (int base = 0; base < width; base += block) {
            butterflies(row, base, block);
            for (int i = 0; i < block; i++) {
                int o = base + i;
                float v = row[o] * scale;
                y[yBase + o] = after ? v * spec.sign(o) : v;
            }
        }
    }

    /** Unnormalized in-place FWHT stages over {@code v[off .. off+n)}, in the kernel's pair order. */
    private static void butterflies(float[] v, int off, int n) {
        final int pairs = n >> 1;
        for (int h = 1; h < n; h <<= 1) {
            for (int p = 0; p < pairs; p++) {
                int i = off + (((p & ~(h - 1)) << 1) | (p & (h - 1)));
                float a = v[i];
                float c = v[i + h];
                v[i] = a + c;
                v[i + h] = a - c;
            }
        }
    }

    private static void checkRange(String what, int length, int off, int nRows, int width) {
        long end = (long) off + (long) nRows * width;
        if (off < 0 || end > length) {
            throw new IllegalArgumentException("Hadamard " + what + " range [" + off + ", " + end + ") exceeds length "
                    + length);
        }
    }
}

package com.eonmux.neuralfree.gguf.ops.gemma3n;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Names, argument layout, argument lists and launch geometry of the Gemma3n kernels in
 * {@code kernels/llm/gemma3n.cl} (GLSL {@code gemma3n.comp}); the semantics and llama.cpp references
 * ({@code src/models/gemma3n.cpp}) are documented on each kernel and on the CPU twins {@link CpuAltUp} and
 * {@link CpuActivationSparsity}.
 * <pre>
 * LLM_AltUp_Predict (i n_tokens*, i n_embd, i n_altup, i stream_stride, i coefs_off, i streams_off, i pred_off,
 *                    R coefs, Rn streams, Wn pred)
 * LLM_AltUp_Correct (i n_tokens*, i n_embd, i n_altup, i act, i stream_stride, i coefs_off, i activated_off,
 *                    i pred_off, i out_off, R coefs, R activated, Rn pred, Wn streams_out)
 * LLM_MagnitudeMatch(i n_tokens*, i n_embd, i n_x, i x_stride, i ref_off, i x_off, R ref, RWn x)
 * LLM_GaussianTopK  (i n_tokens*, i n, i x_stride, i x_off, f k_std, RWn x)
 * LLM_StreamsMean   (i n_tokens*, i n_embd, i n_altup, i stream_stride, i streams_off, i out_off, Rn streams, Wn out)
 * </pre>
 * {@code *} marks the per-step argument ({@link #ARG_N_TOKENS} for every kernel). Offsets and strides are in floats.
 *
 * <p><b>Stream layout.</b> The AltUp state of one ubatch is {@code n_altup} streams of {@code n_tokens} token-major
 * rows of {@code n_embd} floats: element (stream {@code k}, token {@code t}, column {@code e}) is at
 * {@code off + k*stream_stride + t*n_embd + e}, with {@code stream_stride >= n_tokens*n_embd} so a buffer sized for
 * the largest ubatch serves every step with fixed offsets. This is llama.cpp's {@code [n_embd, n_tokens, n_altup]}
 * tensor; each stream is a contiguous token-major activation block that the generic matvec and norm kernels consume
 * directly.
 *
 * <p><b>Aliasing.</b> In-place use is exact when the output view coincides with the aliased input view:
 * {@code pred == streams} with equal offsets, {@code streams_out == pred} with equal offsets, {@code out} equal to
 * stream 0 of {@code streams}. Every other pair of views that shares a buffer must be disjoint; the argument builders
 * reject overlaps.
 *
 * <p><b>Dispatch.</b> Element-wise kernels (predict, correct, stream mean): one work-item per (token, column),
 * {@link #elementwiseGlobal}. Row kernels (magnitude match, Gaussian top-k): one workgroup per row,
 * {@link #rowGlobal}, with {@code local} from {@link #rowLocal} (a power of two in [{@value #ROW_MIN_LOCAL},
 * {@code LLM_DIMX}], which satisfies the subgroup-reduction rule: at least 32 and a multiple of 32).
 */
public final class Gemma3nKernels {

    public static final String ALTUP_PREDICT = "LLM_AltUp_Predict";
    public static final String ALTUP_CORRECT = "LLM_AltUp_Correct";
    public static final String MAGNITUDE_MATCH = "LLM_MagnitudeMatch";
    public static final String GAUSSIAN_TOPK = "LLM_GaussianTopK";
    public static final String STREAMS_MEAN = "LLM_StreamsMean";

    /** Every kernel of the family, for prewarming. */
    public static final List<String> ALL = Collections.unmodifiableList(
            Arrays.asList(ALTUP_PREDICT, ALTUP_CORRECT, MAGNITUDE_MATCH, GAUSSIAN_TOPK, STREAMS_MEAN));

    /** Largest stream count ({@code NF_G3N_MAX_ALTUP}). */
    public static final int MAX_ALTUP = 8;

    /** Index of {@code n_tokens} in every kernel of the family. */
    public static final int ARG_N_TOKENS = 0;
    /** {@link #GAUSSIAN_TOPK} {@code k_std}. */
    public static final int ARG_TOPK_K_STD = 4;

    /** {@link #ALTUP_PREDICT} argument count (7 scalars, 3 buffers). */
    public static final int PREDICT_ARG_COUNT = 10;
    /** {@link #ALTUP_CORRECT} argument count (9 scalars, 4 buffers). */
    public static final int CORRECT_ARG_COUNT = 13;
    /** {@link #MAGNITUDE_MATCH} argument count (6 scalars, 2 buffers). */
    public static final int MAGNITUDE_ARG_COUNT = 8;
    /** {@link #GAUSSIAN_TOPK} argument count (5 scalars, 1 buffer). */
    public static final int TOPK_ARG_COUNT = 6;
    /** {@link #STREAMS_MEAN} argument count (6 scalars, 2 buffers). */
    public static final int MEAN_ARG_COUNT = 8;

    /** Smallest row-kernel workgroup: one full CUDA warp. */
    public static final int ROW_MIN_LOCAL = 32;
    /** Largest row-kernel workgroup ({@code LLM_DIMX}, the size of the kernels' reduction buffer). */
    public static final int ROW_MAX_LOCAL = LlmKernelConstants.LLM_DIMX;

    private static final int FLOAT_BYTES = Float.BYTES;

    private Gemma3nKernels() {
    }

    /**
     * Arguments of {@link #ALTUP_PREDICT}. {@code coefs} holds {@code n_altup^2} floats per token at
     * {@code coefsOff + t*n_altup^2} (the {@code altup_predict_coef} matvec output).
     *
     * @throws IllegalArgumentException on an invalid shape, a view outside its buffer, or overlapping views
     */
    public static Object[] predictArgs(int nTokens, int nEmbd, int nAltup, int streamStride,
                                       GpuBuffer coefs, int coefsOff, GpuBuffer streams, int streamsOff,
                                       GpuBuffer pred, int predOff) {
        checkStreams(nTokens, nEmbd, nAltup, streamStride);
        long rowFloats = (long) nTokens * nEmbd;
        long coefFloats = elementCount(nTokens, (long) nAltup * nAltup);
        checkView("coefs", coefs, coefsOff, 0, 1, coefFloats);
        checkView("streams", streams, streamsOff, streamStride, nAltup, rowFloats);
        checkView("pred", pred, predOff, streamStride, nAltup, rowFloats);
        checkInPlaceOrDisjoint("pred", pred, predOff, "streams", streams, streamsOff, streamStride, nAltup, rowFloats);
        checkDisjoint("coefs", coefs, coefsOff, 0, 1, coefFloats, "pred", pred, predOff, streamStride, nAltup, rowFloats);
        return new Object[]{nTokens, nEmbd, nAltup, streamStride, coefsOff, streamsOff, predOff, coefs, streams, pred};
    }

    /**
     * Arguments of {@link #ALTUP_CORRECT}. {@code coefs} holds the raw {@code altup_correct_coef} matvec output,
     * {@code n_altup} floats per token at {@code coefsOff + t*n_altup}; the kernel adds 1. {@code activated} is a
     * token-major {@code [nTokens][nEmbd]} view.
     *
     * @throws IllegalArgumentException on an invalid shape or active index, a view outside its buffer, or
     *                                  overlapping views
     */
    public static Object[] correctArgs(int nTokens, int nEmbd, int nAltup, int act, int streamStride,
                                       GpuBuffer coefs, int coefsOff, GpuBuffer activated, int activatedOff,
                                       GpuBuffer pred, int predOff, GpuBuffer out, int outOff) {
        checkStreams(nTokens, nEmbd, nAltup, streamStride);
        checkActive(act, nAltup);
        long rowFloats = (long) nTokens * nEmbd;
        long coefFloats = elementCount(nTokens, nAltup);
        checkView("coefs", coefs, coefsOff, 0, 1, coefFloats);
        checkView("activated", activated, activatedOff, 0, 1, rowFloats);
        checkView("pred", pred, predOff, streamStride, nAltup, rowFloats);
        checkView("streams_out", out, outOff, streamStride, nAltup, rowFloats);
        checkInPlaceOrDisjoint("streams_out", out, outOff, "pred", pred, predOff, streamStride, nAltup, rowFloats);
        checkDisjoint("coefs", coefs, coefsOff, 0, 1, coefFloats, "streams_out", out, outOff, streamStride, nAltup, rowFloats);
        checkDisjoint("activated", activated, activatedOff, 0, 1, rowFloats,
                "streams_out", out, outOff, streamStride, nAltup, rowFloats);
        return new Object[]{nTokens, nEmbd, nAltup, act, streamStride, coefsOff, activatedOff, predOff, outOff,
                coefs, activated, pred, out};
    }

    /**
     * Arguments of {@link #MAGNITUDE_MATCH}: {@code nX} rescaled rows per token, row {@code (j, t)} at
     * {@code xOff + j*xStride + t*nEmbd}, matched to {@code ref[refOff + t*nEmbd ..]}.
     *
     * @throws IllegalArgumentException on an invalid shape, a view outside its buffer, rows of {@code x} that overlap
     *                                  one another, or {@code x} overlapping {@code ref}
     */
    public static Object[] magnitudeMatchArgs(int nTokens, int nEmbd, int nX, int xStride,
                                              GpuBuffer ref, int refOff, GpuBuffer x, int xOff) {
        checkMagnitudeShape(nTokens, nEmbd, nX, xStride);
        long rowFloats = (long) nTokens * nEmbd;
        checkView("ref", ref, refOff, 0, 1, rowFloats);
        checkView("x", x, xOff, xStride, nX, rowFloats);
        checkDisjoint("ref", ref, refOff, 0, 1, rowFloats, "x", x, xOff, xStride, nX, rowFloats);
        return new Object[]{nTokens, nEmbd, nX, xStride, refOff, xOff, ref, x};
    }

    /**
     * Arguments of {@link #GAUSSIAN_TOPK} over rows {@code t < nTokens} of {@code n} floats at {@code xOff + t*xStride}
     * (for example the gate half of a fused gate/up row: {@code n = n_ff}, {@code xStride = 2*n_ff}).
     *
     * @throws IllegalArgumentException on an invalid shape or multiplier, or a view outside its buffer
     */
    public static Object[] gaussianTopKArgs(int nTokens, int n, int xStride, int xOff, float kStd, GpuBuffer x) {
        checkTopKShape(nTokens, n, xStride, kStd);
        checkView("x", x, xOff, xStride, nTokens, n);
        return new Object[]{nTokens, n, xStride, xOff, kStd, x};
    }

    /**
     * Arguments of {@link #STREAMS_MEAN}; {@code out} is a token-major {@code [nTokens][nEmbd]} view.
     *
     * @throws IllegalArgumentException on an invalid shape, a view outside its buffer, or {@code out} overlapping a
     *                                  stream other than in place on stream 0
     */
    public static Object[] streamsMeanArgs(int nTokens, int nEmbd, int nAltup, int streamStride,
                                           GpuBuffer streams, int streamsOff, GpuBuffer out, int outOff) {
        checkStreams(nTokens, nEmbd, nAltup, streamStride);
        long rowFloats = (long) nTokens * nEmbd;
        checkView("streams", streams, streamsOff, streamStride, nAltup, rowFloats);
        checkView("out", out, outOff, 0, 1, rowFloats);
        if (!(out == streams && outOff == streamsOff)) {
            checkDisjoint("out", out, outOff, 0, 1, rowFloats, "streams", streams, streamsOff, streamStride, nAltup, rowFloats);
        }
        return new Object[]{nTokens, nEmbd, nAltup, streamStride, streamsOff, outOff, streams, out};
    }

    /** Global size of an element-wise kernel: {@code roundUp(nTokens*nEmbd, local)}, at least one workgroup. */
    public static long[] elementwiseGlobal(int nTokens, int nEmbd, int local) {
        if (local <= 0) {
            throw new IllegalArgumentException("local size must be positive, got " + local);
        }
        long n = elementCount(nTokens, nEmbd);
        return new long[]{Math.max(1L, (n + local - 1) / local) * local};
    }

    /**
     * Global size of a row kernel: one workgroup of {@code local} per row, at least one workgroup
     * ({@code rows = nX*nTokens} for {@link #MAGNITUDE_MATCH}, {@code nTokens} for {@link #GAUSSIAN_TOPK}).
     */
    public static long[] rowGlobal(long rows, int local) {
        if (rows < 0 || rows > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("row count must be in [0, 2^31), got " + rows);
        }
        checkRowLocal(local);
        return new long[]{Math.max(1L, rows) * local};
    }

    /**
     * Row-kernel local size for a backend's preferred size: the largest power of two not above
     * {@code min(preferred, LLM_DIMX)}, raised to {@value #ROW_MIN_LOCAL}.
     */
    public static int rowLocal(int preferred) {
        int cap = Math.min(Math.max(preferred, ROW_MIN_LOCAL), ROW_MAX_LOCAL);
        return Integer.highestOneBit(cap);
    }

    /**
     * Validates a row-kernel local size: a power of two in [{@value #ROW_MIN_LOCAL}, {@value #ROW_MAX_LOCAL}].
     */
    public static void checkRowLocal(int local) {
        if (local < ROW_MIN_LOCAL || local > ROW_MAX_LOCAL || Integer.bitCount(local) != 1) {
            throw new IllegalArgumentException("row kernel local size must be a power of two in [" + ROW_MIN_LOCAL
                    + ", " + ROW_MAX_LOCAL + "], got " + local);
        }
    }

    /**
     * Validates an AltUp stream block shape, shared by the kernels and {@link CpuAltUp}: {@code nTokens >= 0},
     * {@code nEmbd > 0}, {@code 1 <= nAltup <= MAX_ALTUP}, {@code streamStride >= nTokens*nEmbd} and every index
     * below 2^31.
     *
     * @throws IllegalArgumentException naming the offending values
     */
    public static void checkStreams(int nTokens, int nEmbd, int nAltup, int streamStride) {
        if (nTokens < 0 || nEmbd <= 0) {
            throw new IllegalArgumentException("AltUp needs n_tokens >= 0 and n_embd > 0: n_tokens=" + nTokens
                    + " n_embd=" + nEmbd);
        }
        if (nAltup < 1 || nAltup > MAX_ALTUP) {
            throw new IllegalArgumentException("AltUp n_altup=" + nAltup + " is outside [1, " + MAX_ALTUP + "]");
        }
        long rowFloats = elementCount(nTokens, nEmbd);
        if (streamStride < rowFloats) {
            throw new IllegalArgumentException("AltUp stream_stride=" + streamStride + " is below n_tokens*n_embd="
                    + rowFloats);
        }
        if ((long) (nAltup - 1) * streamStride + rowFloats > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("AltUp streams of " + nAltup + " x stride " + streamStride
                    + " exceed the 32-bit index range");
        }
    }

    /**
     * Validates the active stream index.
     *
     * @throws IllegalArgumentException when {@code act} is outside {@code [0, nAltup)}
     */
    public static void checkActive(int act, int nAltup) {
        if (act < 0 || act >= nAltup) {
            throw new IllegalArgumentException("AltUp active index " + act + " is outside [0, " + nAltup + ")");
        }
    }

    /**
     * Validates a magnitude-match shape: {@code nTokens >= 0}, {@code nEmbd > 0}, {@code nX >= 1}, and with several
     * rows per token {@code xStride >= nTokens*nEmbd} so the rows are disjoint.
     *
     * @throws IllegalArgumentException naming the offending values
     */
    public static void checkMagnitudeShape(int nTokens, int nEmbd, int nX, int xStride) {
        if (nTokens < 0 || nEmbd <= 0 || nX < 1) {
            throw new IllegalArgumentException("magnitude match needs n_tokens >= 0, n_embd > 0 and n_x >= 1: n_tokens="
                    + nTokens + " n_embd=" + nEmbd + " n_x=" + nX);
        }
        long rowFloats = elementCount(nTokens, nEmbd);
        if (nX > 1 && xStride < rowFloats) {
            throw new IllegalArgumentException("magnitude match x_stride=" + xStride + " is below n_tokens*n_embd="
                    + rowFloats);
        }
        if ((long) nX * nTokens > Integer.MAX_VALUE || (long) (nX - 1) * xStride + rowFloats > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("magnitude match of " + nX + " x " + rowFloats
                    + " floats exceeds the 32-bit index range");
        }
    }

    /**
     * Validates a Gaussian top-k shape: {@code nTokens >= 0}, {@code n >= 2} (the unbiased deviation divides by
     * {@code n - 1}), {@code xStride >= n} with several rows, and a finite non-negative multiplier.
     *
     * @throws IllegalArgumentException naming the offending values
     */
    public static void checkTopKShape(int nTokens, int n, int xStride, float kStd) {
        if (nTokens < 0 || n < 2) {
            throw new IllegalArgumentException("Gaussian top-k needs n_tokens >= 0 and a row width n >= 2: n_tokens="
                    + nTokens + " n=" + n);
        }
        if (nTokens > 1 && xStride < n) {
            throw new IllegalArgumentException("Gaussian top-k x_stride=" + xStride + " is below the row width " + n);
        }
        if (nTokens > 0 && (long) (nTokens - 1) * xStride + n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Gaussian top-k rows exceed the 32-bit index range");
        }
        if (!(kStd >= 0f) || Float.isInfinite(kStd)) {
            throw new IllegalArgumentException("Gaussian top-k std multiplier must be finite and non-negative, got " + kStd);
        }
    }

    /**
     * {@code rows * cols} as a count below 2^31.
     *
     * @throws IllegalArgumentException when a factor is negative or the product reaches 2^31
     */
    static long elementCount(long rows, long cols) {
        if (rows < 0 || cols < 0) {
            throw new IllegalArgumentException("shape must be non-negative, got " + rows + " x " + cols);
        }
        long n = rows * cols;
        if (rows != 0 && n / rows != cols || n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("element count " + rows + " x " + cols + " exceeds 2^31 - 1");
        }
        return n;
    }

    /**
     * True when any of the {@code aCount} spans {@code [aOff + i*aStride, + aLen)} intersects any of the
     * {@code bCount} spans {@code [bOff + j*bStride, + bLen)}.
     */
    static boolean spansOverlap(long aOff, long aStride, int aCount, long aLen,
                                long bOff, long bStride, int bCount, long bLen) {
        if (aLen <= 0 || bLen <= 0) {
            return false;
        }
        for (int i = 0; i < aCount; i++) {
            long a = aOff + i * aStride;
            for (int j = 0; j < bCount; j++) {
                long b = bOff + j * bStride;
                if (a < b + bLen && b < a + aLen) {
                    return true;
                }
            }
        }
        return false;
    }

    private static void checkView(String name, GpuBuffer buf, long off, long stride, int count, long len) {
        if (buf == null) {
            throw new IllegalArgumentException(name + " buffer is null");
        }
        if (off < 0) {
            throw new IllegalArgumentException(name + ": offset must be non-negative, got " + off);
        }
        if (len == 0) {
            return;
        }
        long end = off + (long) (count - 1) * stride + len;
        long floats = buf.sizeInBytes() / FLOAT_BYTES;
        if (end > floats) {
            throw new IllegalArgumentException(name + ": view ends at float " + end + " but the buffer holds "
                    + floats + " floats");
        }
    }

    private static void checkInPlaceOrDisjoint(String outName, GpuBuffer out, long outOff, String inName, GpuBuffer in,
                                               long inOff, long stride, int count, long len) {
        if (out == in && outOff != inOff && spansOverlap(outOff, stride, count, len, inOff, stride, count, len)) {
            throw new IllegalArgumentException(outName + " overlaps " + inName
                    + " without coinciding with it (in-place use needs equal offsets)");
        }
    }

    private static void checkDisjoint(String aName, GpuBuffer a, long aOff, long aStride, int aCount, long len,
                                      String bName, GpuBuffer b, long bOff, long bStride, int bCount, long bLen) {
        if (a != b) {
            return;
        }
        if (spansOverlap(aOff, aStride, aCount, len, bOff, bStride, bCount, bLen)) {
            throw new IllegalArgumentException(aName + " overlaps " + bName + " in the same buffer");
        }
    }
}

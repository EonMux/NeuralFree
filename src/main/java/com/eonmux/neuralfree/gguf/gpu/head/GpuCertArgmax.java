package com.eonmux.neuralfree.gguf.gpu.head;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CertifiedArgmaxScreen;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * The certified exact-argmax readout of an LM head on the device ({@code kernels/llm/cert_argmax.cl}):
 * {@code LLM_CertScreen} computes every row's INT8 screen logit against the hidden rows, then the
 * stored type's {@code LLM_CertRescore_*} recomputes the threshold from the per-row error bounds,
 * gathers the candidates that can still reach it into its workgroup's local memory, rescores exactly
 * those rows through the matvec family's arithmetic at the family's work-group size, and falls back to
 * the full exact scan in-kernel when a row's candidates overflow the cap or hold none. The ids land at
 * element 0 of the caller's id buffer; no call waits on the device or reads weights back after the build.
 *
 * <p>The build (first bind) reads the stored head bytes back once, screens them on the host through
 * {@link CertifiedArgmaxScreen} and uploads the codes, scales and bounds. The dispatches are prepared
 * for up to {@code maxRows} rows at the matvec work-group size, so a rescored row's value equals the
 * value the plain family matvec of the same row produces. A backend without the kernels binds nothing
 * ({@link #bind} returns null), and its caller keeps the exact readout. Not thread-safe.</p>
 */
public final class GpuCertArgmax implements AutoCloseable {

    /** The INT8 screen kernel ({@code kernels/llm/cert_argmax.cl}). */
    public static final String SCREEN = "LLM_CertScreen";
    /** The exact rescore kernel of an F16 head. */
    public static final String RESCORE_F16 = "LLM_CertRescore_F16";
    /** The exact rescore kernel of a BF16 head. */
    public static final String RESCORE_BF16 = "LLM_CertRescore_BF16";
    /** The exact rescore kernel of a Q8_0 head. */
    public static final String RESCORE_Q8_0 = "LLM_CertRescore_Q8_0";

    /** Candidate capacity of the rescore kernel's local gather; the configured cap clamps to this. */
    public static final int KERNEL_KMAX = 256;

    /** Screen kernel argument indexes ({@link #SCREEN}). */
    private static final int SCREEN_ARG_VOCAB = 0;
    private static final int SCREEN_ARG_COLS = 1;
    private static final int SCREEN_ARG_N_ROWS = 2;
    private static final int SCREEN_ARG_CODES = 3;
    private static final int SCREEN_ARG_SCALES = 4;
    private static final int SCREEN_ARG_HIDDEN = 5;
    private static final int SCREEN_ARG_OUT = 6;

    /** Rescore kernel argument indexes ({@code LLM_CertRescore_*}). */
    private static final int RESCORE_ARG_VOCAB = 0;
    private static final int RESCORE_ARG_COLS = 1;
    private static final int RESCORE_ARG_N_ROWS = 2;
    private static final int RESCORE_ARG_KMAX = 3;
    private static final int RESCORE_ARG_EBMAX = 4;
    private static final int RESCORE_ARG_WEIGHT = 5;
    private static final int RESCORE_ARG_BOUNDS = 6;
    private static final int RESCORE_ARG_HIDDEN = 7;
    private static final int RESCORE_ARG_SCREENED = 8;
    private static final int RESCORE_ARG_IDS = 9;

    /** The rescore kernel of a head stored as {@code type}. */
    public static String rescoreKernel(GGMLType type) {
        switch (type) {
            case F16:
                return RESCORE_F16;
            case BF16:
                return RESCORE_BF16;
            case Q8_0:
                return RESCORE_Q8_0;
            default:
                throw new IllegalArgumentException(type + " is not a certified head type");
        }
    }

    private final GpuOps ops;
    private final PreparedDispatch screenDispatch;
    private final PreparedDispatch rescoreDispatch;
    private final GpuBuffer codes;
    private final GpuBuffer scales;
    private final GpuBuffer bounds;
    private final GpuBuffer screened;
    private final int vocab;
    private final int cols;
    private final int maxRows;
    private final int kmax;
    private int lastRows = -1;

    private GpuCertArgmax(GpuOps ops, PreparedDispatch screenDispatch, PreparedDispatch rescoreDispatch,
                          GpuBuffer codes, GpuBuffer scales, GpuBuffer bounds, GpuBuffer screened,
                          int vocab, int cols, int maxRows, int kmax) {
        this.ops = ops;
        this.screenDispatch = screenDispatch;
        this.rescoreDispatch = rescoreDispatch;
        this.codes = codes;
        this.scales = scales;
        this.bounds = bounds;
        this.screened = screened;
        this.vocab = vocab;
        this.cols = cols;
        this.maxRows = maxRows;
        this.kmax = kmax;
    }

    /**
     * Binds the certified readout of {@code output}: the head weight's stored rows are read back once
     * and screened into INT8 codes, scales and bounds, and both kernels are prepared at the matvec
     * work-group size for up to {@code maxRows} hidden rows. The greedy ids land at element 0 of
     * {@code ids}, which must hold {@code maxRows} ints.
     *
     * @param kmax the configured candidate cap; clamped to {@link #KERNEL_KMAX}, which only forces
     *             more full-scan fallbacks (the certification stays exact)
     * @return null when the backend provides no certified kernel at the launch's work-group size; the
     *         caller keeps its exact readout
     * @throws IllegalArgumentException for non-positive sizes or a cap above {@link #KERNEL_KMAX} after
     *                                  clamping being negative
     */
    public static GpuCertArgmax bind(GpuForwardContext ctx, GpuWeight output, int maxRows, int kmax,
                                     GpuBuffer ids) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(ids, "ids");
        if (ids.sizeInBytes() < (long) maxRows * Integer.BYTES) {
            throw new IllegalArgumentException("certified argmax ids buffer holds " + ids.sizeInBytes()
                    + " bytes, " + maxRows + " ids need " + ((long) maxRows * Integer.BYTES));
        }
        if (maxRows < 1) {
            throw new IllegalArgumentException("certified argmax needs positive rows, got " + maxRows);
        }
        GpuOps ops = ctx.ops();
        int local = ops.matvecLocal();
        String rescore = rescoreKernel(output.type());
        if (!ops.prewarm(SCREEN, new long[]{local}) || !ops.prewarm(rescore, new long[]{local})) {
            return null;
        }
        CertifiedArgmaxScreen screen = screenOf(ctx.queue(), output);
        int rows = Math.toIntExact(output.rows());
        int cols = Math.toIntExact(output.cols());
        int cap = Math.min(kmax, KERNEL_KMAX);
        if (cap < 1) {
            throw new IllegalArgumentException("certified argmax needs a positive candidate cap, got " + kmax);
        }
        GpuBuffer codes = null;
        GpuBuffer scales = null;
        GpuBuffer bounds = null;
        GpuBuffer screened = null;
        try {
            codes = allocate(ctx.backend(), (long) rows * cols);
            scales = allocate(ctx.backend(), (long) rows * Float.BYTES);
            bounds = allocate(ctx.backend(), (long) rows * Float.BYTES);
            screened = allocate(ctx.backend(), (long) maxRows * rows * Float.BYTES);
            upload(ctx.queue(), codes, screen.codesView(), (long) rows * cols);
            upload(ctx.queue(), scales, screen.scalesCopy(), rows);
            upload(ctx.queue(), bounds, screen.boundsCopy(), rows);
            Object[] screenArgs = new Object[7];
            screenArgs[SCREEN_ARG_VOCAB] = rows;
            screenArgs[SCREEN_ARG_COLS] = cols;
            screenArgs[SCREEN_ARG_N_ROWS] = maxRows;
            screenArgs[SCREEN_ARG_CODES] = codes;
            screenArgs[SCREEN_ARG_SCALES] = scales;
            // the hidden rows arrive per run(); the placeholder only satisfies the prepare-time binding check
            screenArgs[SCREEN_ARG_HIDDEN] = ctx.placeholders().readOnlyFloats();
            screenArgs[SCREEN_ARG_OUT] = screened;
            PreparedDispatch screenDispatch = ops.prepare(SCREEN, new long[]{global(rows, maxRows, local)},
                    new long[]{local}, screenArgs);
            Object[] rescoreArgs = new Object[10];
            rescoreArgs[RESCORE_ARG_VOCAB] = rows;
            rescoreArgs[RESCORE_ARG_COLS] = cols;
            rescoreArgs[RESCORE_ARG_N_ROWS] = maxRows;
            rescoreArgs[RESCORE_ARG_KMAX] = cap;
            rescoreArgs[RESCORE_ARG_EBMAX] = screen.maxBound();
            rescoreArgs[RESCORE_ARG_WEIGHT] = output.shard(0);
            rescoreArgs[RESCORE_ARG_BOUNDS] = bounds;
            rescoreArgs[RESCORE_ARG_HIDDEN] = ctx.placeholders().readOnlyFloats();
            rescoreArgs[RESCORE_ARG_SCREENED] = screened;
            rescoreArgs[RESCORE_ARG_IDS] = ids;
            PreparedDispatch rescoreDispatch;
            try {
                rescoreDispatch = ops.prepare(rescore, new long[]{(long) maxRows * local}, new long[]{local},
                        rescoreArgs);
            } catch (RuntimeException e) {
                screenDispatch.close();
                throw e;
            }
            return new GpuCertArgmax(ops, screenDispatch, rescoreDispatch, codes, scales, bounds, screened,
                    rows, cols, maxRows, cap);
        } catch (RuntimeException e) {
            closeAll(codes, scales, bounds, screened);
            throw e;
        }
    }

    /**
     * The head weight's stored bytes, read back once and screened: per row the absmax scale, the
     * rounded codes and the error bound.
     */
    private static CertifiedArgmaxScreen screenOf(ComputeQueue queue, GpuWeight output) {
        long bytes = output.sizeInBytes();
        byte[] raw = new byte[Math.toIntExact(bytes)];
        GpuEvent ev = queue.enqueueReadBuffer(output.shard(0), 0, bytes, raw);
        if (ev != null) {
            ev.close();
        }
        queue.finish();
        CpuTensorRef weight = CpuTensorRef.of(output.type(), ByteBuffer.wrap(raw), output.rows(), output.cols());
        return CertifiedArgmaxScreen.of(weight);
    }

    /** Enqueues one host-to-device upload and lets the in-order queue order it before the first dispatch. */
    private static void upload(ComputeQueue queue, GpuBuffer dst, ByteBuffer data, long elements) {
        GpuEvent ev = queue.enqueueWriteBuffer(dst, 0, elements, data);
        if (ev != null) {
            ev.close();
        }
    }

    /** Enqueues one host-to-device upload of {@code floats} float values and orders it before the first dispatch. */
    private static void upload(ComputeQueue queue, GpuBuffer dst, float[] data, long floats) {
        GpuEvent ev = queue.enqueueWriteBuffer(dst, 0, floats * Float.BYTES, data);
        if (ev != null) {
            ev.close();
        }
    }

    /**
     * Enqueues the certified greedy choice of the first {@code rows} hidden rows: the screen of every
     * vocabulary row against them, then the rescore into the id buffer.
     *
     * @param hidden rows of {@code hiddenWidth} floats at element 0 of their buffer, as the head's
     *               {@code hidden} returns them
     * @throws IllegalArgumentException when {@code rows} is outside {@code [1, maxRows]}
     */
    public void run(int rows, ScratchSlot hidden) {
        Objects.requireNonNull(hidden, "hidden");
        if (rows < 1 || rows > maxRows) {
            throw new IllegalArgumentException("certified argmax of " + rows + " rows outside [1, " + maxRows + "]");
        }
        if (hidden.offset() != 0) {
            throw new IllegalArgumentException("the certified kernels read the hidden rows from element 0, "
                    + hidden.name() + " starts at " + hidden.offset());
        }
        hidden.require((long) rows * cols);
        if (rows != lastRows) {
            screenDispatch.setInt(SCREEN_ARG_N_ROWS, rows);
            screenDispatch.setGlobal(screenGlobal(rows));
            rescoreDispatch.setInt(RESCORE_ARG_N_ROWS, rows);
            rescoreDispatch.setGlobal((long) rows * dispatchLocal());
            lastRows = rows;
        }
        screenDispatch.setBuffer(SCREEN_ARG_HIDDEN, hidden.buffer());
        screenDispatch.dispatch();
        rescoreDispatch.setBuffer(RESCORE_ARG_HIDDEN, hidden.buffer());
        rescoreDispatch.dispatch();
    }

    /** The configured candidate cap, clamped to the kernel's gather capacity. */
    public int kmax() {
        return kmax;
    }

    /** Screen dispatch global size (threads) for {@code rows} hidden rows: one workgroup per (row, v). */
    private long screenGlobal(int rows) {
        return global(vocab, rows, dispatchLocal());
    }

    private int dispatchLocal() {
        return ops.matvecLocal();
    }

    /** Global size in threads of {@code n * rows} one-workgroup-per-row launches at {@code local}. */
    private static long global(int n, int rows, int local) {
        return (long) n * rows * local;
    }

    private static GpuBuffer allocate(ComputeBackend backend, long bytes) {
        return backend.allocateBuffer(Math.max(bytes, Float.BYTES));
    }

    private static void closeAll(GpuBuffer... buffers) {
        RuntimeException first = null;
        for (GpuBuffer b : buffers) {
            if (b == null) {
                continue;
            }
            try {
                b.close();
            } catch (RuntimeException e) {
                if (first == null) {
                    first = e;
                } else {
                    first.addSuppressed(e);
                }
            }
        }
        if (first != null) {
            throw first;
        }
    }

    /** Releases the prepared dispatches and the screen buffers. Idempotent. */
    @Override
    public void close() {
        RuntimeException first = null;
        try {
            screenDispatch.close();
        } catch (RuntimeException e) {
            first = e;
        }
        try {
            rescoreDispatch.close();
        } catch (RuntimeException e) {
            if (first == null) {
                first = e;
            } else {
                first.addSuppressed(e);
            }
        }
        try {
            closeAll(codes, scales, bounds, screened);
        } catch (RuntimeException e) {
            if (first == null) {
                first = e;
            } else {
                first.addSuppressed(e);
            }
        }
        if (first != null) {
            throw first;
        }
    }
}

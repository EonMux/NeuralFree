package com.eonmux.neuralfree.gguf.gpu.head;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.engine.DryBreakers;
import com.eonmux.neuralfree.gguf.engine.LogitPrefilter;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * Per-row pre-top-k logit edits on the device ({@value GemmKernels#LOGITS_PREFILTER},
 * {@code kernels/llm/sampling_topk.cl}): each of the first {@code rows} logits rows takes its
 * {@link LogitPrefilter.Row} — the fixed logit-bias table (a NaN bias is the {@code -inf} ban), the
 * repetition/presence/frequency penalties over the row's distinct window tokens, and DRY over its window and
 * restart sequences — before the row top-k readout of the same buffer. The logits rows sit at the buffer's
 * element 0, {@code vocab} floats apart.
 *
 * <p>The row tables are uploaded per call. Buffers grow to the widest call seen and the prepared dispatch is
 * rebuilt when one grows; a backend without the kernel binds nothing ({@link #bind} returns null), and its
 * caller keeps the whole-row path. Not thread-safe.</p>
 */
public final class GpuLogitPrefilter implements AutoCloseable {

    /** Ints per row of the metadata table ({@code NF_PF_META}). */
    static final int META_STRIDE = 16;
    /** Floats per row of the parameter table ({@code NF_PF_FCFG}). */
    static final int FCFG_STRIDE = 8;
    /** Ints per restart sequence of the breaker table ({@code NF_PF_BR}). */
    static final int BREAKER_STRIDE = 4;
    /** Longest DRY window the kernel's local Z-length table holds ({@code NF_PF_DRY_MAX}). */
    public static final int DRY_WINDOW_MAX = 256;

    private final GpuOps ops;
    private final ComputeBackend backend;
    private final ComputeQueue queue;
    private final GpuBuffer logits;
    private final int vocab;
    private final int maxRows;
    private final int local;

    private GpuBuffer meta;
    private GpuBuffer fcfg;
    private GpuBuffer biasIds;
    private GpuBuffer biasVals;
    private GpuBuffer tokens;
    private GpuBuffer breakerMeta;
    private GpuBuffer breakerTails;
    private int metaRows;
    private int fcfgRows;
    private int biasCapacity;
    private int tokenCapacity;
    private int breakerCapacity;
    private int tailCapacity;
    private PreparedDispatch dispatch;
    private int lastRows = -1;
    /**
     * The breaker table of each row of the last call (null for a row without DRY): the device breaker arrays hold
     * exactly these tables in this order until a buffer is reallocated, so a call over the same tables skips them.
     */
    private DryBreakers[] lastBreakers;
    private boolean buffersChanged;
    private long breakerUploads;
    private long breakerUploadsSkipped;

    private GpuLogitPrefilter(GpuOps ops, ComputeBackend backend, ComputeQueue queue, GpuBuffer logits, int vocab,
                              int maxRows, int local) {
        this.ops = ops;
        this.backend = backend;
        this.queue = queue;
        this.logits = logits;
        this.vocab = vocab;
        this.maxRows = maxRows;
        this.local = local;
    }

    /**
     * Binds the prefilter of up to {@code maxRows} logits rows of {@code vocab} floats over {@code logits}.
     *
     * @return null when the backend provides no {@value GemmKernels#LOGITS_PREFILTER} kernel
     * @throws IllegalArgumentException for non-positive sizes or a logits buffer smaller than one row
     */
    public static GpuLogitPrefilter bind(GpuOps ops, ComputeBackend backend, ComputeQueue queue, GpuBuffer logits,
                                         int maxRows, int vocab) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(queue, "queue");
        Objects.requireNonNull(logits, "logits");
        if (maxRows < 1 || vocab < 1) {
            throw new IllegalArgumentException("logit prefilter needs positive rows and vocabulary, got " + maxRows
                    + "/" + vocab);
        }
        if (logits.sizeInBytes() < (long) vocab * Float.BYTES) {
            throw new IllegalArgumentException("logit prefilter buffer holds " + logits.sizeInBytes()
                    + " bytes, one row of " + vocab + " needs " + (long) vocab * Float.BYTES);
        }
        int local = Math.min(ops.local(), LlmKernelConstants.LLM_DIMX);
        if (!ops.prewarm(GemmKernels.LOGITS_PREFILTER, new long[]{local})) {
            return null;
        }
        GpuLogitPrefilter prefilter = new GpuLogitPrefilter(ops, backend, queue, logits, vocab, maxRows, local);
        try {
            prefilter.ensure(1, 0, 0, 0, 0);
        } catch (RuntimeException e) {
            prefilter.close();
            throw e;
        }
        return prefilter;
    }

    /**
     * The calls that uploaded the DRY breaker tables and the calls that found the device already holding them, in
     * that order, since the prefilter was bound.
     */
    public long[] breakerUploadCounts() {
        return new long[]{breakerUploads, breakerUploadsSkipped};
    }

    /**
     * Applies {@code prefilter} to the first {@code prefilter.rows()} logits rows of the bound buffer.
     *
     * @throws IllegalArgumentException when the rows exceed the bound width, a row names an out-of-range token, or
     *                                  a DRY window is longer than {@value #DRY_WINDOW_MAX}
     */
    public void run(LogitPrefilter prefilter) {
        Objects.requireNonNull(prefilter, "prefilter");
        int rows = prefilter.rows();
        if (rows < 1 || rows > maxRows) {
            throw new IllegalArgumentException("logit prefilter of " + rows + " rows outside [1, " + maxRows + "]");
        }
        if (prefilter.isEmpty()) {
            return;
        }
        int[][] biasRowIds = new int[rows][];
        float[][] biasRowValues = new float[rows][];
        int[][] penaltyRowIds = new int[rows][];
        int[][] penaltyRowCounts = new int[rows][];
        int[][] dryRowTokens = new int[rows][];
        DryBreakers[] dryRowBreakers = new DryBreakers[rows];
        int totalBias = 0;
        int totalTokens = 0;
        int totalBreakers = 0;
        int totalTails = 0;
        for (int r = 0; r < rows; r++) {
            LogitPrefilter.Row row = prefilter.row(r);
            biasRowIds[r] = row.biasIds();
            biasRowValues[r] = row.biasValues();
            checkIds(biasRowIds[r], biasRowValues[r].length, "bias");
            totalBias += biasRowIds[r].length;
            if (row.penalties()) {
                penaltyRowIds[r] = row.penaltyTokens();
                penaltyRowCounts[r] = row.penaltyCounts();
                checkIds(penaltyRowIds[r], penaltyRowCounts[r].length, "penalty");
                totalTokens += 2 * penaltyRowIds[r].length;
            }
            if (row.dry()) {
                dryRowTokens[r] = row.dryTokens();
                if (dryRowTokens[r].length > DRY_WINDOW_MAX) {
                    throw new IllegalArgumentException("DRY window of " + dryRowTokens[r].length
                            + " tokens exceeds the device window of " + DRY_WINDOW_MAX);
                }
                checkIds(dryRowTokens[r], -1, "DRY");
                totalTokens += dryRowTokens[r].length;
                DryBreakers breakers = row.dryBreakerTable();
                if (!breakers.inVocabulary(vocab)) {
                    throw new IllegalArgumentException("a DRY breaker token id lies outside the vocabulary of "
                            + vocab);
                }
                dryRowBreakers[r] = breakers;
                totalBreakers += breakers.count();
                totalTails += breakers.tailCount();
            }
        }
        ensure(rows, totalBias, totalTokens, totalBreakers, totalTails);

        int[] metaHost = new int[rows * META_STRIDE];
        float[] cfgHost = new float[rows * FCFG_STRIDE];
        int[] biasIdHost = new int[totalBias];
        float[] biasValHost = new float[totalBias];
        int[] tokenHost = new int[totalTokens];
        boolean sameBreakers = Arrays.equals(dryRowBreakers, lastBreakers) && !buffersChanged;
        int[] breakerHost = sameBreakers ? null : new int[totalBreakers * BREAKER_STRIDE];
        int[] tailHost = sameBreakers ? null : new int[totalTails];
        int biasAt = 0;
        int tokenAt = 0;
        int breakerAt = 0;
        int tailAt = 0;
        for (int r = 0; r < rows; r++) {
            LogitPrefilter.Row row = prefilter.row(r);
            int m = r * META_STRIDE;
            int f = r * FCFG_STRIDE;
            int[] biasIdsRow = biasRowIds[r];
            float[] biasValsRow = biasRowValues[r];
            metaHost[m] = biasAt;
            metaHost[m + 1] = biasIdsRow.length;
            System.arraycopy(biasIdsRow, 0, biasIdHost, biasAt, biasIdsRow.length);
            System.arraycopy(biasValsRow, 0, biasValHost, biasAt, biasValsRow.length);
            biasAt += biasIdsRow.length;
            int penAt = tokenAt;
            int penCount = 0;
            if (row.penalties()) {
                int[] ids = penaltyRowIds[r];
                int[] counts = penaltyRowCounts[r];
                penCount = ids.length;
                for (int i = 0; i < ids.length; i++) {
                    tokenHost[tokenAt++] = ids[i];
                    tokenHost[tokenAt++] = counts[i];
                }
                cfgHost[f] = row.repeatPenalty();
                cfgHost[f + 1] = row.frequencyPenalty();
                cfgHost[f + 2] = row.presencePenalty();
            }
            metaHost[m + 2] = penAt;
            metaHost[m + 3] = penCount;
            metaHost[m + 4] = row.penalties() ? 1 : 0;
            int dryAt = tokenAt;
            int dryCount = 0;
            int brAt = breakerAt;
            int brCount = 0;
            if (row.dry()) {
                int[] window = dryRowTokens[r];
                System.arraycopy(window, 0, tokenHost, tokenAt, window.length);
                tokenAt += window.length;
                dryCount = window.length;
                cfgHost[f + 3] = row.dryMultiplier();
                cfgHost[f + 4] = row.dryBase();
                DryBreakers breakers = dryRowBreakers[r];
                brCount = breakers.count();
                if (!sameBreakers) {
                    for (int i = 0; i < brCount; i++) {
                        int b = (breakerAt + i) * BREAKER_STRIDE;
                        breakerHost[b] = breakers.head(i);
                        breakerHost[b + 1] = tailAt + breakers.tailFrom(i);
                        breakerHost[b + 2] = breakers.tailLength(i);
                    }
                    breakers.copyTails(tailHost, tailAt);
                }
                breakerAt += brCount;
                tailAt += breakers.tailCount();
            }
            metaHost[m + 5] = dryAt;
            metaHost[m + 6] = dryCount;
            metaHost[m + 7] = row.dryAllowedLength();
            metaHost[m + 8] = row.dry() ? 1 : 0;
            metaHost[m + 9] = brAt;
            metaHost[m + 10] = brCount;
            metaHost[m + 11] = row.penaltiesFirst() ? 1 : 0;
        }
        writeInts(meta, metaHost);
        writeFloats(fcfg, cfgHost);
        writeInts(biasIds, biasIdHost);
        writeFloats(biasVals, biasValHost);
        writeInts(tokens, tokenHost);
        if (!sameBreakers) {
            writeInts(breakerMeta, breakerHost);
            writeInts(breakerTails, tailHost);
            lastBreakers = dryRowBreakers;
            breakerUploads++;
        } else {
            breakerUploadsSkipped++;
        }
        buffersChanged = false;

        if (rows != lastRows) {
            dispatch.setInt(1, rows);
            dispatch.setGlobal((long) rows * local);
            lastRows = rows;
        }
        dispatch.dispatch();
    }

    /** Whether the prefilter kernel runs on this backend. */
    public boolean available() {
        return dispatch != null;
    }

    private void checkIds(int[] ids, int otherLength, String what) {
        if (otherLength >= 0 && otherLength != ids.length) {
            throw new IllegalArgumentException(what + " ids and counts disagree in length");
        }
        for (int id : ids) {
            if (id < 0 || id >= vocab) {
                throw new IllegalArgumentException(what + " token id " + id + " outside the vocabulary of " + vocab);
            }
        }
    }

    /**
     * Grows the row tables and the flat arrays to hold one call, rebuilding the prepared dispatch when a buffer
     * was (re)allocated.
     */
    private void ensure(int rows, int bias, int tokenInts, int breakers, int tails) {
        boolean changed = false;
        if (metaRows < rows) {
            closeBuffer(meta);
            meta = allocate((long) rows * META_STRIDE * Integer.BYTES);
            metaRows = rows;
            changed = true;
        }
        if (fcfgRows < rows) {
            closeBuffer(fcfg);
            fcfg = allocate((long) rows * FCFG_STRIDE * Float.BYTES);
            fcfgRows = rows;
            changed = true;
        }
        if (biasIds == null || biasCapacity < bias) {
            closeBuffer(biasIds);
            closeBuffer(biasVals);
            biasIds = allocate((long) bias * Integer.BYTES);
            biasVals = allocate((long) bias * Float.BYTES);
            biasCapacity = bias;
            changed = true;
        }
        if (tokens == null || tokenCapacity < tokenInts) {
            closeBuffer(tokens);
            tokens = allocate((long) tokenInts * Integer.BYTES);
            tokenCapacity = tokenInts;
            changed = true;
        }
        if (breakerMeta == null || breakerCapacity < breakers) {
            closeBuffer(breakerMeta);
            breakerMeta = allocate((long) breakers * BREAKER_STRIDE * Integer.BYTES);
            breakerCapacity = breakers;
            changed = true;
        }
        if (breakerTails == null || tailCapacity < tails) {
            closeBuffer(breakerTails);
            breakerTails = allocate((long) tails * Integer.BYTES);
            tailCapacity = tails;
            changed = true;
        }
        buffersChanged |= changed;
        if (dispatch == null || changed) {
            if (dispatch != null) {
                dispatch.close();
            }
            Object[] args = new Object[]{vocab, rows, logits, meta, fcfg, biasIds, biasVals, tokens, breakerMeta,
                    breakerTails};
            dispatch = ops.prepare(GemmKernels.LOGITS_PREFILTER, new long[]{(long) rows * local},
                    new long[]{local}, args);
            lastRows = rows;
        }
    }

    private GpuBuffer allocate(long bytes) {
        return backend.allocateBuffer(Math.max(bytes, Integer.BYTES));
    }

    private void writeInts(GpuBuffer buffer, int[] values) {
        if (values.length == 0) {
            return;
        }
        byte[] bytes = new byte[values.length * Integer.BYTES];
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : values) {
            bb.putInt(v);
        }
        GpuEvent event = queue.enqueueWriteBuffer(buffer, 0, bytes.length, bytes);
        if (event != null) {
            event.close();
        }
    }

    private void writeFloats(GpuBuffer buffer, float[] values) {
        if (values.length == 0) {
            return;
        }
        GpuEvent event = queue.enqueueWriteBuffer(buffer, 0, (long) values.length * Float.BYTES, values);
        if (event != null) {
            event.close();
        }
    }

    private static void closeBuffer(GpuBuffer buffer) {
        if (buffer != null) {
            buffer.close();
        }
    }

    /** Releases the prepared dispatch and the tables. Idempotent. */
    @Override
    public void close() {
        if (dispatch != null) {
            dispatch.close();
            dispatch = null;
        }
        for (GpuBuffer b : new GpuBuffer[]{meta, fcfg, biasIds, biasVals, tokens, breakerMeta, breakerTails}) {
            closeBuffer(b);
        }
        meta = null;
        fcfg = null;
        biasIds = null;
        biasVals = null;
        tokens = null;
        breakerMeta = null;
        breakerTails = null;
    }

    @Override
    public String toString() {
        return "GpuLogitPrefilter{vocab=" + vocab + ", maxRows=" + maxRows + '}';
    }
}

package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.quant.write.ImatrixFile;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * The in-engine imatrix collector (llama.cpp {@code tools/imatrix/imatrix.cpp}
 * {@code collect_imatrix}): accumulates, per weight tensor, the column sums of the squared matmul activations, the
 * column sums of the plain activations and the activation-row counts — the
 * {@code <tensor>.in_sum2}/{@code <tensor>.in_sum}/{@code <tensor>.counts} statistics {@link ImatrixFile} stores.
 * A calibration run installs {@code this} through {@link CpuMatVec#observeGlobally} (the CPU matmuls) and feeds
 * the device matmul inputs of a GPU run through {@link #accumulate} (the GPU tap reads them back), then reads
 * {@link #entries()} afterwards.
 *
 * <p>The CPU observer fires inside every CPU matvec and matmul after the weight's input transform and before the
 * vec-dot quantization (the {@code src1} llama.cpp's graph callback sees). Per matrix of a 3-D expert tensor the
 * statistics are separate: {@code CpuMatVec.matvecExpert} dispatches one expert per call, whose index is
 * {@code rowStart / rows} ({@code rows} is the per-expert row count of that call shape), so a collection over a MoE
 * model fills one matrix per expert of every expert tensor it ran. The accumulation is synchronized: the observer
 * is called from {@code CpuParallel} workers on the per-token transform paths, and llama.cpp's collector holds a
 * mutex too. A matrix whose squared sums turn non-finite fails the collection naming the tensor, as the reference
 * aborts on non-finite values.</p>
 *
 * <p>CPU weights are named through the {@code namer} given at construction; a calibration tool passes the loaded
 * weights' catalog lookup ({@code CpuWeights.describe(w).name()}), the stored GGUF name llama.cpp keys its entries
 * by. A weight the namer cannot name fails the collection with the reason, as an unnamed statistic cannot be
 * quantized against. Entries are keyed by that name, so the CPU and GPU feeds of one run share them.</p>
 *
 * <p>Which products are collected is decided by the ask ({@link #wants(String, boolean, int)}), which both taps
 * make once per product before handing its rows over: llama.cpp's {@code collect_imatrix} with {@code ask = true}
 * takes every indirect product ({@code MUL_MAT_ID}), and a dense one ({@code MUL_MAT}) only over at least
 * {@link #LLAMA_MIN_DENSE_ROWS} input rows and only for the names the collection keeps.</p>
 */
public final class ImatrixCollector implements CpuMatVec.InputObserver {

    /**
     * The fewest input rows of a dense product llama.cpp's imatrix collects ({@code src1->ne[1] < 16} is declined):
     * single-token decodes and other small batches do not count.
     */
    public static final int LLAMA_MIN_DENSE_ROWS = 16;

    private final Function<CpuTensorRef, String> namer;
    /** The weight names whose dense products are kept; the others pass through uncollected. */
    private final Predicate<String> accept;
    /** The fewest input rows of a dense product that is collected. */
    private final int minDenseRows;
    private final Map<String, Accum> accums = new LinkedHashMap<>();
    /** The CPU weight each name was first observed for: two weights under one name fail the collection. */
    private final Map<CpuTensorRef, String> cpuWeights = new IdentityHashMap<>();
    private final Map<String, CpuTensorRef> cpuNames = new LinkedHashMap<>();

    /**
     * A collector that names weights through {@code namer} (the stored GGUF tensor name, llama.cpp's entry key)
     * and collects every product of every one of them.
     */
    public ImatrixCollector(Function<CpuTensorRef, String> namer) {
        this(namer, n -> true, 1);
    }

    /**
     * A collector with llama.cpp's ask: every indirect product, and the dense products of at least
     * {@code minDenseRows} input rows over the names {@code accept} keeps (the block tensors, with the output and
     * NextN tensors only when the run asks for them). A declined product produces no statistics, exactly as
     * llama.cpp declines the tensor before its collection.
     *
     * @throws IllegalArgumentException when {@code minDenseRows} is below 1
     */
    public ImatrixCollector(Function<CpuTensorRef, String> namer, Predicate<String> accept, int minDenseRows) {
        this.namer = Objects.requireNonNull(namer, "namer");
        this.accept = Objects.requireNonNull(accept, "accept");
        if (minDenseRows < 1) {
            throw new IllegalArgumentException("the fewest dense rows must be at least 1, got " + minDenseRows);
        }
        this.minDenseRows = minDenseRows;
    }

    /** Removes every collected statistic; the collector starts a fresh run. */
    public synchronized void reset() {
        accums.clear();
        cpuWeights.clear();
        cpuNames.clear();
    }

    /**
     * The ask half of llama.cpp's callback: true when a product of the weight {@code name} over {@code rows} input
     * rows is collected — every {@code indirect} (routed expert) product, a dense one over at least the minimum rows
     * of a kept name.
     */
    public boolean wants(String name, boolean indirect, int rows) {
        Objects.requireNonNull(name, "name");
        return indirect || (rows >= minDenseRows && accept.test(name));
    }

    /**
     * {@link #wants(String, boolean, int)} for a CPU weight, named through the namer.
     *
     * @throws IllegalStateException when the namer cannot name the weight, or names two weights alike
     */
    @Override
    public synchronized boolean wants(CpuTensorRef weight, boolean indirect, int rows) {
        return wants(nameOf(weight), indirect, rows);
    }

    /**
     * The collected statistics, one {@link ImatrixFile.Entry} per weight the matmuls saw, with the plain
     * activation sums, in first-seen order ({@code ImatrixFile.writeGguf} sorts by name). A weight whose matmuls
     * never ran holds no entry, as llama.cpp writes no entry for an uncollected tensor.
     */
    public synchronized List<ImatrixFile.Entry> entries() {
        List<ImatrixFile.Entry> out = new ArrayList<>(accums.size());
        for (Accum a : accums.values()) {
            out.add(new ImatrixFile.Entry(a.name, a.nMat, a.cols, a.sumSq, a.sums, a.counts));
        }
        return out;
    }

    /** How many weight tensors were collected so far. */
    public synchronized int tensorCount() {
        return accums.size();
    }

    /** Number of activation rows collected for {@code matrix} of the named weight, or 0 when none were. */
    public synchronized long count(String tensor, int matrix) {
        Accum a = accums.get(tensor);
        return a != null && matrix >= 0 && matrix < a.nMat ? a.counts[matrix] : 0L;
    }

    /** Adds the rows of a product the ask took ({@link #wants(CpuTensorRef, boolean, int)}). */
    @Override
    public synchronized void observe(CpuTensorRef weight, long rowStart, int rows, int experts, float[] x, int xOff,
                                     int xStride, int cols, int nTokens) {
        String name = nameOf(weight);
        int expert = experts > 0 ? (int) (rowStart / rows) : 0;
        accumulate(name, experts, expert, x, xOff, xStride, cols, nTokens);
    }

    /** The entry name of a CPU weight, resolved through the namer on first sight. */
    private String nameOf(CpuTensorRef weight) {
        if (weight == null) {
            throw new IllegalArgumentException("the imatrix observer received no weight");
        }
        String name = cpuWeights.get(weight);
        if (name == null) {
            name = namer.apply(weight);
            if (name == null || name.isEmpty()) {
                throw new IllegalStateException("the imatrix collector cannot name the weight " + weight);
            }
            if (cpuNames.containsKey(name)) {
                throw new IllegalStateException("two weights map to the imatrix entry '" + name + "'");
            }
            cpuWeights.put(weight, name);
            cpuNames.put(name, weight);
        }
        return name;
    }

    /**
     * Adds {@code nTokens} activation rows of {@code cols} values at {@code x[xOff + t * xStride]} to matrix
     * {@code expert} of the weight {@code name}: their squares and plain values to the column sums, their count to
     * the matrix count. {@code experts} is the matrix count of a 3-D weight (0 for a 2-D weight, whose one matrix
     * is 0). The caller asked first ({@link #wants(String, boolean, int)}).
     *
     * @throws IllegalArgumentException when the rows do not fit {@code x}
     * @throws IllegalStateException    when a later call reports another column or expert count for the weight, the
     *                                  expert is outside it, or the squared sums turn non-finite
     */
    public synchronized void accumulate(String name, int experts, int expert, float[] x, int xOff, int xStride,
                                        int cols, int nTokens) {
        Objects.requireNonNull(name, "name");
        if (nTokens < 0 || cols < 1 || x.length < (long) Math.max(nTokens - 1, 0) * xStride + xOff + cols) {
            throw new IllegalArgumentException("the imatrix rows of '" + name + "' do not carry " + nTokens
                    + " rows of " + cols + " values");
        }
        Accum a = accumulator(name, experts, cols);
        if (expert < 0 || expert >= a.nMat) {
            throw new IllegalStateException("the dispatch of '" + name + "' names expert " + expert
                    + " outside [0, " + a.nMat + ")");
        }
        int base = expert * cols;
        for (int t = 0; t < nTokens; t++) {
            int off = xOff + t * xStride;
            for (int j = 0; j < cols; j++) {
                float v = x[off + j];
                a.sumSq[base + j] += v * v;
                a.sums[base + j] += v;
            }
        }
        for (int j = base; j < base + cols; j++) {
            if (!Float.isFinite(a.sumSq[j])) {
                throw new IllegalStateException("non-finite values detected in the activations of '" + name + "'");
            }
        }
        a.counts[expert] += nTokens;
    }

    /** The accumulator of {@code name}, created on first use and checked against the call's shape after. */
    private Accum accumulator(String name, int experts, int cols) {
        Accum a = accums.get(name);
        if (a == null) {
            a = new Accum(name, Math.max(experts, 1), cols);
            accums.put(name, a);
            return a;
        }
        if (a.cols != cols) {
            throw new IllegalStateException("the matmuls of '" + name + "' collect over " + cols
                    + " columns now, first over " + a.cols);
        }
        if (experts > a.nMat) {
            throw new IllegalStateException("the matmuls of '" + name + "' report " + experts + " experts, first "
                    + a.nMat);
        }
        return a;
    }

    @Override
    public String toString() {
        return "ImatrixCollector{" + tensorCount() + " tensors}";
    }

    /** One weight's accumulated statistics. */
    private static final class Accum {
        final String name;
        final int cols;
        final int nMat;
        final float[] sumSq;
        final float[] sums;
        final long[] counts;

        Accum(String name, int nMat, int cols) {
            this.name = name;
            this.cols = cols;
            this.nMat = nMat;
            this.sumSq = new float[Math.multiplyExact(nMat, cols)];
            this.sums = new float[sumSq.length];
            this.counts = new long[nMat];
        }
    }
}

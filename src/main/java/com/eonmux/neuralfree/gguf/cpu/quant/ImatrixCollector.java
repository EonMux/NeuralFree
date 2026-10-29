package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.quant.write.ImatrixFile;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * The in-engine imatrix collector (PARITY.md P5, llama.cpp {@code tools/imatrix/imatrix.cpp}): a
 * {@link CpuMatVec.InputObserver} that accumulates, per weight tensor, the column sums of the squared matmul
 * activations and the activation-row counts — the {@code <tensor>.in_sum2}/{@code <tensor>.counts} statistics
 * {@link ImatrixFile} stores. A calibration run installs {@code this} through {@link CpuMatVec#observeGlobally},
 * prefills the calibration text, and reads {@link #entries()} afterwards; {@code ImatrixFile.writeGguf} writes them.
 *
 * <p>The observer fires inside every CPU matvec and matmul after the weight's input transform and before the vec-dot
 * quantization (the {@code src1} llama.cpp's graph callback sees). Per matrix of a 3-D expert tensor the statistics
 * are separate: {@code CpuMatVec.matvecExpert} dispatches one expert per call, whose index is {@code rowStart / rows}
 * ({@code rows} is the per-expert row count of that call shape), so a collection over a MoE model fills one matrix
 * per expert of every expert tensor it ran. The accumulation is synchronized: the observer is called from
 * {@code CpuParallel} workers on the per-token transform paths, and llama.cpp's collector holds a mutex too.</p>
 *
 * <p>Weights are named through the {@code namer} given at construction; a calibration tool passes the loaded
 * weights' catalog lookup ({@code CpuWeights.describe(w).name()}), the stored GGUF name llama.cpp keys its entries
 * by. A weight the namer cannot name fails the collection with the reason, as an unnamed statistic cannot be
 * quantized against.</p>
 */
public final class ImatrixCollector implements CpuMatVec.InputObserver {

    private final Function<CpuTensorRef, String> namer;
    private final Map<CpuTensorRef, Accum> accums = new IdentityHashMap<>();
    private final java.util.Set<String> names = new java.util.HashSet<>();

    /**
     * A collector that names weights through {@code namer} (the stored GGUF tensor name, llama.cpp's entry key).
     *
     * @throws IllegalArgumentException when a named weight's matrices do not split into the expert count its
     *                                  dispatch reports, or a later call reports another column count
     */
    public ImatrixCollector(Function<CpuTensorRef, String> namer) {
        this.namer = Objects.requireNonNull(namer, "namer");
    }

    /** Removes every collected statistic; the collector starts a fresh run. */
    public synchronized void reset() {
        accums.clear();
        names.clear();
    }

    /**
     * The collected statistics, one {@link ImatrixFile.Entry} per weight the matmuls saw, in first-seen order
     * ({@code ImatrixFile.writeGguf} sorts by name). A weight whose matmuls never ran holds no entry, as llama.cpp
     * writes no entry for an uncollected tensor.
     */
    public synchronized List<ImatrixFile.Entry> entries() {
        List<ImatrixFile.Entry> out = new ArrayList<>(accums.size());
        for (Accum a : accums.values()) {
            out.add(new ImatrixFile.Entry(a.name, a.nMat, a.cols, a.sumSq, a.counts));
        }
        return out;
    }

    /** How many weight tensors were collected so far. */
    public synchronized int tensorCount() {
        return accums.size();
    }

    /** Number of activation rows collected for {@code matrix} of the named weight, or 0 when none were. */
    public synchronized long count(String tensor, int matrix) {
        for (Accum a : accums.values()) {
            if (a.name.equals(tensor)) {
                return matrix >= 0 && matrix < a.nMat ? a.counts[matrix] : 0L;
            }
        }
        return 0L;
    }

    @Override
    public synchronized void observe(CpuTensorRef weight, long rowStart, int rows, int experts, float[] x, int xOff,
                                     int xStride, int cols, int nTokens) {
        if (weight == null) {
            throw new IllegalArgumentException("the imatrix observer received no weight");
        }
        if (nTokens < 0 || x.length < (long) Math.max(nTokens - 1, 0) * xStride + xOff + cols) {
            throw new IllegalArgumentException("the imatrix observer's activation rows do not carry "
                    + nTokens + " rows of " + cols + " values");
        }
        String name = namer.apply(weight);
        if (name == null || name.isEmpty()) {
            throw new IllegalStateException("the imatrix collector cannot name the weight " + weight);
        }
        if (!names.contains(name)) {
            names.add(name);
        } else if (!accums.containsKey(weight)) {
            throw new IllegalStateException("two weights map to the imatrix entry '" + name + "'");
        }
        Accum a = accums.get(weight);
        if (a == null) {
            int nMat = Math.max(experts, 1);
            a = new Accum(name, nMat, cols);
            accums.put(weight, a);
        } else {
            if (!a.name.equals(name)) {
                throw new IllegalStateException("two weights map to the imatrix entry '" + a.name + "'");
            }
            if (a.cols != cols) {
                throw new IllegalStateException("the matmuls of '" + name + "' collect over " + cols
                        + " columns now, first over " + a.cols);
            }
            if (experts > a.nMat) {
                throw new IllegalStateException("the matmuls of '" + name + "' report " + experts
                        + " experts, first " + a.nMat);
            }
        }
        int expert = experts > 0 ? (int) (rowStart / rows) : 0;
        if (expert < 0 || expert >= a.nMat) {
            throw new IllegalStateException("the dispatch of '" + name + "' at rows " + rowStart + ".."
                    + (rowStart + rows) + " names expert " + expert + " outside [0, " + a.nMat + ")");
        }
        float[] sumSq = a.sumSq;
        int base = expert * cols;
        for (int t = 0; t < nTokens; t++) {
            int off = xOff + t * xStride;
            for (int j = 0; j < cols; j++) {
                float v = x[off + j];
                sumSq[base + j] += v * v;
            }
        }
        a.counts[expert] += nTokens;
    }

    @Override
    public String toString() {
        return "ImatrixCollector{" + tensorCount() + " tensors}";
    }

    /** One weight's accumulated statistics; grows its matrices when an expert dispatch first reaches one. */
    private static final class Accum {
        final String name;
        final int cols;
        int nMat;
        float[] sumSq;
        long[] counts;

        Accum(String name, int nMat, int cols) {
            this.name = name;
            this.cols = cols;
            this.nMat = nMat;
            this.sumSq = new float[nMat * cols];
            this.counts = new long[nMat];
        }
    }
}

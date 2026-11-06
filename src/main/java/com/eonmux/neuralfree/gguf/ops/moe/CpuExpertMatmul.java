package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuActivationQuantizer;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.InputTransform;

import java.util.Arrays;
import java.util.Objects;

/**
 * Expert products of one MoE step grouped by expert, the CPU twin of {@code ggml_mul_mat_id}
 * (llama.cpp ggml/src/ggml-cpu/ggml-cpu.c:1554-1735):
 * <ol>
 *   <li>every distinct source row is transformed by the weight's input transform ({@link InputTransform}) when it
 *       has one, in the transform order without its PERMUTE part (a 3-D expert tensor folds its input permutation
 *       into the weights): the PRE_SCALE multipliers run per source row (or per gathered row with the pair's
 *       expert when each expert has its own), the HADAMARD rotation runs on the gathered rows once, and the
 *       ACT_QUANT fake-quantization runs per source row, or, when each expert has its own {@code .input_scale},
 *       per gathered row with that expert's parameters; the rows are then quantized once into the weight type's
 *       {@code vec_dot_type} (the {@code from_float} pass, :1609-1644);</li>
 *   <li>the step's (token, slot) pairs are grouped by expert ({@link Groups}, the {@code matrix_row_counts} /
 *       {@code matrix_rows} pass, :1646-1661);</li>
 *   <li>each active expert's rows are multiplied with all of its pairs in tiles of {@value #ROW_TILE} weight rows by
 *       at most {@value #PAIR_BLOCK} pairs (the {@code blck_0 x blck_1} chunks of
 *       {@code ggml_compute_forward_mul_mat_id_one_chunk}, :1483-1545), so an expert's weights are streamed once per
 *       step and stay in cache across its pairs; all tiles of all experts run in one {@link CpuParallel} loop.</li>
 * </ol>
 * The output of pair {@code p} is {@code y[yOff + p*R + r] = scale(e) * sum_c W[e*R + r][c] * xs(p)[c]} for
 * {@code r < R = rows / nExpert}, with {@code e} the pair's expert, {@code xs(p)} the pair's source row and
 * {@code scale(e)} the weight's output scale {@code w_s} (per expert or per tensor,
 * llama-graph.cpp:1722-1729) resolved through the {@link CpuMatVec} transforms. The source row of pair {@code p}
 * is {@code p / pairsPerRow} of the row-major input, or {@code srcRows[p]} when the pairs carry their rows
 * (a hybrid step's host pairs are a subset of the routed pairs). Each output value is the same
 * vec-dot of the same quantized row as a per-pair {@link CpuMatVec#matvecExpert} call, so grouping changes the memory
 * traffic and the fork/join count, never the result.
 *
 * <p>Instances are immutable and thread-safe; each call allocates its quantized activations and tile index.</p>
 */
public final class CpuExpertMatmul {

    /** Weight rows per tile ({@code ggml_compute_forward_mul_mat_id} chunk size for one-row operands, :1690-1693). */
    static final int ROW_TILE = 64;
    /** Pairs per tile, the {@code blck_1} block of {@code ggml_compute_forward_mul_mat_id_one_chunk}. */
    static final int PAIR_BLOCK = 16;
    /** Multiply-accumulates per parallel task below which tiles are not split across workers. */
    private static final long MIN_TASK_WORK = 1L << 15;

    private final CpuMatVec transforms;

    /** A grouped product that resolves each weight's input Hadamard and output scale through {@code transforms}. */
    public CpuExpertMatmul(CpuMatVec transforms) {
        this.transforms = Objects.requireNonNull(transforms, "transforms");
    }

    /**
     * Receives every output tile of a grouped product once its rows are written, with the pair and the row range of
     * the weight the tile covers. The fused int8 expert pass applies its bias and gated activation here, so the tile
     * is consumed while it is still in cache. Implementations must be thread-safe and write only the rows of that
     * pair's own output.
     */
    @FunctionalInterface
    public interface TileEpilogue {

        /** Epilogue of a product that needs none. */
        TileEpilogue NONE = (pair, rowStart, rows) -> { };

        /**
         * @param pair     pair index into the grouped output, {@code y[yOff + pair * perExpert + rowStart ..]}
         * @param rowStart first weight row of the tile, relative to the expert's row block
         * @param rows     weight rows the tile covers
         */
        void tile(int pair, int rowStart, int rows);
    }

    /**
     * The pairs of one step grouped by expert: a counting sort of the pair ids. Within an expert the pairs keep
     * ascending pair order. Immutable after creation.
     */
    public static final class Groups {
        final int nExpert;
        final int pairs;
        /** Experts with at least one pair, ascending. */
        final int[] active;
        /** Pairs of {@code active[a]} are {@code order[start[a] .. start[a + 1])}. */
        final int[] start;
        /** Pair indices grouped by expert. */
        final int[] order;

        private Groups(int nExpert, int pairs, int[] active, int[] start, int[] order) {
            this.nExpert = nExpert;
            this.pairs = pairs;
            this.active = active;
            this.start = start;
            this.order = order;
        }

        /**
         * Groups pairs {@code [0, pairs)} whose experts are {@code ids[p]}.
         *
         * @throws IllegalArgumentException for a negative pair count, an id array shorter than the pairs, a
         *                                  non-positive expert count, or an id outside {@code [0, nExpert)}
         */
        public static Groups of(int[] ids, int pairs, int nExpert) {
            Objects.requireNonNull(ids, "ids");
            if (nExpert < 1) {
                throw new IllegalArgumentException("expert count must be positive, got " + nExpert);
            }
            if (pairs < 0 || pairs > ids.length) {
                throw new IllegalArgumentException("pair count " + pairs + " outside [0, " + ids.length + "]");
            }
            int[] counts = new int[nExpert];
            for (int p = 0; p < pairs; p++) {
                int e = ids[p];
                if (e < 0 || e >= nExpert) {
                    throw new IllegalArgumentException("pair " + p + " selects expert " + e + ", outside [0, "
                            + nExpert + ")");
                }
                counts[e]++;
            }
            int nActive = 0;
            for (int c : counts) {
                if (c > 0) {
                    nActive++;
                }
            }
            int[] active = new int[nActive];
            int[] start = new int[nActive + 1];
            int[] next = new int[nExpert];
            int a = 0;
            for (int e = 0; e < nExpert; e++) {
                if (counts[e] > 0) {
                    active[a] = e;
                    next[e] = start[a];
                    start[a + 1] = start[a] + counts[e];
                    a++;
                }
            }
            int[] order = new int[pairs];
            for (int p = 0; p < pairs; p++) {
                order[next[ids[p]]++] = p;
            }
            return new Groups(nExpert, pairs, active, start, order);
        }

        /** Number of pairs routed to {@code active[a]}. */
        int count(int a) {
            return start[a + 1] - start[a];
        }

        /** The group's pair count. */
        public int pairs() {
            return pairs;
        }

        @Override
        public String toString() {
            return "Groups{pairs=" + pairs + ", experts=" + Arrays.toString(active) + '}';
        }
    }

    /**
     * Grouped product of the 3-D expert tensor {@code w3d} holding {@code nExpert} experts stacked along its rows:
     * for every pair {@code p} of {@code g}, {@code y[yOff + p*R .. + R)} receives expert {@code ids[p]}'s rows applied
     * to source row {@code p / pairsPerRow}, which starts at {@code x[xOff + (p / pairsPerRow) * cols]}.
     *
     * @param pairsPerRow pairs sharing one source row: {@code k} when the pairs read their token's row, 1 when every
     *                    pair has its own row
     * @throws IllegalArgumentException when the rows do not split into {@code nExpert} experts, {@code nExpert}
     *                                  contradicts the weight's known expert count or the grouping, or a view
     *                                  exceeds its array
     */
    public void run(CpuTensorRef w3d, int nExpert, Groups g, float[] x, int xOff, int pairsPerRow, float[] y,
                    int yOff) {
        run(w3d, nExpert, g, x, xOff, pairsPerRow, y, yOff, TileEpilogue.NONE);
    }

    /**
     * {@link #run(CpuTensorRef, int, Groups, float[], int, int, float[], int)} with an epilogue over every written
     * tile, the hook the fused int8 expert pass applies its bias and gated activation through.
     */
    public void run(CpuTensorRef w3d, int nExpert, Groups g, float[] x, int xOff, int pairsPerRow, float[] y,
                    int yOff, TileEpilogue epilogue) {
        if (pairsPerRow < 1) {
            throw new IllegalArgumentException("pairs per source row must be positive, got " + pairsPerRow);
        }
        run(w3d, nExpert, g, x, xOff, null, y, yOff, pairsPerRow, Objects.requireNonNull(epilogue, "epilogue"));
    }

    /**
     * Grouped product over pairs with explicit source rows: pair {@code p} reads source row {@code srcRows[p]},
     * which starts at {@code x[xOff + srcRows[p] * cols]}. A hybrid step's host pairs are a subset of the routed
     * pairs, so their token rows are not at {@code p / pairsPerRow} of the compacted list.
     *
     * @param srcRows source row of every pair; entries may repeat (the pairs of one token share their row)
     * @throws IllegalArgumentException as {@link #run(CpuTensorRef, int, Groups, float[], int, int, float[], int)}
     */
    public void run(CpuTensorRef w3d, int nExpert, Groups g, float[] x, int xOff, int[] srcRows, float[] y,
                    int yOff) {
        run(w3d, nExpert, g, x, xOff, srcRows, y, yOff, TileEpilogue.NONE);
    }

    /** {@link #run(CpuTensorRef, int, Groups, float[], int, int[], float[], int)} with a tile epilogue. */
    public void run(CpuTensorRef w3d, int nExpert, Groups g, float[] x, int xOff, int[] srcRows, float[] y,
                    int yOff, TileEpilogue epilogue) {
        if (srcRows == null) {
            throw new IllegalArgumentException("srcRows must not be null; use the pairsPerRow overload");
        }
        run(w3d, nExpert, g, x, xOff, srcRows, y, yOff, 0, Objects.requireNonNull(epilogue, "epilogue"));
    }

    /**
     * The grouped product with one of the two source-row rules: {@code srcRows != null} gives every pair its own
     * row, otherwise {@code pairsPerRow} pairs share a row.
     */
    private void run(CpuTensorRef w3d, int nExpert, Groups g, float[] x, int xOff, int[] srcRows, float[] y,
                     int yOff, int pairsPerRow, TileEpilogue epilogue) {
        Objects.requireNonNull(w3d, "weight");
        Objects.requireNonNull(g, "groups");
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        if (nExpert < 1 || w3d.rows() % nExpert != 0) {
            throw new IllegalArgumentException("expert count " + nExpert + " does not divide the " + w3d.rows()
                    + " rows of " + w3d);
        }
        if (g.nExpert != nExpert) {
            throw new IllegalArgumentException("pairs grouped over " + g.nExpert + " experts, the weight " + w3d
                    + " holds " + nExpert);
        }
        CpuMatVec.Transforms t = transforms.transformsOf(w3d);
        InputTransform transform = t.transform();
        if (transform.hasPerm()) {
            throw new IllegalArgumentException("the input transform of " + w3d + " permutes its input; a 3-D expert"
                    + " tensor folds its input permutation into the weights (TP-1)");
        }
        if (t.experts() != 0 && t.experts() != nExpert) {
            throw new IllegalArgumentException("expert count " + nExpert + " contradicts the " + t.experts()
                    + " experts of " + w3d);
        }
        int cols = toInt(w3d.cols(), w3d);
        int perExpert = toInt(w3d.rows() / nExpert, w3d);
        // The row of the input that pair p reads, and the number of input rows in play.
        int maxRow = -1;
        if (srcRows != null) {
            if (g.pairs > srcRows.length) {
                throw new IllegalArgumentException("the grouping holds " + g.pairs + " pairs, srcRows covers "
                        + srcRows.length);
            }
            for (int p = 0; p < g.pairs; p++) {
                if (srcRows[p] < 0) {
                    throw new IllegalArgumentException("pair " + p + " reads source row " + srcRows[p]);
                }
                maxRow = Math.max(maxRow, srcRows[p]);
            }
        }
        int nSrc = srcRows != null ? maxRow + 1 : (g.pairs + pairsPerRow - 1) / pairsPerRow;
        checkView("input", x, xOff, (long) nSrc * cols);
        checkView("output", y, yOff, (long) g.pairs * perExpert);
        if (g.pairs == 0 || perExpert == 0) {
            return;
        }
        CpuVecDot dot = CpuMatVec.vecDot(w3d.type());
        HadamardSpec hadamard = transform.hadamard();
        InputTransform.ActQuant actQuant = transform.actQuant();
        float[] preScale = transform.preScale();
        boolean prePerExpert = preScale != null && preScale.length > cols;
        boolean actPerExpert = actQuant != null && actQuant.entries() > 1;
        // A weight whose transform gathers or rewrites whole rows (shared pre-scale, HADAMARD rotation, shared
        // ACT_QUANT) reads them contiguously; explicit source rows are gathered into this call's scratch first, one
        // row per pair. A per-expert pre-scale gives every pair its own scaled row, so it runs per pair.
        boolean gather = srcRows != null && !prePerExpert
                && (hadamard != null || preScale != null || (actQuant != null && !actPerExpert));
        float[] rows = x;
        int rowsOff = xOff;
        if (gather) {
            rows = new float[Math.multiplyExact(g.pairs, cols)];
            rowsOff = 0;
            for (int p = 0; p < g.pairs; p++) {
                System.arraycopy(x, xOff + srcRows[p] * cols, rows, p * cols, cols);
            }
            nSrc = g.pairs;
        }
        // The pre-scale and the HADAMARD rotation write this call's scratch; the ACT_QUANT fake-quantization acts in
        // place on a gathered or copied set of rows when every expert shares its parameters (once per source row),
        // and per gathered row (once per pair, with the expert's .input_scale) when each expert has its own.
        float[] scratch = null;
        if (!prePerExpert && preScale != null) {
            if (rows == x) {
                scratch = new float[Math.multiplyExact(nSrc, cols)];
                for (int r = 0; r < nSrc; r++) {
                    System.arraycopy(x, xOff + r * cols, scratch, r * cols, cols);
                }
                rows = scratch;
                rowsOff = 0;
            }
            transform.applyPreScale(rows, rowsOff, nSrc, cols, 0);
        }
        if (!prePerExpert && hadamard != null) {
            scratch = new float[Math.multiplyExact(nSrc, cols)];
            CpuHadamard.apply(hadamard, rows, rowsOff, scratch, 0, nSrc, cols);
            rows = scratch;
            rowsOff = 0;
        }
        // Row of the working set that pair p reads: its own gathered row, its explicit source row, or its position.
        final int pairsPerSource = gather ? 1 : pairsPerRow;
        final int[] pairRow = gather ? null : srcRows;
        final QuantizedActivations[] rowActs;
        final QuantizedActivations[] pairActs;
        if (prePerExpert || actPerExpert) {
            QuantizedActivations[] acts = new QuantizedActivations[g.pairs];
            int[] pairExpert = pairExperts(g);
            final float[] buf = new float[Math.multiplyExact(g.pairs, cols)];
            final float[] perPairRotated = prePerExpert && hadamard != null
                    ? new float[Math.multiplyExact(g.pairs, cols)] : null;
            final float[] rotated = rows;
            final int rotOff = rowsOff;
            final GGMLType wType = w3d.type();
            final InputTransform.ActQuant q = actQuant;
            final InputTransform pre = transform;
            final boolean perExpertScale = prePerExpert;
            final int[] expertsOfPair = pairExpert;
            CpuParallel.parallelFor(g.pairs, 1, (from, to) -> {
                for (int p = from; p < to; p++) {
                    int bufOff = p * cols;
                    int row = pairRow != null ? pairRow[p] : p / pairsPerSource;
                    int e = expertsOfPair[p];
                    System.arraycopy(rotated, rotOff + row * cols, buf, bufOff, cols);
                    if (perExpertScale) {
                        pre.applyPreScale(buf, bufOff, 1, cols, e);
                    }
                    if (perPairRotated != null) {
                        CpuHadamard.apply(hadamard, buf, bufOff, perPairRotated, bufOff, 1, cols);
                        if (q != null) {
                            q.apply(perPairRotated, bufOff, 1, cols, e);
                        }
                        acts[p] = CpuActivationQuantizer.quantizeFor(wType, perPairRotated, bufOff, cols);
                    } else {
                        if (q != null) {
                            q.apply(buf, bufOff, 1, cols, e);
                        }
                        acts[p] = CpuActivationQuantizer.quantizeFor(wType, buf, bufOff, cols);
                    }
                }
            });
            observePairs(w3d, nExpert, perExpert, g, perPairRotated != null ? perPairRotated : buf, 0, null, 1,
                    cols);
            pairActs = acts;
            rowActs = null;
        } else {
            if (actQuant != null) {
                if (rows == x) {
                    scratch = new float[Math.multiplyExact(nSrc, cols)];
                    for (int r = 0; r < nSrc; r++) {
                        System.arraycopy(x, xOff + r * cols, scratch, r * cols, cols);
                    }
                    rows = scratch;
                    rowsOff = 0;
                }
                actQuant.apply(rows, rowsOff, nSrc, cols, 0);
            }
            QuantizedActivations[] acts = new QuantizedActivations[nSrc];
            final float[] s = rows;
            final int so = rowsOff;
            CpuParallel.parallelFor(nSrc, 1, (from, to) -> {
                for (int r = from; r < to; r++) {
                    acts[r] = CpuActivationQuantizer.quantizeFor(w3d.type(), s, so + r * cols, cols);
                }
            });
            observePairs(w3d, nExpert, perExpert, g, rows, rowsOff, pairRow, pairsPerSource, cols);
            rowActs = acts;
            pairActs = null;
        }
        int[] tileStart = tileStarts(g, perExpert);
        int nTiles = tileStart[g.active.length];
        int grain = (int) Math.max(1L, MIN_TASK_WORK / ((long) ROW_TILE * cols));
        CpuParallel.parallelFor(nTiles, grain, (from, to) -> {
            int a = slotOf(tileStart, from);
            for (int tile = from; tile < to; tile++) {
                while (tile >= tileStart[a + 1]) {
                    a++;
                }
                int blocks = (g.count(a) + PAIR_BLOCK - 1) / PAIR_BLOCK;
                int local = tile - tileStart[a];
                int r0 = (local / blocks) * ROW_TILE;
                int nr = Math.min(ROW_TILE, perExpert - r0);
                int q0 = g.start[a] + (local % blocks) * PAIR_BLOCK;
                int q1 = Math.min(q0 + PAIR_BLOCK, g.start[a + 1]);
                int e = g.active[a];
                long wRow = (long) e * perExpert + r0;
                float s = t.scale(e);
                for (int q = q0; q < q1; q++) {
                    int p = g.order[q];
                    int yBase = yOff + p * perExpert + r0;
                    int row = pairRow != null ? pairRow[p] : p / pairsPerSource;
                    QuantizedActivations act = rowActs != null ? rowActs[row] : pairActs[p];
                    dot.dotRows(w3d, wRow, nr, act, y, yBase);
                    if (t.hasScale()) {
                        for (int i = 0; i < nr; i++) {
                            y[yBase + i] *= s;
                        }
                    }
                    epilogue.tile(p, r0, nr);
                }
            }
        });
    }

    /**
     * Prefix of the tile counts of the active experts: expert slot {@code a} owns tiles
     * {@code [starts[a], starts[a + 1])}, ordered row tile major and pair block minor so that consecutive tiles
     * reuse the same weight rows.
     */
    private static int[] tileStarts(Groups g, int perExpert) {
        long rowTiles = (perExpert + ROW_TILE - 1) / ROW_TILE;
        int[] starts = new int[g.active.length + 1];
        long total = 0;
        for (int a = 0; a < g.active.length; a++) {
            total += rowTiles * ((g.count(a) + PAIR_BLOCK - 1) / PAIR_BLOCK);
            if (total > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("grouped expert product of " + g.pairs + " pairs x " + perExpert
                        + " rows needs more than " + Integer.MAX_VALUE + " tiles");
            }
            starts[a + 1] = (int) total;
        }
        return starts;
    }

    /** The expert slot owning {@code tile}: the last {@code a} with {@code starts[a] <= tile}. */
    private static int slotOf(int[] starts, int tile) {
        int lo = 0;
        int hi = starts.length - 2;
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (starts[mid] <= tile) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    /**
     * Hands every pair's transformed input row to the installed {@link CpuMatVec.InputObserver}, as a per-pair
     * {@link CpuMatVec#matvecExpert} call would (llama.cpp's imatrix callback walks the {@code MUL_MAT_ID} ids the
     * same way): pair {@code p} reports its row of {@code rows} — {@code srcRow[p]}, else {@code p / perSource} —
     * with its expert's row range of {@code w3d}.
     */
    private static void observePairs(CpuTensorRef w3d, int nExpert, int perExpert, Groups g, float[] rows,
                                     int rowsOff, int[] srcRow, int perSource, int cols) {
        CpuMatVec.InputObserver observer = CpuMatVec.askedObserver(w3d, true, g.pairs);
        if (observer == null) {
            return;
        }
        int[] expert = pairExperts(g);
        for (int p = 0; p < g.pairs; p++) {
            int row = srcRow != null ? srcRow[p] : p / perSource;
            observer.observe(w3d, (long) expert[p] * perExpert, perExpert, nExpert, rows, rowsOff + row * cols,
                    cols, cols, 1);
        }
    }

    /** The expert of every pair: {@code pairExperts(g)[p]} is {@code g}'s expert slot of pair {@code p}. */
    private static int[] pairExperts(Groups g) {
        int[] expert = new int[g.pairs];
        for (int a = 0; a < g.active.length; a++) {
            int e = g.active[a];
            for (int i = g.start[a]; i < g.start[a + 1]; i++) {
                expert[g.order[i]] = e;
            }
        }
        return expert;
    }

    private static int toInt(long v, CpuTensorRef w) {
        if (v > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("dimension " + v + " of " + w + " exceeds one float array");
        }
        return (int) v;
    }

    private static void checkView(String what, float[] a, int off, long length) {
        if (off < 0 || off + length > a.length) {
            throw new IllegalArgumentException(what + " view [" + off + ", " + (off + length)
                    + ") exceeds the array length " + a.length);
        }
    }

    @Override
    public String toString() {
        return "CpuExpertMatmul{" + transforms + '}';
    }
}

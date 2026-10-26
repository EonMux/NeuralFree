package com.eonmux.neuralfree.gguf.engine;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * Ports of llama.cpp's {@code llama_batch_allocr} split modes (src/llama-batch.cpp:585, 619, 790), over the MS
 * batch model where each token belongs to exactly one sequence (design invariant 5; llama.cpp allows several
 * sequences per token, NT invariant does not need it).
 *
 * <ul>
 *   <li>{@link Mode#SIMPLE} (split_simple): the next unused tokens in batch order, up to {@code nUbatch} per
 *       ubatch.</li>
 *   <li>{@link Mode#EQUAL} (split_equal, used when the memory has recurrent state, MS invariant 6): each ubatch
 *       takes {@code floor(nUbatch / nSeqs)} tokens from every sequence of its group, the group being the
 *       sequences that still hold tokens, in order of first appearance in the batch and capped at {@code
 *       nUbatch} sequences; a sequence is always allowed to finish its part, so the per-sequence counts of an
 *       ubatch differ by at most one.</li>
 *   <li>{@link Mode#SEQ} (split_seq): one sequence per ubatch, up to {@code nUbatch} tokens of it.</li>
 * </ul>
 *
 * <p>{@link #split} drains the batch: the union of the ubatches is the batch, every row once. Rows inside an
 * ubatch are ordered by sequence then position (the {@code UBatch} rule), whatever order the mode collected them
 * in, so a call to {@link BatchSplitter#split} with @code paged=false} on a single-sequence batch produces exactly
 * today's contiguous prefill chunks.</p>
 */
public final class BatchSplitter {

    /** Split modes (llama.cpp src/llama-batch.cpp:585 split_simple, :619 split_equal, :790 split_seq). */
    public enum Mode { SIMPLE, EQUAL, SEQ }

    private final boolean paged;

    /** A splitter over un-paged memory: a single-sequence ubatch is the contiguous path of today. */
    public BatchSplitter() {
        this(false);
    }

    /**
     * A splitter whose {@code StepLayout}s mark whether the memory is paged: a single-sequence ubatch over paged
     * memory is handled by the paged step path, not by the contiguous delegation.
     */
    public BatchSplitter(boolean pagedMemory) {
        this.paged = pagedMemory;
    }

    /**
     * Splits {@code batch} into ubatches by {@code mode}, draining its rows.
     *
     * @param batch   the batch to split, never empty
     * @param nUbatch maximum rows per ubatch, positive
     * @param mode    the split mode
     * @return one non-empty ubatch per element; their {@code batchRows()} partition the batch rows
     * @throws IllegalArgumentException when the batch is empty, {@code nUbatch} is not positive or {@code mode}
     *                                  is null
     */
    public List<UBatch> split(ForwardBatch batch, int nUbatch, Mode mode) {
        Objects.requireNonNull(batch, "batch");
        Objects.requireNonNull(mode, "mode");
        if (nUbatch <= 0) {
            throw new IllegalArgumentException("nUbatch must be positive, got " + nUbatch);
        }
        if (batch.nTokens() <= 0) {
            throw new IllegalArgumentException("the batch holds at least one row");
        }
        List<int[]> rowLists;
        switch (mode) {
            case SIMPLE:
                rowLists = simple(batch, nUbatch);
                break;
            case EQUAL:
                rowLists = equal(batch, nUbatch);
                break;
            case SEQ:
                rowLists = seq(batch, nUbatch);
                break;
            default:
                throw new IllegalArgumentException("unknown split mode " + mode);
        }
        int[] seqs = batch.seqIds();
        int[] pos = batch.positions();
        List<UBatch> out = new ArrayList<>(rowLists.size());
        for (int[] rows : rowLists) {
            out.add(UBatch.of(batch, sorted(rows, seqs, pos), paged));
        }
        return out;
    }

    /** llama.cpp split_simple: chunks of at most {@code nUbatch} rows in batch order. */
    private List<int[]> simple(ForwardBatch batch, int nUbatch) {
        int n = batch.nTokens();
        List<int[]> out = new ArrayList<>((n + nUbatch - 1) / nUbatch);
        for (int cur = 0; cur < n; cur += nUbatch) {
            int take = Math.min(nUbatch, n - cur);
            int[] rows = new int[take];
            for (int i = 0; i < take; i++) {
                rows[i] = cur + i;
            }
            out.add(rows);
        }
        return out;
    }

    /** llama.cpp split_equal: each round takes one token per remaining sequence, up to the round cap. */
    private List<int[]> equal(ForwardBatch batch, int nUbatch) {
        int n = batch.nTokens();
        int[] seqs = batch.seqIds();
        // sequences in first-appearance order: the non-overlapping sequence sets of llama.cpp
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (!order.contains(seqs[i])) {
                order.add(seqs[i]);
            }
        }
        int nSeqs = order.size();
        int[][] rowsOfSeq = rowsBySeq(batch, order);
        int[] cursor = new int[nSeqs];
        List<int[]> out = new ArrayList<>();
        while (true) {
            // the group: sequences that still hold tokens, in first-appearance order, capped at nUbatch
            List<Integer> group = new ArrayList<>();
            for (int s = 0; s < nSeqs && group.size() < nUbatch; s++) {
                if (cursor[s] < rowsOfSeq[s].length) {
                    group.add(s);
                }
            }
            if (group.isEmpty()) {
                break;
            }
            List<Integer> rows = new ArrayList<>();
            int rounds = 0;
            while (true) {
                boolean allHoldTokens = true;
                for (int s : group) {
                    allHoldTokens &= cursor[s] < rowsOfSeq[s].length;
                }
                if (!allHoldTokens || (rounds + 1) * group.size() > nUbatch) {
                    break;
                }
                for (int s : group) {
                    rows.add(rowsOfSeq[s][cursor[s]++]);
                }
                rounds++;
            }
            // at least one round always runs: every group member holds a token and fits the cap
            out.add(intArray(rows));
        }
        return out;
    }

    /** llama.cpp split_seq: runs of one sequence, at most nUbatch tokens per ubatch. */
    private List<int[]> seq(ForwardBatch batch, int nUbatch) {
        int n = batch.nTokens();
        int[] seqs = batch.seqIds();
        List<int[]> out = new ArrayList<>();
        int cur = 0;
        while (cur < n) {
            int run = seqs[cur];
            int take = 0;
            while (cur < n && seqs[cur] == run && take < nUbatch) {
                cur++;
                take++;
            }
            if (take == 0) {
                break;
            }
            int[] rows = new int[take];
            for (int i = 0; i < take; i++) {
                rows[i] = cur - take + i;
            }
            out.add(rows);
        }
        return out;
    }

    /** The batch rows of each sequence in {@code order}, batch ordered. */
    private int[][] rowsBySeq(ForwardBatch batch, List<Integer> order) {
        int n = batch.nTokens();
        int[] seqs = batch.seqIds();
        int[] len = new int[order.size()];
        for (int i = 0; i < n; i++) {
            len[order.indexOf(seqs[i])]++;
        }
        int[][] out = new int[order.size()][];
        for (int s = 0; s < order.size(); s++) {
            out[s] = new int[len[s]];
            len[s] = 0;
        }
        for (int i = 0; i < n; i++) {
            int s = order.indexOf(seqs[i]);
            out[s][len[s]++] = i;
        }
        return out;
    }

    private int[] intArray(List<Integer> rows) {
        int[] out = new int[rows.size()];
        int i = 0;
        for (int row : rows) {
            out[i++] = row;
        }
        return out;
    }

    /** Rows ordered by sequence then position (stable: batch order breaks ties, which cannot occur). */
    private static int[] sorted(int[] rows, int[] seqs, int[] pos) {
        if (rows.length <= 1) {
            return rows;
        }
        Integer[] boxed = new Integer[rows.length];
        for (int i = 0; i < rows.length; i++) {
            boxed[i] = rows[i];
        }
        Arrays.sort(boxed, Comparator.comparingInt((Integer r) -> seqs[r]).thenComparingInt(r -> pos[r]));
        int[] out = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            out[i] = boxed[i];
        }
        return out;
    }
}

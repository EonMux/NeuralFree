package com.eonmux.neuralfree.gguf.engine;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Ports of llama.cpp's {@code llama_batch_allocr} split modes (src/llama-batch.cpp:585, 619, 790), over this
 * engine's batch model where each token belongs to exactly one sequence (llama.cpp allows several
 * sequences per token; the splits do not need that).
 *
 * <ul>
 *   <li>{@link Mode#SIMPLE} (split_simple): the next unused tokens in batch order, up to {@code nUbatch} per
 *       ubatch.</li>
 *   <li>{@link Mode#EQUAL} (split_equal, used when the memory has recurrent state): each ubatch
 *       takes {@code floor(nUbatch / nSeqs)} tokens from every sequence of its group, the group being the
 *       sequences that still hold tokens, in order of first appearance in the batch and capped at {@code
 *       nUbatch} sequences; a sequence is always allowed to finish its part, so the per-sequence counts of an
 *       ubatch differ by at most one.</li>
 *   <li>{@link Mode#SEQ} (split_seq): one sequence per ubatch, up to {@code nUbatch} tokens of it.</li>
 * </ul>
 *
 * <p>A group of one sequence — a prompt's rows — splits into as few ubatches as {@code nUbatch} allows, of equal
 * size up to one token ({@link StepSizes}), under {@link Mode#SIMPLE} and {@link Mode#EQUAL}: no short ubatch
 * trails the prompt, and every ubatch pays its fixed cost (the expert upload of a streaming model) in full. A batch
 * of several sequences keeps the llama.cpp split, whose row sets a mode chooses.
 *
 * <p>{@link #split} drains the batch: the union of the ubatches is the batch, every row once. Rows inside an
 * ubatch are ordered by sequence then position (the {@code UBatch} rule), whatever order the mode collected them
 * in, so a call to {@link BatchSplitter#split} with @code paged=false} on a single-sequence batch produces
 * contiguous prefill chunks, of equal size (the paragraph above).</p>
 *
 * <p>A slot-uniform split ({@link #split(ForwardBatch, int, Mode, boolean)}) first groups the rows by their adapter
 * slot ({@code ForwardBatch.adapterSlots()}, in order of first appearance) and splits each group by the mode, so
 * every ubatch runs rows of one slot: the form a model whose blocks cannot mix adapter slots in one product
 * runs per-request adapters in ({@code AdapterRows.mixable}).</p>
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
        return split(batch, nUbatch, mode, false);
    }

    /**
     * Splits {@code batch} into ubatches by {@code mode}; with {@code slotUniform} every ubatch holds rows of one
     * adapter slot (the groups of each slot split by {@code mode}, the slots in order of first appearance).
     *
     * @throws IllegalArgumentException as {@link #split(ForwardBatch, int, Mode)}
     */
    public List<UBatch> split(ForwardBatch batch, int nUbatch, Mode mode, boolean slotUniform) {
        Objects.requireNonNull(batch, "batch");
        Objects.requireNonNull(mode, "mode");
        if (nUbatch <= 0) {
            throw new IllegalArgumentException("nUbatch must be positive, got " + nUbatch);
        }
        if (batch.nTokens() <= 0) {
            throw new IllegalArgumentException("the batch holds at least one row");
        }
        List<int[]> rowLists = new ArrayList<>();
        for (int[] group : slotUniform ? slotGroups(batch) : Arrays.asList(allRows(batch.nTokens()))) {
            switch (mode) {
                case SIMPLE:
                    rowLists.addAll(simple(batch, group, nUbatch));
                    break;
                case EQUAL:
                    rowLists.addAll(equal(batch, group, nUbatch));
                    break;
                case SEQ:
                    rowLists.addAll(seq(batch, group, nUbatch));
                    break;
                default:
                    throw new IllegalArgumentException("unknown split mode " + mode);
            }
        }
        int[] seqs = batch.seqIds();
        int[] pos = batch.positions();
        List<UBatch> out = new ArrayList<>(rowLists.size());
        for (int[] rows : rowLists) {
            out.add(UBatch.of(batch, sorted(rows, seqs, pos), paged));
        }
        return out;
    }

    /** Rows {@code 0..n-1}. */
    private static int[] allRows(int n) {
        int[] rows = new int[n];
        for (int i = 0; i < n; i++) {
            rows[i] = i;
        }
        return rows;
    }

    /** The batch rows grouped by adapter slot, slots in order of first appearance, rows in batch order. */
    private static List<int[]> slotGroups(ForwardBatch batch) {
        int[] slots = batch.adapterSlots();
        if (slots == null) {
            return Collections.singletonList(allRows(batch.nTokens()));
        }
        Map<Integer, List<Integer>> bySlot = new LinkedHashMap<>();
        for (int i = 0; i < slots.length; i++) {
            bySlot.computeIfAbsent(slots[i], s -> new ArrayList<>()).add(i);
        }
        List<int[]> out = new ArrayList<>(bySlot.size());
        for (List<Integer> rows : bySlot.values()) {
            out.add(intArray(rows));
        }
        return out;
    }

    /** llama.cpp split_simple: chunks of at most {@code nUbatch} rows of {@code group} in batch order. */
    private static List<int[]> simple(ForwardBatch batch, int[] group, int nUbatch) {
        if (singleSequence(batch.seqIds(), group)) {
            return equalChunks(group, nUbatch);
        }
        int n = group.length;
        List<int[]> out = new ArrayList<>((n + nUbatch - 1) / nUbatch);
        for (int cur = 0; cur < n; cur += nUbatch) {
            out.add(Arrays.copyOfRange(group, cur, Math.min(n, cur + nUbatch)));
        }
        return out;
    }

    /**
     * {@code group} in order, split into as few ubatches as {@code nUbatch} allows, of equal size up to one token
     * ({@link StepSizes}): the rows of one sequence are contiguous, so any boundary between them is a valid prompt
     * split, and equal ubatches avoid a trailing one that is nearly as expensive as a full one.
     */
    private static List<int[]> equalChunks(int[] group, int nUbatch) {
        int n = group.length;
        List<int[]> out = new ArrayList<>((n + nUbatch - 1) / nUbatch);
        for (int cur = 0; cur < n; ) {
            int take = StepSizes.next(n - cur, nUbatch);
            out.add(Arrays.copyOfRange(group, cur, cur + take));
            cur += take;
        }
        return out;
    }

    /** Whether every row of {@code group} belongs to one sequence. */
    private static boolean singleSequence(int[] seqs, int[] group) {
        for (int i = 1; i < group.length; i++) {
            if (seqs[group[i]] != seqs[group[0]]) {
                return false;
            }
        }
        return true;
    }

    /** llama.cpp split_equal over {@code group}: each round takes one token per remaining sequence, up to the cap. */
    private static List<int[]> equal(ForwardBatch batch, int[] group, int nUbatch) {
        int[] seqs = batch.seqIds();
        // sequences in first-appearance order: the non-overlapping sequence sets of llama.cpp
        List<Integer> order = new ArrayList<>();
        for (int row : group) {
            if (!order.contains(seqs[row])) {
                order.add(seqs[row]);
            }
        }
        if (order.size() == 1) {
            return equalChunks(group, nUbatch); // one sequence's contiguous rows: equally sized ubatches
        }
        int nSeqs = order.size();
        int[][] rowsOfSeq = rowsBySeq(seqs, group, order);
        int[] cursor = new int[nSeqs];
        List<int[]> out = new ArrayList<>();
        while (true) {
            // the group: sequences that still hold tokens, in first-appearance order, capped at nUbatch
            List<Integer> members = new ArrayList<>();
            for (int s = 0; s < nSeqs && members.size() < nUbatch; s++) {
                if (cursor[s] < rowsOfSeq[s].length) {
                    members.add(s);
                }
            }
            if (members.isEmpty()) {
                break;
            }
            List<Integer> rows = new ArrayList<>();
            int rounds = 0;
            while (true) {
                boolean allHoldTokens = true;
                for (int s : members) {
                    allHoldTokens &= cursor[s] < rowsOfSeq[s].length;
                }
                if (!allHoldTokens || (rounds + 1) * members.size() > nUbatch) {
                    break;
                }
                for (int s : members) {
                    rows.add(rowsOfSeq[s][cursor[s]++]);
                }
                rounds++;
            }
            // at least one round always runs: every group member holds a token and fits the cap
            out.add(intArray(rows));
        }
        return out;
    }

    /** llama.cpp split_seq over {@code group}: runs of one sequence, at most nUbatch tokens per ubatch. */
    private static List<int[]> seq(ForwardBatch batch, int[] group, int nUbatch) {
        int n = group.length;
        int[] seqs = batch.seqIds();
        List<int[]> out = new ArrayList<>();
        int cur = 0;
        while (cur < n) {
            int run = seqs[group[cur]];
            int from = cur;
            while (cur < n && seqs[group[cur]] == run && cur - from < nUbatch) {
                cur++;
            }
            out.add(Arrays.copyOfRange(group, from, cur));
        }
        return out;
    }

    /** The rows of {@code group} of each sequence in {@code order}, batch ordered. */
    private static int[][] rowsBySeq(int[] seqs, int[] group, List<Integer> order) {
        int[] len = new int[order.size()];
        for (int row : group) {
            len[order.indexOf(seqs[row])]++;
        }
        int[][] out = new int[order.size()][];
        for (int s = 0; s < order.size(); s++) {
            out[s] = new int[len[s]];
            len[s] = 0;
        }
        for (int row : group) {
            int s = order.indexOf(seqs[row]);
            out[s][len[s]++] = row;
        }
        return out;
    }

    private static int[] intArray(List<Integer> rows) {
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

package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;

import java.util.function.IntUnaryOperator;

/**
 * The paged-memory plumbing of {@link CpuAttentionMixer}: the store and the attention of one paged step, segment by
 * segment. Segment {@code seg} stores at the physical cells the step's {@code cellIdx} names and attends through its
 * sequence's page-table slice and slot count (the unified multi-sequence ubatch, llama.cpp
 * src/llama-kv-cache.cpp:761, whose reads resolve {@code kv_pos[nf_kv_row(...)]}), so the rows of one sequence never
 * see another's keys; the mixer rows follow the ubatch order, so a segment's rows sit at
 * {@code [segStart, segStart + segLen)}. Not thread-safe; driven by the owning mixer's step thread.
 */
final class CpuPagedAttention {

    private final CpuAttentionMixer mixer;
    /** Scratch the page-table slice of one segment's sequence is built in. */
    private int[] pagedSlice;

    CpuPagedAttention(CpuAttentionMixer mixer) {
        this.mixer = mixer;
    }

    /**
     * Paged store: row {@code t} goes to the physical cell the step's {@code cellIdx} names for the token (the step
     * layout rows follow the ubatch order, the mixer rows the same order). The context shift rules do not apply
     * here: a paged step never wraps, its slots are append ordered ({@code PagedSequences}).
     */
    void store(PagedStep step, int n, int[] pos, float[] k, float[] v) {
        KvAccessor kv = mixer.kv();
        int[] cells = step.cellIdx(kv.group());
        for (int t = 0; t < n; t++) {
            HostKvAttention.divideByScales(k, t * mixer.spec().kDim(), mixer.spec().kDim(), mixer.spec().headDimK(),
                    mixer.kScale());
            HostKvAttention.divideByScales(v, t * mixer.spec().vDim(), mixer.spec().vDim(), mixer.spec().headDimV(),
                    mixer.vScale());
            kv.writeK(cells[t], k, t * mixer.spec().kDim());
            kv.writeV(cells[t], v, t * mixer.spec().vDim());
        }
    }

    /**
     * Paged attention of every segment's rows: segment {@code seg} attends through its sequence's page-table slice
     * and slot count, so the rows of one sequence never see another's keys. {@code stepParams}
     * carries the step's mask rule (a non-causal media chunk flags its step).
     */
    void attend(CpuAttentionCore.Params stepParams, float[] q, int[] pos, float[] attn, int group,
                CpuAttentionCore.TreeMask tree) {
        PagedStep step = mixer.pagedStep();
        int shift = pageShift(step);
        for (int seg = 0; seg < step.nSeg(); seg++) {
            int seq = step.segSeq(seg);
            int start = step.segStart(seg);
            int len = step.segLen(seg);
            if (mixer.singlePosition() && pos[start] != step.segPos0(seg)) {
                throw new IllegalStateException("layer " + mixer.layer().il() + " runs segment " + seg + " (sequence "
                        + seq + ") at position " + pos[start] + ", the step says " + step.segPos0(seg));
            }
            int[] pageTable = pageTable(step, group, seq);
            int slots = slots(step, group, seq);
            /* A tree verify's fresh-row visibility is the mask, which the fused int8 kernel takes no part in; the
             * dequantized core below serves the same cache. */
            if (mixer.int8Kv() != null && tree == null && stepParams.sinkTokens() == 0) {
                CpuInt8Attention.attendRows(stepParams, len, q, start * mixer.spec().qDim(), mixer.spec().qDim(), pos,
                        start, mixer.int8Kv(), mixer.rowPos(), rowOf(shift, pageTable), 0, slots, attn,
                        start * mixer.spec().outDim(), mixer.spec().outDim());
                continue;
            }
            if (tree == null) {
                CpuAttentionCore.attendRows(stepParams, len, q, start * mixer.spec().qDim(), mixer.spec().qDim(), pos,
                        start, mixer.kv(), mixer.rowPos(), shift, pageTable, 0, slots, attn,
                        start * mixer.spec().outDim(), mixer.spec().outDim());
            } else {
                // the step's bits name the ubatch's rows; the segment's call sees its own rows shifted local
                CpuAttentionCore.attendRows(stepParams, len, q, start * mixer.spec().qDim(), mixer.spec().qDim(), pos,
                        start, mixer.kv(), mixer.rowPos(), shift, pageTable, 0, slots, attn,
                        start * mixer.spec().outDim(), mixer.spec().outDim(), segmentTreeMask(tree, start, len));
            }
        }
    }

    /**
     * The step's tree mask relativized to one segment's rows: the segment's rows are the call's fresh rows, its
     * queries sit at {@code [start, start + len)} and the core indexes the mask by the call's absolute query row,
     * so the segment's shifted bits land at {@code [start, start + len)} of the slice (the masked batch never
     * splits over ubatches — at most 64 rows — and a sequence's rows hold one contiguous segment, so the shift is
     * exact).
     */
    static CpuAttentionCore.TreeMask segmentTreeMask(CpuAttentionCore.TreeMask tree, int start, int len) {
        long[] bits = tree.mask();
        long[] rel = new long[start + len];
        for (int t = 0; t < len; t++) {
            rel[start + t] = bits[start + t] >>> start;
        }
        return new CpuAttentionCore.TreeMask(rel, len);
    }

    /**
     * The page-table slice of {@code seq} ({@code PagedStep.pageTable[g][seq * maxPages + k]}), the table
     * {@code CpuAttentionCore.attendRows}' paged form indexes, built in a scratch reused across the segments: the
     * entries the sequence's {@link #slots} address, the rest of the scratch unread.
     */
    int[] pageTable(PagedStep step, int group, int seq) {
        if (pagedSlice == null || pagedSlice.length != step.maxPages()) {
            pagedSlice = new int[step.maxPages()];
        }
        step.pageTableRow(group, seq, pagedSlice);
        return pagedSlice;
    }

    /** The slots {@code seq} holds in this cache's group, the attend span (contiguous slot space). */
    static int slots(PagedStep step, int group, int seq) {
        return step.seqSlot(group, seq);
    }

    /**
     * The slot-to-row map of a page table ({@code nf_kv_row}): slot {@code s} is the physical row
     * {@code (pageTable[s >> shift] << shift) | (s & ((1 << shift) - 1))}, the view {@link CpuInt8Attention} reads.
     */
    static IntUnaryOperator rowOf(int shift, int[] pageTable) {
        final int pageMask = (1 << shift) - 1;
        return slot -> (pageTable[slot >> shift] << shift) | (slot & pageMask);
    }

    /** {@code log2(pageSize)}: slot {@code s} sits in page {@code s >> shift} (the kernel's page-table shift). */
    static int pageShift(PagedStep step) {
        int size = step.pageSize();
        if (Integer.bitCount(size) != 1) {
            throw new IllegalArgumentException("page size " + size + " is not a power of two");
        }
        return Integer.numberOfTrailingZeros(size);
    }
}

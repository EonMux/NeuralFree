package com.eonmux.neuralfree.gguf.memory.paged;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import org.junit.Test;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * A step's page tables stored compacted per sequence ({@code PagedStep}): the per-step readers' non-copying
 * accessors ({@code pageTableLength}, {@code pageTableRow}, {@code pageTableEntry}, {@code seqSlot}) agree with the
 * whole-matrix {@code pageTable(g)} after prefills, decodes, page crossings, a removal and a copy-on-write
 * privatization; a table holds exactly the sequence's pages, and the memory keeps no state shared between instances.
 */
public class PagedStepTablesTest {

    private static final int PAGE = 4;
    private static final int CTX = 64;
    private static final int PARALLEL = 3;

    private static PagedSequences memory() {
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16)
                .nCtx(CTX).ubatch(16).parallel(PARALLEL).kvPageSize(PAGE).build();
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, CTX, 16)
                .group(MemoryPlan.KvGroup.of(0, CTX, false, KvWindow.FULL_CAUSAL, PAGE, CTX / PAGE))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV)
                        .kv(KvLayout.of(CTX, false, 8, 8, KvDType.F16, KvDType.F16, KvWindow.FULL_CAUSAL), 0)
                        .build())
                .config(cfg)
                .build();
        return new PagedSequences(plan, PARALLEL);
    }

    /** One finished step over {@code (seq, len, pos0)} segments, checked before its commit and after it. */
    private static PagedStep write(PagedSequences mem, int[]... segments) {
        int n = 0;
        int[] seq = new int[segments.length];
        int[] start = new int[segments.length];
        int[] len = new int[segments.length];
        int[] pos0 = new int[segments.length];
        for (int s = 0; s < segments.length; s++) {
            seq[s] = segments[s][0];
            len[s] = segments[s][1];
            pos0[s] = segments[s][2];
            start[s] = n;
            n += len[s];
        }
        PagedStep step = mem.prepare(StepLayout.of(n, seq, start, len, pos0, new int[0], true));
        assertAccessorsAgree(step);
        mem.commit(step);
        mem.releaseCopies(mem.pendingCopies().toArray(new PageCopy[0]));
        mem.finish(step);
        return step;
    }

    private static void assertAccessorsAgree(PagedStep step) {
        int maxPages = step.maxPages();
        for (int g = 0; g < step.groups(); g++) {
            int[] whole = step.pageTable(g);
            int[] slots = step.seqSlots(g);
            for (int seq = 0; seq < step.maxSequences(); seq++) {
                int len = step.pageTableLength(g, seq);
                int[] row = new int[maxPages];
                Arrays.fill(row, -7);
                assertEquals(len, step.pageTableRow(g, seq, row));
                for (int k = 0; k < maxPages; k++) {
                    int entry = whole[seq * maxPages + k];
                    assertEquals("entry " + k + " of sequence " + seq, entry, step.pageTableEntry(g, seq, k));
                    if (k < len) {
                        assertEquals(entry, row[k]);
                    } else {
                        assertEquals("entries past the length are -1", -1, entry);
                        assertEquals("the row copy leaves the rest of the destination", -7, row[k]);
                    }
                }
                assertEquals(slots[seq], step.seqSlot(g, seq));
                assertEquals("a table holds the pages of the sequence's slots", (slots[seq] + PAGE - 1) / PAGE,
                        len);
            }
        }
    }

    @Test
    public void theCompactedTablesAgreeWithTheWholeMatrixAcrossTheOperations() {
        PagedSequences mem = memory();
        write(mem, new int[]{0, 6, 0}, new int[]{1, 3, 0}, new int[]{2, 9, 0});
        for (int t = 0; t < 5; t++) {
            write(mem, new int[]{0, 1, 6 + t}, new int[]{1, 1, 3 + t}, new int[]{2, 1, 9 + t});
        }
        mem.remove(1, 5, -1);
        write(mem, new int[]{1, 2, 5});
        mem.remove(2, 0, -1);
        mem.copy(0, 2, 0, 6);
        write(mem, new int[]{0, 1, 11}, new int[]{2, 1, 6});
    }

    @Test
    public void theRowCopyRefusesAShortDestinationAndTheAccessorsCheckTheirBounds() {
        PagedSequences mem = memory();
        PagedStep step = write(mem, new int[]{0, 9, 0});
        assertThrows(IllegalArgumentException.class, () -> step.pageTableRow(0, 0, new int[2]));
        assertThrows(IndexOutOfBoundsException.class, () -> step.seqSlot(0, PARALLEL));
        assertThrows(IndexOutOfBoundsException.class, () -> step.pageTableLength(1, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> step.pageTableEntry(0, 0, step.maxPages()));
    }

    /**
     * Two memories (two engines, a draft model, context-parallel ranks) run on their own pump threads: nothing of
     * one memory's bookkeeping may live in shared static state.
     */
    @Test
    public void thePagedMemoryHoldsNoMutableStaticState() {
        for (Field f : PagedSequences.class.getDeclaredFields()) {
            if (!Modifier.isStatic(f.getModifiers())) {
                continue;
            }
            assertTrue("static field " + f.getName() + " is not a constant",
                    Modifier.isFinal(f.getModifiers())
                            && (f.getType().isPrimitive() || f.getType() == String.class));
        }
    }
}

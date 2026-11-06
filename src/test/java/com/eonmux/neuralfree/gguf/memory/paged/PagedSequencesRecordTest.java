package com.eonmux.neuralfree.gguf.memory.paged;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * The token record of a paged sequence follows its rows' cells, as llama.cpp's cells carry {@code ext.tok}: one
 * entry per committed row at its stored position (media rows of one shared position included), dropped by
 * {@link PagedSequences#remove}, copied by {@link PagedSequences#copy} and moved by {@link PagedSequences#add}; media
 * rows past the rows a sequence holds refuse before anything is reserved.
 */
public class PagedSequencesRecordTest {

    private static final int PAGE = 4;
    private static final int N_CTX = 32;
    private static final int STAND_IN = 7;

    private static PagedSequences memory() {
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16).nCtx(N_CTX).ubatch(8)
                .parallel(2).kvPageSize(PAGE).kvUnified(true).build();
        MemoryPlan.KvGroup g = MemoryPlan.KvGroup.of(0, N_CTX, false, KvWindow.FULL_CAUSAL, PAGE, N_CTX / PAGE);
        KvLayout kv = KvLayout.of(N_CTX, false, 8, 8, KvDType.F16, KvDType.F16, KvWindow.FULL_CAUSAL);
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, N_CTX, cfg.ubatch()).group(g)
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(kv, 0).build()).config(cfg).build();
        return new PagedSequences(plan, 2);
    }

    /**
     * Commits one segment of {@code seq} whose rows store {@code positions}, and records {@code tokens} (an
     * embedding row, negative, records {@link #STAND_IN}).
     */
    private static void write(PagedSequences mem, int seq, int[] tokens, int[] positions) {
        int n = tokens.length;
        StepLayout layout = StepLayout.of(n, new int[] {seq}, new int[] {0}, new int[] {n},
                new int[] {positions[0]}, new int[] {n - 1}, true);
        mem.commit(mem.prepare(layout, positions));
        mem.appendRows(seq, tokens, positions, 0, n, STAND_IN);
    }

    /** Text at 0..2, four media rows at 3, text at 5..6. */
    private static PagedSequences withMedia() {
        PagedSequences mem = memory();
        write(mem, 0, new int[] {10, 11, 12}, new int[] {0, 1, 2});
        write(mem, 0, new int[] {-1, -1, -1, -1}, new int[] {3, 3, 3, 3});
        write(mem, 0, new int[] {20, 21}, new int[] {5, 6});
        return mem;
    }

    private static int[] positions(SequenceState s) {
        int[] out = new int[s.nPast()];
        for (int i = 0; i < out.length; i++) {
            out[i] = s.position(i);
        }
        return out;
    }

    @Test
    public void everyCommittedRowRecordsItsTokenAtItsPosition() {
        PagedSequences mem = withMedia();
        SequenceState record = mem.sequence(0);
        assertArrayEquals(new int[] {10, 11, 12, 7, 7, 7, 7, 20, 21}, record.tokens());
        assertArrayEquals(new int[] {0, 1, 2, 3, 3, 3, 3, 5, 6}, positions(record));
        assertArrayEquals("the text view leaves the media rows out", new int[] {10, 11, 12, 20, 21},
                record.textTokens());
        assertEquals(6, mem.posMax(0));
    }

    @Test
    public void aTailRemovalDropsTheRecordFromItsFirstPosition() {
        PagedSequences mem = withMedia();
        mem.remove(0, 3, -1);
        assertArrayEquals(new int[] {10, 11, 12}, mem.sequence(0).tokens());
        assertEquals(2, mem.posMax(0));
    }

    @Test
    public void anInteriorRemovalKeepsTheRowsAroundIt() {
        PagedSequences mem = withMedia();
        mem.remove(0, 1, 3);
        SequenceState record = mem.sequence(0);
        assertArrayEquals(new int[] {10, 7, 7, 7, 7, 20, 21}, record.tokens());
        assertEquals("the n-gram lookup reads the kept media row before the gap", STAND_IN,
                record.tokenAtOrBefore(4));
        assertEquals("nothing is left at position 1", 10, record.tokenAtOrBefore(2));
    }

    @Test
    public void aShiftMovesTheRecordWithTheRows() {
        PagedSequences mem = withMedia();
        mem.remove(0, 1, 3);
        mem.add(0, 3, -1, -2);
        SequenceState record = mem.sequence(0);
        assertArrayEquals(new int[] {10, 7, 7, 7, 7, 20, 21}, record.tokens());
        assertArrayEquals(new int[] {0, 1, 1, 1, 1, 3, 4}, positions(record));
        assertEquals(4, mem.posMax(0));
    }

    @Test
    public void aForkCopiesTheRecordOfTheCopiedRange() {
        PagedSequences mem = withMedia();
        mem.copy(0, 1, 0, 4);
        SequenceState fork = mem.sequence(1);
        assertArrayEquals(new int[] {10, 11, 12, 7, 7, 7, 7}, fork.tokens());
        assertArrayEquals(new int[] {0, 1, 2, 3, 3, 3, 3}, positions(fork));
        assertArrayEquals("the copied entries stay embedding rows", new int[] {10, 11, 12}, fork.textTokens());
        write(mem, 1, new int[] {30}, new int[] {4});
        assertEquals("the fork's next row follows the copied record", 30, fork.tokenAtOrBefore(4));
        assertArrayEquals("the source keeps its record", new int[] {10, 11, 12, 7, 7, 7, 7, 20, 21},
                mem.sequence(0).tokens());
    }

    @Test
    public void mediaRowsBeyondTheSequencesRowsRefuseBeforeReserving() {
        PagedSequences mem = memory();
        int capacity = mem.sequence(0).capacity();
        int[] shared = new int[8];
        int[] tokens = new int[8];
        Arrays.fill(tokens, -1);
        int pos = 0;
        while (mem.sequence(0).nPast() + 8 <= capacity) {
            Arrays.fill(shared, pos++);
            write(mem, 0, tokens, shared);
        }
        int free = mem.freeCells();
        int stored = mem.sequence(0).nPast();
        Arrays.fill(shared, pos);
        StepLayout layout = StepLayout.of(8, new int[] {0}, new int[] {0}, new int[] {8}, new int[] {pos},
                new int[] {7}, true);
        assertThrows(MemoryFullException.class, () -> mem.prepare(layout, shared));
        assertEquals(free, mem.freeCells());
        assertEquals(stored, mem.sequence(0).nPast());
    }
}

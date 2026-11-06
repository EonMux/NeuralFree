package com.eonmux.neuralfree.gguf.memory.paged;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The position division of {@link PagedSequences} (llama.cpp {@code llama_memory_seq_div}): every stored position
 * of a range divides by the factor, several rows can come to share one position, each moved row reports its own
 * RoPE shift, a shared page privatizes before it changes, and the positions past the range stay put. Rows keep
 * their cells, so the next append continues above the divided positions.
 */
public class PagedSequencesDivTest {

    private static final int PAGE = 4;

    private static PagedSequences memory(int nCtx, int parallel) {
        MemoryConfig cfg = MemoryConfig.builder()
                .typeK(KvDType.F16).typeV(KvDType.F16)
                .nCtx(nCtx).ubatch(8).parallel(parallel).kvPageSize(PAGE).kvUnified(true)
                .build();
        KvWindow w = KvWindow.FULL_CAUSAL;
        MemoryPlan.KvGroup g = MemoryPlan.KvGroup.of(0, nCtx, false, w, PAGE, nCtx / PAGE);
        KvLayout kv = KvLayout.of(nCtx, false, 8, 8, KvDType.F16, KvDType.F16, w);
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, nCtx, cfg.ubatch())
                .group(g)
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(kv, 0).build())
                .config(cfg)
                .build();
        return new PagedSequences(plan, parallel);
    }

    /** Appends positions {@code [pos0, pos0 + len)} to {@code seq}, step and token record alike. */
    private static void append(PagedSequences mem, int seq, int pos0, int len) {
        int[] positions = new int[len];
        int[] tokens = new int[len];
        for (int i = 0; i < len; i++) {
            positions[i] = pos0 + i;
            tokens[i] = pos0 + i;
        }
        StepLayout layout = StepLayout.of(len, new int[]{seq}, new int[]{0}, new int[]{len},
                new int[]{pos0}, new int[]{0}, true);
        mem.commit(mem.prepare(layout));
        mem.appendRows(seq, tokens, positions, 0, len, -1);
    }

    @Test
    public void divisionDividesEveryStoredPositionAndReportsEachRowsOwnShift() {
        PagedSequences mem = memory(64, 1);
        append(mem, 0, 0, 8);
        mem.clearReports();

        mem.div(0, 0, -1, 2);

        assertEquals(0, mem.posMin(0));
        assertEquals(3, mem.posMax(0));
        // the token record follows the rows: row i now sits at position i / 2
        SequenceState state = mem.sequence(0);
        for (int i = 0; i < 8; i++) {
            assertEquals("row " + i, i / 2, state.position(i));
        }
        // one report per page, each carrying the rows and their own shift (old - old / 2)
        List<RowReport> reports = mem.pendingReports();
        assertEquals(2, reports.size());
        int[] values = new int[8];
        int[] deltas = new int[8];
        int n = 0;
        for (RowReport r : reports) {
            assertTrue("a division is relative: " + r, !r.absolute());
            int[] rows = r.rows();
            int[] reported = r.values();
            for (int i = 0; i < rows.length; i++) {
                values[n] = reported[i];
                deltas[n] = r.rowDelta(i);
                n++;
            }
        }
        assertEquals(8, n);
        assertArrayEquals(new int[]{0, 0, 1, 1, 2, 2, 3, 3}, values);
        assertArrayEquals(new int[]{0, 1, 1, 2, 2, 3, 3, 4}, deltas);
        mem.clearReports();
    }

    @Test
    public void divisionOfARangeLeavesThePositionsOutsideIt() {
        PagedSequences mem = memory(64, 1);
        append(mem, 0, 0, 8);

        mem.div(0, 4, 8, 2);

        // positions 0..3 stay; 4..7 become 2, 2, 3, 3
        SequenceState state = mem.sequence(0);
        int[] positions = new int[8];
        for (int i = 0; i < 8; i++) {
            positions[i] = state.position(i);
        }
        // the record follows the rows by position: three entries at 2 and three at 3
        assertArrayEquals(new int[]{0, 1, 2, 2, 2, 3, 3, 3}, positions);
        assertEquals(3, mem.posMax(0));
        assertEquals(0, mem.posMin(0));
    }

    @Test
    public void theRowsKeepTheirCellsSoTheNextAppendContinuesAboveTheDividedPositions() {
        PagedSequences mem = memory(64, 1);
        append(mem, 0, 0, 8);
        mem.div(0, 0, -1, 2);

        append(mem, 0, 4, 2);

        assertEquals(5, mem.posMax(0));
        SequenceState state = mem.sequence(0);
        assertEquals(4, state.position(8));
        assertEquals(5, state.position(9));
    }

    @Test
    public void aSharedPagePrivatizesBeforeItsPositionsChange() {
        PagedSequences mem = memory(64, 2);
        append(mem, 0, 0, 4);
        mem.copy(0, 1, 0, -1);
        mem.clearReports();

        mem.div(0, 0, -1, 2);

        List<PageCopy> copies = mem.pendingCopies();
        assertEquals(1, copies.size());
        assertEquals(PAGE, copies.get(0).rows());
        assertEquals(1, mem.posMax(0));
        assertEquals("the other sequence keeps its positions", 3, mem.posMax(1));
        mem.releaseCopies(copies.get(0));
    }

    @Test
    public void rowsSharingADividedPositionLeaveTogether() {
        PagedSequences mem = memory(64, 1);
        append(mem, 0, 0, 4);
        mem.div(0, 0, -1, 2);            // positions 0, 0, 1, 1

        mem.remove(0, 1, 2);             // both rows at position 1 leave

        assertEquals(0, mem.posMax(0));
        assertEquals(2, mem.sequence(0).nPast());
    }

    @Test
    public void divisionRefusesADivisorBelowTwoAndANegativeStart() {
        PagedSequences mem = memory(64, 1);
        append(mem, 0, 0, 4);
        assertThrows(IllegalArgumentException.class, () -> mem.div(0, 0, -1, 1));
        assertThrows(IllegalArgumentException.class, () -> mem.div(0, -1, -1, 2));
        assertEquals("nothing moved", 3, mem.posMax(0));
    }

    @Test
    public void divisionNeedsNoShiftRuleAndReportsNoneForAnEmptySequence() {
        PagedSequences mem = memory(64, 1);
        assertNull(mem.divRefusal());
        assertNull(mem.shiftRefusal());
        mem.div(0, 0, -1, 4);            // an empty sequence divides nothing
        assertTrue(mem.pendingReports().isEmpty());
        assertEquals(-1, mem.posMax(0));
    }
}

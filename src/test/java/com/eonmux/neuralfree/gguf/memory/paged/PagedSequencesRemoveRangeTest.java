package com.eonmux.neuralfree.gguf.memory.paged;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;

/**
 * The stored range {@link PagedSequences#remove} leaves: the removal's own walk gathers the kept rows' positions, and
 * after every removal of a randomized history (appends, tail and interior removals, whole-sequence copies that share
 * pages, a trimming window group beside a full one) the range equals a walk over every row the sequence holds
 * ({@link PagedSequences#scannedRange}).
 */
public class PagedSequencesRemoveRangeTest {

    private static final int PAGE = 4;
    private static final int CTX = 128;
    private static final int SWA_ROWS = 64;
    private static final int PARALLEL = 3;

    private static PagedSequences memory() {
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16)
                .nCtx(CTX).ubatch(8).parallel(PARALLEL).kvPageSize(PAGE).kvUnified(true).build();
        KvWindow swa = KvWindow.of(SwaType.STANDARD, 6, true);
        KvWindow full = KvWindow.FULL_CAUSAL;
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, CTX, cfg.ubatch())
                .group(MemoryPlan.KvGroup.of(0, SWA_ROWS, false, swa, PAGE, SWA_ROWS / PAGE))
                .group(MemoryPlan.KvGroup.of(1, CTX, false, full, PAGE, CTX / PAGE))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV)
                        .kv(KvLayout.of(SWA_ROWS, false, 8, 8, KvDType.F16, KvDType.F16, swa), 0).build())
                .layer(LayerMemoryPlan.builder(1, LayerMemoryPlan.Kind.KV)
                        .kv(KvLayout.of(CTX, false, 8, 8, KvDType.F16, KvDType.F16, full), 1).build())
                .config(cfg)
                .build();
        return new PagedSequences(plan, PARALLEL);
    }

    private static void append(PagedSequences mem, int seq, int len) {
        int from = mem.posMax(seq) + 1;
        StepLayout layout = StepLayout.of(len, new int[] {seq}, new int[] {0}, new int[] {len}, new int[] {from},
                new int[] {0}, true);
        PagedStep step = mem.prepare(layout);
        mem.commit(step);
        mem.releaseCopies(mem.pendingCopies().toArray(new PageCopy[0]));
        mem.finish(step);
    }

    private static void remove(PagedSequences mem, int seq, int p0, int p1, String at) {
        mem.remove(seq, p0, p1);
        mem.releaseCopies(mem.pendingCopies().toArray(new PageCopy[0]));
        assertArrayEquals(at + ": remove(" + seq + ", " + p0 + ", " + p1 + ")", mem.scannedRange(seq),
                new int[] {mem.posMin(seq), mem.posMax(seq)});
    }

    @Test
    public void everyRemovalLeavesTheRangeAWalkOverTheKeptRowsFinds() {
        for (int seed = 0; seed < 40; seed++) {
            Random r = new Random(seed);
            PagedSequences mem = memory();
            for (int op = 0; op < 400; op++) {
                int seq = r.nextInt(PARALLEL);
                int max = mem.posMax(seq);
                int min = mem.posMin(seq);
                String at = "seed " + seed + " op " + op;
                int kind = r.nextInt(10);
                if (kind < 4 || max < 0) {
                    try {
                        append(mem, seq, 1 + r.nextInt(7));
                    } catch (MemoryFullException full) {
                        remove(mem, seq, 0, -1, at);
                    }
                } else if (kind < 7) {
                    remove(mem, seq, min + r.nextInt(max - min + 1), -1, at);
                } else if (kind < 9) {
                    int lo = min + r.nextInt(max - min + 1);
                    remove(mem, seq, lo, lo + 1 + r.nextInt(6), at);
                } else {
                    int dst = (seq + 1 + r.nextInt(PARALLEL - 1)) % PARALLEL;
                    remove(mem, dst, 0, -1, at);
                    mem.copy(seq, dst, 0, -1);
                    mem.releaseCopies(mem.pendingCopies().toArray(new PageCopy[0]));
                }
            }
        }
    }
}

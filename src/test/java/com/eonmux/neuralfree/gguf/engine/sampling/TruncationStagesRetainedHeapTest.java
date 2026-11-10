package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import org.junit.Test;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.ref.Reference;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * The heap a top-p or locally typical stage keeps between applies: an open request whose chain holds them keeps their
 * parameters, not row-sized scratch. Each stage truncates a whole row of a 151,669-token vocabulary to a nucleus
 * wider than the first selection (the path that walks the bucketed order), and the heap retained by the stages is
 * read after a collection while they stay reachable and their samplers do not.
 */
public class TruncationStagesRetainedHeapTest {

    private static final int VOCAB_SIZE = 151_669;
    private static final int PAIRS = 32;
    /** Retained bytes allowed per top-p and typical pair: far below one row of sort keys (8 bytes a token). */
    private static final long BOUND_PER_PAIR = 256L * 1024;
    /** A sampled chain with no truncation of its own, so the stage under test is the one that truncates. */
    private static final SamplerConfig UNTRUNCATED = SamplerConfig.builder().topK(0).topP(1f).minP(0f)
            .typicalP(1f).temperature(1f).seed(1L).build();

    @Test
    public void anOpenRequestKeepsNoRowSizedTruncationScratch() {
        float[] row = row();
        Reference.reachabilityFence(drive(new TopPStage(0.95f, 1), row)); // the pooled scratch, warm
        Reference.reachabilityFence(drive(new TypicalStage(0.95f, 1), row));
        long before = usedAfterCollection();
        List<SamplerStage> open = new ArrayList<>();
        for (int i = 0; i < PAIRS; i++) {
            open.add(drive(new TopPStage(0.95f, 1), row));
            open.add(drive(new TypicalStage(0.95f, 1), row));
        }
        long retained = usedAfterCollection() - before;
        Reference.reachabilityFence(open);
        long perPair = retained / PAIRS;
        assertTrue("a top-p and typical pair between applies retains " + perPair + " bytes, bound " + BOUND_PER_PAIR,
                perPair <= BOUND_PER_PAIR);
    }

    /** Applies {@code stage} to {@code row} through a sampler that is dropped afterwards. */
    private static SamplerStage drive(SamplerStage stage, float[] row) {
        Sampler sampler = Sampler.create(UNTRUNCATED, new int[0], List.of(stage));
        int kept = sampler.apply(row).size();
        assertTrue(stage.kind() + " keeps " + kept + " candidates, wider than the first selection",
                kept > SmallestKeys.FIRST && kept < VOCAB_SIZE);
        return stage;
    }

    /** Unit-variance noise: a nucleus of tens of thousands of candidates. */
    private static float[] row() {
        Random rnd = new Random(3);
        float[] row = new float[VOCAB_SIZE];
        for (int i = 0; i < row.length; i++) {
            row[i] = (float) rnd.nextGaussian();
        }
        return row;
    }

    private static long usedAfterCollection() {
        MemoryMXBean memory = ManagementFactory.getMemoryMXBean();
        long used = Long.MAX_VALUE;
        for (int i = 0; i < 4; i++) {
            System.gc();
            used = Math.min(used, memory.getHeapMemoryUsage().getUsed());
        }
        return used;
    }
}

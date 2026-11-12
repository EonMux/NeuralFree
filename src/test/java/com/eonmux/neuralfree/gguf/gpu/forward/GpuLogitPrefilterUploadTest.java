package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.engine.LogitPrefilter;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import org.junit.After;
import org.junit.Test;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The prefiltered decode readout's table uploads over the real forward backend on the fake queue: the per-step
 * tables (row metadata, penalty tokens) go up every step, while the tables a request fixes (row parameters and the
 * logit-bias table) go up once and again only when a step brings other ones.
 */
public class GpuLogitPrefilterUploadTest {

    private static final int K = 4;

    private final GpuDecodeReplayTest rig = new GpuDecodeReplayTest();

    @After
    public void tearDown() throws Exception {
        rig.tearDown();
    }

    private static Sampler sampler(float repeatPenalty, Map<Integer, Float> bias) {
        SamplerConfig config = SamplerConfig.builder().temperature(0.9f).topK(K).penaltyLastN(8)
                .repeatPenalty(repeatPenalty).seed(3).build();
        return Sampler.create(config, new int[0], bias);
    }

    @Test
    public void fixedTablesGoUpOnlyWhenAStepChangesThem() {
        GpuForwardBackend fb = rig.open(true);
        GpuLogitsHead head = fb.logitsHead();
        Sampler sampler = sampler(1.3f, Map.of(3, 4.0f));
        int pos = 0;
        long keptStepWrites = 0;
        for (int step = 0; step < 4; step++) {
            sampler.accept(step + 5);
            long before = writes();
            fb.decodeTopKOnDevice(step + 5, pos++, K, LogitPrefilter.of(List.of(sampler.prefilterRow())));
            keptStepWrites = writes() - before;
        }
        assertArrayEquals("the first step uploads the fixed tables, the later ones keep them", new long[]{1, 3},
                head.prefilterFixedUploads());
        assertTrue("a step still uploads its own tables: " + keptStepWrites, keptStepWrites > 0);

        Sampler rebiased = sampler(1.3f, Map.of(4, 2.0f));
        rebiased.accept(9);
        long before = writes();
        fb.decodeTopKOnDevice(9, pos++, K, LogitPrefilter.of(List.of(rebiased.prefilterRow())));
        assertEquals("another bias table goes up", 2, head.prefilterFixedUploads()[0]);
        assertEquals("the parameter, bias id and bias value tables are three writes", keptStepWrites + 3,
                writes() - before);

        Sampler repenalized = sampler(1.5f, Map.of(4, 2.0f));
        repenalized.accept(9);
        fb.decodeTopKOnDevice(9, pos, K, LogitPrefilter.of(List.of(repenalized.prefilterRow())));
        assertEquals("other row parameters go up", 3, head.prefilterFixedUploads()[0]);
        assertEquals(3, head.prefilterFixedUploads()[1]);
    }

    /**
     * The reused staging uploads exactly a step's tables: after a step with a two-entry bias table, a step whose bias
     * table is that table's first entry alone is another table (one 4-byte id and one 4-byte value go up), and a
     * step repeating it keeps it.
     */
    @Test
    public void aShorterTableAfterALongerOneIsAnotherTable() {
        GpuForwardBackend fb = rig.open(true);
        GpuLogitsHead head = fb.logitsHead();
        Sampler both = sampler(1.3f, Map.of(3, 4.0f, 5, 1.0f));
        both.accept(5);
        fb.decodeTopKOnDevice(5, 0, K, LogitPrefilter.of(List.of(both.prefilterRow())));
        Sampler first = sampler(1.3f, Map.of(3, 4.0f));
        first.accept(5);
        long changed = fourByteWrites(() -> fb.decodeTopKOnDevice(5, 1, K,
                LogitPrefilter.of(List.of(first.prefilterRow()))));
        assertEquals(2, head.prefilterFixedUploads()[0]);
        long kept = fourByteWrites(() -> fb.decodeTopKOnDevice(5, 2, K,
                LogitPrefilter.of(List.of(first.prefilterRow()))));
        assertEquals("one 4-byte bias id and one 4-byte bias value more than a kept table", 2, changed - kept);
        assertEquals("the same one-entry table stays", 2, head.prefilterFixedUploads()[0]);
    }

    private long fourByteWrites(Runnable step) {
        int from = rig.backend.ops().size();
        step.run();
        return rig.backend.ops().subList(from, rig.backend.ops().size()).stream()
                .filter(op -> op.kind() == FakeOpKind.WRITE && op.bytes() == 4).count();
    }

    private long writes() {
        return rig.backend.ops().stream().filter(op -> op.kind() == FakeOpKind.WRITE).count();
    }
}

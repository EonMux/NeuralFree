package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link BatchSegments}: the sequence runs of a memoryless batch, the packed-run rules (each sequence's rows
 * together and numbered from 0) and which models pack.
 */
public class BatchSegmentsTest {

    /** Rows of {@code seq[i]} at {@code pos[i]}. */
    private static ForwardBatch batch(int[] seq, int[] pos) {
        ForwardBatch.Builder b = ForwardBatch.builder(seq.length);
        for (int i = 0; i < seq.length; i++) {
            b.add(7, pos[i], seq[i], true);
        }
        return b.build();
    }

    @Test
    public void runsFollowTheSequencesInRowOrder() {
        BatchSegments s = BatchSegments.of(batch(new int[]{4, 4, 4, 0, 2, 2}, new int[]{0, 1, 2, 0, 0, 1}));
        assertEquals(3, s.count());
        assertEquals(0, s.start(0));
        assertEquals(3, s.length(0));
        assertEquals(3, s.start(1));
        assertEquals(1, s.length(1));
        assertEquals(4, s.start(2));
        assertEquals(2, s.length(2));
        assertEquals(3, s.maxLength());
    }

    @Test
    public void oneSequenceKeepsItsOwnPositions() {
        BatchSegments s = BatchSegments.of(batch(new int[]{1, 1, 1}, new int[]{5, 6, 9}));
        assertEquals(1, s.count());
        assertEquals(3, s.length(0));
    }

    @Test
    public void aSplitSequenceRefusesByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> BatchSegments.of(batch(new int[]{0, 1, 0}, new int[]{0, 0, 1})));
        assertTrue(e.getMessage(), e.getMessage().contains("2 sequences run in 3 row runs"));
    }

    @Test
    public void aPackedRunStartsAtPositionZero() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> BatchSegments.of(batch(new int[]{0, 0, 1, 1}, new int[]{0, 1, 3, 4})));
        assertTrue(e.getMessage(), e.getMessage().contains("row 2 of sequence 1 sits at position 3, not 0"));
    }

    @Test
    public void anEncoderPacksAndAModelWithMemoryDoesNot() throws IOException {
        EngineConfig cpu = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(16).build()).build();
        Path bert = SyntheticModels.bert().writeTempFile();
        Path llama = SyntheticModels.llama().writeTempFile();
        try (GGUFModel encoder = new GGUFModel(bert.toString(), cpu);
             GGUFModel decoder = new GGUFModel(llama.toString(), cpu)) {
            ModelSpec enc = encoder.spec();
            assertNull(BatchSegments.packingRefusal(enc));
            String refusal = BatchSegments.packingRefusal(decoder.spec());
            assertNotNull(refusal);
            assertTrue(refusal, refusal.contains("keeps a memory"));
            assertEquals(1, decoder.forwardBackend().memorylessSequences());
            assertEquals(0, decoder.forwardBackend().memorylessBatchRows());

            int n = BatchSegments.MAX_PACKED + 1;
            int[] seq = new int[n];
            int[] pos = new int[n];
            for (int i = 0; i < n; i++) {
                seq[i] = i;
            }
            IllegalArgumentException tooMany = assertThrows(IllegalArgumentException.class,
                    () -> BatchSegments.packed(batch(seq, pos), enc));
            assertTrue(tooMany.getMessage(), tooMany.getMessage().contains("at most " + BatchSegments.MAX_PACKED));
            assertNull("one sequence needs no runs", BatchSegments.packed(batch(new int[]{3, 3}, new int[]{4, 5}),
                    enc));
        }
    }
}

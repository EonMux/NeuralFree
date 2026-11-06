package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The load warmup of {@link LLMModelFactory} over paged memory: a model opened with {@code neuralfree.parallel > 1}
 * or {@code neuralfree.kv.paged} warms through {@code forwardBatch} on sequence 0 and hands back an empty memory,
 * so the paged sequences decode right after the factory opens the model.
 */
public class LLMModelFactoryPagedWarmupTest {

    private static EngineConfig paged(int parallel) {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(parallel).kvPaged(true).build())
                .build();
    }

    @Test
    public void aPagedModelWarmsAndOpensWithAnEmptyMemory() throws IOException {
        Path file = SyntheticModels.llama().writeTempFile();
        for (int parallel : new int[] {1, 3}) {
            try (EngineModel model = LLMModelFactory.open(file.toString(), paged(parallel))) {
                PagedSequences seqs = ((CpuMemory) model.forwardBackend().memory()).pagedSequences();
                assertNotNull("the paged plan", seqs);
                assertEquals("the warmup row was reset", seqs.totalCells(), seqs.freeCells());
                assertEquals(-1, seqs.posMax(0));

                ForwardBatch.Builder batch = ForwardBatch.builder(parallel);
                for (int seq = 0; seq < parallel; seq++) {
                    batch.add(1, 0, seq, true);
                }
                float[][] rows = model.forwardBackend().forwardBatch(batch.build());
                assertEquals(parallel, rows.length);
                assertTrue(rows[0].length > 0);
            }
        }
    }
}

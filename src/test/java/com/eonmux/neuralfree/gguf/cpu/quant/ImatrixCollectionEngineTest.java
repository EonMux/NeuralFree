package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.quant.write.ImatrixFile;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The imatrix collection over a running engine: every CPU matmul of a prefill feeds the collector through
 * {@link CpuMatVec#observeGlobally}, the entries carry the stored GGUF names the loaded weights' catalog gives
 * (the weights the matvecs run on), the per-tensor column sums cover the prefill's activation rows, and
 * {@code ImatrixFile.writeGguf} writes what {@code ImatrixFile.read} reads back.
 */
public class ImatrixCollectionEngineTest {

    private static EngineConfig config() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    @Test
    public void aPrefillCollectsEveryBlockMatmulUnderItsStoredNameAndTheFileRoundTrips() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path dir = java.nio.file.Files.createTempDirectory("nf-imatrix-");
        Path out = dir.resolve("imatrix.gguf");
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            CpuForwardContext ctx = m.forwardContext();
            CpuWeights weights = ctx.weights();
            ImatrixCollector collector = new ImatrixCollector(w -> weights.describe(w).name());
            int[] tokens = {4, 9, 17, 33, 65, 129, 7, 11};
            com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec.observeGlobally(collector);
            try {
                m.forwardBackend().prefill(tokens, 0);
            } finally {
                CpuMatVec.observeGlobally(null);
            }
            List<ImatrixFile.Entry> entries = collector.entries();
            Set<String> names = new HashSet<>();
            for (ImatrixFile.Entry e : entries) {
                names.add(e.tensor());
            }
            // Every block's projections and FFN matrices ran; the embedding table (a lookup, no matmul) did not.
            assertTrue("the collection names the projections: " + names, names.contains("blk.0.attn_q.weight"));
            assertTrue(names.contains("blk.0.attn_k.weight"));
            assertTrue(names.contains("blk.0.attn_v.weight"));
            assertTrue(names.contains("blk.0.attn_output.weight"));
            assertTrue(names.contains("blk.0.ffn_gate.weight"));
            assertTrue(names.contains("blk.1.ffn_down.weight"));
            assertTrue(names.contains("output.weight"));
            assertTrue("no entry for the lookup table", !names.contains("token_embd.weight"));
            assertEquals("one entry per collected tensor", collector.tensorCount(), entries.size());
            // The column width of an entry is the matmul's input width: n_embd for the projections, n_ff for
            // the down projection, n_embd for the head.
            for (ImatrixFile.Entry e : entries) {
                if (e.tensor().endsWith("ffn_down.weight")) {
                    assertEquals(128, e.cols());
                } else if (e.tensor().endsWith("attn_k.weight") || e.tensor().endsWith("attn_v.weight")) {
                    assertEquals(64, e.cols()); // 2 KV heads of 16 dims... n_embd input side
                } else {
                    assertEquals("the projections and the head take the residual width", 64, e.cols());
                }
                assertTrue("every matrix counted activation rows", e.count(0) > 0L);
            }
            long headRows = collector.count("output.weight", 0);
            assertEquals("the head runs for the last row's logits only", 1, headRows);
            assertEquals("every block projection counted one activation row per token",
                    tokens.length, collector.count("blk.0.attn_q.weight", 0));

            ImatrixFile.writeGguf(out, entries, List.of("calibration.txt"), 1, 512);
            ImatrixFile.Read read = ImatrixFile.read(out);
            assertEquals(entries.size(), read.entries().size());
            List<String> sortedNames = new ArrayList<>(names);
            java.util.Collections.sort(sortedNames);
            for (int i = 0; i < entries.size(); i++) {
                assertEquals("entries read back in name order, sums intact",
                        sortedNames.get(i), read.entries().get(i).tensor());
                assertTrue(read.entries().get(i).count(0) > 0L);
            }
            assertEquals(List.of("calibration.txt"), read.datasets());
            assertEquals(1, read.chunkCount());
            assertEquals(512, read.chunkSize());
        } finally {
            java.nio.file.Files.deleteIfExists(out);
            java.nio.file.Files.deleteIfExists(dir);
        }
    }
}

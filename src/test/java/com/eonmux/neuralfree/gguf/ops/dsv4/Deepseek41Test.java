package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.ops.sparse.CpuDsaIndexer.RowReader;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticRoutedModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * DeepSeek-V4.1: the synthetic model's spec and memory sharing, the Engram extra, the two-level candidate selection,
 * full CPU forwards over both fixtures, the GPU mixers that accept every V4.1 layer, and the MTP route's refusal.
 */
public class Deepseek41Test {

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(32).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    /** Two paged sequences of up to 32 tokens each. */
    private static EngineConfig pagedConfig() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    private static GGUFModel open(GgufTestWriter w) throws IOException {
        return new GGUFModel(w.writeTempFile().toString(), config());
    }

    @Test
    public void theSpecSharesCompressedRowsAndCarriesEngram() throws IOException {
        try (GGUFModel m = open(SyntheticRoutedModels.deepseek41())) {
            ModelSpec spec = m.spec();
            assertEquals(Arch.DEEPSEEK41, spec.arch());
            assertEquals(4, spec.nLayer());
            Dsv4Spec l0 = spec.layer(0).dsv4();
            Dsv4Spec l1 = spec.layer(1).dsv4();
            Dsv4Spec l2 = spec.layer(2).dsv4();
            Dsv4Spec l3 = spec.layer(3).dsv4();
            assertTrue(l0.dsv41().ownsCompressed());
            assertTrue(l0.dsv41().computesTopk());
            assertFalse(l1.dsv41().ownsCompressed());
            assertEquals(0, l1.dsv41().kvSourceLayer());
            assertEquals(0, l1.dsv41().topkSourceLayer());
            assertTrue(l2.dsv41().ownsCompressed());
            assertFalse(l3.dsv41().ownsCompressed());
            assertEquals(2, l3.dsv41().kvSourceLayer());
            assertEquals(2, l3.dsv41().topkSourceLayer());
            assertEquals(2, l0.compressRatio());
            assertEquals(1, l2.compressRatio());
            assertTrue(spec.layer(1).hasExtra(LayerSpec.Extras.ENGRAM));
            assertEquals(ResidualKind.HYPER_CONNECTIONS, spec.layer(0).residual().kind());

            List<LayerMemoryPlan.CompressedKvPlan> p0 = m.forwardBackend().memory().plan().layer(0).compressed();
            List<LayerMemoryPlan.CompressedKvPlan> p1 = m.forwardBackend().memory().plan().layer(1).compressed();
            assertFalse(p0.isEmpty());
            assertTrue(p1.isEmpty());
            assertEquals("32 KV values plus the 64-value index key", 96, p0.get(0).width());
            assertEquals(2, p0.get(0).ratio());

            float[][] out = m.forwardBackend().forwardRows(new int[]{1, 2, 3, 4}, 0, new int[]{3});
            assertEquals(1, out.length);
            assertEquals(spec.head().vocabRows(), out[0].length);
            for (float v : out[0]) {
                assertTrue("finite logit", Float.isFinite(v));
            }
        }
    }

    @Test
    public void theCandidateFixtureLoadsAndRuns() throws IOException {
        try (GGUFModel m = open(SyntheticRoutedModels.deepseek41Candidate())) {
            ModelSpec spec = m.spec();
            Dsv4Spec l0 = spec.layer(0).dsv4();
            Dsv4Spec l2 = spec.layer(2).dsv4();
            assertTrue(l0.dsv41().picksCandidateBlocks());
            assertTrue(l2.dsv41().usesCandidateBlocks());
            assertEquals(0, l2.dsv41().candidateSourceLayer());
            float[][] out = m.forwardBackend().forwardRows(new int[]{1, 2, 3, 4, 5, 6}, 0, new int[]{5});
            assertEquals(1, out.length);
            assertNotNull(out[0]);
        }
    }

    /**
     * The Engram extra really feeds the residual stream: zeroing layer 1's {@code engram_kv} removes the projection
     * (the only tensor that differs), and the logits then match a model whose {@code ple.layers} is empty, i.e. one
     * that carries no Engram extra at all.
     */
    @Test
    public void theEngramContributionReachesTheResidualStream() throws IOException {
        float[] withEngram = firstLogits(SyntheticRoutedModels.deepseek41());
        GgufTestWriter muted = SyntheticRoutedModels.deepseek41();
        muted.tensorFilled("blk.1.engram_kv.weight", GGMLType.F32, 0f, 64, (long) 64 * (4 + 1));
        float[] zeroKv = firstLogits(muted);
        boolean differs = false;
        for (int i = 0; i < withEngram.length; i++) {
            // random engram_kv weights ensure some logit moves; an unapplied extra could not move any
            differs |= Math.abs(withEngram[i] - zeroKv[i]) > 1e-6f;
        }
        assertTrue("a zero engram_kv must silence the extra's contribution", differs);

        GgufTestWriter noEngram = SyntheticRoutedModels.deepseek41();
        noEngram.u32Array(noEngram.archKey("ple.layers"));
        noEngram.removeTensor("per_layer_token_embd.weight");
        noEngram.removeTensor("blk.1.engram_kv.weight");
        noEngram.removeTensor("blk.1.engram_gate.weight");
        assertArrayEquals("zero engram_kv leaves exactly the model without the Engram extra", zeroKv,
                firstLogits(noEngram), 0f);
    }

    private static float[] firstLogits(GgufTestWriter w) throws IOException {
        try (GGUFModel m = open(w)) {
            return m.forwardBackend().forwardRows(new int[]{1, 2, 3, 4}, 0, new int[]{3})[0];
        }
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** Two interleaved sequences of the same length; each sequence's last row is its output row. */
    private static ForwardBatch interleave(int[] a, int[] b) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length + b.length);
        for (int i = 0; i < a.length; i++) {
            batch.add(a[i], i, 0, i == a.length - 1);
            batch.add(b[i], i, 1, i == b.length - 1);
        }
        return batch.build();
    }

    private static float[] contiguous(GGUFModel model, int[] t) {
        model.forwardBackend().reset();
        return model.forwardBackend().forwardRows(t, 0, new int[]{t.length - 1})[0];
    }

    /**
     * The paged two-sequence step runs the Engram extra through its
     * {@link com.eonmux.neuralfree.gguf.memory.StepLayout} hook and each sequence hashes its own token record: both
     * sequences' logits equal their contiguous runs.
     */
    @Test
    public void pagedSequencesMatchTheContiguousRuns() throws IOException {
        Path model = SyntheticRoutedModels.deepseek41().writeTempFile();
        int[] a = tokens(1, 7);
        int[] b = tokens(2, 7);
        float[] refA;
        float[] refB;
        try (GGUFModel one = new GGUFModel(model.toString(), config())) {
            refA = contiguous(one, a);
            refB = contiguous(one, b);
        }
        try (GGUFModel two = new GGUFModel(model.toString(), pagedConfig())) {
            HeapModelMemory mem = (HeapModelMemory) two.forwardBackend().memory();
            assertNotNull(mem.pagedSequences());
            float[][] out = two.forwardBackend().forwardBatch(interleave(a, b));
            assertEquals(2, out.length);
            assertArrayEquals("sequence A logits match its contiguous run", refA, out[0], 1e-4f);
            assertArrayEquals("sequence B logits match its contiguous run", refB, out[1], 1e-4f);
        }
    }

    /**
     * The candidate fixture over paged sequences: layer 2 reads layer 0's top-k and the candidate blocks of its own
     * segments, so the paged logits equal the contiguous runs.
     */
    @Test
    public void pagedCandidateSequencesMatchTheContiguousRuns() throws IOException {
        Path model = SyntheticRoutedModels.deepseek41Candidate().writeTempFile();
        int[] a = tokens(3, 6);
        int[] b = tokens(4, 6);
        float[] refA;
        float[] refB;
        try (GGUFModel one = new GGUFModel(model.toString(), config())) {
            refA = contiguous(one, a);
            refB = contiguous(one, b);
        }
        try (GGUFModel two = new GGUFModel(model.toString(), pagedConfig())) {
            float[][] out = two.forwardBackend().forwardBatch(interleave(a, b));
            assertEquals(2, out.length);
            assertArrayEquals("sequence A logits match its contiguous run", refA, out[0], 1e-4f);
            assertArrayEquals("sequence B logits match its contiguous run", refB, out[1], 1e-4f);
        }
    }

    /**
     * A prompt longer than one ubatch splits into several steps; the second step's attention reads the compressed
     * rows the first step wrote, so the split run equals the same tokens in one step.
     */
    @Test
    public void multipleUbatchStepsMatchASingleStep() throws IOException {
        Path model = SyntheticRoutedModels.deepseek41().writeTempFile();
        int[] a = tokens(5, 9);
        EngineConfig wide = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(32).ubatch(32)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
        float[] reference;
        float[] split;
        try (GGUFModel one = new GGUFModel(model.toString(), wide)) {
            reference = contiguous(one, a);
        }
        try (GGUFModel two = new GGUFModel(model.toString(), config())) {
            split = contiguous(two, a);
        }
        assertArrayEquals("the 8 + 1 token split equals the single 9-token step", reference, split, 1e-4f);
    }

    /**
     * The GPU DeepSeek-V4 mixer accepts a V4.1 layer: the shared compressed rows, the candidate selection, the index
     * key stored beside each latent and the Engram extra all have GPU kernels.
     */
    @Test
    public void theGpuMixerAcceptsV41Layers() throws IOException {
        try (GGUFModel m = open(SyntheticRoutedModels.deepseek41())) {
            for (int il = 0; il < m.spec().nLayer(); il++) {
                assertNotNull("GPU mixer of layer " + il,
                        Dsv4Blocks.gpuMixers().get(0).create(m.spec().layer(il)));
            }
        }
    }

    /**
     * DeepSeek-V4.1 has no MTP graph (models/deepseek41.cpp:221-224): its DSpark draft is a separate DFLASH file, so
     * the MTP route over a V4.1 target is refused by name and points at that draft type.
     */
    @Test
    public void theMtpRouteRefusesAV41TargetByName() throws IOException {
        Path model = SyntheticRoutedModels.deepseek41().writeTempFile();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(32).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .mtp(true).build();
        try (EngineModel m = LLMModelFactory.open(model.toString(), config())) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> LLMModelFactory.speculate(m, cfg));
            assertTrue(e.getMessage(), e.getMessage().contains("DSpark"));
        }
    }

    /**
     * A token's logits do not depend on how many rows the step carries after it, nor on whether the step asked for it
     * by row index: the V4.1 head collapses each output row with the last FFN pre-mix saved for that row. The logits
     * of row 2 of a four-token step equal the decode step at position 2 after a two-token prefill.
     */
    @Test
    public void anOutputRowMatchesItsDecodeStep() throws IOException {
        Path model = SyntheticRoutedModels.deepseek41().writeTempFile();
        int[] prompt = tokens(6, 4);
        float[] row2;
        float[] decoded;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            row2 = m.forwardBackend().forwardRows(prompt, 0, new int[]{2})[0];
        }
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            m.forwardBackend().forwardRows(Arrays.copyOf(prompt, 2), 0, new int[]{1});
            decoded = m.forwardBackend().forwardRows(new int[]{prompt[2]}, 2, new int[]{0})[0];
        }
        assertArrayEquals("row 2 of the step equals the decode step at position 2", row2, decoded, 1e-4f);
    }

    private static void assertRefused(GgufTestWriter w, String key) {
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> open(w));
        assertTrue(e.getMessage(), e.getMessage().contains(key));
    }

    @Test
    public void aSourceLayerOfRatioZeroIsRefused() {
        GgufTestWriter w = SyntheticRoutedModels.deepseek41();
        w.u32Array(w.archKey("attention.compress_ratios"), 2, 0, 1, 1);
        w.u32Array(w.archKey("attention.kv_source_layers"), 0, 1, 2);
        w.u32Array(w.archKey("attention.indexer.source_layers"), 0, 1, 2);
        assertRefused(w, "attention.kv_source_layers");
    }

    @Test
    public void aLayerWithoutAMatchingSourceIsRefused() {
        GgufTestWriter w = SyntheticRoutedModels.deepseek41();
        w.u32Array(w.archKey("attention.kv_source_layers"), 0);
        w.u32Array(w.archKey("attention.indexer.source_layers"), 0);
        assertRefused(w, "attention.kv_source_layers");
    }

    @Test
    public void sourcesOfDifferentRatiosAreRefused() {
        GgufTestWriter w = SyntheticRoutedModels.deepseek41();
        w.u32Array(w.archKey("attention.indexer.source_layers"), 0, 1);
        assertRefused(w, "attention.indexer.source_layers");
    }

    @Test
    public void anInvalidCandidateSourceIsRefused() {
        GgufTestWriter w = SyntheticRoutedModels.deepseek41Candidate();
        w.u32(w.archKey("attention.indexer.candidate_source_layer"), 1);
        assertRefused(w, "attention.indexer.candidate_source_layer");
    }

    @Test
    public void anEmptyCandidateBlockIsRefused() {
        GgufTestWriter w = SyntheticRoutedModels.deepseek41Candidate();
        w.u32(w.archKey("attention.indexer.candidate_block_size"), 0);
        assertRefused(w, "attention.indexer.candidate_block_size");
    }

    @Test
    public void aCandidateConsumerOfAnotherRatioIsRefused() {
        GgufTestWriter w = SyntheticRoutedModels.deepseek41Candidate();
        w.u32Array(w.archKey("attention.compress_ratios"), 2, 2, 1);
        assertRefused(w, "attention.indexer.candidate_source_layer");
    }

    @Test
    public void anUnsupportedCompressRatioIsRefused() {
        GgufTestWriter w = SyntheticRoutedModels.deepseek41();
        w.u32Array(w.archKey("attention.compress_ratios"), 2, 2, 5, 1);
        assertRefused(w, "attention.compress_ratios");
    }

    @Test
    public void anEmptyIndexerTopKIsRefused() {
        GgufTestWriter w = SyntheticRoutedModels.deepseek41();
        w.u32(w.archKey("attention.indexer.top_k"), 0);
        assertRefused(w, "attention.indexer.top_k");
    }

    @Test
    public void inconsistentEngramMetadataIsRefused() {
        GgufTestWriter w = SyntheticRoutedModels.deepseek41();
        w.u64Array(w.archKey("ple.head_offsets"), 0);
        assertRefused(w, "ple.head_offsets");
    }
}

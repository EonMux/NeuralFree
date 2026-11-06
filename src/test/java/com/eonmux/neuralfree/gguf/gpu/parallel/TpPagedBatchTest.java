package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Paged multi-sequence serving under tensor parallelism on the OpenCL backend: a two-rank split whose main device
 * serves {@code forwardBatch} over a paged memory produces the same logits as the whole-device load of the same
 * model, with every rank cache carrying the step mirror's page tables, cell rows and page copies. A prefix shared
 * from one sequence to another and then privatized by a write also matches the whole-device run, which exercises the
 * engine-side page copies the mirror forwards to the ranks. The GPU cases are skipped when no OpenCL device is
 * available.
 */
public class TpPagedBatchTest {

    private static final int UBATCH = 8;
    private static final int N_CTX = 64;
    private static final int RANKS = 2;
    private static final int N_TOKENS = 5;
    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float TOL = 1e-3f;

    private Path file;
    private HWAcceleration hw;
    private OclComputeBackend main;
    private OclComputeBackend peer;
    private LlmKernels kernels;
    private ComputeQueue queue;
    private GpuPlaceholders placeholders;
    private GpuScratch scratch;
    private GpuOps ops;
    private GGUFFileParser parser;
    private WeightStore store;
    private GpuWeightUploader referenceUploader;
    private GpuWeightUploader splitUploader;
    private ModelSpec spec;
    private MemoryPlan referencePlan;
    private MemoryPlan splitPlan;
    private DeviceModelMemory referenceMemory;
    private DeviceModelMemory splitMemory;
    private TensorParallelGroup group;
    private GpuForwardBackend referenceRaw;
    private ForwardBackend reference;
    private ForwardBackend split;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(1).build()).writeTempFile();
        parser = new GGUFFileParser(file.toString());
        hw = new HWAcceleration();
        Assume.assumeFalse("no OpenCL device", hw.getCLDevices().isEmpty());
        main = new OclComputeBackend(hw.getCLDevices().get(0));
        peer = new OclComputeBackend(hw.getCLDevices().get(0));
        kernels = LlmKernels.open(main);
        queue = main.computeQueue();
        placeholders = GpuPlaceholders.allocate(main, queue);
        MemoryConfig mem = MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH).parallel(RANKS)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), mem);
        scratch = GpuScratch.forModel(main, spec, UBATCH);
        ops = new GpuOps(queue, kernels, scratch, placeholders);
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        GpuWeightUploader.Config cfg = GpuWeightUploader.config(
                GpuWeightUploader.TypePolicy.forKernels(kernels)).build();
        referenceUploader = new GpuWeightUploader(main, queue, parser.getTensors(), store, "llama", cfg);
        referenceUploader.uploadResident(t -> false);
        splitUploader = new GpuWeightUploader(main, queue, parser.getTensors(), store, "llama", cfg);
        TpAttentionPlan attentionPlan = TpAttentionPlan.of(spec, RANKS, splitUploader);
        TpVocabPlan vocabPlan = TpVocabPlan.of(spec, RANKS, splitUploader);
        splitUploader.uploadResident(t -> attentionPlan.splits(t.name()) || vocabPlan.splits(t.name()));
        referencePlan = MemoryPlanner.planForContext(spec, mem, N_CTX);
        referenceMemory = DeviceModelMemory.allocate(main, queue, referencePlan,
                spec.hparams().mropeTextPositions(), null);
        splitPlan = MemoryPlanner.planForContext(spec, mem, N_CTX);
        splitMemory = DeviceModelMemory.allocate(main, queue, splitPlan,
                spec.hparams().mropeTextPositions(), null);
        group = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same device")));
        group.init(spec, UBATCH, kernels, queue, placeholders, null, attentionPlan, vocabPlan, null, mem, N_CTX);
        group.upload(store, splitUploader);
        group.verifyPagedMemory(splitMemory);
        GpuForwardContext referenceContext = GpuForwardContext.builder(kernels, ops, referenceUploader, spec)
                .memory(referenceMemory).build();
        referenceRaw = new GpuForwardBackend(referenceContext, GpuFactoryCatalog.defaults(),
                GpuFactoryCatalog.headPreprocessors());
        reference = referenceRaw;
        GpuBlockRegistry decorated = GpuFactoryCatalog.defaults()
                .withMixerDecorator(group.attentionDecorator())
                .withFfnDecorator(group.decorator());
        GpuForwardContext splitContext = GpuForwardContext.builder(kernels, ops, splitUploader, spec)
                .memory(splitMemory).build();
        GpuForwardBackend splitRaw = new GpuForwardBackend(splitContext, decorated,
                GpuFactoryCatalog.headPreprocessors(), group.vocabEmbedding(), group.vocabHead());
        split = new TensorParallelBackend(splitRaw, group);
    }

    @After
    public void close() throws IOException {
        try {
            if (split != null) {
                split.close();
            }
            if (referenceRaw != null) {
                referenceRaw.close();
            }
            if (group != null) {
                group.close();
            }
            if (splitMemory != null) {
                splitMemory.close();
            }
            if (referenceMemory != null) {
                referenceMemory.close();
            }
            splitUploader.close();
            referenceUploader.close();
            ops.close();
            scratch.close();
            store.close();
            placeholders.close();
            main.close();
            peer.close();
            parser.close();
        } finally {
            if (hw != null) {
                hw.shutdown();
            }
            Files.deleteIfExists(file);
        }
    }

    /** Tokens a sequence runs (deterministic ids inside the synthetic vocab). */
    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** A batch whose rows interleave two prompts; the last row of each sequence is its output row. */
    private static ForwardBatch interleave(int[] a, int[] b) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length + b.length);
        for (int i = 0; i < a.length; i++) {
            batch.add(a[i], i, SEQ_A, i == a.length - 1);
            if (i < b.length) {
                batch.add(b[i], i, SEQ_B, i == b.length - 1);
            }
        }
        return batch.build();
    }

    /** A batch holding one prompt of {@code seq}; the last row is its output row. */
    private static ForwardBatch prompt(int[] tokens, int seq) {
        return prompt(tokens, 0, seq);
    }

    /** {@link #prompt(int[], int)} with its first row at {@code pos0}. */
    private static ForwardBatch prompt(int[] tokens, int pos0, int seq) {
        ForwardBatch.Builder batch = ForwardBatch.builder(tokens.length);
        for (int i = 0; i < tokens.length; i++) {
            batch.add(tokens[i], pos0 + i, seq, i == tokens.length - 1);
        }
        return batch.build();
    }

    /** One row of each of the two sequences at the same position. */
    private static ForwardBatch pairAt(int tokenA, int tokenB, int position) {
        ForwardBatch.Builder batch = ForwardBatch.builder(2);
        batch.add(tokenA, position, SEQ_A, true);
        batch.add(tokenB, position, SEQ_B, true);
        return batch.build();
    }

    private static ForwardBatch row(int token, int position, int seq) {
        ForwardBatch.Builder batch = ForwardBatch.builder(1);
        batch.add(token, position, seq, true);
        return batch.build();
    }

    @Test
    public void thePagedBatchMatchesTheWholeDeviceLoad() throws IOException {
        int[] a = tokens(1, N_TOKENS);
        int[] b = tokens(2, N_TOKENS);
        assertEquals(RANKS, split.maxSequences());
        assertNull("the split mixer runs paged steps", split.batchRefusal());
        float[][] expected = reference.forwardBatch(interleave(a, b));
        float[][] actual = split.forwardBatch(interleave(a, b));
        assertEquals(2, actual.length);
        assertArrayEquals("sequence A logits match the whole-device load", expected[0], actual[0], TOL);
        assertArrayEquals("sequence B logits match the whole-device load", expected[1], actual[1], TOL);

        float[][] expectedNext = reference.forwardBatch(pairAt(a[0], b[0], N_TOKENS));
        float[][] actualNext = split.forwardBatch(pairAt(a[0], b[0], N_TOKENS));
        assertArrayEquals("the next batched decode step matches", expectedNext[0], actualNext[0], TOL);
        assertArrayEquals("the next batched decode step matches", expectedNext[1], actualNext[1], TOL);
    }

    @Test
    public void aSharedPrefixPrivatizedOnWriteMatchesTheWholeDeviceLoad() throws IOException {
        int[] a = tokens(3, N_TOKENS);
        ForwardBatch prompt = prompt(a, SEQ_A);
        float[][] expectedPrompt = reference.forwardBatch(prompt);
        float[][] actualPrompt = split.forwardBatch(prompt);
        assertArrayEquals("the donor prompt matches", expectedPrompt[0], actualPrompt[0], TOL);
        int forkPos = N_TOKENS - 1;
        int fork = tokens(5, 1)[0];
        referenceMemory.sequences().copy(SEQ_A, SEQ_B, 0, -1);
        splitMemory.sequences().copy(SEQ_A, SEQ_B, 0, -1);
        assertTrue("the fork shares the donor's cells", splitMemory.sequences().sharesPages(SEQ_B));
        referenceMemory.sequences().remove(SEQ_B, forkPos, N_TOKENS);
        splitMemory.sequences().remove(SEQ_B, forkPos, N_TOKENS);

        float[][] expected = reference.forwardBatch(row(fork, forkPos, SEQ_B));
        float[][] actual = split.forwardBatch(row(fork, forkPos, SEQ_B));
        assertEquals(1, actual.length);
        assertArrayEquals("the privatized row matches the whole-device load", expected[0], actual[0], TOL);
    }

    @Test
    public void aPagedSequenceShiftMatchesTheWholeDeviceLoad() throws IOException {
        int[] a = tokens(6, 8);
        ForwardBatch prompt = prompt(a, SEQ_A);
        float[][] expectedPrompt = reference.forwardBatch(prompt);
        float[][] actualPrompt = split.forwardBatch(prompt);
        assertArrayEquals("the prompt before the shift matches", expectedPrompt[0], actualPrompt[0], TOL);
        assertNull("the split shifts a paged sequence", split.shiftSequenceRefusal());
        reference.shiftSequence(SEQ_A, 2, 2);
        split.shiftSequence(SEQ_A, 2, 2);

        ForwardBatch tail = prompt(tokens(7, 2), 6, SEQ_A);
        float[][] expected = reference.forwardBatch(tail);
        float[][] actual = split.forwardBatch(tail);
        assertEquals(1, actual.length);
        assertArrayEquals("the step after the shifted sequence matches", expected[0], actual[0], TOL);
    }

    @Test
    public void aPageGeometryMismatchIsRefusedByName() {
        MemoryConfig wider = MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH).parallel(RANKS * 2)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        MemoryPlan otherPlan = MemoryPlanner.planForContext(spec, wider, N_CTX);
        try (DeviceModelMemory other = DeviceModelMemory.allocate(main, queue, otherPlan,
                spec.hparams().mropeTextPositions(), null)) {
            UnsupportedOperationException refusal = assertThrows(UnsupportedOperationException.class,
                    () -> group.verifyPagedMemory(other));
            assertNotNull("the refusal names the rank", refusal.getMessage());
        }
    }
}

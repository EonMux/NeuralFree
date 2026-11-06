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
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Prefill context parallelism (vLLM's {@code prefill_context_parallel_size}, the DualChunkSwap query partition) over
 * the paged memory's batched steps on the OpenCL backend: the coordinator cuts every multi-row segment of a batched
 * prefill step into the ranks' chunk pairs over the page-table shards ({@code CpPageShard}), replicates the
 * single-row segments (the decode rows of a mixed step), gathers the queries, scatters the cache rows by translated
 * cell and merges the per-rank attention partitions, and the logits match the whole-device load. The steps cover a
 * batched prefill of sequences of different lengths, a mixed prefill and decode step, a three-sequence step whose
 * pages both ranks own, and a prefix forked by copy-on-write and privatized on either rank's page. The GPU cases are
 * skipped when no OpenCL device is available.
 */
public class PcpPagedBatchTest {

    private static final int UBATCH = 8;
    private static final int N_CTX = 64;
    private static final int RANKS = 2;
    private static final int SLOTS = 4;
    private static final int PAGE_SIZE = 8;
    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final int SEQ_C = 2;
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
    private GpuWeightUploader uploader;
    private ModelSpec spec;
    private MemoryConfig mem;
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
        mem = MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH).parallel(SLOTS).kvPageSize(PAGE_SIZE)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), mem);
        scratch = GpuScratch.forModel(main, spec, UBATCH);
        ops = new GpuOps(queue, kernels, scratch, placeholders);
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        uploader = new GpuWeightUploader(main, queue, parser.getTensors(), store, "llama",
                GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(kernels)).build());
        uploader.uploadResident(t -> false);
        referenceMemory = DeviceModelMemory.allocate(main, queue,
                MemoryPlanner.planForContext(spec, mem, N_CTX), spec.hparams().mropeTextPositions(), null);
        splitMemory = DeviceModelMemory.allocate(main, queue,
                MemoryPlanner.planForContext(spec, mem, N_CTX), spec.hparams().mropeTextPositions(), null);
        TpAttentionPlan attentionPlan = TpAttentionPlan.context(spec, RANKS, uploader);
        group = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same device")));
        group.prefillContextParallelSize(RANKS);
        group.init(spec, UBATCH, kernels, queue, placeholders, null, attentionPlan, mem, N_CTX);
        group.upload(store, uploader);
        group.verifyPagedMemory(splitMemory);
        GpuForwardContext referenceContext = GpuForwardContext.builder(kernels, ops, uploader, spec)
                .memory(referenceMemory).build();
        referenceRaw = new GpuForwardBackend(referenceContext, GpuFactoryCatalog.defaults(),
                GpuFactoryCatalog.headPreprocessors());
        reference = referenceRaw;
        GpuBlockRegistry decorated = GpuFactoryCatalog.defaults()
                .withMixerDecorator(group.attentionDecorator());
        GpuForwardContext splitContext = GpuForwardContext.builder(kernels, ops, uploader, spec)
                .memory(splitMemory).build();
        GpuForwardBackend splitRaw = new GpuForwardBackend(splitContext, decorated,
                GpuFactoryCatalog.headPreprocessors());
        /* The DualChunkSwap prefill over the paged memory: the coordinator carries the whole batched prefill step,
         * the plain executor keeps the decode steps. */
        splitRaw.batchTrunkRunner(group.pcpBatchTrunkRunner(splitContext, splitRaw.layerExecutor()));
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
            uploader.close();
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

    /** A batch holding the whole prompt of every sequence back to back; each prompt's last row is its output row. */
    private static ForwardBatch prompts(int[]... sequences) {
        int rows = 0;
        for (int[] s : sequences) {
            rows += s.length;
        }
        ForwardBatch.Builder batch = ForwardBatch.builder(rows);
        int seq = 0;
        for (int[] s : sequences) {
            for (int i = 0; i < s.length; i++) {
                batch.add(s[i], i, seq, i == s.length - 1);
            }
            seq++;
        }
        return batch.build();
    }

    /** A mixed step: {@code head} continuation rows of one sequence, then one decode row of the next. */
    private static ForwardBatch mixed(int[] continuation, int position, int seq, int token, int decodePosition,
                                      int decodeSeq) {
        ForwardBatch.Builder batch = ForwardBatch.builder(continuation.length + 1);
        for (int i = 0; i < continuation.length; i++) {
            batch.add(continuation[i], position + i, seq, i == continuation.length - 1);
        }
        batch.add(token, decodePosition, decodeSeq, true);
        return batch.build();
    }

    /** One decode row of each of two sequences, each at its own next position. */
    private static ForwardBatch decodePair(int tokenA, int posA, int tokenB, int posB) {
        ForwardBatch.Builder batch = ForwardBatch.builder(2);
        batch.add(tokenA, posA, SEQ_A, true);
        batch.add(tokenB, posB, SEQ_B, true);
        return batch.build();
    }

    private static ForwardBatch row(int token, int position, int seq) {
        ForwardBatch.Builder batch = ForwardBatch.builder(1);
        batch.add(token, position, seq, true);
        return batch.build();
    }

    @Test
    public void theBatchedPagedPrefillMatchesTheWholeDeviceLoad() throws IOException {
        int[] a = tokens(1, 12);
        int[] b = tokens(2, 5);
        assertEquals(SLOTS, split.maxSequences());
        assertNull("the prefill query split runs the paged batched steps", split.batchRefusal());
        /* One batch of seventeen rows over ubatches of eight: sequence A's whole prefill, then a step whose rows
         * split between the sequences (four of A over rank 1's page, four of B over rank 0's page), then B's last
         * row alone. Every page the steps write belongs to the rank owning its first stored position, so the chunk
         * pairs scatter in both directions between the ranks. */
        float[][] expected = reference.forwardBatch(prompts(a, b));
        float[][] actual = split.forwardBatch(prompts(a, b));
        assertEquals(2, actual.length);
        assertArrayEquals("sequence A logits match the whole-device load", expected[0], actual[0], TOL);
        assertArrayEquals("sequence B logits match the whole-device load", expected[1], actual[1], TOL);

        float[][] expectedNext = reference.forwardBatch(decodePair(a[0], 12, b[0], 5));
        float[][] actualNext = split.forwardBatch(decodePair(a[0], 12, b[0], 5));
        assertEquals(2, actualNext.length);
        assertArrayEquals("the next batched decode step matches", expectedNext[0], actualNext[0], TOL);
        assertArrayEquals("the next batched decode step matches", expectedNext[1], actualNext[1], TOL);
    }

    @Test
    public void aMixedPrefillAndDecodeStepMatchesTheWholeDeviceLoad() throws IOException {
        int[] a = tokens(3, 12);
        int[] b = tokens(4, 5);
        float[][] expectedFirst = reference.forwardBatch(prompts(a, b));
        float[][] actualFirst = split.forwardBatch(prompts(a, b));
        assertArrayEquals("sequence A logits match the whole-device load", expectedFirst[0], actualFirst[0], TOL);
        assertArrayEquals("sequence B logits match the whole-device load", expectedFirst[1], actualFirst[1], TOL);

        /* One step: four continuation rows of sequence A (whose page the second rank owns) beside one decode row of
         * sequence B (whose page the first rank owns) — the decode row replicates onto the ranks while A's rows cut
         * into chunk pairs. */
        int[] more = tokens(5, 4);
        float[][] expected = reference.forwardBatch(mixed(more, 12, SEQ_A, b[0], 5, SEQ_B));
        float[][] actual = split.forwardBatch(mixed(more, 12, SEQ_A, b[0], 5, SEQ_B));
        assertEquals(2, actual.length);
        assertArrayEquals("the continuation rows match the whole-device load", expected[0], actual[0], TOL);
        assertArrayEquals("the decode row matches the whole-device load", expected[1], actual[1], TOL);

        float[][] expectedNext = reference.forwardBatch(decodePair(tokens(6, 1)[0], 16, tokens(7, 1)[0], 6));
        float[][] actualNext = split.forwardBatch(decodePair(tokens(6, 1)[0], 16, tokens(7, 1)[0], 6));
        assertArrayEquals("the decode after the mixed step matches", expectedNext[0], actualNext[0], TOL);
        assertArrayEquals("the decode after the mixed step matches", expectedNext[1], actualNext[1], TOL);
    }

    @Test
    public void aThreeSegmentStepWithReplicatedRowMatchesTheWholeDeviceLoad() throws IOException {
        int[] a = tokens(8, 4);
        int[] b = tokens(9, 3);
        int[] c = tokens(10, 1);
        /* One ubatch of eight rows: two multi-row segments cut into chunk pairs and one single-row segment that
         * replicates onto the ranks. */
        float[][] expected = reference.forwardBatch(prompts(a, b, c));
        float[][] actual = split.forwardBatch(prompts(a, b, c));
        assertEquals(3, actual.length);
        assertArrayEquals("sequence A logits match the whole-device load", expected[0], actual[0], TOL);
        assertArrayEquals("sequence B logits match the whole-device load", expected[1], actual[1], TOL);
        assertArrayEquals("sequence C logits match the whole-device load", expected[2], actual[2], TOL);

        float[][] expectedNext = reference.forwardBatch(row(tokens(11, 1)[0], 1, SEQ_C));
        float[][] actualNext = split.forwardBatch(row(tokens(11, 1)[0], 1, SEQ_C));
        assertEquals(1, actualNext.length);
        assertArrayEquals("the decode of the replicated row's sequence matches", expectedNext[0], actualNext[0], TOL);
    }

    @Test
    public void aSharedPrefixPrivatizedOnWriteMatchesTheWholeDeviceLoad() throws IOException {
        int[] a = tokens(12, 12);
        ForwardBatch prompt = prompts(a);
        float[][] expectedPrompt = reference.forwardBatch(prompt);
        float[][] actualPrompt = split.forwardBatch(prompt);
        assertArrayEquals("the donor prompt matches", expectedPrompt[0], actualPrompt[0], TOL);

        /* A fork on the second rank's page (position 11, whose page the second rank owns): the copy shares the
         * donor's cells until the decode row privatizes its position. */
        int fork = tokens(13, 1)[0];
        referenceMemory.sequences().copy(SEQ_A, SEQ_B, 0, -1);
        splitMemory.sequences().copy(SEQ_A, SEQ_B, 0, -1);
        assertTrue("the fork shares the donor's cells", splitMemory.sequences().sharesPages(SEQ_B));
        referenceMemory.sequences().remove(SEQ_B, 11, 12);
        splitMemory.sequences().remove(SEQ_B, 11, 12);
        float[][] expected = reference.forwardBatch(row(fork, 11, SEQ_B));
        float[][] actual = split.forwardBatch(row(fork, 11, SEQ_B));
        assertEquals(1, actual.length);
        assertArrayEquals("the privatized row on the second rank's page matches", expected[0], actual[0], TOL);

        /* A fork on the first rank's page (position 3): the same privatization where the first rank owns the cells.
         * The donor's rows leave first (a fresh prompt cannot start below a stored position), and the recycled
         * pages still shard by their new first stored position. */
        int[] c = tokens(14, 12);
        referenceMemory.sequences().remove(SEQ_A, 0, 12);
        splitMemory.sequences().remove(SEQ_A, 0, 12);
        float[][] expectedDonor = reference.forwardBatch(prompts(c));
        float[][] actualDonor = split.forwardBatch(prompts(c));
        assertArrayEquals("the second donor prompt matches", expectedDonor[0], actualDonor[0], TOL);
        int lowFork = tokens(15, 1)[0];
        referenceMemory.sequences().copy(SEQ_A, SEQ_C, 0, -1);
        splitMemory.sequences().copy(SEQ_A, SEQ_C, 0, -1);
        referenceMemory.sequences().remove(SEQ_C, 3, 12);
        splitMemory.sequences().remove(SEQ_C, 3, 12);
        float[][] expectedLow = reference.forwardBatch(row(lowFork, 3, SEQ_C));
        float[][] actualLow = split.forwardBatch(row(lowFork, 3, SEQ_C));
        assertEquals(1, actualLow.length);
        assertArrayEquals("the privatized row on the first rank's page matches", expectedLow[0], actualLow[0], TOL);
    }
}

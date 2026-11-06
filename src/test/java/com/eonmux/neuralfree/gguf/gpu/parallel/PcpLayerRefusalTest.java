package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
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
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The load-level layer refusals of a prefill query split (vLLM's {@code prefill_context_parallel_size}, driven
 * through {@link TensorParallelGroup#init} with the property's rank count) on the OpenCL backend: before either
 * PCP runner is built, every rank's bound mixer of the first refusing layer names the layer, the property and the
 * reason, and the group releases everything the init built, so the same devices carry the next load. A TurboQuant
 * key stream is refused by name on both runners — its side planes live outside the cache rows the scatter moves.
 * The quantized-KV Hadamard rotation is not a refusal: every rank rotates its own packed rows before the stores
 * the shadow store re-runs, so the split over a Q8_0 rotated cache keeps the single-device logits and the
 * byte-exact cache rows. The GPU cases are skipped when no OpenCL device is available.
 */
public class PcpLayerRefusalTest {

    private static final int UBATCH = 4;
    private static final int RANKS = 2;
    private static final int CONTIGUOUS_CTX = 10;
    private static final int PAGED_CTX = 64;
    private static final int PAGED_UBATCH = 8;
    private static final int SLOTS = 4;
    private static final int PAGE_SIZE = 8;
    private static final int N_LAYER = 1;
    /** Head dim of the rotation fixture: a multiple of the Hadamard base block, so the cache rotates. */
    private static final int ROT_HEAD_DIM = 64;
    private static final float TOL = 2e-3f;

    private Path file;
    private HWAcceleration hw;
    private OclComputeBackend main;
    private OclComputeBackend peer;
    private LlmKernels kernels;
    private ComputeQueue queue;
    private GpuPlaceholders placeholders;
    private GGUFFileParser parser;
    private WeightStore store;
    private GpuWeightUploader uploader;
    private MemoryConfig mem;
    private ModelSpec spec;
    private GpuScratch scratch;
    private GpuOps ops;
    private DeviceModelMemory mainMemory;
    private GpuForwardContext mainContext;
    private DeviceModelMemory referenceMemory;
    private DeviceModelMemory splitMemory;
    private GpuForwardContext splitContext;
    private final List<AutoCloseable> outer = new ArrayList<>();
    private final List<GpuForwardBackend> backends = new ArrayList<>();
    private final List<TensorParallelGroup> groups = new ArrayList<>();
    private final List<DeviceModelMemory> memories = new ArrayList<>();

    @After
    public void close() throws IOException {
        try {
            closeAll(outer);
            closeAll(backends);
            closeAll(groups);
            closeAll(memories);
            if (uploader != null) {
                uploader.close();
            }
            if (ops != null) {
                ops.close();
            }
            if (scratch != null) {
                scratch.close();
            }
            if (store != null) {
                store.close();
            }
            if (placeholders != null) {
                placeholders.close();
            }
            if (main != null) {
                main.close();
            }
            if (peer != null) {
                peer.close();
            }
            if (parser != null) {
                parser.close();
            }
        } finally {
            if (hw != null) {
                hw.shutdown();
            }
            if (file != null) {
                Files.deleteIfExists(file);
            }
        }
    }

    /** Closes every closeable of a phase, keeping the first failure and suppressing the rest. */
    private static void closeAll(List<? extends AutoCloseable> list) throws IOException {
        Throwable first = null;
        List<AutoCloseable> remaining = new ArrayList<>(list);
        list.clear();
        for (AutoCloseable c : remaining) {
            try {
                c.close();
            } catch (Throwable t) {
                if (first == null) {
                    first = t;
                } else {
                    first.addSuppressed(t);
                }
            }
        }
        if (first instanceof RuntimeException) {
            throw (RuntimeException) first;
        }
        if (first instanceof Error) {
            throw (Error) first;
        }
        if (first != null) {
            throw new IOException("closing the fixture failed", first);
        }
    }

    // ------------------------------------------------------------------ fixtures

    /** The contiguous fixture of {@code CpPrefillSplitTest} over one cache type pair and one head dim. */
    private void openContiguous(KvDType typeK, KvDType typeV, int headDim) throws IOException {
        openBase(headDim);
        mem = MemoryConfig.defaults().toBuilder().ubatch(UBATCH).typeK(typeK).typeV(typeV).build();
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), mem);
        scratch = GpuScratch.forModel(main, spec, UBATCH);
        ops = new GpuOps(queue, kernels, scratch, placeholders);
        mainMemory = DeviceModelMemory.allocate(main, queue, MemoryPlanner.planForContext(spec, mem, CONTIGUOUS_CTX),
                spec.hparams().mropeTextPositions(), null);
        memories.add(mainMemory);
        mainContext = GpuForwardContext.builder(kernels, ops, uploader, spec).memory(mainMemory).build();
    }

    /** The paged fixture of {@code PcpPagedBatchTest}: the group is up, the trunk runner is left unbuilt. */
    private void openPaged(KvDType typeK, KvDType typeV, int headDim) throws IOException {
        openBase(headDim);
        mem = MemoryConfig.builder().nCtx(PAGED_CTX).ubatch(PAGED_UBATCH).parallel(SLOTS).kvPageSize(PAGE_SIZE)
                .typeK(typeK).typeV(typeV).build();
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), mem);
        scratch = GpuScratch.forModel(main, spec, PAGED_UBATCH);
        ops = new GpuOps(queue, kernels, scratch, placeholders);
        referenceMemory = DeviceModelMemory.allocate(main, queue,
                MemoryPlanner.planForContext(spec, mem, PAGED_CTX), spec.hparams().mropeTextPositions(), null);
        memories.add(referenceMemory);
        splitMemory = DeviceModelMemory.allocate(main, queue, MemoryPlanner.planForContext(spec, mem, PAGED_CTX),
                spec.hparams().mropeTextPositions(), null);
        memories.add(splitMemory);
        TensorParallelGroup group = contiguousGroup(spec, mem, PAGED_UBATCH, PAGED_CTX);
        group.verifyPagedMemory(splitMemory);
        GpuForwardContext referenceContext = GpuForwardContext.builder(kernels, ops, uploader, spec)
                .memory(referenceMemory).build();
        backends.add(new GpuForwardBackend(referenceContext, GpuFactoryCatalog.defaults(),
                GpuFactoryCatalog.headPreprocessors()));
        splitContext = GpuForwardContext.builder(kernels, ops, uploader, spec).memory(splitMemory).build();
    }

    /** The device-level fixture both runners share: one synthetic file, one device as both ranks. */
    private void openBase(int headDim) throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(N_LAYER)
                .nEmbd(4 * headDim).headDim(headDim).nRot(headDim).build()).writeTempFile();
        parser = new GGUFFileParser(file.toString());
        hw = new HWAcceleration();
        Assume.assumeFalse("no OpenCL device", hw.getCLDevices().isEmpty());
        main = new OclComputeBackend(hw.getCLDevices().get(0));
        peer = new OclComputeBackend(hw.getCLDevices().get(0));
        kernels = LlmKernels.open(main);
        queue = main.computeQueue();
        placeholders = GpuPlaceholders.allocate(main, queue);
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        uploader = new GpuWeightUploader(main, queue, parser.getTensors(), store, "llama",
                GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(kernels)).build());
        uploader.uploadResident(t -> false);
    }

    /** The initialized, uploaded group of the prefill query split over {@code RANKS} ranks. */
    private TensorParallelGroup contiguousGroup(ModelSpec groupSpec, MemoryConfig groupMem, int ubatch, int nCtx) {
        TpAttentionPlan plan = TpAttentionPlan.context(groupSpec, RANKS, uploader);
        TensorParallelGroup group = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same device")));
        group.prefillContextParallelSize(RANKS);
        group.init(groupSpec, ubatch, kernels, queue, placeholders, null, plan, groupMem, nCtx);
        group.upload(store, uploader);
        groups.add(group);
        return group;
    }

    /** The decorated backend whose registry binds the split's mixers, without the trunk runner. */
    private GpuForwardBackend decoratedBackend(TensorParallelGroup group, GpuForwardContext context) {
        GpuBlockRegistry registry = GpuFactoryCatalog.defaults().withMixerDecorator(group.attentionDecorator());
        GpuForwardBackend backend = new GpuForwardBackend(context, registry, GpuFactoryCatalog.headPreprocessors());
        backends.add(backend);
        return backend;
    }

    /** The single-device reference backend: the plain registry, no trunk runner. */
    private GpuForwardBackend wholeBackend(GpuForwardContext context) {
        GpuForwardBackend backend = new GpuForwardBackend(context, GpuFactoryCatalog.defaults(),
                GpuFactoryCatalog.headPreprocessors());
        backends.add(backend);
        return backend;
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void aTurboQuantCacheIsRefusedByNameOnTheContiguousRunner() throws IOException {
        openContiguous(KvDType.TQ_4BIT_KEYS, KvDType.TQ_4BIT_VALUES, 16);
        TensorParallelGroup group = contiguousGroup(spec, mem, UBATCH, CONTIGUOUS_CTX);
        GpuForwardBackend backend = decoratedBackend(group, mainContext);
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> group.pcpTrunkRunner(mainContext, backend.layerExecutor()));
        assertRefusalNames(e, "vendor planes");
    }

    @Test
    public void aTurboQuantCacheIsRefusedByNameOnThePagedRunner() throws IOException {
        openPaged(KvDType.TQ_4BIT_KEYS, KvDType.TQ_4BIT_VALUES, 16);
        GpuForwardBackend splitRaw = decoratedBackend(groups.get(0), splitContext);
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> groups.get(0).pcpBatchTrunkRunner(splitContext, splitRaw.layerExecutor()));
        assertRefusalNames(e, "vendor planes");
    }

    /** The refusal names the refusing layer, the property and the reason. */
    private static void assertRefusalNames(UnsupportedOperationException e, String reason) {
        assertTrue(e.getMessage(), e.getMessage().contains("layer 0"));
        assertTrue(e.getMessage(),
                e.getMessage().contains(EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL + "=" + RANKS));
        assertTrue(e.getMessage(), e.getMessage().contains(reason));
    }

    /**
     * A refused init leaves nothing allocated: the TurboQuant load refuses at the runner and the same devices
     * carry the next load, a plain-cache prefill query split, to the single-device parity.
     */
    @Test
    public void aRefusedInitReleasesTheDevicesForTheNextLoad() throws IOException {
        openContiguous(KvDType.TQ_4BIT_KEYS, KvDType.TQ_4BIT_VALUES, 16);
        TensorParallelGroup refused = contiguousGroup(spec, mem, UBATCH, CONTIGUOUS_CTX);
        GpuForwardBackend refusedBackend = decoratedBackend(refused, mainContext);
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> refused.pcpTrunkRunner(mainContext, refusedBackend.layerExecutor()));
        assertRefusalNames(e, "vendor planes");

        MemoryConfig plainMem = MemoryConfig.defaults().toBuilder().ubatch(UBATCH)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        ModelSpec plainSpec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), plainMem);
        DeviceModelMemory plainMemory = DeviceModelMemory.allocate(main, queue,
                MemoryPlanner.planForContext(plainSpec, plainMem, CONTIGUOUS_CTX),
                plainSpec.hparams().mropeTextPositions(), null);
        memories.add(plainMemory);
        GpuForwardContext plainContext = GpuForwardContext.builder(kernels, ops, uploader, plainSpec)
                .memory(plainMemory).build();

        int il = plainSpec.layers().get(0).il();
        int[] tokens = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
        GpuForwardBackend whole = wholeBackend(plainContext);
        float[][] wholeLogits = runSequence(whole, tokens);
        float[][][] wholeKv = readWholeF32Rows(plainMemory, il, CONTIGUOUS_CTX);
        plainMemory.reset();

        TensorParallelGroup retry = contiguousGroup(plainSpec, plainMem, UBATCH, CONTIGUOUS_CTX);
        GpuForwardBackend split = decoratedBackend(retry, plainContext);
        split.trunkRunner(retry.pcpTrunkRunner(plainContext, split.layerExecutor()));
        float[][] splitLogits = runSequence(split, tokens);
        for (int s = 0; s < wholeLogits.length; s++) {
            assertArrayEquals("the load after the refusal keeps the single-device logits at step " + s,
                    wholeLogits[s], splitLogits[s], TOL);
        }
        for (int rank = 0; rank < RANKS; rank++) {
            GpuKvLayer kv = retry.rankMemory(rank).kv(il);
            for (int p = retry.contextBlockStart(rank); p < retry.contextBlockEnd(rank); p++) {
                int row = kv.layout().rowOf(p);
                assertArrayEquals("rank " + rank + " K row of position " + p + " after the refused init",
                        wholeKv[0][p], readF32Row(kv.k(), kv.layout().rowBytesK(), row), TOL);
                assertArrayEquals("rank " + rank + " V row of position " + p + " after the refused init",
                        wholeKv[1][p], readF32Row(kv.v(), kv.layout().rowBytesV(), row), TOL);
            }
        }
    }

    // ------------------------------------------------------------------ the rotation is carried

    @Test
    public void theQuantizedRotationCacheRunsTheContiguousPrefillQuerySplit() throws IOException {
        openContiguous(KvDType.Q8_0, KvDType.Q8_0, ROT_HEAD_DIM);
        int il = spec.layers().get(0).il();
        assertTrue("the fixture rotates the cache", mainMemory.kv(il).rotation().rotateK());
        int[] tokens = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
        GpuForwardBackend whole = wholeBackend(mainContext);
        float[][] wholeLogits = runSequence(whole, tokens);
        byte[][][] wholeKv = readWholeCacheRows(mainMemory, il, CONTIGUOUS_CTX);
        mainMemory.reset();

        TensorParallelGroup group = contiguousGroup(spec, mem, UBATCH, CONTIGUOUS_CTX);
        GpuForwardBackend split = decoratedBackend(group, mainContext);
        split.trunkRunner(group.pcpTrunkRunner(mainContext, split.layerExecutor()));
        float[][] splitLogits = runSequence(split, tokens);
        for (int s = 0; s < wholeLogits.length; s++) {
            assertArrayEquals("the rotated cache keeps the single-device logits at step " + s,
                    wholeLogits[s], splitLogits[s], TOL);
        }
        for (int rank = 0; rank < RANKS; rank++) {
            GpuKvLayer kv = group.rankMemory(rank).kv(il);
            for (int p = group.contextBlockStart(rank); p < group.contextBlockEnd(rank); p++) {
                int row = kv.layout().rowOf(p);
                assertArrayEquals("rank " + rank + " rotated K row of position " + p, wholeKv[0][p],
                        readRowBytes(kv.k(), kv.layout().rowBytesK(), row));
                assertArrayEquals("rank " + rank + " rotated V row of position " + p, wholeKv[1][p],
                        readRowBytes(kv.v(), kv.layout().rowBytesV(), row));
            }
        }
    }

    @Test
    public void theQuantizedRotationCacheRunsThePagedBatchedPrefill() throws IOException {
        openPaged(KvDType.Q8_0, KvDType.Q8_0, ROT_HEAD_DIM);
        TensorParallelGroup group = groups.get(0);
        int il = spec.layers().get(0).il();
        assertTrue("the fixture rotates the cache", splitMemory.kv(il).rotation().rotateK());
        GpuForwardBackend splitRaw = decoratedBackend(group, splitContext);
        splitRaw.batchTrunkRunner(group.pcpBatchTrunkRunner(splitContext, splitRaw.layerExecutor()));
        GpuForwardBackend reference = backends.get(0);
        ForwardBackend split = new TensorParallelBackend(splitRaw, group);
        outer.add(split);

        int[] a = tokens(1, 12);
        int[] b = tokens(2, 5);
        float[][] expected = reference.forwardBatch(prompts(a, b));
        float[][] actual = split.forwardBatch(prompts(a, b));
        assertEquals(2, actual.length);
        assertArrayEquals("the rotated cache keeps sequence A's logits", expected[0], actual[0], TOL);
        assertArrayEquals("the rotated cache keeps sequence B's logits", expected[1], actual[1], TOL);

        float[][] expectedNext = reference.forwardBatch(decodePair(a[0], 12, b[0], 5));
        float[][] actualNext = split.forwardBatch(decodePair(a[0], 12, b[0], 5));
        assertEquals(2, actualNext.length);
        assertArrayEquals("the decode after the rotated prefill matches", expectedNext[0], actualNext[0], TOL);
        assertArrayEquals("the decode after the rotated prefill matches", expectedNext[1], actualNext[1], TOL);
    }

    // ------------------------------------------------------------------ helpers

    /** Every step's logits of the sequence: one prefill of 4, one boundary-crossing prefill of 4, two decodes. */
    private float[][] runSequence(GpuForwardBackend backend, int[] tokens) {
        int nVocab = spec.hparams().nVocab();
        int[][] steps = {{0, UBATCH}, {UBATCH, UBATCH}, {2 * UBATCH, 1}};
        float[][] out = new float[steps.length + 1][];
        for (int s = 0; s < steps.length; s++) {
            int pos0 = steps[s][0];
            int n = steps[s][1];
            int[] chunk = new int[n];
            for (int r = 0; r < n; r++) {
                chunk[r] = tokens[pos0 + r];
            }
            int[] rows = new int[n];
            for (int r = 0; r < n; r++) {
                rows[r] = r;
            }
            float[][] logits = backend.forwardRows(chunk, pos0, rows);
            float[] flat = new float[n * nVocab];
            for (int r = 0; r < n; r++) {
                System.arraycopy(logits[r], 0, flat, r * nVocab, nVocab);
            }
            out[s] = flat;
        }
        out[steps.length] = backend.decode(tokens[CONTIGUOUS_CTX - 1], CONTIGUOUS_CTX - 1);
        return out;
    }

    /** Tokens a sequence runs (deterministic ids inside the synthetic vocab). */
    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** A batch holding the whole prompt of every sequence back to back. */
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

    /** One decode row of each of two sequences, each at its own next position. */
    private static ForwardBatch decodePair(int tokenA, int posA, int tokenB, int posB) {
        ForwardBatch.Builder batch = ForwardBatch.builder(2);
        batch.add(tokenA, posA, 0, true);
        batch.add(tokenB, posB, 1, true);
        return batch.build();
    }

    private byte[] readRowBytes(GpuBuffer buffer, long rowBytes, int row) {
        byte[] bytes = new byte[(int) rowBytes];
        try (GpuEvent e = queue.enqueueReadBuffer(buffer, row * rowBytes, rowBytes, bytes)) {
            queue.finish();
        }
        return bytes;
    }

    /** One cache row as F32 values. */
    private float[] readF32Row(GpuBuffer buffer, long rowBytes, int row) {
        byte[] bytes = readRowBytes(buffer, rowBytes, row);
        float[] out = new float[bytes.length / 4];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out);
        return out;
    }

    /** The K and V row of every position of a contiguous cache, as the byte reference of one side's run. */
    private byte[][][] readWholeCacheRows(DeviceModelMemory memory, int il, int positions) {
        GpuKvLayer kv = memory.kv(il);
        byte[][][] out = new byte[2][positions][];
        for (int p = 0; p < positions; p++) {
            int row = kv.layout().rowOf(p);
            out[0][p] = readRowBytes(kv.k(), kv.layout().rowBytesK(), row);
            out[1][p] = readRowBytes(kv.v(), kv.layout().rowBytesV(), row);
        }
        return out;
    }

    /** The K and V row of every position of an F32 contiguous cache, as the float reference of one side's run. */
    private float[][][] readWholeF32Rows(DeviceModelMemory memory, int il, int positions) {
        GpuKvLayer kv = memory.kv(il);
        float[][][] out = new float[2][positions][];
        for (int p = 0; p < positions; p++) {
            int row = kv.layout().rowOf(p);
            out[0][p] = readF32Row(kv.k(), kv.layout().rowBytesK(), row);
            out[1][p] = readF32Row(kv.v(), kv.layout().rowBytesV(), row);
        }
        return out;
    }
}

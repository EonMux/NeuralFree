package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
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
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The prefill query split of a context-parallel split (vLLM's {@code prefill_context_parallel_size}) on the OpenCL
 * backend: over 1, every prefill's query rows cut into the DualChunkSwap chunk pairs and the ranks run the stack
 * through {@link PcpPrefillCoordinator}'s trunk — every rank projects, stores and attends its own packed rows while
 * the exchanges gather the queries, scatter the cache rows into the owning blocks and merge the attention
 * partitions. A prefill through the split keeps the single-device logits, the single-device KV rows in every rank's
 * block (the rows another rank projected land there through the scatter), the coordinator packs each rank's
 * chunk-pair windows into its step inputs, and a size that does not match the split's ranks — or a paged memory —
 * is refused by name. The GPU cases are skipped when no OpenCL device is available.
 */
public class CpPrefillSplitTest {

    private static final int UBATCH = 4;
    private static final int N_EMBD = 64;
    private static final int RANKS = 2;
    private static final int N_CTX = 10;
    private static final float TOL = 2e-3f;

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
    private DeviceModelMemory mainMemory;
    private GpuForwardContext mainContext;

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
        mem = MemoryConfig.defaults().toBuilder().ubatch(UBATCH)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), mem);
        scratch = GpuScratch.forModel(main, spec, UBATCH);
        ops = new GpuOps(queue, kernels, scratch, placeholders);
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        uploader = new GpuWeightUploader(main, queue, parser.getTensors(), store, "llama",
                GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(kernels)).build());
        uploader.uploadResident(t -> false);
        MemoryPlan mainPlan = MemoryPlanner.planForContext(spec, mem, N_CTX);
        mainMemory = DeviceModelMemory.allocate(main, queue, mainPlan, spec.hparams().mropeTextPositions(), null);
        mainContext = GpuForwardContext.builder(kernels, ops, uploader, spec).memory(mainMemory).build();
    }

    @After
    public void close() throws IOException {
        try {
            if (mainMemory != null) {
                mainMemory.close();
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

    private TensorParallelGroup newGroup(boolean prefillSplit) {
        TpAttentionPlan plan = TpAttentionPlan.context(spec, RANKS, uploader);
        TensorParallelGroup group = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same device")));
        if (prefillSplit) {
            group.prefillContextParallelSize(RANKS);
        }
        group.init(spec, UBATCH, kernels, queue, placeholders, null, plan, mem, N_CTX);
        group.upload(store, uploader);
        return group;
    }

    /** The single-device reference forward pass: the plain registry, no trunk runner. */
    private GpuForwardBackend wholeBackend() {
        return new GpuForwardBackend(mainContext, GpuFactoryCatalog.defaults(),
                GpuFactoryCatalog.headPreprocessors());
    }

    /** The split forward pass: the decorated registry binds the prefill query split, whose trunk runner interlaces
     * the ranks; the model's prefill driver dispatches every multi-token prefill through the coordinator. */
    private GpuForwardBackend pcpBackend(TensorParallelGroup group) {
        GpuBlockRegistry registry = GpuFactoryCatalog.defaults().withMixerDecorator(group.attentionDecorator());
        GpuForwardBackend backend = new GpuForwardBackend(mainContext, registry,
                GpuFactoryCatalog.headPreprocessors());
        backend.trunkRunner(group.pcpTrunkRunner(mainContext, backend.layerExecutor()));
        return backend;
    }

    private void write(GpuBuffer buffer, float[] values) {
        try (GpuEvent e = queue.enqueueWriteBuffer(buffer, 0, values.length * (long) Float.BYTES, values)) {
            queue.finish();
        }
    }

    private float[] read(GpuBuffer buffer, int floats) {
        float[] out = new float[floats];
        try (GpuEvent e = queue.enqueueReadBuffer(buffer, 0, floats * (long) Float.BYTES, out)) {
            queue.finish();
        }
        return out;
    }

    /** One cache row as F32 values. */
    private float[] readRow(GpuBuffer buffer, long rowBytes, int row) {
        byte[] bytes = new byte[(int) rowBytes];
        try (GpuEvent e = queue.enqueueReadBuffer(buffer, row * rowBytes, rowBytes, bytes)) {
            queue.finish();
        }
        float[] out = new float[bytes.length / 4];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out);
        return out;
    }

    /** Every step's logits of the sequence: one prefill of 4, one boundary-crossing prefill of 4, two decodes. */
    private float[][] runSequence(GpuForwardBackend backend, int[] tokens) {
        int nVocab = spec.hparams().nVocab();
        int[][] steps = {{0, UBATCH}, {UBATCH, UBATCH}, {2 * UBATCH, 1}};
        float[][] out = new float[steps.length + 1][];
        for (int s = 0; s < steps.length; s++) {
            int pos0 = steps[s][0];
            int n = steps[s][1];
            int[] chunk = Arrays.copyOfRange(tokens, pos0, pos0 + n);
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
        out[steps.length] = backend.decode(tokens[N_CTX - 1], N_CTX - 1);
        return out;
    }

    /** The K and V row of every position of a contiguous cache, as the host reference of one side's run. */
    private float[][][] readWholeKv(DeviceModelMemory memory, int il, int positions) {
        GpuKvLayer kv = memory.kv(il);
        Assume.assumeTrue("F32 KV rows", kv.layout().dtypeK() == KvDType.F32
                && kv.layout().dtypeV() == KvDType.F32);
        float[][][] out = new float[2][positions][];
        for (int p = 0; p < positions; p++) {
            int row = kv.layout().rowOf(p);
            out[0][p] = readRow(kv.k(), kv.layout().rowBytesK(), row);
            out[1][p] = readRow(kv.v(), kv.layout().rowBytesV(), row);
        }
        return out;
    }

    @Test
    public void thePrefillQuerySplitKeepsTheSingleDeviceParity() throws IOException {
        int[] tokens = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
        int il = spec.layers().get(0).il();
        TensorParallelGroup group = null;
        GpuForwardBackend whole = wholeBackend();
        try {
            float[][] wholeLogits = runSequence(whole, tokens);
            float[][][] wholeKv = readWholeKv(mainMemory, il, N_CTX);
            mainMemory.reset();

            group = newGroup(true);
            GpuForwardBackend split = pcpBackend(group);
            try {
                float[][] splitLogits = runSequence(split, tokens);
                for (int s = 0; s < wholeLogits.length; s++) {
                    assertArrayEquals("the prefill query split keeps the single-device logits at step " + s,
                            wholeLogits[s], splitLogits[s], TOL);
                }
                for (int rank = 0; rank < RANKS; rank++) {
                    GpuKvLayer kv = group.rankMemory(rank).kv(il);
                    int from = group.contextBlockStart(rank);
                    int to = group.contextBlockEnd(rank);
                    for (int p = from; p < to; p++) {
                        int row = kv.layout().rowOf(p);
                        assertArrayEquals("rank " + rank + " K row of position " + p + " (through the query"
                                        + " gather, the cache scatter and the merged attention)",
                                wholeKv[0][p], readRow(kv.k(), kv.layout().rowBytesK(), row), TOL);
                        assertArrayEquals("rank " + rank + " V row of position " + p,
                                wholeKv[1][p], readRow(kv.v(), kv.layout().rowBytesV(), row), TOL);
                    }
                }
            } finally {
                split.close();
            }
        } finally {
            whole.close();
            if (group != null) {
                group.close();
            }
        }
    }

    @Test
    public void theCoordinatorPacksTheChunkPairWindows() throws IOException {
        TensorParallelGroup group = null;
        GpuForwardBackend split = null;
        try {
            group = newGroup(true);
            split = pcpBackend(group);
            // the first prefill fills positions [0, 4); the boundary prefill at [4, 8) cuts into chunk size
            // ceil(4 / 4) = 1: rank 0 packs rows 4 and 7 (its chunk pair), rank 1 packs rows 5 and 6 — the
            // packed first position and count land in its step inputs
            split.forwardRows(new int[]{3, 1, 4, 1}, 0, new int[]{UBATCH - 1});
            split.forwardRows(new int[]{5, 9, 2, 6}, UBATCH, new int[]{UBATCH - 1});
            assertEquals("rank 0's packed window starts at its first chunk row", UBATCH,
                    group.rankMemory(0).stepInputs().pos0());
            assertEquals("rank 0 runs its chunk pair", 2, group.rankMemory(0).stepInputs().nTokens());
            assertEquals("rank 1's packed window starts at its first chunk row", UBATCH + 1,
                    group.rankMemory(1).stepInputs().pos0());
            assertEquals("rank 1 runs its chunk pair", 2, group.rankMemory(1).stepInputs().nTokens());
        } finally {
            if (split != null) {
                split.close();
            }
            if (group != null) {
                group.close();
            }
        }
    }

    @Test
    public void aPrefillSplitSizeBeyondTheRanksIsRefusedByName() {
        TpAttentionPlan plan = TpAttentionPlan.context(spec, RANKS, uploader);
        TensorParallelGroup group = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same device")));
        group.prefillContextParallelSize(RANKS + 1);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> group.init(spec, UBATCH, kernels, queue, placeholders, null, plan, mem, N_CTX));
        assertTrue(e.getMessage(), e.getMessage().contains("prefill context-parallel size"));
    }

}

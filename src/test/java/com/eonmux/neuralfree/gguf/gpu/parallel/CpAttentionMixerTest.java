package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.attn.GpuAttentionMixer;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * End-to-end parity of context-parallel attention on the OpenCL backend: a synthetic llama's attention layer runs
 * once as the whole-device {@link GpuAttentionMixer} and once as the {@link CpAttentionMixer} of a two-rank group
 * whose peer runs on a second backend of the same device. The ranks divide the context positions between them, so a
 * prefill that crosses the block boundary and the decodes after it merge two partial softmaxes into the single-device
 * result, and a decode before the peer's block merges an empty partition. The GPU cases are skipped when no OpenCL
 * device is available.
 */
public class CpAttentionMixerTest {

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
    private TpAttentionPlan plan;
    private TensorParallelGroup group;
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
        mem = MemoryConfig.defaults().toBuilder().ubatch(UBATCH).build();
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
        plan = TpAttentionPlan.context(spec, RANKS, uploader);
        group = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same device")));
        group.init(spec, UBATCH, kernels, queue, placeholders, null, plan, mem, N_CTX);
        group.upload(store, uploader);
    }

    @After
    public void close() throws IOException {
        try {
            if (group != null) {
                group.close();
            }
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

    private static float[] input(int step, int nTokens) {
        float[] values = new float[nTokens * N_EMBD];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) Math.sin(i * 0.37 + step * 7.0);
        }
        return values;
    }

    /** The step sequence of the parity run: a prefill crossing rank 1's block start, then decodes after it. */
    private static int[][] steps() {
        return new int[][]{{0, UBATCH}, {UBATCH, 1}, {UBATCH + 1, 3}, {2 * UBATCH, 1}, {2 * UBATCH + 1, 1}};
    }

    @Test
    public void theContextParallelAttentionMatchesTheWholeDeviceAttention() {
        LayerSpec layer = spec.layers().get(0);
        GpuAttentionMixer whole = new GpuAttentionMixer();
        whole.bind(mainContext, layer);
        GpuMixer split = group.attentionDecorator().mixer(layer, null);
        assertNotNull("the context plan replaces the device mixer", split);
        split.bind(mainContext, layer);
        try (GpuBuffer in = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true);
             GpuBuffer outWhole = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true);
             GpuBuffer outSplit = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true)) {
            ScratchSlot inSlot = ScratchSlot.of("test.cp.in", in, 0, (long) UBATCH * N_EMBD);
            ScratchSlot wholeSlot = ScratchSlot.of("test.cp.whole", outWhole, 0, (long) UBATCH * N_EMBD);
            ScratchSlot splitSlot = ScratchSlot.of("test.cp.split", outSplit, 0, (long) UBATCH * N_EMBD);
            /* The decode at position 4 falls before rank 1's block, so its partition is empty; the prefill at 5
             * then fills rank 1's block and the decodes after it merge both partitions. */
            int[][] sequence = steps();
            for (int step = 0; step < sequence.length; step++) {
                int pos0 = sequence[step][0];
                int n = sequence[step][1];
                write(in, input(step, n));
                mainMemory.stepInputs().upload(pos0, n);
                if (n == 1) {
                    whole.decode(pos0, inSlot, wholeSlot);
                    split.decode(pos0, inSlot, splitSlot);
                } else {
                    whole.prefill(pos0, n, inSlot, wholeSlot);
                    split.prefill(pos0, n, inSlot, splitSlot);
                }
                assertArrayEquals("step " + step + " (" + n + " token(s) at " + pos0 + "): the context-parallel"
                                + " attention matches the whole-device branch",
                        read(outWhole, n * N_EMBD), read(outSplit, n * N_EMBD), TOL);
            }
        } finally {
            whole.close();
            split.close();
        }
    }

    @Test
    public void aRankCacheHoldsOnlyItsContextBlock() {
        LayerSpec layer = spec.layers().get(0);
        GpuAttentionMixer whole = new GpuAttentionMixer();
        whole.bind(mainContext, layer);
        GpuMixer split = group.attentionDecorator().mixer(layer, null);
        split.bind(mainContext, layer);
        try (GpuBuffer in = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true);
             GpuBuffer outWhole = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true);
             GpuBuffer outSplit = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true)) {
            ScratchSlot inSlot = ScratchSlot.of("test.cp.in", in, 0, (long) UBATCH * N_EMBD);
            ScratchSlot wholeSlot = ScratchSlot.of("test.cp.whole", outWhole, 0, (long) UBATCH * N_EMBD);
            ScratchSlot splitSlot = ScratchSlot.of("test.cp.split", outSplit, 0, (long) UBATCH * N_EMBD);
            int[][] sequence = steps();
            for (int step = 0; step < sequence.length; step++) {
                int pos0 = sequence[step][0];
                int n = sequence[step][1];
                write(in, input(step, n));
                mainMemory.stepInputs().upload(pos0, n);
                if (n == 1) {
                    whole.decode(pos0, inSlot, wholeSlot);
                    split.decode(pos0, inSlot, splitSlot);
                } else {
                    whole.prefill(pos0, n, inSlot, wholeSlot);
                    split.prefill(pos0, n, inSlot, splitSlot);
                }
            }
            DeviceModelMemory m0 = group.rankMemories().get(0);
            DeviceModelMemory m1 = group.rankMemories().get(1);
            int rowsPerRank = m0.kv(0).layout().rows();
            int rowsMain = mainMemory.kv(0).layout().rows();
            assertTrue("a rank of " + RANKS + " holds " + rowsPerRank + " of the " + rowsMain + " context rows",
                    rowsPerRank * RANKS >= rowsMain && rowsPerRank < rowsMain);
            assertTrue("the rank key cache of " + m1.kv(0).k().sizeInBytes() + " bytes stays below the whole"
                            + " device's " + mainMemory.kv(0).k().sizeInBytes(),
                    m1.kv(0).k().sizeInBytes() < mainMemory.kv(0).k().sizeInBytes());
            assertTrue("the rank value cache of " + m1.kv(0).v().sizeInBytes() + " bytes stays below the whole"
                            + " device's " + mainMemory.kv(0).v().sizeInBytes(),
                    m1.kv(0).v().sizeInBytes() < mainMemory.kv(0).v().sizeInBytes());
            /* The caches hold the raw element rows, so reading both as floats compares their bytes. */
            int rowFloats = (int) (mainMemory.kv(0).layout().rowBytesV() / Float.BYTES);
            float[] wholeV = read(mainMemory.kv(0).v(), rowsMain * rowFloats);
            float[] rank0V = read(m0.kv(0).v(), rowsPerRank * rowFloats);
            float[] rank1V = read(m1.kv(0).v(), rowsPerRank * rowFloats);
            int kRowFloats = (int) (mainMemory.kv(0).layout().rowBytesK() / Float.BYTES);
            float[] wholeK = read(mainMemory.kv(0).k(), rowsMain * kRowFloats);
            float[] rank0K = read(m0.kv(0).k(), rowsPerRank * kRowFloats);
            float[] rank1K = read(m1.kv(0).k(), rowsPerRank * kRowFloats);
            for (int r = 0; r < rowsPerRank; r++) {
                assertArrayEquals("rank 0 key row " + r + " is the whole device's row " + r,
                        copyRange(wholeK, r * kRowFloats, kRowFloats),
                        copyRange(rank0K, r * kRowFloats, kRowFloats), 0.0f);
                assertArrayEquals("rank 0 value row " + r + " is the whole device's row " + r,
                        copyRange(wholeV, r * rowFloats, rowFloats),
                        copyRange(rank0V, r * rowFloats, rowFloats), 0.0f);
                assertArrayEquals("rank 1 key row " + r + " is the whole device's row " + (r + rowsPerRank),
                        copyRange(wholeK, (r + rowsPerRank) * kRowFloats, kRowFloats),
                        copyRange(rank1K, r * kRowFloats, kRowFloats), 0.0f);
                assertArrayEquals("rank 1 value row " + r + " is the whole device's row " + (r + rowsPerRank),
                        copyRange(wholeV, (r + rowsPerRank) * rowFloats, rowFloats),
                        copyRange(rank1V, r * rowFloats, rowFloats), 0.0f);
            }
            assertArrayEquals("rank 0's position mirror lists its own block's positions",
                    new int[]{0, 1, 2, 3, 4}, readInts(m0.stepInputs().kvPos(0), rowsPerRank));
            assertArrayEquals("rank 1's position mirror lists its own block's positions",
                    new int[]{5, 6, 7, 8, 9}, readInts(m1.stepInputs().kvPos(0), rowsPerRank));
        } finally {
            whole.close();
            split.close();
        }
    }

    private static float[] copyRange(float[] values, int from, int length) {
        float[] out = new float[length];
        System.arraycopy(values, from, out, 0, length);
        return out;
    }

    private int[] readInts(GpuBuffer buffer, int ints) {
        float[] raw = new float[ints];
        try (GpuEvent e = queue.enqueueReadBuffer(buffer, 0, ints * (long) Integer.BYTES, raw)) {
            queue.finish();
        }
        int[] out = new int[ints];
        for (int i = 0; i < ints; i++) {
            out[i] = Float.floatToRawIntBits(raw[i]);
        }
        return out;
    }

    @Test
    public void aLayerOutsideThePlanKeepsItsDeviceMixer() {
        LayerSpec unsplit = LayerSpec.builder(spec.layers().get(0)).stack("alt").build();
        GpuMixer marker = new GpuAttentionMixer();
        assertSame("a layer outside the plan keeps the whole-device mixer",
                marker, group.attentionDecorator().mixer(unsplit, marker));
        GpuMixer split = group.attentionDecorator().mixer(spec.layers().get(0), marker);
        split.close();
    }

    @Test
    public void aPagedBatchRunsThroughTheMixer() {
        GpuMixer split = group.attentionDecorator().mixer(spec.layers().get(0), null);
        try {
            split.bind(mainContext, spec.layers().get(0));
            assertNull("a paged multi-sequence batch carries through the context-parallel merge",
                    split.batchRefusal());
            assertNull("a paged cache write carries through the context-parallel merge", split.kvRefusal());
        } finally {
            split.close();
        }
    }
}

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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

/**
 * End-to-end parity of a tensor-parallel attention layer on the OpenCL backend: a synthetic llama's attention layer
 * runs once as the whole-device {@link GpuAttentionMixer} and once as the {@link TpAttentionMixer} of a two-rank group
 * whose peer runs on a second backend of the same device, with per-rank shard KV caches. Two contiguous steps match
 * the whole-device branch within the layer-split tolerance. The GPU cases are skipped when no OpenCL device is
 * available.
 */
public class TpAttentionMixerTest {

    private static final int UBATCH = 3;
    private static final int N_EMBD = 64;
    private static final int RANKS = 2;
    private static final int N_CTX = 32;
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
        plan = TpAttentionPlan.of(spec, RANKS, uploader);
        group = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same device")));
        group.init(spec, UBATCH, kernels, queue, placeholders, null, plan, mem, N_CTX);
        group.upload(store, uploader);
    }

    @After
    public void close() throws IOException {
        try {
            if (mainContext != null) {
                mainContext = null;
            }
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

    private static float[] input(int step) {
        float[] values = new float[UBATCH * N_EMBD];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) Math.sin(i * 0.37 + step * 7.0);
        }
        return values;
    }

    @Test
    public void theTensorParallelAttentionMatchesTheWholeDeviceAttention() {
        LayerSpec layer = spec.layers().get(0);
        GpuAttentionMixer whole = new GpuAttentionMixer();
        whole.bind(mainContext, layer);
        GpuMixer split = group.attentionDecorator().mixer(layer, null);
        assertNotNull("the split attention replaces the device mixer", split);
        split.bind(mainContext, layer);
        try (GpuBuffer in = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true);
             GpuBuffer outWhole = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true);
             GpuBuffer outSplit = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true)) {
            ScratchSlot inSlot = ScratchSlot.of("test.attn.in", in, 0, (long) UBATCH * N_EMBD);
            ScratchSlot wholeSlot = ScratchSlot.of("test.attn.whole", outWhole, 0, (long) UBATCH * N_EMBD);
            ScratchSlot splitSlot = ScratchSlot.of("test.attn.split", outSplit, 0, (long) UBATCH * N_EMBD);
            for (int step = 0; step < 2; step++) {
                int pos0 = step * UBATCH;
                write(in, input(step));
                mainMemory.stepInputs().upload(pos0, UBATCH);
                whole.prefill(pos0, UBATCH, inSlot, wholeSlot);
                split.prefill(pos0, UBATCH, inSlot, splitSlot);
                assertArrayEquals("step " + step + ": the tensor-parallel attention matches the whole-device branch",
                        read(outWhole, UBATCH * N_EMBD), read(outSplit, UBATCH * N_EMBD), TOL);
            }
        } finally {
            whole.close();
            split.close();
        }
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
    public void thePerRankSessionsRoundTrip() throws IOException {
        Path session = Files.createTempFile("tp-session", ".nfs");
        try {
            mainMemory.sequence().append(new int[]{11, 12, 13}, 0, 3);
            group.appendStep(new int[]{11, 12, 13}, 0, 3);
            mainMemory.stepInputs().upload(0, 3);
            group.uploadStepInputs(0, 3);
            byte[] mainK = fill(mainMemory.kv(0).k(), queue, (byte) 0x21);
            byte[] rank0K = fill(group.rankMemories().get(0).kv(0).k(), queue, (byte) 0x31);
            byte[] rank1K = fill(group.rankMemories().get(1).kv(0).k(), peer.computeQueue(), (byte) 0x41);

            group.saveSession(mainMemory, session);
            fill(mainMemory.kv(0).k(), queue, (byte) 0);
            fill(group.rankMemories().get(0).kv(0).k(), queue, (byte) 0);
            fill(group.rankMemories().get(1).kv(0).k(), peer.computeQueue(), (byte) 0);
            group.loadSession(mainMemory, session);

            assertEquals(3, mainMemory.sequence().nPast());
            for (int rank = 0; rank < RANKS; rank++) {
                assertEquals("rank " + rank + " sequence", 3, group.rankMemories().get(rank).sequence().nPast());
            }
            assertArrayEquals("rank 0's KV rows survive the round trip", rank0K,
                    readBytes(group.rankMemories().get(0).kv(0).k(), queue));
            assertArrayEquals("rank 1's KV rows survive the round trip", rank1K,
                    readBytes(group.rankMemories().get(1).kv(0).k(), peer.computeQueue()));
            assertArrayEquals("the main memory's KV rows survive the round trip", mainK,
                    readBytes(mainMemory.kv(0).k(), queue));
        } finally {
            Files.deleteIfExists(session);
        }
    }

    private static byte[] fill(com.eonmux.neuralfree.acceleration.backend.GpuBuffer buffer, ComputeQueue q,
                               byte value) {
        byte[] data = new byte[(int) buffer.sizeInBytes()];
        java.util.Arrays.fill(data, value);
        ByteBuffer staged = ByteBuffer.allocateDirect(data.length).order(ByteOrder.LITTLE_ENDIAN);
        staged.put(data).flip();
        try (GpuEvent e = q.enqueueWriteBuffer(buffer, 0, data.length, staged)) {
            q.finish();
        }
        return data;
    }

    private static byte[] readBytes(com.eonmux.neuralfree.acceleration.backend.GpuBuffer buffer, ComputeQueue q) {
        byte[] data = new byte[(int) buffer.sizeInBytes()];
        try (GpuEvent e = q.enqueueReadBuffer(buffer, 0, data.length, data)) {
            q.finish();
        }
        return data;
    }

    @Test
    public void theGroupExposesItsRankState() {
        assertEquals(RANKS, group.rankCount());
        assertEquals(RANKS - 1, group.peerCount());
        assertSame(plan, group.attentionPlan());
        assertNotNull(group.rankContext(0));
        assertNotNull(group.rankContext(1));
        assertNotNull(group.inputSlot(1));
        assertNotNull(group.partialSlot(1));
        assertNotNull(group.allReduceTemporary());
    }
}

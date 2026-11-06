package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.act.GpuDenseFfn;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * End-to-end parity of a feed-forward row split on the OpenCL backend: a synthetic llama's FFN of one layer runs
 * once as the whole-device {@link GpuDenseFfn} and once as the split block of a two-rank group whose peer rank runs
 * on a second backend of the same device, and both steps match within the layer-split tolerance. The group refuses
 * a second initialization and releases its rank infrastructure on close. The GPU cases are skipped when no OpenCL
 * device is available.
 */
public class FfnRowSplitBlockTest {

    private static final int UBATCH = 4;
    private static final int N_EMBD = 64;
    private static final int N_FF = 128;
    private static final int RANKS = 2;
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
    private FfnRowSplitPlan plan;
    private TensorParallelGroup group;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2).nFf(N_FF).build())
                .writeTempFile();
        parser = new GGUFFileParser(file.toString());
        hw = new HWAcceleration();
        Assume.assumeFalse("no OpenCL device", hw.getCLDevices().isEmpty());
        main = new OclComputeBackend(hw.getCLDevices().get(0));
        peer = new OclComputeBackend(hw.getCLDevices().get(0));
        kernels = LlmKernels.open(main);
        queue = main.computeQueue();
        placeholders = GpuPlaceholders.allocate(main, queue);
        scratch = GpuScratch.builder(main, UBATCH)
                .slot(ScratchSlots.NORMED, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.BRANCH, 2L * UBATCH * N_EMBD)
                .slot(ScratchSlots.FFN_GATE_UP, 2L * UBATCH * N_FF)
                .build();
        ops = new GpuOps(queue, kernels, scratch, placeholders);
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        uploader = new GpuWeightUploader(main, queue, parser.getTensors(), store, "llama",
                GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(kernels)).build());
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser),
                MemoryConfig.defaults().toBuilder().ubatch(UBATCH).build());
        plan = FfnRowSplitPlan.of(spec, RANKS, uploader);
        group = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same device")));
        group.init(spec, UBATCH, kernels, queue, placeholders, plan);
        group.upload(store, uploader);
    }

    @After
    public void close() throws IOException {
        try {
            if (group != null) {
                group.close();
            }
            uploader.close();
            ops.close();
            scratch.close();
            placeholders.close();
            store.close();
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

    private float[] read(GpuBuffer buffer, int floats) {
        float[] out = new float[floats];
        try (GpuEvent e = queue.enqueueReadBuffer(buffer, 0, floats * (long) Float.BYTES, out)) {
            queue.finish();
        }
        return out;
    }

    private void write(GpuBuffer buffer, float... values) {
        try (GpuEvent e = queue.enqueueWriteBuffer(buffer, 0, values.length * (long) Float.BYTES, values)) {
            queue.finish();
        }
    }

    /** Two inputs of values in [-1, 1); the model order fixes every weight. */
    private static float[][] inputs() {
        float[][] inputs = new float[2][UBATCH * N_EMBD];
        for (int step = 0; step < inputs.length; step++) {
            for (int i = 0; i < inputs[step].length; i++) {
                inputs[step][i] = (float) Math.sin(i * 0.7 + step * 13.0);
            }
        }
        return inputs;
    }

    @Test
    public void theSplitBlockMatchesTheWholeDeviceFfn() {
        LayerSpec layer = spec.layers().get(0);
        uploader.uploadResident(t -> false);
        GpuForwardContext ctx = GpuForwardContext.builder(kernels, ops, uploader, spec).build();
        GpuDenseFfn full = new GpuDenseFfn();
        full.bind(ctx, layer);
        GpuFfnBlock split = group.decorator().ffn(layer, null);
        assertNotNull("a split layer gets a split block", split);
        split.bind(ctx, layer);
        try (GpuBuffer in = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true);
             GpuBuffer outFull = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true);
             GpuBuffer outSplit = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true)) {
            ScratchSlot inSlot = ScratchSlot.of("test.ffn.in", in, 0, (long) UBATCH * N_EMBD);
            ScratchSlot fullSlot = ScratchSlot.of("test.ffn.full", outFull, 0, (long) UBATCH * N_EMBD);
            ScratchSlot splitSlot = ScratchSlot.of("test.ffn.split", outSplit, 0, (long) UBATCH * N_EMBD);
            float[][] steps = inputs();
            for (float[] step : steps) {
                write(in, step);
                full.forward(UBATCH, inSlot, fullSlot);
                split.forward(UBATCH, inSlot, splitSlot);
                float[] reference = read(outFull, UBATCH * N_EMBD);
                float[] got = read(outSplit, UBATCH * N_EMBD);
                assertArrayEquals("the split FFN matches the whole-device FFN", reference, got, TOL);
            }
        } finally {
            full.close();
            split.close();
        }
    }

    @Test
    public void theProportionalSplitMatchesTheWholeDeviceFfn() throws IOException {
        FfnRowSplitPlan uneven = FfnRowSplitPlan.of(spec, RANKS, uploader, new double[]{3, 1});
        TensorParallelGroup proportional = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same device")));
        try {
            proportional.init(spec, UBATCH, kernels, queue, placeholders, uneven);
            proportional.upload(store, uploader);
            LayerSpec layer = spec.layers().get(0);
            uploader.uploadResident(t -> false);
            GpuForwardContext ctx = GpuForwardContext.builder(kernels, ops, uploader, spec).build();
            GpuDenseFfn full = new GpuDenseFfn();
            full.bind(ctx, layer);
            GpuFfnBlock split = proportional.decorator().ffn(layer, null);
            split.bind(ctx, layer);
            try (GpuBuffer in = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true);
                 GpuBuffer outFull = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true);
                 GpuBuffer outSplit = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true)) {
                ScratchSlot inSlot = ScratchSlot.of("test.uneven.in", in, 0, (long) UBATCH * N_EMBD);
                ScratchSlot fullSlot = ScratchSlot.of("test.uneven.full", outFull, 0, (long) UBATCH * N_EMBD);
                ScratchSlot splitSlot = ScratchSlot.of("test.uneven.split", outSplit, 0, (long) UBATCH * N_EMBD);
                float[][] steps = inputs();
                for (float[] step : steps) {
                    write(in, step);
                    full.forward(UBATCH, inSlot, fullSlot);
                    split.forward(UBATCH, inSlot, splitSlot);
                    assertArrayEquals("the 3:1 split FFN matches the whole-device FFN",
                            read(outFull, UBATCH * N_EMBD), read(outSplit, UBATCH * N_EMBD), TOL);
                }
            } finally {
                full.close();
                split.close();
            }
        } finally {
            proportional.close();
        }
    }

    @Test
    public void aLayerOutsideThePlanKeepsItsDeviceBlock() {
        LayerSpec unsplit = LayerSpec.builder(spec.layers().get(1)).stack("alt").build();
        GpuFfnBlock deviceBlock = group.decorator().ffn(unsplit, PASSTHROUGH);
        assertSame("the trunk layer outside the plan keeps the whole-device block", PASSTHROUGH, deviceBlock);
        GpuFfnBlock splitBlock = group.decorator().ffn(spec.layers().get(0), PASSTHROUGH);
        assertNotSame(PASSTHROUGH, splitBlock);
        splitBlock.close();
    }

    /** A marker block whose identity the decorator assertions compare against. */
    private static final GpuFfnBlock PASSTHROUGH = new GpuFfnBlock() {
        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            throw new UnsupportedOperationException("not bound in this test");
        }

        @Override
        public void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
            throw new UnsupportedOperationException("not run in this test");
        }
    };

    @Test
    public void theGroupRefusesASecondInitialization() {
        assertThrows(IllegalStateException.class,
                () -> group.init(spec, UBATCH, kernels, queue, placeholders, plan));
    }

    @Test
    public void theGroupRefusesAPlanOverADifferentRankCount() throws IOException {
        FfnRowSplitPlan four = FfnRowSplitPlan.of(spec, 4, uploader);
        TensorParallelGroup fresh = TensorParallelGroup.forPeers(main,
                Collections.singletonList(new GpuDevices.Selected(peer, "peer")));
        try {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> fresh.init(spec, UBATCH, kernels, queue, placeholders, four));
            assertTrue(e.getMessage(), e.getMessage().contains("4 rank"));
            assertTrue(e.getMessage(), e.getMessage().contains("2 (the model device and 1 peer"));
        } finally {
            fresh.close();
        }
    }

    @Test
    public void theGroupExposesItsRankState() {
        assertEquals(RANKS, group.rankCount());
        assertEquals(RANKS - 1, group.peerCount());
        assertSame(plan, group.plan());
        assertSame(spec, group.spec());
        assertNotNull(group.rankContext(0));
        assertNotNull(group.rankContext(1));
        assertSame(queue, group.rankQueue(0));
        assertTrue(group.allReduceTemporary() != null);
        List<FfnRowSplitPlan.Layer> layers = plan.layers();
        assertEquals(2, layers.size());
        for (int rank = 1; rank < RANKS; rank++) {
            assertNotNull(group.inputSlot(rank));
            assertNotNull(group.partialSlot(rank));
        }
    }
}

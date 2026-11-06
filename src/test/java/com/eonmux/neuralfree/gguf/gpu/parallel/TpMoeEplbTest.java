package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.engine.EplbConfig;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.moe.GpuMoeFfn;
import com.eonmux.neuralfree.gguf.ops.moe.MoeExpertHeat;
import com.eonmux.neuralfree.gguf.ops.moe.RankedExpert;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
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
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * EPLB placement on the OpenCL backend: a synthetic MoE layer whose four experts pack as {@code {0, 2}} and
 * {@code {1, 3}} over two ranks produces the same block output as the whole-device {@link GpuMoeFfn}, each rank
 * holds its experts as separate whole-expert runs at their global ids, and the shard byte estimate equals the
 * uploaded runs. The GPU cases are skipped when no OpenCL device is available.
 */
public class TpMoeEplbTest {

    private static final int UBATCH = 3;
    private static final int RANKS = 2;
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
    private ModelSpec spec;
    private GpuWeightUploader referenceUploader;
    private GpuScratch referenceScratch;
    private GpuOps referenceOps;
    private GpuForwardContext referenceContext;
    private GpuWeightUploader splitUploader;
    private GpuScratch splitScratch;
    private GpuOps splitOps;
    private GpuForwardContext splitContext;
    private TensorParallelGroup group;
    private TpMoePlan plan;
    private MoeExpertHeat heat;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.qwen3moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(1).build())
                .writeTempFile();
        parser = new GGUFFileParser(file.toString());
        hw = new HWAcceleration();
        Assume.assumeFalse("no OpenCL device", hw.getCLDevices().isEmpty());
        main = new OclComputeBackend(hw.getCLDevices().get(0));
        peer = new OclComputeBackend(hw.getCLDevices().get(0));
        kernels = LlmKernels.open(main);
        queue = main.computeQueue();
        placeholders = GpuPlaceholders.allocate(main, queue);
        MemoryConfig mem = MemoryConfig.defaults().toBuilder().ubatch(UBATCH).build();
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), mem);
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);

        referenceScratch = GpuScratch.forModel(main, spec, UBATCH);
        referenceOps = new GpuOps(queue, kernels, referenceScratch, placeholders);
        referenceUploader = uploader();
        referenceUploader.uploadResident(t -> false);
        referenceContext = GpuForwardContext.builder(kernels, referenceOps, referenceUploader, spec).build();

        EplbPlacement placement = EplbPlacement.of(spec, RANKS, loads(100, 100, 1, 1));
        plan = TpMoePlan.of(spec, RANKS, referenceUploader, true, placement);
        splitScratch = GpuScratch.forModel(main, spec, UBATCH);
        splitOps = new GpuOps(queue, kernels, splitScratch, placeholders);
        splitUploader = uploader();
        splitUploader.uploadResident(t -> plan.splits(t.name()));
        group = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same device")));
        heat = new MoeExpertHeat();
        group.expertHeat(heat);
        group.init(spec, UBATCH, kernels, queue, placeholders, null, null, null, plan, null, 0);
        group.upload(store, splitUploader);
        splitContext = GpuForwardContext.builder(kernels, splitOps, splitUploader, spec).build();
    }

    private GpuWeightUploader uploader() {
        return new GpuWeightUploader(main, queue, parser.getTensors(), store, "qwen3moe",
                GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(kernels)).build());
    }

    private static List<RankedExpert> loads(int e0, int e1, int e2, int e3) {
        return Arrays.asList(new RankedExpert(0, 0, e0), new RankedExpert(0, 1, e1),
                new RankedExpert(0, 2, e2), new RankedExpert(0, 3, e3));
    }

    @After
    public void close() throws IOException {
        try {
            if (group != null) {
                group.close();
            }
            splitUploader.close();
            splitOps.close();
            splitScratch.close();
            referenceUploader.close();
            referenceOps.close();
            referenceScratch.close();
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

    @Test
    public void theBalancedRanksMatchTheWholeBlock() {
        LayerSpec layer = spec.layers().get(0);
        GpuMoeFfn whole = new GpuMoeFfn(layer.ffnKind(), layer);
        whole.bind(referenceContext, layer);
        GpuFfnBlock split = group.decorator().ffn(layer, null);
        assertNotNull("the split experts replace the device block", split);
        split.bind(splitContext, layer);
        int width = layer.residual().streamWidth();
        try (GpuBuffer in = main.allocateBuffer((long) UBATCH * width * Float.BYTES, true)) {
            ScratchSlot inSlot = ScratchSlot.of("test.moe.eplb.in", in, 0, (long) UBATCH * width);
            float[] values = new float[UBATCH * width];
            for (int i = 0; i < values.length; i++) {
                values[i] = (float) Math.cos(i * 0.11);
            }
            try (GpuEvent e = queue.enqueueWriteBuffer(in, 0L, (long) values.length * Float.BYTES, values)) {
                queue.finish();
            }
            ScratchSlot wholeOut = referenceScratch.slot(ScratchSlots.FFN_OUT);
            ScratchSlot splitOut = splitScratch.slot(ScratchSlots.FFN_OUT);
            whole.forward(UBATCH, inSlot, wholeOut);
            split.forward(UBATCH, inSlot, splitOut);
            assertArrayEquals("the load-balanced ranks match the whole-device block",
                    read(wholeOut.buffer(), UBATCH * width), read(splitOut.buffer(), UBATCH * width), TOL);
        } finally {
            whole.close();
            split.close();
        }
    }

    @Test
    public void rankZeroCountsTheRoutedRowsForTheNextPlacement() {
        LayerSpec layer = spec.layers().get(0);
        GpuFfnBlock split = group.decorator().ffn(layer, null);
        assertNotNull(split);
        split.bind(splitContext, layer);
        int width = layer.residual().streamWidth();
        try (GpuBuffer in = main.allocateBuffer((long) UBATCH * width * Float.BYTES, true)) {
            ScratchSlot inSlot = ScratchSlot.of("test.moe.eplb.heat", in, 0, (long) UBATCH * width);
            float[] values = new float[UBATCH * width];
            for (int i = 0; i < values.length; i++) {
                values[i] = (float) Math.sin(i * 0.07);
            }
            try (GpuEvent e = queue.enqueueWriteBuffer(in, 0L, (long) values.length * Float.BYTES, values)) {
                queue.finish();
            }
            split.forward(UBATCH, inSlot, splitScratch.slot(ScratchSlots.FFN_OUT));
        } finally {
            split.close();
        }
        long counted = 0;
        for (RankedExpert e : heat.snapshot()) {
            if (e.layer() == layer.weightLayer()) {
                counted += e.heat();
            }
        }
        assertEquals("every routed row of the step is counted once", (long) UBATCH * layer.moe().nUsed(),
                counted);
    }

    @Test
    public void thePlanPacksTheExpertsByTheirLoad() {
        TpMoePlan.Layer layer = plan.layers().get(0);
        assertTrue("the layer is balanced", layer.balanced());
        assertEquals("each rank holds two experts", 2, layer.expertCount());
        assertEquals("rank 0 holds experts 0 and 2 as two runs",
                Arrays.asList(new TpMoePlan.ExpertRun(0, 1), new TpMoePlan.ExpertRun(2, 1)),
                layer.expertRuns(0));
        assertEquals("rank 1 holds experts 1 and 3 as two runs",
                Arrays.asList(new TpMoePlan.ExpertRun(1, 1), new TpMoePlan.ExpertRun(3, 1)),
                layer.expertRuns(1));
    }

    @Test
    public void thePlanRefusesAPlacementWithoutExpertParallelOrOverOtherRanks() {
        EplbPlacement placement = EplbPlacement.of(spec, RANKS, loads(100, 100, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> TpMoePlan.of(spec, RANKS, referenceUploader, false, placement));
        EplbPlacement four = EplbPlacement.of(spec, 4, loads(100, 100, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> TpMoePlan.of(spec, RANKS, referenceUploader, true, four));
    }

    @Test
    public void aPlanWithoutAPlacementKeepsOneContiguousRun() {
        TpMoePlan contiguous = TpMoePlan.of(spec, RANKS, referenceUploader, true);
        TpMoePlan.Layer layer = contiguous.layers().get(0);
        assertFalse("no profile, no balanced layer", layer.balanced());
        assertEquals(Arrays.asList(new TpMoePlan.ExpertRun(0, 2)), layer.expertRuns(0));
        assertEquals(Arrays.asList(new TpMoePlan.ExpertRun(2, 2)), layer.expertRuns(1));
    }

    @Test
    public void theBanksUploadTheirSlotsAtTheGlobalPhysicalIds() throws IOException {
        TpMoePlan.Layer layer = plan.layers().get(0);
        String down = TensorNames.weight(TensorNames.FFN_DOWN_EXPS);
        GpuWeightUploader.TensorPlan p = referenceUploader.plan(layer.tensor(down));
        int cols = (int) p.ne()[0];
        int rowsPerExpert = (int) p.ne()[1];
        int slotRows = rowsPerExpert * cols;
        TpMoeUploader uploader = new TpMoeUploader(plan, referenceUploader, store);
        try {
            Map<String, GpuWeight> shard = uploader.uploadRank(1, main, queue);
            GpuWeight w = shard.get(layer.tensor(down));
            assertNotNull("rank 1 uploads the down projections", w);
            assertArrayEquals("the bank keeps the global physical shape",
                    new long[]{cols, rowsPerExpert, layer.physicalExperts()}, w.ne());
            int firstSlot = 1 * layer.slotsPerRank();
            long bankBytes = 0;
            for (int i = 0; i < w.shardCount(); i++) {
                assertEquals("each chunk holds whole slots", 0, w.shardRows(i) % rowsPerExpert);
                assertTrue("the chunk covers slots of rank 1 only",
                        w.shardRowStart(i) >= (long) firstSlot * rowsPerExpert
                                && w.shardRowStart(i) + w.shardRows(i)
                                <= (long) (firstSlot + layer.slotsPerRank()) * rowsPerExpert);
                bankBytes += w.shard(i).sizeInBytes();
            }
            assertEquals("the chunks cover the rank's whole bank",
                    (long) layer.slotsPerRank() * slotRows * Float.BYTES, bankBytes);
            for (int i = 0; i < w.shardCount(); i++) {
                int first = (int) (w.shardRowStart(i) / rowsPerExpert);
                int slots = (int) (w.shardRows(i) / rowsPerExpert);
                float[] chunk = read(w.shard(i), slots * slotRows);
                for (int j = 0; j < slots; j++) {
                    int slot = first + j;
                    byte[] expected = new byte[slotRows * Float.BYTES];
                    store.readTensor(p.info().offset() + (long) layer.logicalOfSlot(slot) * slotRows * Float.BYTES,
                            slotRows * Float.BYTES, expected);
                    assertArrayEquals("physical slot " + slot + " holds expert " + layer.logicalOfSlot(slot),
                            floats(expected), Arrays.copyOfRange(chunk, j * slotRows, (j + 1) * slotRows), TOL);
                }
            }
            long uploaded = 0;
            for (GpuWeight weight : shard.values()) {
                for (int i = 0; i < weight.shardCount(); i++) {
                    uploaded += weight.shard(i).sizeInBytes();
                }
            }
            long estimated = ShardBytes.moe(plan, 1, referenceUploader)
                    + ShardBytes.moeSupport(plan, 1, referenceUploader);
            assertEquals("the bank byte estimate covers the uploaded shards and support tensors", estimated,
                    uploaded);
        } finally {
            uploader.close();
        }
    }

    @Test
    public void aRedundantPlacementReplicatesItsExpertsAndStillMatchesTheWholeBlock() throws IOException {
        // vLLM: {100,100,1,1} with two redundant slots first replicates e0 and e1, then packs the six slots
        // heaviest first, which spreads them as {0,0,2} and {1,1,3} over the two ranks.
        EplbPlacement placement = EplbPlacement.of(spec, RANKS, loads(100, 100, 1, 1),
                EplbPlacement.Options.of(EplbConfig.Policy.DEFAULT, 2, 0, 1));
        TpMoePlan redundant = TpMoePlan.of(spec, RANKS, referenceUploader, true, placement);
        TpMoePlan.Layer layer = redundant.layers().get(0);
        assertEquals(6, layer.physicalExperts());
        assertEquals("each rank holds three physical slots", 3, layer.expertCount());
        assertArrayEquals(new int[]{0, 0, 2, 1, 1, 3}, layer.phyToLog());
        assertArrayEquals(new int[]{2, 2, 1, 1}, layer.replicaCount());
        assertEquals(2, layer.maxReplicas());
        assertEquals("each slot uploads as its own run, so a replicated expert appears once per slot",
                Arrays.asList(new TpMoePlan.ExpertRun(0, 1), new TpMoePlan.ExpertRun(0, 1),
                        new TpMoePlan.ExpertRun(2, 1)), layer.expertRuns(0));
        assertEquals(Arrays.asList(new TpMoePlan.ExpertRun(1, 1), new TpMoePlan.ExpertRun(1, 1),
                new TpMoePlan.ExpertRun(3, 1)), layer.expertRuns(1));

        GpuScratch splitScratch = GpuScratch.forModel(main, spec, UBATCH);
        GpuOps splitOps = new GpuOps(queue, kernels, splitScratch, placeholders);
        GpuWeightUploader splitUploader = uploader();
        TensorParallelGroup redundantGroup = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same device")));
        try {
            splitUploader.uploadResident(t -> redundant.splits(t.name()));
            redundantGroup.init(spec, UBATCH, kernels, queue, placeholders, null, null, null, redundant, null, 0);
            redundantGroup.upload(store, splitUploader);
            GpuForwardContext splitContext = GpuForwardContext.builder(kernels, splitOps, splitUploader, spec)
                    .build();
            GpuMoeFfn whole = new GpuMoeFfn(spec.layers().get(0).ffnKind(), spec.layers().get(0));
            whole.bind(referenceContext, spec.layers().get(0));
            GpuFfnBlock split = redundantGroup.decorator().ffn(spec.layers().get(0), null);
            assertNotNull("the redundant split replaces the device block", split);
            split.bind(splitContext, spec.layers().get(0));
            int width = spec.layers().get(0).residual().streamWidth();
            try (GpuBuffer in = main.allocateBuffer((long) UBATCH * width * Float.BYTES, true)) {
                ScratchSlot inSlot = ScratchSlot.of("test.moe.eplb.redundant.in", in, 0, (long) UBATCH * width);
                float[] values = new float[UBATCH * width];
                for (int i = 0; i < values.length; i++) {
                    values[i] = (float) Math.sin(i * 0.13);
                }
                try (GpuEvent e = queue.enqueueWriteBuffer(in, 0L, (long) values.length * Float.BYTES, values)) {
                    queue.finish();
                }
                ScratchSlot wholeOut = referenceScratch.slot(ScratchSlots.FFN_OUT);
                ScratchSlot splitOut = splitScratch.slot(ScratchSlots.FFN_OUT);
                whole.forward(UBATCH, inSlot, wholeOut);
                split.forward(UBATCH, inSlot, splitOut);
                assertArrayEquals("the replicated ranks match the whole-device block",
                        read(wholeOut.buffer(), UBATCH * width), read(splitOut.buffer(), UBATCH * width), TOL);
            } finally {
                whole.close();
                split.close();
            }
            // A layout that gives one expert a third replica does not fit the uploaded replica map: refused.
            IllegalArgumentException grown = assertThrows(IllegalArgumentException.class,
                    () -> redundantGroup.rearrangeExperts(Collections.singletonMap(layer.weightLayer(),
                            new int[]{0, 0, 0, 1, 2, 3})));
            assertTrue(grown.getMessage(), grown.getMessage().contains("replica"));
        } finally {
            redundantGroup.close();
            splitUploader.close();
            splitOps.close();
            splitScratch.close();
        }
    }

    @Test
    public void aRuntimeRearrangementMigratesTheBanksAndKeepsTheOutput() {
        TpMoePlan.Layer layer = plan.layers().get(0);
        assertTrue("the balanced layer's routed ids are remapped", layer.remaps());
        int layerId = layer.weightLayer();
        group.rearrangeExperts(Collections.singletonMap(layerId, new int[]{1, 3, 0, 2}));
        LayerSpec specLayer = spec.layers().get(0);
        GpuMoeFfn whole = new GpuMoeFfn(specLayer.ffnKind(), specLayer);
        whole.bind(referenceContext, specLayer);
        GpuFfnBlock split = group.decorator().ffn(specLayer, null);
        assertNotNull(split);
        split.bind(splitContext, specLayer);
        int width = specLayer.residual().streamWidth();
        try (GpuBuffer in = main.allocateBuffer((long) UBATCH * width * Float.BYTES, true)) {
            ScratchSlot inSlot = ScratchSlot.of("test.moe.eplb.rearranged.in", in, 0, (long) UBATCH * width);
            float[] values = new float[UBATCH * width];
            for (int i = 0; i < values.length; i++) {
                values[i] = (float) Math.cos(i * 0.17);
            }
            try (GpuEvent e = queue.enqueueWriteBuffer(in, 0L, (long) values.length * Float.BYTES, values)) {
                queue.finish();
            }
            ScratchSlot wholeOut = referenceScratch.slot(ScratchSlots.FFN_OUT);
            ScratchSlot splitOut = splitScratch.slot(ScratchSlots.FFN_OUT);
            whole.forward(UBATCH, inSlot, wholeOut);
            split.forward(UBATCH, inSlot, splitOut);
            assertArrayEquals("the ranks hold the swapped experts after the rearrangement and match the whole",
                    read(wholeOut.buffer(), UBATCH * width), read(splitOut.buffer(), UBATCH * width), TOL);
        } finally {
            whole.close();
            split.close();
        }
    }

    @Test
    public void aRearrangementWithoutRemapBuffersOrWithTooManyReplicasIsRefused() {
        TpMoePlan contiguous = TpMoePlan.of(spec, RANKS, referenceUploader, true);
        int layerId = contiguous.layers().get(0).weightLayer();
        IllegalArgumentException parallel = assertThrows(IllegalArgumentException.class,
                () -> TpMoePlan.of(spec, RANKS, referenceUploader, false, null, true));
        assertTrue(parallel.getMessage(), parallel.getMessage().contains("expert parallel"));

        TpMoeUploader uploader = new TpMoeUploader(contiguous, referenceUploader, store);
        try {
            // The contiguous split uploaded no remap buffers (not rearrangeable): a rearrangement is refused.
            IllegalStateException noMap = assertThrows(IllegalStateException.class,
                    () -> uploader.rearrange(Collections.singletonMap(layerId, new int[]{3, 2, 1, 0}),
                            group));
            assertTrue(noMap.getMessage(), noMap.getMessage().contains("remap"));
        } finally {
            uploader.close();
        }
    }

    private static float[] floats(byte[] bytes) {
        ByteBuffer wrapped = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[bytes.length / Float.BYTES];
        wrapped.asFloatBuffer().get(out);
        return out;
    }

    private float[] read(GpuBuffer buffer, int floats) {
        float[] out = new float[floats];
        try (GpuEvent e = queue.enqueueReadBuffer(buffer, 0, (long) floats * Float.BYTES, out)) {
            queue.finish();
        }
        return out;
    }
}

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
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.moe.GpuMoeFfn;
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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Tensor-parallel expert parity with a shared expert: a synthetic GLM4-MoE layer runs once as the whole-device
 * {@link GpuMoeFfn} and once as a two-rank {@link TpMoeBlock}, where each rank holds an intermediate-width shard of
 * the routed experts and of the shared expert (which is sharded like a dense FFN). The outputs match. The GPU cases
 * are skipped when no OpenCL device is available.
 */
public class TpMoeSharedExpertTest {

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

    @Before
    public void open() throws IOException {
        file = SyntheticModels.glm4moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(2).build(), 0)
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

        plan = TpMoePlan.of(spec, RANKS, referenceUploader);
        splitScratch = GpuScratch.forModel(main, spec, UBATCH);
        splitOps = new GpuOps(queue, kernels, splitScratch, placeholders);
        splitUploader = uploader();
        splitUploader.uploadResident(t -> plan.splits(t.name()));
        group = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same device")));
        group.init(spec, UBATCH, kernels, queue, placeholders, null, null, null, plan, null, 0);
        group.upload(store, splitUploader);
        splitContext = GpuForwardContext.builder(kernels, splitOps, splitUploader, spec).build();
    }

    private GpuWeightUploader uploader() {
        return new GpuWeightUploader(main, queue, parser.getTensors(), store, "glm4moe",
                GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(kernels)).build());
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
    public void theSharedExpertShardsMatchTheWholeBlock() {
        assertEquals("the plan covers the single MoE trunk layer", 1, plan.layers().size());
        LayerSpec layer = spec.layers().get(1);
        GpuMoeFfn whole = new GpuMoeFfn(layer.ffnKind(), layer);
        whole.bind(referenceContext, layer);
        GpuFfnBlock split = group.decorator().ffn(layer, null);
        assertNotNull("the split experts replace the device block", split);
        split.bind(splitContext, layer);
        int width = layer.residual().streamWidth();
        try (GpuBuffer in = main.allocateBuffer((long) UBATCH * width * Float.BYTES, true)) {
            ScratchSlot inSlot = ScratchSlot.of("test.moe.shared.in", in, 0, (long) UBATCH * width);
            float[] values = new float[UBATCH * width];
            for (int i = 0; i < values.length; i++) {
                values[i] = (float) Math.sin(i * 0.13 + 0.7);
            }
            try (GpuEvent e = queue.enqueueWriteBuffer(in, 0L, (long) values.length * Float.BYTES, values)) {
                queue.finish();
            }
            ScratchSlot wholeOut = referenceScratch.slot(ScratchSlots.FFN_OUT);
            ScratchSlot splitOut = splitScratch.slot(ScratchSlots.FFN_OUT);
            whole.forward(UBATCH, inSlot, wholeOut);
            split.forward(UBATCH, inSlot, splitOut);
            assertArrayEquals("the tensor-parallel shared expert matches the whole-device block",
                    read(wholeOut.buffer(), UBATCH * width), read(splitOut.buffer(), UBATCH * width), TOL);
        } finally {
            whole.close();
            split.close();
        }
    }

    private float[] read(GpuBuffer buffer, int floats) {
        float[] out = new float[floats];
        try (GpuEvent e = queue.enqueueReadBuffer(buffer, 0, (long) floats * Float.BYTES, out)) {
            queue.finish();
        }
        return out;
    }
}

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

/**
 * Context-shift parity of a tensor-parallel attention layer on the OpenCL backend: after both the whole-device mixer
 * and the two-rank {@link TpAttentionMixer} store two steps, the main memory and the per-rank caches shift their rows
 * and each mixer re-rotates its own keys; the next step then produces the same branch output on both. The GPU cases
 * are skipped when no OpenCL device is available.
 */
public class TpContextShiftTest {

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
        MemoryConfig mem = MemoryConfig.defaults().toBuilder().ubatch(UBATCH).build();
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
            mainContext = null;
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
    public void theTensorParallelShiftMatchesTheWholeDeviceShift() {
        LayerSpec layer = spec.layers().get(0);
        GpuAttentionMixer whole = new GpuAttentionMixer();
        whole.bind(mainContext, layer);
        GpuMixer split = group.attentionDecorator().mixer(layer, null);
        assertNotNull("the split attention replaces the device mixer", split);
        split.bind(mainContext, layer);
        try (GpuBuffer in = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true);
             GpuBuffer outWhole = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true);
             GpuBuffer outSplit = main.allocateBuffer((long) UBATCH * N_EMBD * Float.BYTES, true)) {
            ScratchSlot inSlot = ScratchSlot.of("test.shift.in", in, 0, (long) UBATCH * N_EMBD);
            ScratchSlot wholeSlot = ScratchSlot.of("test.shift.whole", outWhole, 0, (long) UBATCH * N_EMBD);
            ScratchSlot splitSlot = ScratchSlot.of("test.shift.split", outSplit, 0, (long) UBATCH * N_EMBD);
            for (int step = 0; step < 2; step++) {
                int pos0 = step * UBATCH;
                int[] tokens = {pos0, pos0 + 1, pos0 + 2};
                write(in, input(step));
                mainMemory.sequence().append(tokens, 0, UBATCH);
                group.appendStep(tokens, 0, UBATCH);
                mainMemory.stepInputs().upload(pos0, UBATCH);
                whole.prefill(pos0, UBATCH, inSlot, wholeSlot);
                split.prefill(pos0, UBATCH, inSlot, splitSlot);
            }
            assertNull("the whole-device mixer shifts", whole.shiftRefusal());
            assertNull("the tensor-parallel mixer shifts", split.shiftRefusal());
            mainMemory.shiftPositions(0, 2);
            group.shiftPositions(0, 2);
            whole.shiftContext(0, 2);
            split.shiftContext(0, 2);

            int pos0 = 2 * UBATCH - 2;
            int[] tokens = {pos0, pos0 + 1, pos0 + 2};
            write(in, input(2));
            mainMemory.sequence().append(tokens, 0, UBATCH);
            group.appendStep(tokens, 0, UBATCH);
            mainMemory.stepInputs().upload(pos0, UBATCH);
            whole.prefill(pos0, UBATCH, inSlot, wholeSlot);
            split.prefill(pos0, UBATCH, inSlot, splitSlot);
            assertArrayEquals("the shifted tensor-parallel attention matches the shifted whole-device branch",
                    read(outWhole, UBATCH * N_EMBD), read(outSplit, UBATCH * N_EMBD), TOL);
        } finally {
            whole.close();
            split.close();
        }
    }
}

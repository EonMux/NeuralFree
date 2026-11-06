package com.eonmux.neuralfree.gguf.gpu.pipeline;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuLayerExecutor;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuSplitForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.head.GpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Numeric parity of a layer split against the single-device forward pass on a real GPU: the same tiny llama runs
 * once on one device and once split over two devices (two contexts on the same OpenCL device, as on a two-GPU
 * host), and prefill, decode and greedy selection must produce the same logits. Skipped when OpenCL is unavailable.
 */
public class GpuSplitParityTest {

    private static final int UBATCH = 8;
    private static final int N_CTX = 64;

    private HWAcceleration acceleration;
    private final List<ComputeBackend> backends = new ArrayList<>();
    private final List<AutoCloseable> closeables = new ArrayList<>();

    @Before
    public void assumeGpu() {
        GpuTestUtil.assumeOpenCLAvailable();
        acceleration = new HWAcceleration(java.util.EnumSet.of(ComputeBackend.BackendType.OPENCL));
    }

    @After
    public void tearDown() throws Exception {
        for (int i = closeables.size() - 1; i >= 0; i--) {
            try {
                closeables.get(i).close();
            } catch (Exception ignored) {
                // teardown only
            }
        }
        for (ComputeBackend backend : backends) {
            try {
                backend.close();
            } catch (RuntimeException ignored) {
                // teardown only
            }
        }
        if (acceleration != null) {
            acceleration.shutdown();
        }
    }

    private ComputeBackend openBackend(CLDevice device) {
        ComputeBackend backend = acceleration.createBackend(device);
        backends.add(backend);
        return backend;
    }

    /** Host values and shape of one tensor, deterministic across the devices it is uploaded to. */
    private static final class HostTensor {
        final long[] dims;
        final float[] data;

        HostTensor(String name, long[] dims) {
            long n = 1;
            for (long d : dims) {
                n *= d;
            }
            this.dims = dims;
            this.data = new float[(int) n];
            Random random = new Random(name.hashCode() * 31 + 7);
            for (int i = 0; i < this.data.length; i++) {
                this.data[i] = random.nextFloat() * 0.4f - 0.2f;
            }
        }

        long bytes() {
            return data.length * (long) Float.BYTES;
        }
    }

    /** One assembled forward pass over the layers {@code [firstIl, endIl)} of the fixture, on its own device. A
     * full-span path also builds the single-device {@link GpuForwardBackend}; a shard path is driven by the split
     * backend's executor alone. */
    private static final class Path implements AutoCloseable {
        final LlmKernels kernels;
        final GpuScratch scratch;
        final GpuPlaceholders placeholders;
        final GpuOps ops;
        final DeviceModelMemory memory;
        final GpuForwardContext context;
        final GpuLayerExecutor executor;
        final GpuForwardBackend forward;

        Path(ComputeBackend backend, ModelSpec spec, int firstIl, int endIl, Map<String, GpuWeight> weights) {
            this.kernels = LlmKernels.open(backend);
            this.scratch = GpuScratch.forModel(backend, spec, UBATCH);
            this.placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
            this.ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
            kernels.prewarm(GpuForwardBackend.kernels(spec));
            MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16).nCtx(N_CTX)
                    .ubatch(UBATCH).build();
            GpuForwardContext.Builder b = GpuForwardContext.builder(kernels, ops, weights::get, spec);
            if (spec.memoryKind() != MemoryKind.NONE) {
                this.memory = DeviceModelMemory.allocate(backend, backend.computeQueue(),
                        MemoryPlanner.planForContext(spec, cfg, N_CTX, il -> il >= firstIl && il < endIl),
                        spec.hparams().mropeTextPositions(), null);
                b.memory(memory);
            } else {
                this.memory = null;
            }
            this.context = b.build();
            this.executor = GpuLayerExecutor.forStack(context, GpuFactoryCatalog.defaults(),
                    spec.layers().subList(firstIl, endIl));
            this.forward = firstIl == 0 && endIl == spec.layers().size()
                    ? new GpuForwardBackend(context, GpuFactoryCatalog.defaults(),
                    GpuFactoryCatalog.headPreprocessors())
                    : null;
        }

        @Override
        public void close() {
            if (forward != null) {
                forward.close();
            }
            executor.close();
            ops.close();
            placeholders.close();
            scratch.close();
            if (memory != null) {
                memory.close();
            }
        }
    }

    private static String blk(int il, String base) {
        return TensorNames.layerName(TensorNames.STACK_BLK, il, TensorNames.weight(base));
    }

    /** Every tensor of the fixture with its GGUF name and shape. */
    private static Map<String, HostTensor> tensorValues(ModelSpec spec) {
        Map<String, HostTensor> out = new HashMap<>();
        out.put(TensorNames.weight(TensorNames.TOKEN_EMBD),
                new HostTensor("token_embd", new long[]{SplitSpecs.N_EMBD, SplitSpecs.VOCAB}));
        out.put(TensorNames.weight(TensorNames.OUTPUT_NORM),
                new HostTensor("output_norm", new long[]{SplitSpecs.N_EMBD}));
        out.put(TensorNames.weight(TensorNames.OUTPUT),
                new HostTensor("output", new long[]{SplitSpecs.N_EMBD, SplitSpecs.VOCAB}));
        long qkv = SplitSpecs.N_HEAD * (long) SplitSpecs.HEAD_DIM;
        for (LayerSpec l : spec.layers()) {
            int il = l.weightLayer();
            out.put(blk(il, "attn_norm"), new HostTensor("attn_norm" + il, new long[]{SplitSpecs.N_EMBD}));
            out.put(blk(il, "ffn_norm"), new HostTensor("ffn_norm" + il, new long[]{SplitSpecs.N_EMBD}));
            out.put(blk(il, "ffn_gate"), new HostTensor("ffn_gate" + il,
                    new long[]{SplitSpecs.N_EMBD, SplitSpecs.N_FF}));
            out.put(blk(il, "ffn_up"), new HostTensor("ffn_up" + il,
                    new long[]{SplitSpecs.N_EMBD, SplitSpecs.N_FF}));
            out.put(blk(il, "ffn_down"), new HostTensor("ffn_down" + il,
                    new long[]{SplitSpecs.N_FF, SplitSpecs.N_EMBD}));
            out.put(blk(il, "attn_q"), new HostTensor("attn_q" + il, new long[]{SplitSpecs.N_EMBD, qkv}));
            out.put(blk(il, "attn_k"), new HostTensor("attn_k" + il, new long[]{SplitSpecs.N_EMBD, qkv}));
            out.put(blk(il, "attn_v"), new HostTensor("attn_v" + il, new long[]{SplitSpecs.N_EMBD, qkv}));
            out.put(blk(il, "attn_output"), new HostTensor("attn_output" + il,
                    new long[]{qkv, SplitSpecs.N_EMBD}));
        }
        return out;
    }

    /** Uploads the tensors of {@code selection} to the backend with their host values. */
    private static Map<String, GpuWeight> upload(ComputeBackend backend, Map<String, HostTensor> tensors,
                                                 Map<String, HostTensor> selection) {
        Map<String, GpuWeight> out = new HashMap<>();
        ComputeQueue queue = backend.computeQueue();
        for (Map.Entry<String, HostTensor> e : selection.entrySet()) {
            HostTensor t = e.getValue();
            GpuWeight w = GpuWeight.of(e.getKey(), GGMLType.F32, t.dims.clone(),
                    backend.allocateBuffer(t.bytes(), true));
            queue.enqueueWriteBuffer(w.shard(0), 0, t.bytes(), t.data).close();
            out.put(e.getKey(), w);
        }
        queue.finish();
        return out;
    }

    /** The tensors device holds: model-level when {@code first < 0}, else the layers of {@code [first, end)}. */
    private static Map<String, HostTensor> subset(Map<String, HostTensor> values, int first, int end) {
        Map<String, HostTensor> out = new HashMap<>();
        for (Map.Entry<String, HostTensor> e : values.entrySet()) {
            String name = e.getKey();
            if (first < 0 && !name.startsWith("blk.")) {
                out.put(name, e.getValue());
            } else if (first >= 0 && name.startsWith("blk.")) {
                int il = Integer.parseInt(name.substring(4, name.indexOf('.', 4)));
                if (il >= first && il < end) {
                    out.put(name, e.getValue());
                }
            }
        }
        return out;
    }

    @Test
    public void splitForwardMatchesTheSingleDeviceForward() {
        ModelSpec spec = SplitSpecs.trunk(2, MemoryKind.KV);
        Map<String, HostTensor> values = tensorValues(spec);
        CLDevice device = acceleration.getCLDevices().get(0);

        ComputeBackend singleBackend = openBackend(device);
        Path single = new Path(singleBackend, spec, 0, 2, upload(singleBackend, values, values));

        ComputeBackend first = openBackend(device);
        ComputeBackend second = openBackend(device);
        Map<String, GpuWeight> firstWeights = upload(first, values, subset(values, 0, 1));
        firstWeights.putAll(upload(first, values, subset(values, -1, 0)));
        Map<String, GpuWeight> secondWeights = upload(second, values, subset(values, 1, 2));
        Path stageOne = new Path(first, spec, 0, 1, firstWeights);
        Path stageTwo = new Path(second, spec, 1, 2, secondWeights);
        GpuEmbeddingStage embedding = new GpuEmbeddingStage(stageOne.context);
        GpuLogitsHead head = new GpuLogitsHead(stageOne.context, GpuFactoryCatalog.headPreprocessors());
        closeables.add(head);
        GpuSplitForwardBackend split = new GpuSplitForwardBackend(spec, Arrays.asList(
                new GpuSplitForwardBackend.Shard(stageOne.context, stageOne.executor, stageOne.memory, 0,
                        SplitSpecs.N_EMBD, SplitSpecs.N_EMBD),
                new GpuSplitForwardBackend.Shard(stageTwo.context, stageTwo.executor, stageTwo.memory, 1,
                        SplitSpecs.N_EMBD, SplitSpecs.N_EMBD)), 0, embedding, head, stageOne.memory);
        closeables.add(split);

        int[] prompt = {1, 2, 3, 4, 5, 6, 7};
        float[] singlePrefill = single.forward.prefill(prompt, 0);
        float[] splitPrefill = split.prefill(prompt, 0);
        assertEquals(singlePrefill.length, splitPrefill.length);
        assertArrayEquals(singlePrefill, splitPrefill, 1e-4f);

        float[] singleDecode = single.forward.decode(9, prompt.length);
        float[] splitDecode = split.decode(9, prompt.length);
        assertArrayEquals(singleDecode, splitDecode, 1e-4f);

        assertEquals(single.forward.decodeGreedyOnDevice(11, prompt.length + 1),
                split.decodeGreedyOnDevice(11, prompt.length + 1));

        assertArrayEquals(single.forward.forwardRows(new int[]{12}, prompt.length + 2, null)[0],
                split.forwardRows(new int[]{12}, prompt.length + 2, null)[0], 1e-4f);

        int[] verify = {13, 14, 15};
        assertArrayEquals(single.forward.forwardRowsGreedy(verify, prompt.length + 3, null),
                split.forwardRowsGreedy(verify, prompt.length + 3, null));

        assertEquals(single.forward.memory().sequence().nPast(), split.memory().sequence().nPast());

        single.close();
        stageOne.close();
        stageTwo.close();
    }
}

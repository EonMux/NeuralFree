package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.gpu.head.GpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.gpu.pipeline.SplitSpecs;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import org.junit.After;
import org.junit.Assume;
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
import static org.junit.Assert.assertTrue;

/**
 * The per-shard decode replay of a layer split on CUDA: two backends on one device stand in for two GPUs, and a tiny
 * two-layer llama decodes the same tokens through a split whose spans run directly and through one whose spans are
 * captured and replayed. The logits must be bit-identical, and every shard of the replaying split must have
 * submitted patched replays. A context shift and a rollback without the chain replay drop every shard's capture; a
 * rollback with it re-arms the captures at the rewound position; a memory reset and a memoryless call keep them and
 * verify them against the body first. Skipped without a CUDA device that serves the LLM kernels.
 */
public class GpuSplitDecodeReplayTest {

    private static final int UBATCH = 8;
    private static final int N_CTX = 64;
    private static final int DECODE_STEPS = 12;

    private HWAcceleration hw;
    private CUDevice device;
    private final List<AutoCloseable> closeables = new ArrayList<>();
    private final List<ComputeBackend> backends = new ArrayList<>();

    @Before
    public void setUp() {
        HWAcceleration.enableCuda = true;
        hw = new HWAcceleration();
        Assume.assumeTrue("no CUDA device", !hw.getCUDevices().isEmpty());
        device = hw.getCUDevices().get(0);
        ComputeBackend probe = openBackend();
        Availability availability = LlmKernels.availability(probe);
        Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
    }

    @After
    public void tearDown() throws Exception {
        for (int i = closeables.size() - 1; i >= 0; i--) {
            closeables.get(i).close();
        }
        for (ComputeBackend backend : backends) {
            backend.close();
        }
        if (hw != null) {
            hw.shutdown();
        }
    }

    @Test
    public void replayedSpansDecodeBitIdenticallyToDirectSpans() {
        ModelSpec spec = SplitSpecs.trunk(2, MemoryKind.KV);
        Map<String, float[]> values = tensorValues(spec);
        GpuSplitForwardBackend direct = withReplay(false, () -> split(spec, values));
        GpuSplitForwardBackend replayed = withReplay(true, () -> split(spec, values));

        int[] prompt = {1, 2, 3, 4, 5};
        assertArrayEquals(direct.prefill(prompt, 0), replayed.prefill(prompt, 0), 0f);
        for (int i = 0; i < DECODE_STEPS; i++) {
            int pos = prompt.length + i;
            int token = 6 + i % 20;
            assertArrayEquals("decode step " + i, direct.decode(token, pos), replayed.decode(token, pos), 0f);
        }

        for (GpuDecodeReplay.Counters c : direct.decodeReplayCounters()) {
            assertEquals("the direct split captures nothing: " + c, 0, c.captures);
        }
        List<GpuDecodeReplay.Counters> counters = replayed.decodeReplayCounters();
        assertEquals(2, counters.size());
        for (GpuDecodeReplay.Counters c : counters) {
            assertEquals("each shard captures its span once: " + c, 1, c.captures);
            assertTrue("each shard submits patched replays: " + c, c.patchedReplays > 0);
        }
    }

    @Test
    public void aContextShiftDropsEveryShardsCaptureAndDecodingStaysExact() {
        ModelSpec spec = SplitSpecs.trunk(2, MemoryKind.KV);
        Map<String, float[]> values = tensorValues(spec);
        GpuSplitForwardBackend direct = withReplay(false, () -> split(spec, values));
        GpuSplitForwardBackend replayed = withReplay(true, () -> split(spec, values));

        int[] prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        direct.prefill(prompt, 0);
        replayed.prefill(prompt, 0);
        int pos = prompt.length;
        for (int i = 0; i < 8; i++, pos++) {
            assertArrayEquals(direct.decode(11, pos), replayed.decode(11, pos), 0f);
        }
        direct.shiftContext(2, 4);
        replayed.shiftContext(2, 4);
        pos -= 4;
        for (int i = 0; i < 8; i++, pos++) {
            assertArrayEquals("after the shift, step " + i, direct.decode(12, pos), replayed.decode(12, pos), 0f);
        }
        for (GpuDecodeReplay.Counters c : replayed.decodeReplayCounters()) {
            assertEquals("the shift drops the capture and the span captures again: " + c, 2, c.captures);
        }
    }

    @Test
    public void aRollbackKeepsEveryShardsCaptureReArmedAtTheRewoundPosition() {
        ModelSpec spec = SplitSpecs.trunk(2, MemoryKind.KV);
        Map<String, float[]> values = tensorValues(spec);
        GpuSplitForwardBackend direct = withReplay(false, () -> split(spec, values));
        GpuSplitForwardBackend replayed = withChainReplay(true, () -> withReplay(true, () -> split(spec, values)));
        int pos = decodeAfterPrompt(direct, replayed);
        List<GpuDecodeReplay.Counters> before = replayed.decodeReplayCounters();

        direct.rollback(3);
        replayed.rollback(3);
        pos -= 3;
        for (int i = 0; i < 6; i++, pos++) {
            assertArrayEquals("after the rollback, step " + i, direct.decode(13 + i, pos),
                    replayed.decode(13 + i, pos), 0f);
        }
        List<GpuDecodeReplay.Counters> after = replayed.decodeReplayCounters();
        for (int s = 0; s < after.size(); s++) {
            assertEquals("the rollback keeps shard " + s + "'s capture: " + after.get(s), 1, after.get(s).captures);
            assertEquals("and replays it trusted from the rewound position: " + after.get(s),
                    before.get(s).patchedReplays + 6, after.get(s).patchedReplays);
        }
    }

    @Test
    public void withoutTheChainReplayARollbackDropsEveryShardsCapture() {
        ModelSpec spec = SplitSpecs.trunk(2, MemoryKind.KV);
        Map<String, float[]> values = tensorValues(spec);
        GpuSplitForwardBackend direct = withReplay(false, () -> split(spec, values));
        GpuSplitForwardBackend replayed = withChainReplay(false, () -> withReplay(true, () -> split(spec, values)));
        int pos = decodeAfterPrompt(direct, replayed);

        direct.rollback(3);
        replayed.rollback(3);
        pos -= 3;
        for (int i = 0; i < 8; i++, pos++) {
            assertArrayEquals("after the rollback, step " + i, direct.decode(13 + i, pos),
                    replayed.decode(13 + i, pos), 0f);
        }
        for (GpuDecodeReplay.Counters c : replayed.decodeReplayCounters()) {
            assertEquals("the rollback drops the capture and the span captures again: " + c, 2, c.captures);
        }
    }

    @Test
    public void aMemoryResetKeepsTheCapturesAndVerifiesThemAgain() {
        ModelSpec spec = SplitSpecs.trunk(2, MemoryKind.KV);
        Map<String, float[]> values = tensorValues(spec);
        GpuSplitForwardBackend direct = withReplay(false, () -> split(spec, values));
        GpuSplitForwardBackend replayed = withReplay(true, () -> split(spec, values));
        decodeAfterPrompt(direct, replayed);
        List<GpuDecodeReplay.Counters> before = replayed.decodeReplayCounters();

        direct.reset();
        replayed.reset();
        for (int pos = 0; pos < 6; pos++) {
            assertArrayEquals("after the reset, step " + pos, direct.decode(20 + pos, pos),
                    replayed.decode(20 + pos, pos), 0f);
        }
        List<GpuDecodeReplay.Counters> after = replayed.decodeReplayCounters();
        for (int s = 0; s < after.size(); s++) {
            assertEquals("the reset keeps shard " + s + "'s capture: " + after.get(s), 1, after.get(s).captures);
            assertEquals("and verifies it against the body before trusting it again: " + after.get(s),
                    before.get(s).bodyReplays + GpuDecodeReplay.VERIFY_STEPS, after.get(s).bodyReplays);
            assertEquals(before.get(s).patchedReplays + 6 - GpuDecodeReplay.VERIFY_STEPS,
                    after.get(s).patchedReplays);
        }
    }

    @Test
    public void aMemorylessCallBetweenDecodeStepsLeavesEverySpanExact() {
        ModelSpec spec = SplitSpecs.trunk(2, MemoryKind.KV);
        Map<String, float[]> values = tensorValues(spec);
        GpuSplitForwardBackend direct = withReplay(false, () -> split(spec, values));
        GpuSplitForwardBackend replayed = withReplay(true, () -> split(spec, values));
        decodeAfterPrompt(direct, replayed);

        direct.reset();
        replayed.reset();
        int[] other = {7, 3, 9, 1};
        float[][] directRows = direct.hiddenRows(other, null);
        float[][] replayedRows = replayed.hiddenRows(other, null);
        assertEquals(directRows.length, replayedRows.length);
        for (int r = 0; r < directRows.length; r++) {
            assertArrayEquals("hidden row " + r, directRows[r], replayedRows[r], 0f);
        }
        int[] prompt = {2, 4, 6};
        assertArrayEquals(direct.prefill(prompt, 0), replayed.prefill(prompt, 0), 0f);
        for (int i = 0, pos = prompt.length; i < 8; i++, pos++) {
            assertArrayEquals("after the memoryless call, step " + i, direct.decode(10 + i, pos),
                    replayed.decode(10 + i, pos), 0f);
        }
        for (GpuDecodeReplay.Counters c : replayed.decodeReplayCounters()) {
            assertEquals("the memoryless call keeps the capture: " + c, 1, c.captures);
        }
    }

    /** Prefills one prompt and decodes eight steps on both splits, which must agree; the next position. */
    private static int decodeAfterPrompt(GpuSplitForwardBackend direct, GpuSplitForwardBackend replayed) {
        int[] prompt = {1, 2, 3, 4, 5};
        assertArrayEquals(direct.prefill(prompt, 0), replayed.prefill(prompt, 0), 0f);
        int pos = prompt.length;
        for (int i = 0; i < 8; i++, pos++) {
            assertArrayEquals("decode step " + i, direct.decode(6 + i, pos), replayed.decode(6 + i, pos), 0f);
        }
        for (GpuDecodeReplay.Counters c : replayed.decodeReplayCounters()) {
            assertEquals("each shard captured its span once: " + c, 1, c.captures);
            assertTrue("and replays it trusted: " + c, c.patchedReplays > 0);
        }
        return pos;
    }

    private static GpuSplitForwardBackend withChainReplay(boolean on, SplitFactory factory) {
        String previous = System.getProperty(GpuForwardBackend.CHAIN_REPLAY_PROPERTY);
        System.setProperty(GpuForwardBackend.CHAIN_REPLAY_PROPERTY, Boolean.toString(on));
        try {
            return factory.make();
        } finally {
            if (previous == null) {
                System.clearProperty(GpuForwardBackend.CHAIN_REPLAY_PROPERTY);
            } else {
                System.setProperty(GpuForwardBackend.CHAIN_REPLAY_PROPERTY, previous);
            }
        }
    }

    private interface SplitFactory {
        GpuSplitForwardBackend make();
    }

    private static GpuSplitForwardBackend withReplay(boolean on, SplitFactory factory) {
        String previous = System.getProperty(GpuForwardBackend.DECODE_REPLAY_PROPERTY);
        System.setProperty(GpuForwardBackend.DECODE_REPLAY_PROPERTY, Boolean.toString(on));
        try {
            return factory.make();
        } finally {
            if (previous == null) {
                System.clearProperty(GpuForwardBackend.DECODE_REPLAY_PROPERTY);
            } else {
                System.setProperty(GpuForwardBackend.DECODE_REPLAY_PROPERTY, previous);
            }
        }
    }

    private ComputeBackend openBackend() {
        ComputeBackend backend = new CudaComputeBackend(device);
        backends.add(backend);
        return backend;
    }

    /** A split of the two layers over two backends of the device; the first holds the embedding and the head. */
    private GpuSplitForwardBackend split(ModelSpec spec, Map<String, float[]> values) {
        ComputeBackend first = openBackend();
        ComputeBackend second = openBackend();
        Map<String, GpuWeight> firstWeights = upload(first, values, 0, 1, true);
        Map<String, GpuWeight> secondWeights = upload(second, values, 1, 2, false);
        Span one = new Span(first, spec, 0, 1, firstWeights);
        Span two = new Span(second, spec, 1, 2, secondWeights);
        closeables.add(one);
        closeables.add(two);
        GpuEmbeddingStage embedding = new GpuEmbeddingStage(one.context);
        GpuLogitsHead head = new GpuLogitsHead(one.context, GpuFactoryCatalog.headPreprocessors());
        GpuSplitForwardBackend split = new GpuSplitForwardBackend(spec, Arrays.asList(
                new GpuSplitForwardBackend.Shard(one.context, one.executor, one.memory, 0,
                        SplitSpecs.N_EMBD, SplitSpecs.N_EMBD),
                new GpuSplitForwardBackend.Shard(two.context, two.executor, two.memory, 1,
                        SplitSpecs.N_EMBD, SplitSpecs.N_EMBD)), 0, embedding, head, one.memory);
        closeables.add(split);
        return split;
    }

    /** One device's span {@code [firstIl, endIl)}: its kernels, scratch, memory, context and executor. */
    private static final class Span implements AutoCloseable {
        final LlmKernels kernels;
        final GpuScratch scratch;
        final GpuPlaceholders placeholders;
        final GpuOps ops;
        final DeviceModelMemory memory;
        final GpuForwardContext context;
        final GpuLayerExecutor executor;

        /** The weights' buffers belong to the backend, which releases them when it closes. */
        Span(ComputeBackend backend, ModelSpec spec, int firstIl, int endIl, Map<String, GpuWeight> weights) {
            this.kernels = LlmKernels.open(backend);
            this.scratch = GpuScratch.forModel(backend, spec, UBATCH);
            this.placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
            this.ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
            kernels.prewarm(GpuForwardBackend.kernels(spec));
            MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16).nCtx(N_CTX)
                    .ubatch(UBATCH).build();
            this.memory = DeviceModelMemory.allocate(backend, backend.computeQueue(),
                    MemoryPlanner.planForContext(spec, cfg, N_CTX, il -> il >= firstIl && il < endIl),
                    spec.hparams().mropeTextPositions(), null);
            this.context = GpuForwardContext.builder(kernels, ops, weights::get, spec).memory(memory).build();
            this.executor = GpuLayerExecutor.forStack(context, GpuFactoryCatalog.defaults(),
                    spec.layers().subList(firstIl, endIl));
        }

        @Override
        public void close() {
            ops.close();
            placeholders.close();
            scratch.close();
            memory.close();
        }
    }

    private static String blk(int il, String base) {
        return TensorNames.layerName(TensorNames.STACK_BLK, il, TensorNames.weight(base));
    }

    /** Every tensor of the fixture by GGUF name, with deterministic values. */
    private static Map<String, float[]> tensorValues(ModelSpec spec) {
        Map<String, float[]> out = new HashMap<>();
        long qkv = SplitSpecs.N_HEAD * (long) SplitSpecs.HEAD_DIM;
        put(out, TensorNames.weight(TensorNames.TOKEN_EMBD), (long) SplitSpecs.N_EMBD * SplitSpecs.VOCAB);
        put(out, TensorNames.weight(TensorNames.OUTPUT_NORM), SplitSpecs.N_EMBD);
        put(out, TensorNames.weight(TensorNames.OUTPUT), (long) SplitSpecs.N_EMBD * SplitSpecs.VOCAB);
        for (LayerSpec l : spec.layers()) {
            int il = l.weightLayer();
            put(out, blk(il, "attn_norm"), SplitSpecs.N_EMBD);
            put(out, blk(il, "ffn_norm"), SplitSpecs.N_EMBD);
            put(out, blk(il, "ffn_gate"), (long) SplitSpecs.N_EMBD * SplitSpecs.N_FF);
            put(out, blk(il, "ffn_up"), (long) SplitSpecs.N_EMBD * SplitSpecs.N_FF);
            put(out, blk(il, "ffn_down"), (long) SplitSpecs.N_FF * SplitSpecs.N_EMBD);
            put(out, blk(il, "attn_q"), SplitSpecs.N_EMBD * qkv);
            put(out, blk(il, "attn_k"), SplitSpecs.N_EMBD * qkv);
            put(out, blk(il, "attn_v"), SplitSpecs.N_EMBD * qkv);
            put(out, blk(il, "attn_output"), qkv * SplitSpecs.N_EMBD);
        }
        return out;
    }

    private static void put(Map<String, float[]> out, String name, long n) {
        float[] data = new float[(int) n];
        Random random = new Random(name.hashCode() * 31L + 7);
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextFloat() * 0.4f - 0.2f;
        }
        out.put(name, data);
    }

    /** The shapes of the fixture's tensors, as {@link SplitSpecs} lays them out. */
    private static long[] dims(String name) {
        long qkv = SplitSpecs.N_HEAD * (long) SplitSpecs.HEAD_DIM;
        String base = name.startsWith("blk.") ? name.substring(name.indexOf('.', 4) + 1) : name;
        switch (base) {
            case "token_embd.weight":
            case "output.weight":
                return new long[]{SplitSpecs.N_EMBD, SplitSpecs.VOCAB};
            case "ffn_gate.weight":
            case "ffn_up.weight":
                return new long[]{SplitSpecs.N_EMBD, SplitSpecs.N_FF};
            case "ffn_down.weight":
                return new long[]{SplitSpecs.N_FF, SplitSpecs.N_EMBD};
            case "attn_q.weight":
            case "attn_k.weight":
            case "attn_v.weight":
                return new long[]{SplitSpecs.N_EMBD, qkv};
            case "attn_output.weight":
                return new long[]{qkv, SplitSpecs.N_EMBD};
            default:
                return new long[]{SplitSpecs.N_EMBD};
        }
    }

    /** Uploads the layers {@code [first, end)} and, when {@code model}, the model-level tensors. */
    private static Map<String, GpuWeight> upload(ComputeBackend backend, Map<String, float[]> values, int first,
                                                 int end, boolean model) {
        Map<String, GpuWeight> out = new HashMap<>();
        ComputeQueue queue = backend.computeQueue();
        for (Map.Entry<String, float[]> e : values.entrySet()) {
            String name = e.getKey();
            boolean layer = name.startsWith("blk.");
            if (layer) {
                int il = Integer.parseInt(name.substring(4, name.indexOf('.', 4)));
                if (il < first || il >= end) {
                    continue;
                }
            } else if (!model) {
                continue;
            }
            float[] data = e.getValue();
            long bytes = (long) data.length * Float.BYTES;
            GpuWeight w = GpuWeight.of(name, GGMLType.F32, dims(name), backend.allocateBuffer(bytes, true));
            queue.enqueueWriteBuffer(w.shard(0), 0, bytes, data).close();
            out.put(name, w);
        }
        queue.finish();
        return out;
    }
}

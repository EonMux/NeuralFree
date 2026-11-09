package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.head.GpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.gpu.pipeline.SplitSpecs;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
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
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import com.eonmux.neuralfree.gguf.testutil.FakeOp;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The pipeline behaviour of the split forward backend over two host-only fake devices: layers run on their own
 * device in order, the residual stream crosses between the devices through one read and one write per boundary, the
 * head and its readbacks stay on the main device, and every device's memory moves in lockstep.
 */
public class GpuSplitForwardBackendTest {

    private static final int UBATCH = 4;
    private static final int N_CTX = 64;

    private final List<Device> devices = new ArrayList<>();
    private final List<AutoCloseable> closeables = new ArrayList<>();
    private GpuSplitForwardBackend backend;
    private ModelSpec spec;

    private static final class Device {
        final String name;
        final FakeComputeBackend backend;
        final LlmKernels kernels;
        final GpuScratch scratch;
        final GpuPlaceholders placeholders;
        final GpuOps ops;
        final Map<String, GpuWeight> weights;
        final DeviceModelMemory memory;
        final GpuLayerExecutor executor;
        final GpuForwardContext context;
        final List<String> events;
        final int firstIl;

        Device(String name, FakeComputeBackend backend, ModelSpec spec, int firstIl, int endIl, boolean withMemory,
                List<String> events, ExecutionObserver observer) {
            this.name = name;
            this.backend = backend;
            this.firstIl = firstIl;
            this.kernels = LlmKernels.open(backend);
            this.scratch = GpuScratch.forModel(backend, spec, UBATCH);
            this.placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
            this.ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
            this.weights = new HashMap<>();
            for (int il = firstIl; il < endIl; il++) {
                weights.putAll(deviceTensors(backend, il));
            }
            MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16).nCtx(N_CTX)
                    .ubatch(UBATCH).build();
            GpuForwardContext.Builder b = GpuForwardContext.builder(kernels, ops, weights::get, spec)
                    .observer(observer);
            if (withMemory) {
                MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX, il -> il >= firstIl && il < endIl);
                this.memory = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan,
                        spec.hparams().mropeTextPositions(), null);
                b.memory(memory);
            } else {
                this.memory = null;
            }
            this.context = b.build();
            this.executor = GpuLayerExecutor.forStack(context, registry(), spec.layers().subList(firstIl, endIl));
            this.events = events;
        }

        FakeOpKind kind(int index) {
            return backend.ops().get(index).kind();
        }

        ScratchSlot xSlot() {
            return context.scratch().slot(ScratchSlots.X);
        }

        FakeGpuBuffer xBuffer() {
            return (FakeGpuBuffer) xSlot().buffer();
        }

        List<FakeDispatch> dispatches() {
            return backend.dispatches();
        }
    }

    private final List<String> mainEvents = new ArrayList<>();
    private final List<String> secondEvents = new ArrayList<>();
    private ExecutionObserver mainObserver = ExecutionObserver.NONE;
    private ExecutionObserver secondObserver = ExecutionObserver.NONE;
    private Device main;
    private Device second;
    private GpuEmbeddingStage embedding;
    private GpuLogitsHead head;

    @After
    public void tearDown() throws Exception {
        close(backend);
        for (int i = closeables.size() - 1; i >= 0; i--) {
            closeables.get(i).close();
        }
        for (Device d : devices) {
            close(d.executor);
            close(d.ops);
            close(d.placeholders);
            close(d.scratch);
            if (d.memory != null) {
                d.memory.close();
            }
            d.backend.close();
        }
    }

    private static void close(AutoCloseable c) {
        try {
            if (c != null) {
                c.close();
            }
        } catch (Exception ignored) {
            // teardown only
        }
    }

    /** Recording stand-ins for the mixers and FFNs: the per-family blocks have their own traces. */
    private static GpuBlockRegistry registry() {
        GpuMixerFactory mixer = new GpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return MixerKind.ATTENTION;
            }

            @Override
            public GpuMixer create(LayerSpec layer) {
                return new RecordingMixer();
            }
        };
        GpuFfnFactory ffn = new GpuFfnFactory() {
            @Override
            public FfnKind kind() {
                return FfnKind.DENSE;
            }

            @Override
            public GpuFfnBlock create(LayerSpec layer) {
                return new RecordingFfn();
            }
        };
        return new GpuBlockRegistry(Collections.singletonList(mixer), Collections.singletonList(ffn),
                BuiltInGpuExtras.factories());
    }

    private static final class RecordingMixer implements GpuMixer {
        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            // nothing to bind; the trace records the calls below
        }

        @Override
        public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        }

        @Override
        public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        }
    }

    private static final class RecordingFfn implements GpuFfnBlock {
        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
        }

        @Override
        public void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
        }
    }

    private static String blk(int il, String base) {
        return TensorNames.layerName(TensorNames.STACK_BLK, il, TensorNames.weight(base));
    }

    /** The F32 device tensors of one layer, allocated on the given fake backend. */
    private static Map<String, GpuWeight> deviceTensors(FakeComputeBackend backend, int il) {
        Map<String, GpuWeight> out = new HashMap<>();
        out.put(blk(il, TensorNames.ATTN_NORM), vector(backend, blk(il, TensorNames.ATTN_NORM),
                SplitSpecs.N_EMBD));
        out.put(blk(il, TensorNames.FFN_NORM), vector(backend, blk(il, TensorNames.FFN_NORM), SplitSpecs.N_EMBD));
        out.put(blk(il, TensorNames.FFN_GATE), matrix(backend, blk(il, TensorNames.FFN_GATE),
                SplitSpecs.N_EMBD, SplitSpecs.N_FF));
        out.put(blk(il, TensorNames.FFN_UP), matrix(backend, blk(il, TensorNames.FFN_UP),
                SplitSpecs.N_EMBD, SplitSpecs.N_FF));
        out.put(blk(il, TensorNames.FFN_DOWN), matrix(backend, blk(il, TensorNames.FFN_DOWN),
                SplitSpecs.N_FF, SplitSpecs.N_EMBD));
        return out;
    }

    private static GpuWeight vector(FakeComputeBackend backend, String name, long n) {
        return GpuWeight.of(name, GGMLType.F32, new long[]{n},
                backend.allocateBuffer(n * Float.BYTES, true));
    }

    private static GpuWeight matrix(FakeComputeBackend backend, String name, long cols, long rows) {
        return GpuWeight.of(name, GGMLType.F32, new long[]{cols, rows},
                backend.allocateBuffer(cols * rows * Float.BYTES, true));
    }

    /** A two-device split of a four-layer trunk: device one holds layers 0-1 plus the model-level tensors. */
    private void openSplit(MemoryKind kind) {
        spec = SplitSpecs.trunk(4, kind);
        main = new Device("one", FakeComputeBackend.builder().strictLlmArgs(true).build(), spec, 0, 2,
                kind == MemoryKind.KV, mainEvents, mainObserver);
        second = new Device("two", FakeComputeBackend.builder().strictLlmArgs(true).build(), spec, 2, 4,
                kind == MemoryKind.KV, secondEvents, secondObserver);
        devices.add(main);
        devices.add(second);
        // model-level tensors live on the main device
        main.weights.put(TensorNames.weight(TensorNames.TOKEN_EMBD),
                matrix(main.backend, TensorNames.weight(TensorNames.TOKEN_EMBD),
                        SplitSpecs.N_EMBD, SplitSpecs.VOCAB));
        main.weights.put(TensorNames.weight(TensorNames.OUTPUT_NORM),
                vector(main.backend, TensorNames.weight(TensorNames.OUTPUT_NORM), SplitSpecs.N_EMBD));
        main.weights.put(TensorNames.weight(TensorNames.OUTPUT),
                matrix(main.backend, TensorNames.weight(TensorNames.OUTPUT), SplitSpecs.N_EMBD, SplitSpecs.VOCAB));
        embedding = new GpuEmbeddingStage(main.context);
        head = new GpuLogitsHead(main.context, Collections.emptyList());
        closeables.add(head);
        GpuSplitForwardBackend.Shard one = new GpuSplitForwardBackend.Shard(main.context, main.executor,
                main.memory, 0, SplitSpecs.N_EMBD, SplitSpecs.N_EMBD);
        GpuSplitForwardBackend.Shard two = new GpuSplitForwardBackend.Shard(second.context, second.executor,
                second.memory, 2, SplitSpecs.N_EMBD, SplitSpecs.N_EMBD);
        backend = new GpuSplitForwardBackend(spec, java.util.Arrays.asList(one, two), 0, embedding, head,
                main.memory);
    }

    private static int countKind(List<FakeOp> ops, FakeOpKind kind) {
        int n = 0;
        for (FakeOp op : ops) {
            if (op.kind() == kind) {
                n++;
            }
        }
        return n;
    }

    /** The READ op of the given source buffer, or null. */
    private static FakeOp readOf(List<FakeOp> ops, FakeGpuBuffer source) {
        for (FakeOp op : ops) {
            if (op.kind() == FakeOpKind.READ && source.equals(op.buffer())) {
                return op;
            }
        }
        return null;
    }

    /** The WRITE ops into the given buffer. */
    private static List<FakeOp> writesTo(List<FakeOp> ops, FakeGpuBuffer dst) {
        List<FakeOp> out = new ArrayList<>();
        for (FakeOp op : ops) {
            if (op.kind() == FakeOpKind.WRITE && dst.equals(op.buffer())) {
                out.add(op);
            }
        }
        return out;
    }

    private static int countWriteTo(List<FakeOp> ops, FakeGpuBuffer dst) {
        return writesTo(ops, dst).size();
    }

    // ------------------------------------------------------------------ traces

    @Test
    public void decodeRunsEachLayerOnItsOwnDeviceWithOneHandOffEachWay() {
        openSplit(MemoryKind.KV);

        backend.decode(3, 0);

        long handOffBytes = (long) SplitSpecs.N_EMBD * Float.BYTES;
        // device one: the stream leaves after its layers, and the logits are read back at the end
        List<FakeOp> oneOps = main.backend.ops();
        assertEquals(2, countKind(oneOps, FakeOpKind.READ));
        FakeOp leave = readOf(oneOps, main.xBuffer());
        assertTrue("device one never reads its stream out", leave != null);
        assertEquals(handOffBytes, leave.bytes());
        // device two: the stream arrives as one write of the same size, and is read back for the way home
        List<FakeOp> twoOps = second.backend.ops();
        FakeOp arrive = writesTo(twoOps, second.xBuffer()).get(0);
        assertTrue("device two never receives the stream", arrive != null);
        assertEquals(handOffBytes, arrive.bytes());
        assertEquals(1, countKind(twoOps, FakeOpKind.READ));
        FakeOp home = readOf(twoOps, second.xBuffer());
        assertTrue(home != null);
        assertEquals(handOffBytes, home.bytes());
        // one finish per device
        assertEquals(1, countKind(oneOps, FakeOpKind.FINISH));
        assertEquals(1, countKind(twoOps, FakeOpKind.FINISH));
    }

    @Test
    public void theHeadStaysOnTheMainDevice() {
        openSplit(MemoryKind.KV);

        float[] logits = backend.decode(1, 0);

        assertEquals(SplitSpecs.VOCAB, logits.length);
        // the logits read is on the main queue, never on the second
        assertTrue(countKind(main.backend.ops(), FakeOpKind.READ) >= 2);
    }

    @Test
    public void everyDeviceHoldsItsOwnLayersKernelsOnly() {
        openSplit(MemoryKind.KV);

        backend.decode(2, 0);

        // the row norms of device one use layers 0-1's weights, device two's use 2-3's
        List<FakeDispatch> one = main.dispatches();
        List<FakeDispatch> two = second.dispatches();
        assertTrue(one.stream().anyMatch(d -> d.bufferArgs().contains(
                main.weights.get(blk(0, TensorNames.ATTN_NORM)).shard(0))));
        assertTrue(one.stream().noneMatch(d -> d.bufferArgs().contains(
                second.weights.get(blk(2, TensorNames.ATTN_NORM)).shard(0))));
        assertTrue(two.stream().anyMatch(d -> d.bufferArgs().contains(
                second.weights.get(blk(3, TensorNames.ATTN_NORM)).shard(0))));
        assertTrue(two.stream().noneMatch(d -> d.bufferArgs().contains(
                main.weights.get(blk(0, TensorNames.ATTN_NORM)).shard(0))));
    }

    @Test
    public void memoriesMoveInLockstep() {
        openSplit(MemoryKind.KV);

        backend.decode(1, 0);
        backend.decode(2, 1);
        assertEquals(2, main.memory.sequence().nPast());
        assertEquals(2, second.memory.sequence().nPast());

        assertTrue(backend.canRollback(1));
        backend.rollback(1);
        assertEquals(1, main.memory.sequence().nPast());
        assertEquals(1, second.memory.sequence().nPast());

        backend.reset();
        assertEquals(0, main.memory.sequence().nPast());
        assertEquals(0, second.memory.sequence().nPast());
    }

    @Test
    public void prefillChunksByTheUbatchAndHandsOffPerStep() {
        openSplit(MemoryKind.KV);
        int[] tokens = {1, 2, 3, 4, 5, 6};

        backend.prefill(tokens, 0);

        assertEquals(6, main.memory.sequence().nPast());
        // two ubatch steps, so the stream crosses to device two and back twice
        assertEquals(2, countWriteTo(second.backend.ops(), second.xBuffer()));
    }

    @Test
    public void tapsSeeTheLayersOfBothDevices() {
        openSplit(MemoryKind.KV);
        List<Integer> tapped = new ArrayList<>();
        backend.setHiddenTap(new HiddenTap() {
            @Override
            public boolean wants(int layer) {
                return true;
            }

            @Override
            public void onHidden(int layer, float[] rows, int nRows, int width) {
                tapped.add(layer);
            }
        });

        backend.decode(1, 0);

        assertEquals(java.util.Arrays.asList(0, 1, 2, 3, 4), tapped);
    }

    @Test
    public void aMemorylessSplitRunsTheWholeCallInOneStep() {
        openSplit(MemoryKind.NONE);

        float[][] rows = backend.forwardAll(new int[]{1, 2, 3}, new int[]{2});

        assertEquals(1, rows.length);
        assertEquals(SplitSpecs.VOCAB, rows[0].length);
        assertEquals(1, countWriteTo(second.backend.ops(), second.xBuffer()));
    }

    @Test
    public void hiddenRowsOfAMemorylessSplitComeFromTheMainDevice() {
        openSplit(MemoryKind.NONE);

        float[][] rows = backend.hiddenRows(new int[]{1, 2}, null);

        assertEquals(2, rows.length);
        assertEquals(SplitSpecs.N_EMBD, rows[0].length);
    }

    @Test
    public void callsBeyondTheUbatchAreRefusedWithoutMemory() {
        openSplit(MemoryKind.NONE);

        try {
            backend.forwardAll(new int[UBATCH + 1], null);
            throw new AssertionError("forwardAll beyond the ubatch must fail");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("ubatch"));
        }
    }

    @Test
    public void reevaluatingStoredPositionsRollsEveryDeviceBack() {
        openSplit(MemoryKind.KV);
        backend.decode(1, 0);
        backend.decode(2, 1);

        backend.decode(3, 1);

        assertEquals(2, main.memory.sequence().nPast());
        assertEquals(2, second.memory.sequence().nPast());
    }

    @Test
    public void aPositionGapIsRefusedBeforeAnythingMoves() {
        openSplit(MemoryKind.KV);
        backend.decode(1, 0);

        try {
            backend.decode(1, 5);
            throw new AssertionError("a gap after the stored positions must fail");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("gap"));
        }
        assertEquals(1, main.memory.sequence().nPast());
        assertEquals(1, second.memory.sequence().nPast());
    }

    // ------------------------------------------------------------------ observers

    /** An enabled observer that records step boundaries and the stages a device's context reports. */
    private static final class Recording implements ExecutionObserver {
        final List<String> seen = new ArrayList<>();

        @Override
        public boolean enabled() {
            return true;
        }

        @Override
        public void onStepStart(int pos0, int nTokens) {
            seen.add("step+" + pos0 + "x" + nTokens);
        }

        @Override
        public void onStageStart(int layer, Stage stage) {
            if (stage == Stage.LAYER || stage == Stage.EMBEDDING || stage == Stage.OUTPUT_NORM) {
                seen.add(stage + ":" + layer);
            }
        }

        @Override
        public void onStepEnd(int pos0, int nTokens) {
            seen.add("step-" + pos0 + "x" + nTokens);
        }
    }

    @Test
    public void everyDeviceObservesTheStepAndItsOwnStages() {
        Recording one = new Recording();
        Recording two = new Recording();
        mainObserver = one;
        secondObserver = two;
        openSplit(MemoryKind.KV);

        backend.decode(3, 0);

        assertEquals(java.util.Arrays.asList("step+0x1", "EMBEDDING:-1", "LAYER:0", "LAYER:1", "OUTPUT_NORM:-1",
                "step-0x1"), one.seen);
        assertEquals(java.util.Arrays.asList("step+0x1", "LAYER:2", "LAYER:3", "step-0x1"), two.seen);
    }

    @Test
    public void aDeviceThatDoesNotObserveSeesNoStep() {
        Recording one = new Recording();
        mainObserver = one;
        openSplit(MemoryKind.KV);

        backend.decode(3, 0);

        assertEquals(java.util.Arrays.asList("step+0x1", "EMBEDDING:-1", "LAYER:0", "LAYER:1", "OUTPUT_NORM:-1",
                "step-0x1"), one.seen);
    }
}

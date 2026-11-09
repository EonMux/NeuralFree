package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.resolve.NormResolver;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeOp;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The decode replay wiring over the real forward backend on the capture-capable fake queue: the first
 * one-token step after a shape change captures the trunk and launches the capture once, the following
 * steps replay it as one REPLAY op with no trunk dispatches, a tap or the disabled property run every
 * trunk directly, and a prefill invalidates the capture so the next decode captures again.
 */
public class GpuDecodeReplayTest {

    private static final int UBATCH = 4;
    private static final int N_EMBD = 64;
    private static final int N_FF = 128;
    private static final int VOCAB = 32;
    private static final int N_HEAD = 2;
    private static final int HEAD_DIM = 32;
    private static final int N_CTX = 64;
    private static final float RMS_EPS = 1e-6f;

    FakeComputeBackend backend;
    private DeviceModelMemory memory;
    private GpuOps ops;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private final List<AutoCloseable> closeables = new ArrayList<>();

    @After
    public void tearDown() throws Exception {
        System.clearProperty(GpuDecodeReplay.PROPERTY);
        System.clearProperty(GpuDecodeReplay.WARMUP_PROPERTY);
        System.clearProperty(GpuForwardBackend.CHAIN_REPLAY_PROPERTY);
        for (int i = closeables.size() - 1; i >= 0; i--) {
            closeables.get(i).close();
        }
        if (memory != null) {
            memory.close();
        }
        if (ops != null) {
            ops.close();
        }
        if (placeholders != null) {
            placeholders.close();
        }
        if (scratch != null) {
            scratch.close();
        }
        if (backend != null) {
            backend.close();
        }
    }

    GpuForwardBackend open(boolean capture) {
        if (System.getProperty(GpuDecodeReplay.WARMUP_PROPERTY) == null) {
            // most cases exercise the capture/replay mechanics, not the warmup
            System.setProperty(GpuDecodeReplay.WARMUP_PROPERTY, "0");
        }
        backend = FakeComputeBackend.builder().strictLlmArgs(true).graphCapture(capture).build();
        LlmKernels kernels = LlmKernels.open(backend);
        ModelSpec spec = llama(2);
        scratch = GpuScratch.forModel(backend, spec, UBATCH);
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
        Map<String, GpuWeight> weightMap = new HashMap<>();
        TensorCatalog tc = spec.tensors();
        for (String name : tc.names()) {
            GGUFTensorInfo info = tc.info(name);
            weightMap.put(name, GpuWeight.of(name, GGMLType.F32, info.dims().clone(),
                    backend.allocateBuffer(info.numElements() * Float.BYTES, true)));
        }
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16).ubatch(UBATCH).build();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX);
        memory = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan,
                spec.hparams().mropeTextPositions(), null);
        GpuForwardContext ctx = GpuForwardContext.builder(kernels, ops, weightMap::get, spec).memory(memory).build();
        GpuForwardBackend fb = new GpuForwardBackend(ctx, registry(), Collections.emptyList());
        closeables.add(fb);
        return fb;
    }

    private GpuBlockRegistry registry() {
        GpuMixerFactory mixer = new GpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return MixerKind.ATTENTION;
            }

            @Override
            public GpuMixer create(LayerSpec layer) {
                return new GpuMixer() {
                    @Override
                    public void bind(GpuForwardContext ctx, LayerSpec l) {
                    }

                    @Override
                    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
                    }

                    @Override
                    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
                    }
                };
            }
        };
        GpuFfnFactory ffn = new GpuFfnFactory() {
            @Override
            public FfnKind kind() {
                return FfnKind.DENSE;
            }

            @Override
            public GpuFfnBlock create(LayerSpec layer) {
                return new GpuFfnBlock() {
                    @Override
                    public void bind(GpuForwardContext ctx, LayerSpec l) {
                    }

                    @Override
                    public void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
                    }
                };
            }
        };
        return new GpuBlockRegistry(Collections.singletonList(mixer), Collections.singletonList(ffn),
                BuiltInGpuExtras.factories());
    }

    private long replayOps() {
        return backend.ops().stream().filter(op -> op.kind() == FakeOpKind.REPLAY).count();
    }

    private long launchOps() {
        return backend.ops().stream().filter(op -> op.kind() == FakeOpKind.LAUNCH).count();
    }

    private long dispatchesOf(String kernel) {
        return backend.dispatchesOf(kernel).size();
    }

    private int trunkDispatches() {
        int n = 0;
        for (FakeOp op : backend.ops()) {
            if (op.kind() == FakeOpKind.DISPATCH && op.dispatch().kernelName().equals(NormKernels.ROW_NORM)) {
                n++;
            }
        }
        return n;
    }

    // ------------------------------------------------------------------ fixtures

    private static String blk(int il, String base) {
        return TensorNames.layerName(TensorNames.STACK_BLK, il, TensorNames.weight(base));
    }

    private static GGUFTensorInfo info(String name, long... dims) {
        return new GGUFTensorInfo(name, dims, GGMLType.F32, 0);
    }

    private static List<GGUFTensorInfo> tensors(int nLayer) {
        List<GGUFTensorInfo> t = new ArrayList<>();
        t.add(info(TensorNames.weight(TensorNames.TOKEN_EMBD), N_EMBD, VOCAB));
        t.add(info(TensorNames.weight(TensorNames.OUTPUT_NORM), N_EMBD));
        t.add(info(TensorNames.weight(TensorNames.OUTPUT), N_EMBD, VOCAB));
        for (int il = 0; il < nLayer; il++) {
            t.add(info(blk(il, TensorNames.ATTN_NORM), N_EMBD));
            t.add(info(blk(il, TensorNames.FFN_NORM), N_EMBD));
            t.add(info(blk(il, TensorNames.FFN_GATE), N_EMBD, N_FF));
            t.add(info(blk(il, TensorNames.FFN_UP), N_EMBD, N_FF));
            t.add(info(blk(il, TensorNames.FFN_DOWN), N_FF, N_EMBD));
        }
        return t;
    }

    private static ModelSpec llama(int nLayer) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.LLAMA)).layerCounts(nLayer, 0)
                .nCtxTrain(4096).nEmbd(N_EMBD).nVocab(VOCAB).normRmsEps(RMS_EPS).normEps(1e-5f)
                .headKFull(HEAD_DIM).headVFull(HEAD_DIM);
        for (int il = 0; il < nLayer; il++) {
            hb.nHead(il, N_HEAD).nHeadKv(il, N_HEAD);
        }
        TensorCatalog tc = TensorCatalog.of(hb.build().arch().ggufName(), tensors(nLayer));
        ModelHparams hp = hb.build();
        NormResolver norms = new NormResolver(hp, tc);
        ModelSpec.Builder b = ModelSpec.builder(hp, tc)
                .schedule(LayerSchedule.linear(nLayer))
                .embedding(EmbeddingSpec.builder(tc.get(TensorNames.weight(TensorNames.TOKEN_EMBD)), N_EMBD)
                        .embdScale(1.0f).build())
                .head(HeadSpec.builder()
                        .outputNorm(tc.get(TensorNames.weight(TensorNames.OUTPUT_NORM)), null)
                        .output(tc.get(TensorNames.weight(TensorNames.OUTPUT)), false).build());
        for (int il = 0; il < nLayer; il++) {
            b.layer(LayerSpec.builder(il, MixerKind.ATTENTION, FfnKind.DENSE)
                    .norm(norms.resolve(il))
                    .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                    .attention(AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD).headDimK(HEAD_DIM)
                            .headDimV(HEAD_DIM).noRope(true).nRot(0)
                            .kqScale(1.0f / (float) Math.sqrt(HEAD_DIM)).kvOwnerLayer(il).build())
                    .ffn(FfnSpec.builder(FfnSpec.Role.DENSE).nFf(N_FF).build())
                    .build());
        }
        b.memoryKind(MemoryKind.KV);
        return b.build();
    }

    // ------------------------------------------------------------------ tests

    @Test
    public void theFirstDecodeCapturesAndTheFollowingOnesReplayTheTrunk() {
        GpuForwardBackend fb = open(true);

        fb.decode(1, 0);
        assertEquals(0, replayOps());
        // the capturing step submits its capture once, and only the head's output norm ran outside it
        assertEquals(1, launchOps());
        assertEquals(1, dispatchesOf(NormKernels.ROW_NORM));

        fb.decode(2, 1);
        fb.decode(3, 2);
        assertEquals(2, replayOps());
        // still one norm per step (the head's): no trunk dispatches on the replayed steps
        assertEquals(3, dispatchesOf(NormKernels.ROW_NORM));
        // the capturing step submits its capture once; the trunk reaches the device every step
        assertEquals(1, launchOps());
        assertEquals(2, replayOps());
    }

    @Test
    public void replayedStepsAndCapturedStepsProduceTheSameLogits() {
        GpuForwardBackend replayed = open(true);
        float[][] withReplay = new float[][]{replayed.decode(1, 0), replayed.decode(2, 1), replayed.decode(3, 2)};
        replayed.close();

        GpuForwardBackend direct = open(false);
        float[][] withoutReplay = new float[][]{direct.decode(1, 0), direct.decode(2, 1), direct.decode(3, 2)};
        direct.close();

        for (int i = 0; i < withReplay.length; i++) {
            assertArrayEquals(withoutReplay[i], withReplay[i], 0.0f);
        }
    }

    @Test
    public void aPrefillInvalidatesTheCaptureAndTheNextDecodeCapturesAgain() {
        GpuForwardBackend fb = open(true);

        fb.decode(1, 0);
        fb.decode(2, 1);
        assertEquals(1, replayOps());

        fb.prefill(new int[]{4, 5, 6}, 2);
        assertEquals(1, replayOps());

        fb.decode(7, 5);
        // the invalidation made this step capture again instead of replaying
        assertEquals(1, replayOps());

        fb.decode(8, 6);
        assertEquals(2, replayOps());
    }

    @Test
    public void aTapThatWantsLayersRunsEveryTrunkDirectly() {
        GpuForwardBackend fb = open(true);
        fb.setHiddenTap(new HiddenTap() {
            @Override
            public void onHidden(int layer, float[] rows, int nRows, int width) {
            }

            @Override
            public boolean wants(int layer) {
                return true;
            }
        });

        fb.decode(1, 0);
        fb.decode(2, 1);
        fb.decode(3, 2);

        assertEquals(0, replayOps());
        // every trunk ran directly: the trunk norm, the head's output norm and the tap's row norm per step
        assertEquals(3 * 3, trunkDispatches());
    }

    @Test
    public void aTapThatWantsNothingCaptures() {
        GpuForwardBackend fb = open(true);
        fb.setHiddenTap(HiddenTap.NONE);

        fb.decode(1, 0);
        fb.decode(2, 1);
        assertEquals(1, replayOps());
    }

    @Test
    public void theDisabledPropertyRunsEveryTrunkDirectly() {
        System.setProperty(GpuDecodeReplay.PROPERTY, "false");
        GpuForwardBackend fb = open(true);

        fb.decode(1, 0);
        fb.decode(2, 1);
        fb.decode(3, 2);

        assertEquals(0, replayOps());
        // every trunk ran directly: the trunk norm and the head's output norm per step
        assertEquals(3 * 2, trunkDispatches());
    }

    @Test
    public void aQueueWithoutCaptureSupportRunsEveryTrunkDirectly() {
        GpuForwardBackend fb = open(false);

        fb.decode(1, 0);
        fb.decode(2, 1);
        fb.decode(3, 2);

        assertEquals(0, replayOps());
        assertEquals(3 * 2, trunkDispatches());
    }

    @Test
    public void theMultiRowStepAndARollbackKeepTheirTrunksOnlyWithTheChainReplay() {
        GpuForwardBackend fb = open(true);

        // the memory-backed multi-row step (a chain round's verify) runs directly by default ...
        fb.prefill(new int[]{4, 5, 6}, 0);
        fb.prefill(new int[]{7, 8, 9}, 3);
        assertEquals(0, launchOps());
        assertEquals(0, replayOps());

        // ... and a rollback drops the decode's capture, as before
        fb.decode(1, 6);
        fb.decode(2, 7);
        assertEquals(1, launchOps());
        assertEquals(1, replayOps());
        fb.rollback(1);
        fb.decode(2, 7);
        assertEquals(2, launchOps());
    }

    @Test
    public void theChainReplayKeepsTheRollbackCaptureAndCapturesTheMultiRowStep() {
        System.setProperty(GpuForwardBackend.CHAIN_REPLAY_PROPERTY, "true");
        GpuForwardBackend fb = open(true);

        // the verify-shaped multi-row step captures and the next one replays it
        fb.prefill(new int[]{4, 5, 6}, 0);
        assertEquals(1, launchOps());
        fb.prefill(new int[]{7, 8, 9}, 3);
        assertEquals(1, replayOps());

        // a decode capture survives a rollback: the truncated state reaches the replayed trunk
        fb.decode(1, 6);
        fb.decode(2, 7);
        assertEquals(2, launchOps());
        assertEquals(2, replayOps());
        fb.rollback(1);
        fb.decode(2, 7);
        assertEquals(2, launchOps());
        assertEquals(3, replayOps());
    }

    @Test
    public void theChainReplayPropertyMustBeABoolean() {
        System.setProperty(GpuForwardBackend.CHAIN_REPLAY_PROPERTY, "sometimes");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> GpuForwardBackend.chainReplayConfigured(ComputeBackend.BackendType.CUDA));
        assertTrue(e.getMessage().contains(GpuForwardBackend.CHAIN_REPLAY_PROPERTY));
        assertThrows(IllegalArgumentException.class, GpuForwardBackend::chainReplayConfigured);
        System.setProperty(GpuForwardBackend.CHAIN_REPLAY_PROPERTY, "true");
        assertTrue(GpuForwardBackend.chainReplayConfigured(ComputeBackend.BackendType.CUDA));
        assertTrue("an explicit true is a GPU-only request", GpuForwardBackend.chainReplayConfigured());
        System.setProperty(GpuForwardBackend.CHAIN_REPLAY_PROPERTY, "false");
        assertFalse(GpuForwardBackend.chainReplayConfigured(ComputeBackend.BackendType.CUDA));
        assertFalse("an explicit false is no GPU-only request", GpuForwardBackend.chainReplayConfigured());
        System.clearProperty(GpuForwardBackend.CHAIN_REPLAY_PROPERTY);
        assertTrue("the default is on for CUDA",
                GpuForwardBackend.chainReplayConfigured(ComputeBackend.BackendType.CUDA));
        assertFalse("the default stays off elsewhere",
                GpuForwardBackend.chainReplayConfigured(ComputeBackend.BackendType.OPENCL));
        assertFalse(GpuForwardBackend.chainReplayConfigured(ComputeBackend.BackendType.HIP));
        assertFalse(GpuForwardBackend.chainReplayConfigured(ComputeBackend.BackendType.VULKAN));
        assertFalse("the implicit per-backend default is not a GPU-only request",
                GpuForwardBackend.chainReplayConfigured());
    }

    @Test
    public void aTrunkDeclaredUncapturableRunsEveryStepDirectly() {
        System.setProperty(GpuDecodeReplay.WARMUP_PROPERTY, "0");
        FakeComputeBackend fake = FakeComputeBackend.builder().graphCapture(true).build();
        try (com.eonmux.neuralfree.acceleration.backend.GpuBuffer buffer = fake.allocateBuffer(16, true)) {
            GpuDecodeReplay replay = new GpuDecodeReplay(fake.computeQueue());
            int[] runs = {0};
            Runnable body = () -> {
                runs[0]++;
                fake.computeQueue().enqueueWriteBuffer(buffer, 0, Integer.BYTES, new byte[]{1, 2, 3, 4});
            };
            replay.trunk(false, body);
            assertEquals("an uncapturable trunk runs once, directly", 1, runs[0]);
            replay.trunk(false, body);
            assertEquals("no warmup or capture is attempted", 2, runs[0]);
            replay.close();
        } finally {
            fake.close();
        }
    }

    @Test
    public void aHostTransferInsideACapturedTrunkAbortsTheCaptureAndRunsTheStepDirectly() {
        System.setProperty(GpuDecodeReplay.WARMUP_PROPERTY, "0");
        FakeComputeBackend fake = FakeComputeBackend.builder().graphCapture(true).build();
        try (com.eonmux.neuralfree.acceleration.backend.GpuBuffer buffer = fake.allocateBuffer(16, true)) {
            GpuDecodeReplay replay = new GpuDecodeReplay(fake.computeQueue());
            int[] runs = {0};
            Runnable body = () -> {
                runs[0]++;
                fake.computeQueue().enqueueWriteBuffer(buffer, 0, Integer.BYTES, new byte[]{1, 2, 3, 4});
            };
            // warmup 0 by default in this class: the first eligible step records the trunk, the host write
            // is refused before the driver sees it, and the replay drops the capture and runs the step
            replay.trunk(true, body);
            assertEquals("the aborted capture ran the body again", 2, runs[0]);
            assertFalse("the aborted capture stays open", fake.computeQueue().isCapturing());
            replay.close();
        } finally {
            fake.close();
        }
    }

    @Test
    public void anInvalidPropertyIsRefusedByName() {
        System.setProperty(GpuDecodeReplay.PROPERTY, "maybe");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                GpuDecodeReplayTest::newGpuDecodeReplay);
        assertTrue(e.getMessage(), e.getMessage().contains(GpuDecodeReplay.PROPERTY));
    }

    @Test
    public void aStableShapeIsCapturedOnlyAfterTheWarmupSteps() {
        System.setProperty(GpuDecodeReplay.WARMUP_PROPERTY, "2");
        GpuForwardBackend fb = open(true);

        // two stable decode steps run directly, without recording a graph
        fb.decode(1, 0);
        fb.decode(2, 1);
        assertEquals(0, launchOps());
        assertEquals(0, replayOps());
        assertEquals(2 * 2, trunkDispatches());

        // the third stable step captures the trunk and submits the capture once; only the head's
        // output norm runs outside it
        fb.decode(3, 2);
        assertEquals(1, launchOps());
        assertEquals(0, replayOps());
        assertEquals(2 * 2 + 1, trunkDispatches());

        // the fourth step replays it as one submission: no trunk dispatches
        fb.decode(4, 3);
        assertEquals(1, replayOps());
        assertEquals(2 * 2 + 2, trunkDispatches());
    }

    @Test
    public void aShapeChangeRestartsTheWarmup() {
        System.setProperty(GpuDecodeReplay.WARMUP_PROPERTY, "2");
        GpuForwardBackend fb = open(true);

        fb.decode(1, 0);
        fb.decode(2, 1);
        // a prefill resets the warmup count and closes nothing (no graph was captured yet)
        fb.prefill(new int[]{4, 5}, 2);
        fb.decode(3, 3);
        fb.decode(4, 4);
        assertEquals(0, launchOps());
        assertEquals(0, replayOps());
        // two stable decode steps after the prefill: the next one captures
        fb.decode(5, 5);
        assertEquals(1, launchOps());
        assertEquals(0, replayOps());
    }

    @Test
    public void anInvalidWarmupPropertyIsRefusedByName() {
        System.setProperty(GpuDecodeReplay.WARMUP_PROPERTY, "-1");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                GpuDecodeReplayTest::newGpuDecodeReplay);
        assertTrue(e.getMessage(), e.getMessage().contains(GpuDecodeReplay.WARMUP_PROPERTY));
        System.setProperty(GpuDecodeReplay.WARMUP_PROPERTY, "many");
        e = assertThrows(IllegalArgumentException.class, GpuDecodeReplayTest::newGpuDecodeReplay);
        assertTrue(e.getMessage(), e.getMessage().contains(GpuDecodeReplay.WARMUP_PROPERTY));
    }

    private static void newGpuDecodeReplay() {
        // constructing a replay over a throwaway queue parses the property eagerly
        new GpuDecodeReplay(FakeComputeBackend.create().computeQueue());
    }
}

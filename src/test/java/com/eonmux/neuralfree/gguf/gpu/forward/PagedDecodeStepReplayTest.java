package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.ops.attn.AttnBlocks;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import org.junit.After;
import org.junit.Test;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The batched decode step of the grouped-query paged decode through a step replay ({@link GpuDecodeReplay} with the
 * attention mixer's paged decode patchers, the way the prefill driver runs a plain paged decode step) on the fake
 * backend's updatable graphs, over a llama layer of a paged F16 cache of two sequences that decode from key 1 past
 * the first split boundary (33 keys). Where the launch aims at more than one split, the one-split steps merge their
 * partial too, so the steps on both sides of the boundary replay one capture; where the sequences and KV heads alone
 * fill the device, the plan keeps one split through these keys and every step writes its output directly, without a
 * combine launch.
 */
public class PagedDecodeStepReplayTest {

    private static final int WIDTH = 256;
    private static final int N_HEAD = 4;
    private static final int N_HEAD_KV = 2;
    private static final int HEAD_DIM = 64;
    private static final int Q_DIM = N_HEAD * HEAD_DIM;
    private static final int KV_DIM = N_HEAD_KV * HEAD_DIM;
    private static final int N_CTX = 128;
    private static final int UBATCH = 8;
    private static final int PARALLEL = 2;
    private static final int PAGE = 16;
    /** Keys the last step reads: past the first split boundary of a splitting plan. */
    private static final int LAST_KEYS = 40;

    private FakeComputeBackend backend;
    private DeviceModelMemory memory;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private GpuMixer mixer;
    private GpuDecodeReplay replay;
    /** Combine launches the emulated device ran, directly or inside a replayed graph. */
    private int combines;
    private final Map<String, GpuWeight> tensors = new HashMap<>();

    @After
    public void tearDown() {
        System.clearProperty(GpuDecodeReplay.PROPERTY);
        System.clearProperty(GpuDecodeReplay.WARMUP_PROPERTY);
        if (replay != null) {
            replay.close();
        }
        if (mixer != null) {
            mixer.close();
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
        if (memory != null) {
            memory.close();
        }
        if (backend != null) {
            backend.close();
        }
    }

    /** 32 compute units: two sequences of two KV heads aim at 16 splits, so the keys split from 33 on. */
    @Test
    public void aSplittingPlanReplaysOneCapturePastTheFirstSplitBoundary() {
        bind(32);
        int combines = decodeSteps();
        assertEquals("one capture serves every step", 1, count(FakeOpKind.LAUNCH));
        assertEquals("the steps after it replay", LAST_KEYS - 1, count(FakeOpKind.REPLAY));
        assertEquals("every step merges, the one-split steps included", LAST_KEYS, combines);
    }

    /** One compute unit: the two sequences' four groups fill it, so the plan keeps one split up to 512 keys. */
    @Test
    public void aPlanWhoseGroupsFillTheDeviceWritesEveryOneSplitStepDirectly() {
        bind(1);
        int combines = decodeSteps();
        assertEquals("no step merges", 0, combines);
        assertEquals("one capture serves every step", 1, count(FakeOpKind.LAUNCH));
        assertEquals(LAST_KEYS - 1, count(FakeOpKind.REPLAY));
    }

    /**
     * Decodes one token per sequence at keys 1 through {@link #LAST_KEYS}, each step's trunk (the mixer's batched
     * launches) through the step replay at the step's key count, and returns the combine launches the device ran.
     */
    private int decodeSteps() {
        PagedSequences paged = memory.pagedSequences();
        backend.clearLog();
        for (int keys = 1; keys <= LAST_KEYS; keys++) {
            int pos = keys - 1;
            StepLayout layout = StepLayout.of(PARALLEL, new int[]{0, 1}, new int[]{0, 1}, new int[]{1, 1},
                    new int[]{pos, pos}, new int[0], true);
            PagedStep step = paged.prepare(layout);
            memory.beginPagedStep(step, new int[]{pos, pos}, null);
            replay.trunk(keys, 1, mixer.pagedDecodeReplayPatchers(), true,
                    () -> mixer.batch(layout, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH)));
            memory.endPagedStep();
            paged.commit(step);
            for (int s = 0; s < PARALLEL; s++) {
                paged.appendTokens(s, new int[1]);
            }
        }
        assertTrue("the patcher's plan agrees with the launch, so the steps replay patched: " + replay,
                replay.toString().contains("patched replay"));
        return combines;
    }

    private long count(FakeOpKind kind) {
        return backend.ops().stream().filter(o -> o.kind() == kind).count();
    }

    private static DeviceCapabilities caps(int computeUnits) {
        return new DeviceCapabilities.Builder()
                .deviceName("FakeTensorCores")
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(computeUnits)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .supportsAsyncTransfer(true)
                .driverVersion("fake-1.0")
                .supportsFloat16(true)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(48L * 1024)
                .maxWorkGroupCount(new long[]{65535, 65535, 65535})
                .computeCapability(120)
                .feature(DeviceFeature.COOPERATIVE_MATRIX)
                .feature(DeviceFeature.BF16_COOPERATIVE_MATRIX)
                .deviceClass(DeviceClass.DISCRETE)
                .identity(new DeviceIdentity("FakeTensorCores", FakeComputeBackend.NVIDIA_VENDOR_ID, null, null))
                .build();
    }

    private static ModelSpec llama() {
        AttentionSpec attention = AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD_KV).headDimK(HEAD_DIM)
                .headDimV(HEAD_DIM).nRot(HEAD_DIM).kqScale(0.125f).kvOwnerLayer(0).build();
        LayerSpec l = LayerSpec.builder(0, MixerKind.ATTENTION, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).qkNorm(attention.qkNorm()).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, WIDTH).build())
                .attention(attention)
                .rope(RopeSpec.builder(RopeType.NORM).nDims(HEAD_DIM).freqBase(10000f).build())
                .build();
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.LLAMA)).layerCounts(1, 0)
                .nCtxTrain(512).nEmbd(WIDTH).headKFull(HEAD_DIM).headVFull(HEAD_DIM).nRotFull(HEAD_DIM)
                .normEps(1e-5f).normRmsEps(1e-6f);
        hb.nHead(0, N_HEAD).nHeadKv(0, N_HEAD_KV);
        return ModelSpec.builder(hb.build(), TensorCatalog.of(Arch.LLAMA.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, WIDTH).build())
                .head(HeadSpec.builder().build())
                .layer(l)
                .build();
    }

    private void matrix(LayerSpec l, String base, int cols, int rows) {
        String n = TensorNames.layerName(l.stack(), l.weightLayer(), TensorNames.weight(base));
        tensors.put(n, GpuWeight.of(n, GGMLType.F32, new long[]{cols, rows},
                backend.allocateBuffer((long) cols * rows * Float.BYTES, true)));
    }

    /**
     * Binds the llama mixer over a paged F16 cache of {@link #PARALLEL} sequences on a CUDA tensor-core backend of
     * {@code computeUnits} units with updatable graphs, whose kernels include the grouped-query paged decode, and a
     * step replay that captures the first step.
     */
    private void bind(int computeUnits) {
        System.setProperty(GpuDecodeReplay.WARMUP_PROPERTY, "0");
        System.setProperty(GpuDecodeReplay.PROPERTY, "true");
        backend = FakeComputeBackend.builder().strictLlmArgs(true).backendType(ComputeBackend.BackendType.CUDA)
                .capabilities(caps(computeUnits)).graphCapture(true).graphArgUpdates(true)
                .emulate(AttnKernels.FLASH_COMBINE, d -> combines++).build();
        ModelSpec spec = llama();
        LayerSpec l = spec.layers().get(0);
        matrix(l, TensorNames.ATTN_Q, WIDTH, Q_DIM);
        matrix(l, TensorNames.ATTN_K, WIDTH, KV_DIM);
        matrix(l, TensorNames.ATTN_V, WIDTH, KV_DIM);
        matrix(l, TensorNames.ATTN_OUT, Q_DIM, WIDTH);
        MemoryConfig cfg = MemoryConfig.builder().ubatch(UBATCH).typeK(KvDType.F16).typeV(KvDType.F16)
                .nCtx(N_CTX).parallel(PARALLEL).kvPageSize(PAGE).build();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX);
        memory = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan, MropeTextPositions.PPP0, null);
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.NORMED, (long) UBATCH * WIDTH)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * WIDTH)
                .slot(ScratchSlots.Q, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.K, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.V, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.ATTN_OUT, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.POS, (long) UBATCH * 4)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        LlmKernels kernels = LlmKernels.open(backend);
        assertTrue("the build holds the grouped-query paged decode",
                kernels.has(AttnKernels.flashDecodePagedGqa(KvDType.F16, KvDType.F16)));
        ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
        GpuForwardContext ctx = GpuForwardContext.builder(kernels, ops, tensors::get, spec).memory(memory).build();
        for (GpuMixerFactory f : AttnBlocks.gpuMixers()) {
            if (f.kind() == MixerKind.ATTENTION) {
                mixer = f.create(l);
            }
        }
        mixer.bind(ctx, l);
        replay = new GpuDecodeReplay(backend.computeQueue());
    }
}

package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.FusedMode;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOpsFixture;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.act.FfnBlocks;
import com.eonmux.neuralfree.gguf.ops.fused.FusedKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
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
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The kernels of one Qwen3-shaped decoder layer's single-token decode step (the attention mixer with its QK norms,
 * RoPE, cache store and flash decode, then the SwiGLU FFN) on the Arrow-Lake iGPU's capabilities, over the Vulkan
 * and the OpenCL backends of the host-only fake with strict LLM argument rules. Both backends must run every
 * projection of a type with subgroup-row forms through them ({@code {T}_MatVecSg_Q8} and
 * {@code {T}_MatVecFusedGateUpSg_Q8} after one Q8_1 quantization of the row; the accumulating
 * {@code {T}_MatVecAddSg_Q8}, which this layer does not launch, is {@code Int8MatvecLaneTest}'s) and the attention
 * through {@code LLM_FlashDecodeSg_F16}: those forms are the iGPU's decode speed (the portable matvecs stream its
 * weights at 5-12 GB/s against 41-61), so a selection change that drops one of them on either backend fails here.
 * The same capabilities without the 16-lane subgroup size keep the portable kernels.
 *
 * <p>The OpenCL program builds the forms whether or not the device reports packed dot products (the kernels then
 * emulate dp4a), and a device without them has no decode evidence for the family's integer-dot lane: there every
 * projection takes the lane only because its subgroup twin serves ({@code GpuProjections.int8LaneUsable} and the
 * fused gate/up selection). The Vulkan modules need the packed dot product, so that variant runs on OpenCL only;
 * on Vulkan, Q4_0 is the type whose recorded evidence rejects the family lane
 * ({@link com.eonmux.neuralfree.gguf.gpu.forward.Int8MatvecTuning}).</p>
 */
public class IntelDecodeKernelSetTest {

    private static final int WIDTH = 512;
    private static final int N_HEAD = 4;
    private static final int N_HEAD_KV = 2;
    private static final int HEAD_DIM = 128;
    private static final int Q_DIM = N_HEAD * HEAD_DIM;
    private static final int KV_DIM = N_HEAD_KV * HEAD_DIM;
    private static final int N_FF = 1024;
    private static final int N_CTX = 64;
    private static final int UBATCH = 8;
    private static final int PCI_VENDOR_INTEL = 0x8086;

    private FakeComputeBackend backend;
    private DeviceModelMemory memory;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private GpuMixer mixer;
    private GpuFfnBlock ffn;
    private final Map<String, GpuWeight> tensors = new HashMap<>();

    @After
    public void tearDown() {
        closeRig();
    }

    /** Closes the bound layer and its backend, so a test can bind the next one. */
    private void closeRig() {
        if (ffn != null) {
            ffn.close();
            ffn = null;
        }
        if (mixer != null) {
            mixer.close();
            mixer = null;
        }
        if (ops != null) {
            ops.close();
            ops = null;
        }
        if (placeholders != null) {
            placeholders.close();
            placeholders = null;
        }
        if (scratch != null) {
            scratch.close();
            scratch = null;
        }
        if (memory != null) {
            memory.close();
            memory = null;
        }
        if (backend != null) {
            backend.close();
            backend = null;
        }
    }

    @Test
    public void vulkanDecodeRunsTheSubgroupForms() {
        for (GGMLType type : new GGMLType[]{GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q5_K,
                GGMLType.Q6_K}) {
            assertEquals(type.name(), subgroupStep(type),
                    decodeStep(BackendType.VULKAN, arrowLake(8, 32, true), type));
            closeRig();
        }
    }

    @Test
    public void openClDecodeRunsTheSubgroupForms() {
        for (GGMLType type : new GGMLType[]{GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q5_K,
                GGMLType.Q6_K}) {
            assertEquals(type.name(), subgroupStep(type),
                    decodeStep(BackendType.OPENCL, arrowLake(8, 32, true), type));
            closeRig();
        }
    }

    @Test
    public void openClWithoutPackedDotsStillRunsTheSubgroupForms() {
        for (GGMLType type : new GGMLType[]{GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q5_K,
                GGMLType.Q6_K}) {
            assertEquals(type.name(), subgroupStep(type),
                    decodeStep(BackendType.OPENCL, arrowLake(8, 32, false), type));
            closeRig();
        }
    }

    @Test
    public void withoutSixteenLaneSubgroupsBothBackendsKeepThePortableForms() {
        for (BackendType api : new BackendType[]{BackendType.VULKAN, BackendType.OPENCL}) {
            List<String> step = decodeStep(api, arrowLake(32, 32, true), GGMLType.Q8_0);
            assertTrue(api + " " + step, step.contains(AttnKernels.flashDecode(KvDType.F16)));
            for (String kernel : step) {
                assertFalse(api + " " + step, kernel.contains("Sg"));
            }
            closeRig();
        }
    }

    /**
     * The decode step of a layer of {@code type} weights whose every projection has a subgroup-row form: Q, K and V
     * over one quantization of the normed row, the fused QK norm and RoPE, the K store with its norm and RoPE and
     * the V store, the subgroup flash decode, the output projection, the fused gate/up SwiGLU and the down
     * projection, each over its own row's quantization.
     */
    private static List<String> subgroupStep(GGMLType type) {
        String t = type.name();
        String matvec = t + "_MatVecSg_Q8";
        List<String> s = new ArrayList<>();
        Collections.addAll(s, GpuOps.QUANTIZE_Q8_1, matvec, matvec, matvec,
                FusedKernels.RMS_NORM_ROPE, FusedKernels.NORM_ROPE_STORE, RopeKernels.kvStore(KvDType.F16),
                SubgroupFlashAttention.kernel(KvDType.F16),
                GpuOps.QUANTIZE_Q8_1, matvec,
                GpuOps.QUANTIZE_Q8_1, t + "_MatVecFusedGateUpSg_Q8",
                GpuOps.QUANTIZE_Q8_1, matvec);
        return s;
    }

    /**
     * Arrow-Lake iGPU capabilities whose subgroup sizes span {@code min..max}, reporting packed dot products when
     * {@code packedDots}.
     */
    private static DeviceCapabilities arrowLake(int min, int max, boolean packedDots) {
        DeviceCapabilities.Builder b = new DeviceCapabilities.Builder()
                .deviceName("Intel(R) Graphics (ARL)")
                .identity(new DeviceIdentity("Intel(R) Graphics (ARL)", PCI_VENDOR_INTEL, null, null))
                .deviceClass(DeviceClass.INTEGRATED)
                .totalVramBytes(16L << 30)
                .maxAllocSizeBytes(4L << 30)
                .computeUnits(0)
                .maxClockFrequencyMhz(2350)
                .maxWorkGroupSize(1024)
                .maxLocalMemoryBytes(64L * 1024)
                .driverVersion("fake-1.0")
                .supportsSubgroups(true)
                .subgroupSize(max)
                .minSubgroupSize(min)
                .maxSubgroupSize(max)
                .feature(DeviceFeature.SUBGROUP_ARITHMETIC)
                .feature(DeviceFeature.STORAGE_8BIT)
                .feature(DeviceFeature.STORAGE_16BIT)
                .feature(DeviceFeature.TIMELINE_SEMAPHORE);
        if (packedDots) {
            b.feature(DeviceFeature.INTEGER_DOT_PRODUCT);
        }
        return b.build();
    }

    /** The kernel names of one decode step at position 5 of a layer bound on {@code api} with {@code caps}. */
    private List<String> decodeStep(BackendType api, DeviceCapabilities caps, GGMLType type) {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).backendType(api).capabilities(caps).build();
        LayerSpec layer = layer();
        ModelSpec spec = model(layer);
        tensors.clear();
        matrix(layer, TensorNames.ATTN_Q, type, WIDTH, Q_DIM);
        matrix(layer, TensorNames.ATTN_K, type, WIDTH, KV_DIM);
        matrix(layer, TensorNames.ATTN_V, type, WIDTH, KV_DIM);
        matrix(layer, TensorNames.ATTN_OUT, type, Q_DIM, WIDTH);
        vector(layer, TensorNames.weight(TensorNames.ATTN_Q_NORM), HEAD_DIM);
        vector(layer, TensorNames.weight(TensorNames.ATTN_K_NORM), HEAD_DIM);
        matrix(layer, TensorNames.FFN_GATE, type, WIDTH, N_FF);
        matrix(layer, TensorNames.FFN_UP, type, WIDTH, N_FF);
        matrix(layer, TensorNames.FFN_DOWN, type, N_FF, WIDTH);
        MemoryConfig cfg = MemoryConfig.builder().ubatch(UBATCH).typeK(KvDType.F16).typeV(KvDType.F16).build();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX);
        memory = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan, MropeTextPositions.PPP0, null);
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.X, (long) UBATCH * WIDTH)
                .slot(ScratchSlots.NORMED, (long) UBATCH * WIDTH)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * WIDTH)
                .slot(ScratchSlots.Q, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.K, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.V, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.ATTN_OUT, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.POS, (long) UBATCH * 4)
                .slot(ScratchSlots.FFN_GATE_UP, 4L * UBATCH * N_FF)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        LlmKernels kernels = LlmKernels.open(backend);
        ops = GpuOpsFixture.withFusedMode(backend.computeQueue(), kernels, scratch, placeholders, FusedMode.AUTO);
        GpuForwardContext ctx = GpuForwardContext.builder(kernels, ops, tensors::get, spec).memory(memory).build();
        mixer = mixerFactory().create(layer);
        mixer.bind(ctx, layer);
        GpuFfnFactory ffnFactory = FfnBlocks.gpuFfns().get(0);
        ffn = ffnFactory.create(layer);
        ffn.bind(ctx, layer);

        ((DeviceStepInputs) memory.stepInputs()).upload(5, 1);
        backend.clearLog();
        mixer.decode(5, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH));
        ffn.forward(1, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH));
        List<String> names = new ArrayList<>();
        for (FakeDispatch d : backend.dispatches()) {
            names.add(d.kernelName());
        }
        return names;
    }

    private static LayerSpec layer() {
        AttentionSpec attention = AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD_KV).headDimK(HEAD_DIM)
                .headDimV(HEAD_DIM).nRot(HEAD_DIM).kqScale((float) (1.0 / Math.sqrt(HEAD_DIM))).kvOwnerLayer(0)
                .qkNorm(QkNormKind.RMS_HEAD).build();
        return LayerSpec.builder(0, MixerKind.ATTENTION, FfnKind.DENSE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).qkNorm(attention.qkNorm()).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, WIDTH).build())
                .attention(attention)
                .rope(RopeSpec.builder(RopeType.NEOX).nDims(HEAD_DIM).freqBase(1_000_000f).build())
                .ffn(FfnSpec.builder(FfnSpec.Role.DENSE).nFf(N_FF).glu(FfnGating.PAR, GluOp.SWIGLU).build())
                .build();
    }

    private static ModelSpec model(LayerSpec l) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.QWEN3)).layerCounts(1, 0)
                .nCtxTrain(4096).nEmbd(WIDTH).headKFull(HEAD_DIM).headVFull(HEAD_DIM).nRotFull(HEAD_DIM)
                .normEps(1e-6f).normRmsEps(1e-6f);
        hb.nHead(0, N_HEAD).nHeadKv(0, N_HEAD_KV);
        return ModelSpec.builder(hb.build(), TensorCatalog.of(Arch.QWEN3.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, WIDTH).build())
                .head(HeadSpec.builder().build())
                .layer(l)
                .build();
    }

    private static GpuMixerFactory mixerFactory() {
        for (GpuMixerFactory f : AttnBlocks.gpuMixers()) {
            if (f.kind() == MixerKind.ATTENTION) {
                return f;
            }
        }
        throw new AssertionError("no GPU attention mixer factory");
    }

    private void matrix(LayerSpec l, String base, GGMLType type, int cols, int rows) {
        String n = TensorNames.layerName(l.stack(), l.weightLayer(), TensorNames.weight(base));
        long bytes = (long) rows * (cols / type.getBlockSize()) * type.getTypeSize();
        tensors.put(n, GpuWeight.of(n, type, new long[]{cols, rows}, backend.allocateBuffer(bytes, true)));
    }

    private void vector(LayerSpec l, String suffix, int n) {
        String name = TensorNames.layerName(l.stack(), l.weightLayer(), suffix);
        tensors.put(name, GpuWeight.of(name, GGMLType.F32, new long[]{n},
                backend.allocateBuffer((long) n * Float.BYTES, true)));
    }
}

package com.eonmux.neuralfree.gguf.ops.act;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.FusedMode;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOpsFixture;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
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
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import org.junit.After;
import org.junit.Before;
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
 * Which form a llama SwiGLU dense FFN takes for a prompt on a CUDA tensor-tile device with Q8_0 weights: the fused
 * stream-k gate/up GEMM, which writes {@code silu(g) * u} itself, at a prompt-sized row count, and the separate
 * gate and up products followed by {@code LLM_GLU} while a LoRA delta sits on either product (the fused kernel
 * computes the bare {@code W x}) or at a row count the fused GEMM does not serve. The fused-kernel mode is the
 * default {@code auto}; {@code off} refuses every fused form.
 */
public class GpuDenseFfnPrefillFusionTest {

    private static final int N_EMBD = 512;
    private static final int N_FF = 320;
    private static final int UBATCH = 256;
    private static final int PROMPT = 200;

    private final Map<String, GpuWeight> tensors = new HashMap<>();
    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private GpuForwardContext ctx;
    private GpuDenseFfn block;
    private LayerSpec layer;

    @Before
    public void setUp() {
        backend = FakeComputeBackend.builder().backendType(ComputeBackend.BackendType.CUDA).strictLlmArgs(true)
                .capabilities(cudaTensorTile()).build();
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.NORMED, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.BRANCH, 2L * UBATCH * N_EMBD)
                .slot(ScratchSlots.FFN_GATE_UP, 4L * UBATCH * N_FF)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = GpuOpsFixture.withFusedMode(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders,
                FusedMode.AUTO);
        q80("ffn_gate.weight", N_EMBD, N_FF);
        q80("ffn_up.weight", N_EMBD, N_FF);
        q80("ffn_down.weight", N_FF, N_EMBD);
        layer = LayerSpec.builder(0, MixerKind.NONE, FfnKind.DENSE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-5f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .ffn(FfnSpec.builder(FfnSpec.Role.DENSE).nFf(N_FF).glu(FfnGating.PAR, GluOp.SWIGLU).build())
                .build();
        ctx = context();
        GpuFfnFactory f = FfnBlocks.gpuFfns().get(0);
        block = (GpuDenseFfn) f.create(layer);
        block.bind(ctx, layer);
    }

    @After
    public void tearDown() {
        for (AutoCloseable c : new AutoCloseable[]{block, ops, placeholders, scratch, backend}) {
            if (c != null) {
                try {
                    c.close();
                } catch (Exception e) {
                    throw new IllegalStateException(e);
                }
            }
        }
    }

    @Test
    public void aPromptRunsTheFusedGateUpGemmWithoutTheGluPass() {
        List<String> names = names(run(PROMPT));
        assertTrue(names.toString(), names.contains(GemmKernels.matmulMmqStreamKGateUp(GGMLType.Q8_0)));
        assertFalse(names.toString(), names.contains(ActKernels.GLU));
    }

    @Test
    public void aDeltaOnEitherProductKeepsThePromptOnTheSeparateProducts() {
        for (String product : new String[]{TensorNames.FFN_GATE, TensorNames.FFN_UP}) {
            int[] applied = {0};
            GpuProjectionDeltas.Delta delta = (scale, n, x, xOff, xStride, y, yOff, yStride) -> applied[0] += n;
            ctx.projectionDeltas().register(layer.tensorName(TensorNames.weight(product)), delta);
            List<String> names = names(run(PROMPT));
            assertFalse(product + ": " + names,
                    names.contains(GemmKernels.matmulMmqStreamKGateUp(GGMLType.Q8_0)));
            assertTrue(product + ": " + names, names.contains(ActKernels.GLU));
            assertTrue(product + ": the delta applies to the prompt's rows", applied[0] >= PROMPT);
        }
    }

    @Test
    public void aSmallBatchKeepsTheSeparateProducts() {
        List<String> names = names(run(4));
        assertFalse(names.toString(), names.contains(GemmKernels.matmulMmqStreamKGateUp(GGMLType.Q8_0)));
        assertTrue(names.toString(), names.contains(ActKernels.GLU));
    }

    private List<FakeDispatch> run(int nTokens) {
        backend.clearLog();
        block.forward(nTokens, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH));
        return backend.dispatches();
    }

    private static List<String> names(List<FakeDispatch> ds) {
        List<String> out = new ArrayList<>();
        for (FakeDispatch d : ds) {
            out.add(d.kernelName());
        }
        return out;
    }

    private void q80(String suffix, int cols, int rows) {
        String name = TensorNames.layerName(0, suffix);
        tensors.put(name, GpuWeight.of(name, GGMLType.Q8_0, new long[]{cols, rows},
                backend.allocateBuffer((long) rows * cols / 32 * 34, true)));
    }

    private GpuForwardContext context() {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(Arch.LLAMA)).layerCounts(1, 0).nCtxTrain(4096)
                .nEmbd(N_EMBD).build();
        ModelSpec spec = ModelSpec.builder(hp, TensorCatalog.of(Arch.LLAMA.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build())
                .layer(layer)
                .build();
        GpuWeights weights = new GpuWeights() {
            @Override
            public GpuWeight find(String name) {
                return tensors.get(name);
            }
        };
        return GpuForwardContext.builder(ops.kernels(), ops, weights, spec).build();
    }

    /** An sm_120 device with both matrix features, as the CUDA backend reports one. */
    private static DeviceCapabilities cudaTensorTile() {
        return new DeviceCapabilities.Builder()
                .deviceName("TensorDevice")
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(60)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .driverVersion("fake-1.0")
                .supportsFloat16(true)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(48L * 1024)
                .computeCapability(120)
                .feature(DeviceFeature.COOPERATIVE_MATRIX)
                .feature(DeviceFeature.BF16_COOPERATIVE_MATRIX)
                .deviceClass(DeviceClass.DISCRETE)
                .identity(new DeviceIdentity("TensorDevice", FakeComputeBackend.NVIDIA_VENDOR_ID, null, null))
                .build();
    }
}

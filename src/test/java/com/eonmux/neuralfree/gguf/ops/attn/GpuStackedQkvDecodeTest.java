package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
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
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
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
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The attention layer's Q, K and V products on a faked Vulkan NVIDIA device ({@link GpuQkvProjections}): a
 * single-token decode step of three projections of one type runs them as one stacked {@code {T}_MatVecQKV} launch
 * into the layer's Q, K and V rows, and each projection's LoRA delta then adds into its own rows, after that launch;
 * a prompt, mixed weight types and a layer that reads another layer's K and V keep the separate products.
 */
public class GpuStackedQkvDecodeTest {

    private static final int WIDTH = 256;
    private static final int N_HEAD = 4;
    private static final int N_HEAD_KV = 2;
    private static final int HEAD_DIM = 64;
    private static final int Q_DIM = N_HEAD * HEAD_DIM;
    private static final int KV_DIM = N_HEAD_KV * HEAD_DIM;
    private static final int N_CTX = 64;
    private static final int UBATCH = 8;
    private static final String STACKED = "Q8_0_MatVecQKV";

    private FakeComputeBackend backend;
    private DeviceModelMemory memory;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private GpuMixer mixer;
    private GpuForwardContext ctx;
    private Map<String, GpuWeight> tensors;

    @After
    public void tearDown() {
        for (AutoCloseable c : new AutoCloseable[]{mixer, ops, placeholders, scratch, memory, backend}) {
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
    public void aDecodeStepRunsTheThreeProductsAsOneStackedLaunchIntoTheLayerRows() {
        ModelSpec spec = model(1, 0);
        bind(spec, 0, GGMLType.Q8_0, GGMLType.Q8_0);
        List<FakeDispatch> ds = decode(5);

        List<FakeDispatch> stacked = named(ds, STACKED);
        assertEquals(names(ds).toString(), 1, stacked.size());
        assertFalse(names(ds).toString(), names(ds).contains(GpuOps.matVecKernel(GGMLType.Q8_0)));
        FakeDispatch d = stacked.get(0);
        assertEquals(Q_DIM, d.intArg(0));
        assertEquals(WIDTH, d.intArg(1));
        assertEquals(KV_DIM, d.intArg(2));
        assertEquals(KV_DIM, d.intArg(3));
        assertSame(weight(0, TensorNames.ATTN_Q).shard(0), d.bufferArg(4));
        assertSame(weight(0, TensorNames.ATTN_K).shard(0), d.bufferArg(5));
        assertSame(weight(0, TensorNames.ATTN_V).shard(0), d.bufferArg(6));
        assertSame(scratch.slot(ScratchSlots.Q).buffer(), d.bufferArg(8));
        assertSame(scratch.slot(ScratchSlots.K).buffer(), d.bufferArg(9));
        assertSame(scratch.slot(ScratchSlots.V).buffer(), d.bufferArg(10));
    }

    /** Each delta adds one token row into its own projection's rows once the stacked launch has been queued. */
    @Test
    public void eachDeltaAddsIntoItsOwnRowsAfterTheStackedLaunch() {
        ModelSpec spec = model(1, 0);
        bind(spec, 0, GGMLType.Q8_0, GGMLType.Q8_0);
        List<String> order = new ArrayList<>();
        String[] products = {TensorNames.ATTN_Q, TensorNames.ATTN_K, TensorNames.ATTN_V};
        String[] slots = {ScratchSlots.Q, ScratchSlots.K, ScratchSlots.V};
        for (int i = 0; i < products.length; i++) {
            String product = products[i];
            GpuBuffer rows = scratch.slot(slots[i]).buffer();
            GpuProjectionDeltas.Delta delta = (scale, n, x, xOff, xStride, y, yOff, yStride) -> {
                assertEquals(product, 1, n);
                assertSame(product + " adds into its own rows", rows, y);
                assertEquals(product, 0, yOff);
                assertEquals(product + " after the stacked launch", 1, named(backend.dispatches(), STACKED).size());
                order.add(product);
            };
            ctx.projectionDeltas().register(spec.layers().get(0).tensorName(TensorNames.weight(product)), delta);
        }
        List<FakeDispatch> ds = decode(5);

        assertEquals(1, named(ds, STACKED).size());
        assertEquals(java.util.Arrays.asList(products), order);
    }

    @Test
    public void aPromptKeepsTheSeparateProducts() {
        ModelSpec spec = model(1, 0);
        bind(spec, 0, GGMLType.Q8_0, GGMLType.Q8_0);
        steps().upload(0, 3);
        backend.clearLog();
        mixer.prefill(0, 3, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH));
        List<FakeDispatch> ds = backend.dispatches();
        assertTrue(names(ds).toString(), named(ds, STACKED).isEmpty());
    }

    @Test
    public void mixedWeightTypesKeepTheSeparateProducts() {
        ModelSpec spec = model(1, 0);
        bind(spec, 0, GGMLType.Q8_0, GGMLType.F32);
        List<FakeDispatch> ds = decode(5);
        assertTrue(names(ds).toString(), named(ds, STACKED).isEmpty());
        assertEquals(names(ds).toString(), 2, named(ds, GpuOps.matVecKernel(GGMLType.Q8_0)).size()
                + named(ds, pairedMatvec(GGMLType.Q8_0)).size());
    }

    /** A layer that reads layer 0's K and V cache projects Q alone (no K or V weight): no stacked launch. */
    @Test
    public void aLayerSharingAnotherLayersCacheKeepsItsQProduct() {
        ModelSpec spec = model(2, 0);
        bind(spec, 1, GGMLType.Q8_0, GGMLType.Q8_0);
        List<FakeDispatch> ds = decode(5);
        assertTrue(names(ds).toString(), named(ds, STACKED).isEmpty());
        assertEquals(names(ds).toString(), 1, named(ds, GpuOps.matVecKernel(GGMLType.Q8_0)).size()
                + named(ds, pairedMatvec(GGMLType.Q8_0)).size());
    }

    // ================================================================== rig

    private List<FakeDispatch> decode(int pos) {
        steps().upload(pos, 1);
        backend.clearLog();
        mixer.decode(pos, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH));
        return backend.dispatches();
    }

    private DeviceStepInputs steps() {
        return (DeviceStepInputs) memory.stepInputs();
    }

    private static String pairedMatvec(GGMLType type) {
        return GpuOps.matVecKernel(type).replace("MatVec", "MatVec2");
    }

    private static List<FakeDispatch> named(List<FakeDispatch> ds, String kernel) {
        List<FakeDispatch> out = new ArrayList<>();
        for (FakeDispatch d : ds) {
            if (d.kernelName().equals(kernel)) {
                out.add(d);
            }
        }
        return out;
    }

    private static List<String> names(List<FakeDispatch> ds) {
        List<String> out = new ArrayList<>();
        for (FakeDispatch d : ds) {
            out.add(d.kernelName());
        }
        return out;
    }

    /**
     * {@code layers} attention layers; every layer from 1 up reads the K and V cache of layer {@code kvOwner}.
     */
    private static ModelSpec model(int layers, int kvOwner) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.LLAMA)).layerCounts(layers, 0)
                .nCtxTrain(512).nEmbd(WIDTH).headKFull(HEAD_DIM).headVFull(HEAD_DIM).nRotFull(HEAD_DIM)
                .normEps(1e-5f).normRmsEps(1e-6f);
        List<LayerSpec> specs = new ArrayList<>();
        for (int il = 0; il < layers; il++) {
            hb.nHead(il, N_HEAD).nHeadKv(il, N_HEAD_KV);
            AttentionSpec attention = AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD_KV).headDimK(HEAD_DIM)
                    .headDimV(HEAD_DIM).nRot(HEAD_DIM).kqScale(0.125f).kvOwnerLayer(il == 0 ? 0 : kvOwner).build();
            specs.add(LayerSpec.builder(il, MixerKind.ATTENTION, FfnKind.NONE)
                    .norm(NormSpec.builder(NormKind.RMS, 1e-6f).qkNorm(attention.qkNorm()).build())
                    .residual(ResidualSpec.builder(ResidualKind.STANDARD, WIDTH).build())
                    .attention(attention)
                    .rope(RopeSpec.builder(RopeType.NORM).nDims(HEAD_DIM).freqBase(10000f).build())
                    .build());
        }
        ModelSpec.Builder b = ModelSpec.builder(hb.build(), TensorCatalog.of(Arch.LLAMA.ggufName(),
                        Collections.emptyList()))
                .schedule(LayerSchedule.linear(layers))
                .embedding(EmbeddingSpec.builder(null, WIDTH).build())
                .head(HeadSpec.builder().build());
        for (LayerSpec l : specs) {
            b.layer(l);
        }
        return b.build();
    }

    /** Binds the mixer of layer {@code il} on a Vulkan NVIDIA device; Q and K take {@code qk}, V takes {@code v}. */
    private void bind(ModelSpec spec, int il, GGMLType qk, GGMLType v) {
        backend = FakeComputeBackend.builder().backendType(BackendType.VULKAN).strictLlmArgs(true)
                .capabilities(nvidiaDiscrete()).build();
        tensors = new HashMap<>();
        for (LayerSpec l : spec.layers()) {
            matrix(l, TensorNames.ATTN_Q, qk, Q_DIM);
            if (!l.attention().sharesKv(l.il())) {
                matrix(l, TensorNames.ATTN_K, qk, KV_DIM);
                matrix(l, TensorNames.ATTN_V, v, KV_DIM);
            }
            matrix(l, TensorNames.ATTN_OUT, GGMLType.F32, WIDTH);
        }
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
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        LlmKernels kernels = LlmKernels.open(backend);
        ops = GpuOpsFixture.withFusedMode(backend.computeQueue(), kernels, scratch, placeholders, FusedMode.AUTO);
        ctx = GpuForwardContext.builder(kernels, ops, tensors::get, spec).memory(memory).build();
        LayerSpec l = spec.layers().get(il);
        mixer = factory().create(l);
        mixer.bind(ctx, l);
    }

    private static GpuMixerFactory factory() {
        for (GpuMixerFactory f : AttnBlocks.gpuMixers()) {
            if (f.kind() == MixerKind.ATTENTION) {
                return f;
            }
        }
        throw new AssertionError("no GPU attention mixer factory");
    }

    private void matrix(LayerSpec l, String base, GGMLType type, int rows) {
        int cols = base.equals(TensorNames.ATTN_OUT) ? Q_DIM : WIDTH;
        String n = TensorNames.layerName(l.stack(), l.weightLayer(), TensorNames.weight(base));
        long bytes = (long) rows * (cols / type.getBlockSize()) * type.getTypeSize();
        tensors.put(n, GpuWeight.of(n, type, new long[]{cols, rows}, backend.allocateBuffer(bytes, true)));
    }

    private GpuWeight weight(int il, String base) {
        return tensors.get(TensorNames.layerName(il, TensorNames.weight(base)));
    }

    /** An RTX-class discrete NVIDIA Vulkan device with the packed integer dot product. */
    private static DeviceCapabilities nvidiaDiscrete() {
        return new DeviceCapabilities.Builder()
                .deviceName("QkvDevice")
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(60)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .driverVersion("fake-1.0")
                .supportsFloat16(true)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(48L * 1024)
                .feature(DeviceFeature.INTEGER_DOT_PRODUCT)
                .feature(DeviceFeature.STORAGE_8BIT)
                .feature(DeviceFeature.STORAGE_16BIT)
                .feature(DeviceFeature.TIMELINE_SEMAPHORE)
                .deviceClass(DeviceClass.DISCRETE)
                .identity(new DeviceIdentity("QkvDevice", FakeComputeBackend.NVIDIA_VENDOR_ID, null, null))
                .build();
    }
}

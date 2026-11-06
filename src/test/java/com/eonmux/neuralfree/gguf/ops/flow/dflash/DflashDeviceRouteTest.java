package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.Closeables;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The dflash drafter's bind routes ({@link DflashDrafter#onDevice}): a GPU target drafts through the device stack
 * ({@link DflashDevice}, {@link GpuDflashBackbone} over a {@link GpuDflashKvCache}) and a CPU target through the host
 * backbone ({@link DflashBackbone} over a {@link DflashKvCache}); both draft real tokens over the target's forwards,
 * a draft the device refuses reports the reason by name, and the backbones of the same weights fed identical features
 * agree on the block values over F16, Q8_0 and Q4_0 caches. GPU tests are skipped when OpenCL is unavailable.
 */
public class DflashDeviceRouteTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26};
    private static final int[] TARGET_LAYERS = {0, 2};
    private static final int BLOCK = 8;
    private static final float F16_TOL = 2e-2f;
    private static final float Q8_TOL = 5e-3f;
    private static final float Q4_TOL = 2e-2f;

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    private final List<Path> files = new ArrayList<>();
    private Path target;
    private Path draft;

    @Before
    public void writeModels() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        target = written(SyntheticModels.llama(c.toBuilder().nLayer(4).build()));
        draft = written(SyntheticModels.dflashDrafter(c.toBuilder().nLayer(2).build(), TARGET_LAYERS, BLOCK));
    }

    @After
    public void cleanUp() throws IOException {
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
    }

    private Path written(GgufTestWriter w) throws IOException {
        Path p = w.writeTempFile();
        files.add(p);
        return p;
    }

    private static EngineConfig.Builder gpu() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(MemoryConfig.defaults().toBuilder().nCtx(128).ubatch(32).build());
    }

    private static EngineConfig.Builder cpu() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.defaults().toBuilder().nCtx(128).ubatch(32).build());
    }

    @Test
    public void aGpuTargetTakesTheDeviceStackAndDrafts() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        try (GGUFModelGPU gpuTarget = new GGUFModelGPU(target.toString(), null, gpu().build())) {
            ModelSource dp = ModelSources.open(draft.toString());
            List<AutoCloseable> owned = upload(gpuTarget, dp, DflashDrafter.Options.defaults());
            DflashDrafter d = null;
            try {
                d = DflashDrafter.open(dp, null, DflashDrafter.Options.defaults(), new DflashDevice(
                        gpuTarget.forwardContext(), (GpuWeightUploader) owned.get(owned.size() - 1)));
                assertTrue("a GPU target binds the device stack", d.onDevice());
                d.bindTarget(gpuTarget);
                gpuTarget.forwardBackend().prefill(PROMPT, 0);
                int[] drafts = d.draft(PROMPT[PROMPT.length - 1], 4);
                assertTrue("the device stack drafts tokens", drafts.length > 0);
                assertTrue(drafts.length <= 4);
                for (int token : drafts) {
                    assertTrue(token >= 0 && token < d.spec().nVocab());
                }
            } finally {
                closeAll(d, owned, dp);
            }
        }
    }

    @Test
    public void aCpuTargetKeepsTheHostBackbone() throws IOException {
        try (GGUFModel cpuTarget = new GGUFModel(target.toString(), cpu().build())) {
            ModelSource dp = ModelSources.open(draft.toString());
            DflashDrafter d = null;
            try {
                d = DflashDrafter.open(dp, null, DflashDrafter.Options.defaults());
                assertFalse("a CPU target keeps the host backbone", d.onDevice());
                d.bindTarget(cpuTarget);
                cpuTarget.forwardBackend().prefill(PROMPT, 0);
                int[] drafts = d.draft(PROMPT[PROMPT.length - 1], 4);
                assertTrue("the host backbone drafts tokens", drafts.length > 0);
                for (int token : drafts) {
                    assertTrue(token >= 0 && token < d.spec().nVocab());
                }
            } finally {
                Closeables.close(d, Closeables.close(dp, null));
            }
        }
    }

    /** A quantized cache over the synthetic head size of 16 is refused, naming the head size. */
    @Test
    public void aRefusedCacheNamesItsField() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        try (GGUFModelGPU gpuTarget = new GGUFModelGPU(target.toString(), null, gpu().build())) {
            ModelSource dp = ModelSources.open(draft.toString());
            List<AutoCloseable> owned = upload(gpuTarget, dp, DflashDrafter.Options.of(0, 0.0f, KvDType.Q8_0, true));
            try {
                GpuWeightUploader uploader = (GpuWeightUploader) owned.get(owned.size() - 1);
                UnsupportedOperationException refused = assertThrows(UnsupportedOperationException.class,
                        () -> DflashDrafter.open(dp, null, DflashDrafter.Options.of(0, 0.0f, KvDType.Q8_0, true),
                                new DflashDevice(gpuTarget.forwardContext(), uploader)));
                assertTrue("the refusal names the head size: " + refused.getMessage(),
                        refused.getMessage().contains("multiple of 32"));
            } finally {
                closeAll(null, owned, dp);
            }
        }
    }

    /**
     * A sliding-window causal draft over a wide context: layer 0 keeps a ring of {@code nSwa + 2 * block} padded rows
     * while layer 1 keeps every position, and the device stack drafts over both row schemes.
     */
    @Test
    public void aSlidingWindowDraftKeepsARingBesideFullLayers() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path swaDraft = written(SyntheticModels.dflashDrafter(
                SyntheticModels.Config.llama().toBuilder().nLayer(2).build(), TARGET_LAYERS, BLOCK, true, 16,
                new boolean[] {true, false}));
        EngineConfig wide = EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(MemoryConfig.defaults().toBuilder().nCtx(512).ubatch(32).build()).build();
        try (GGUFModelGPU gpuTarget = new GGUFModelGPU(target.toString(), null, wide)) {
            ModelSource dp = ModelSources.open(swaDraft.toString());
            List<AutoCloseable> owned = upload(gpuTarget, dp, DflashDrafter.Options.defaults());
            DflashDrafter d = null;
            try {
                d = DflashDrafter.open(dp, null, DflashDrafter.Options.defaults(), new DflashDevice(
                        gpuTarget.forwardContext(), (GpuWeightUploader) owned.get(owned.size() - 1)));
                assertTrue(d.onDevice());
                d.bindTarget(gpuTarget);
                assertTrue(d.spec().isSwa(0));
                assertFalse(d.spec().isSwa(1));
                gpuTarget.forwardBackend().prefill(PROMPT, 0);
                int[] drafts = d.draft(PROMPT[PROMPT.length - 1], 4);
                assertTrue("the sliding-window draft drafts tokens", drafts.length > 0);
            } finally {
                closeAll(d, owned, dp);
            }
        }
    }

    /**
     * A DFlash2 draft drafts on the device through its convolution stack and the selector readout: the same
     * synthetic draft opened twice over GPU targets of the same weights — once on the host backbone, once on the
     * device stack — drafts the same ids, with and without a {@code p_min} stop.
     */
    @Test
    public void aDflash2DraftDraftsTheHostsIdsOnTheDevice() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        SyntheticModels.Config c = SyntheticModels.Config.llama().toBuilder().nLayer(2).build();
        Path d2 = written(SyntheticModels.dflashDrafter(c, TARGET_LAYERS, BLOCK, false, 0, null,
                new SyntheticModels.Dflash2(2, 4, 8, 4)));
        for (float pMin : new float[] {0.0f, 0.3f}) {
            DflashDrafter.Options options = pMin == 0.0f ? DflashDrafter.Options.defaults()
                    : DflashDrafter.Options.of(0, pMin, KvDType.F16);
            int[] host = draftsOn(target, d2, options, false);
            int[] device = draftsOn(target, d2, options, true);
            if (pMin == 0.0f) {
                assertTrue("the host DFlash2 draft drafts tokens, got " + host.length, host.length > 0);
            }
            // a p_min above 0 stops before the first row whose top probability falls below it, which over random
            // weights can be the first row: only the p_min-free round asserts that tokens are drafted at all
            assertArrayEquals("the device draft ids at p_min " + pMin, host, device);
        }
    }

    /** The draft ids of one drafter over a fresh GPU target of {@code targetFile}, on the host or on the device. */
    private int[] draftsOn(Path targetFile, Path draftFile, DflashDrafter.Options options, boolean device)
            throws IOException {
        try (GGUFModelGPU gpuTarget = new GGUFModelGPU(targetFile.toString(), null, gpu().build())) {
            ModelSource dp = ModelSources.open(draftFile.toString());
            List<AutoCloseable> owned = device ? upload(gpuTarget, dp, options) : new ArrayList<>();
            DflashDrafter d = null;
            try {
                d = device
                        ? DflashDrafter.open(dp, null, options, new DflashDevice(gpuTarget.forwardContext(),
                                (GpuWeightUploader) owned.get(owned.size() - 1)))
                        : DflashDrafter.open(dp, null, options);
                assertEquals(device, d.onDevice());
                d.bindTarget(gpuTarget);
                gpuTarget.forwardBackend().prefill(PROMPT, 0);
                return d.draft(PROMPT[PROMPT.length - 1], 4);
            } finally {
                closeAll(d, owned, dp);
            }
        }
    }

    /**
     * The host and device backbones of the same draft weights produce the same block values over the same prompt,
     * within the rounding difference of the two float paths (F16 cache).
     */
    @Test
    public void bothBackbonesProduceTheSameBlockValues() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        float[] features = capturedFeatures();
        DflashStack.Block host = hostBlock(features, draft, DflashDrafter.Options.defaults());
        DflashStack.Block device = deviceBlock(features, draft, DflashDrafter.Options.defaults());
        compare("the two backbones agree on the block logits", host.logits(), device.logits(), F16_TOL);
        compare("the two backbones agree on the block t_embd", host.tEmbd(), device.tEmbd(), F16_TOL);
    }

    /** A quantized draft cache over a head size of 32: device and host block values agree within its rounding. */
    @Test
    public void aQuantizedDraftCacheMatchesTheHostBackbone() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path wideDraft = written(wideHeadDraft());
        float[] features = capturedFeatures();
        DflashStack.Block host = hostBlock(features, wideDraft, DflashDrafter.Options.of(0, 0.0f, KvDType.Q8_0));
        DflashStack.Block device = deviceBlock(features, wideDraft, DflashDrafter.Options.of(0, 0.0f, KvDType.Q8_0));
        compare("the quantized device cache matches the host backbone logits", host.logits(), device.logits(),
                Q8_TOL);
        compare("the quantized device cache matches the host backbone t_embd", host.tEmbd(), device.tEmbd(), Q8_TOL);
    }

    /** A Q4_0 draft cache over a head size of 32: device and host block values agree within its rounding. */
    @Test
    public void aQ4_0DraftCacheMatchesTheHostBackbone() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path wideDraft = written(wideHeadDraft());
        float[] features = capturedFeatures();
        DflashStack.Block host = hostBlock(features, wideDraft, DflashDrafter.Options.of(0, 0.0f, KvDType.Q4_0));
        DflashStack.Block device = deviceBlock(features, wideDraft, DflashDrafter.Options.of(0, 0.0f, KvDType.Q4_0));
        compare("the Q4_0 device cache matches the host backbone logits", host.logits(), device.logits(), Q4_TOL);
        compare("the Q4_0 device cache matches the host backbone t_embd", host.tEmbd(), device.tEmbd(), Q4_TOL);
    }

    /**
     * A Q8_0 draft cache over a head size of 64 carries the Hadamard rotation: the device rotates Q, K and V before
     * the store and the attention output back ({@code GpuOps#hadamard}) like the host cache, and the block values
     * agree within the quantization rounding.
     */
    @Test
    public void aRotatedQuantizedDraftCacheMatchesTheHostBackbone() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path rotatedDraft = written(SyntheticModels.dflashDrafter(
                SyntheticModels.Config.llama().toBuilder().nLayer(2).headDim(64).nRot(64).build(), TARGET_LAYERS,
                BLOCK));
        float[] features = capturedFeatures();
        DflashDrafter.Options q8 = DflashDrafter.Options.of(0, 0.0f, KvDType.Q8_0);
        DflashStack.Block host = hostBlock(features, rotatedDraft, q8);
        DflashStack.Block device = deviceBlock(features, rotatedDraft, q8);
        assertNotNull("the host cache rotates K", hostRotationK(rotatedDraft, q8));
        compare("the rotated device cache matches the host backbone logits", host.logits(), device.logits(), Q4_TOL);
        compare("the rotated device cache matches the host backbone t_embd", host.tEmbd(), device.tEmbd(), Q4_TOL);
    }

    /** The K rotation of a host cache over {@code draftFile}, null when the rows are stored unrotated. */
    private static HadamardSpec hostRotationK(Path draftFile, DflashDrafter.Options options) throws IOException {
        ModelSource dp = ModelSources.open(draftFile.toString());
        try {
            GgufKv kv = GgufKv.of(dp);
            DflashSpec spec = DflashDrafter.readSpec(dp);
            DflashKvCache cache = new DflashKvCache(spec.blockSize() + 1, spec.blockSize(), spec.nSwa(),
                    swaLayers(spec), spec.causal(), spec.dsv4(), kvHeads(spec), spec.headDim(), options.kvType(),
                    options.kvRotation());
            return cache.rotationK();
        } finally {
            dp.close();
        }
    }

    /** The head-32 draft the quantized cache tests run over. */
    private GgufTestWriter wideHeadDraft() {
        return SyntheticModels.dflashDrafter(
                SyntheticModels.Config.llama().toBuilder().nLayer(2).headDim(32).nRot(32).build(), TARGET_LAYERS,
                BLOCK);
    }

    /**
     * The draft's uploaded stack resources beside {@code gpuTarget}: the mapped store and the uploader
     * ({@link DflashDevice#reads}), in closing order, the uploader last.
     */
    private static List<AutoCloseable> upload(GGUFModelGPU gpuTarget, ModelSource dp, DflashDrafter.Options options)
            throws IOException {
        GpuForwardContext ctx = gpuTarget.forwardContext();
        GgufKv kv = GgufKv.of(dp);
        TensorCatalog catalog = TensorCatalog.of(dp);
        WeightStore store = dp.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        GpuWeightUploader uploader = new GpuWeightUploader(gpuTarget.backend(), ctx.queue(), dp.getTensors(), store,
                "dflash", GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(ctx.kernels()))
                .transforms(InputTransformLoader.load(dp, kv, catalog, options.activationQuant())).build());
        uploader.uploadResident(t -> !DflashDevice.reads(t.name()));
        List<AutoCloseable> owned = new ArrayList<>();
        owned.add(store);
        owned.add(uploader);
        return owned;
    }

    /** Closes {@code first}, then {@code owned} in reverse, then {@code last}, joining failures. */
    private static void closeAll(AutoCloseable first, List<AutoCloseable> owned, AutoCloseable last) {
        RuntimeException failure = null;
        if (first != null) {
            failure = Closeables.close(first, failure);
        }
        for (int i = owned.size() - 1; i >= 0; i--) {
            failure = Closeables.close(owned.get(i), failure);
        }
        failure = Closeables.close(last, failure);
        if (failure != null) {
            throw failure;
        }
    }

    /** The fused target features of one prefill of {@link #PROMPT}, captured on the GPU target. */
    private float[] capturedFeatures() throws IOException {
        final int rowWidth = TARGET_LAYERS.length * 64;
        float[][] captured = new float[1][];
        try (GGUFModelGPU gpuTarget = new GGUFModelGPU(target.toString(), null, gpu().build())) {
            DflashTargetFeatures tap = new DflashTargetFeatures(TARGET_LAYERS, 64, (feats, n) ->
                    captured[0] = java.util.Arrays.copyOf(feats, n * rowWidth));
            gpuTarget.forwardBackend().setHiddenTap(tap);
            try {
                gpuTarget.forwardBackend().prefill(PROMPT, 0);
            } finally {
                gpuTarget.forwardBackend().setHiddenTap(HiddenTap.NONE);
            }
        }
        assertNotNull("the target tap captured no features", captured[0]);
        return captured[0];
    }

    /** The host backbone over {@code draftFile}, fed {@code features} and then one noise block. */
    private DflashStack.Block hostBlock(float[] features, Path draftFile, DflashDrafter.Options options)
            throws IOException {
        ModelSource dp = ModelSources.open(draftFile.toString());
        try {
            GgufKv kv = GgufKv.of(dp);
            ModelHparams hp = HparamsLoader.load(kv);
            TensorCatalog catalog = TensorCatalog.of(dp);
            CpuWeights weights = CpuWeights.open(dp, catalog,
                    InputTransformLoader.load(dp, kv, catalog, options.activationQuant()));
            try {
                DflashSpec spec = DflashSpec.of(hp, kv, weights);
                int nCtx = 128;
                DflashKvCache cache = new DflashKvCache(nCtx + spec.blockSize(), spec.blockSize(), spec.nSwa(),
                        swaLayers(spec), spec.causal(), spec.dsv4(), kvHeads(spec), spec.headDim(), options.kvType(),
                        options.kvRotation());
                DflashBackbone backbone = new DflashBackbone(spec, hp, weights,
                        DflashMatrix.require(weights, TensorNames.weight(TensorNames.TOKEN_EMBD)),
                        DflashMatrix.require(weights, TensorNames.weight(TensorNames.OUTPUT)), cache, nCtx);
                injectFused(backbone, weights, spec, features);
                return block(backbone, spec);
            } finally {
                weights.close();
            }
        } finally {
            dp.close();
        }
    }

    /** The device stack over {@code draftFile}, fed the same features and block. */
    private DflashStack.Block deviceBlock(float[] features, Path draftFile, DflashDrafter.Options options)
            throws IOException {
        try (GGUFModelGPU gpuTarget = new GGUFModelGPU(target.toString(), null, gpu().build())) {
            ModelSource dp = ModelSources.open(draftFile.toString());
            GgufKv kv = GgufKv.of(dp);
            ModelHparams hp = HparamsLoader.load(kv);
            TensorCatalog catalog = TensorCatalog.of(dp);
            CpuWeights weights = CpuWeights.open(dp, catalog,
                    InputTransformLoader.load(dp, kv, catalog, options.activationQuant()));
            GpuForwardContext ctx = gpuTarget.forwardContext();
            WeightStore store = dp.createMappedStore(StoreUse.RESIDENT_UPLOAD);
            GpuWeightUploader uploader = new GpuWeightUploader(gpuTarget.backend(), ctx.queue(), dp.getTensors(),
                    store, "dflash", GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(ctx.kernels()))
                    .transforms(InputTransformLoader.load(dp, kv, catalog, options.activationQuant())).build());
            GpuDflashKvCache cache = null;
            GpuDflashBackbone dev = null;
            try {
                uploader.uploadResident(t -> !DflashDevice.reads(t.name()));
                DflashSpec spec = DflashSpec.of(hp, kv, weights);
                int nCtx = gpuTarget.forwardBackend().memory().sequence().nCtx();
                cache = new GpuDflashKvCache(ctx.backend(), ctx.ops(), nCtx + spec.blockSize(), spec.blockSize(),
                        spec.nSwa(), swaLayers(spec), spec.causal(), kvHeads(spec), spec.headDim(), options.kvType(),
                        options.kvRotation(), Math.max(DflashDrafter.INJECT_CHUNK, spec.blockSize()));
                dev = new GpuDflashBackbone(spec, hp, ctx, uploader, weights,
                        DflashMatrix.require(weights, TensorNames.weight(TensorNames.TOKEN_EMBD)),
                        new DflashDevice(ctx, uploader).output(), cache, fcCols(weights), nCtx, 0);
                dev.injectFeatures(features, fcCols(weights), features.length / (spec.nTargets() * spec.nEmbd()), 0,
                        null);
                return block(dev, spec);
            } finally {
                RuntimeException failure = Closeables.close(dev, null);
                failure = Closeables.close(cache, failure);
                failure = Closeables.close(uploader, failure);
                failure = Closeables.close(weights, failure);
                failure = Closeables.close(dp, failure);
                if (failure != null) {
                    throw failure;
                }
            }
        }
    }

    /** The host encoder of the drafter over {@code features}, injected into {@code backbone} at position 0. */
    private static void injectFused(DflashBackbone backbone, CpuWeights weights, DflashSpec spec, float[] features) {
        int nRows = features.length / (spec.nTargets() * spec.nEmbd());
        float[] g = new float[nRows * spec.nEmbd()];
        DflashMatrix.require(weights, TensorNames.weight(TensorNames.FC)).matmul(features, 0, g, 0, nRows);
        CpuOps.rmsNorm(g, 0, g, 0, nRows, spec.nEmbd(),
                weights.floats(TensorNames.weight(TensorNames.ENC_OUTPUT_NORM)), spec.rmsEps());
        backbone.inject(g, nRows, 0);
    }

    /** The noise block after the injected rows, its first token the prompt's last one. */
    private static DflashStack.Block block(DflashStack backbone, DflashSpec spec) {
        int[] tokens = new int[spec.blockTokens(2)];
        java.util.Arrays.fill(tokens, spec.maskTokenId());
        tokens[0] = PROMPT[PROMPT.length - 1];
        return backbone.forward(tokens, PROMPT.length);
    }

    /** The encoder input width of {@code spec}: {@code nTargets * n_embd} of the fc matrix. */
    private static int fcCols(CpuWeights weights) {
        return DflashMatrix.require(weights, TensorNames.weight(TensorNames.FC)).cols();
    }

    private static boolean[] swaLayers(DflashSpec spec) {
        boolean[] swa = new boolean[spec.nLayer()];
        for (int il = 0; il < swa.length; il++) {
            swa[il] = spec.isSwa(il);
        }
        return swa;
    }

    private static int[] kvHeads(DflashSpec spec) {
        int[] heads = new int[spec.nLayer()];
        for (int il = 0; il < heads.length; il++) {
            heads[il] = spec.nHeadKv(il);
        }
        return heads;
    }

    /** The two arrays agree within {@code tol} (special values agree exactly). */
    private static void compare(String what, float[] host, float[] device, float tol) {
        org.junit.Assert.assertEquals(what + ": length", host.length, device.length);
        float worst = 0.0f;
        int worstAt = -1;
        for (int i = 0; i < host.length; i++) {
            float diff = Math.abs(host[i] - device[i]);
            if (diff > worst) {
                worst = diff;
                worstAt = i;
            }
        }
        System.out.println(what + ": max diff " + worst + " at " + worstAt + " (host " + (worstAt >= 0
                ? host[worstAt] : 0) + ", device " + (worstAt >= 0 ? device[worstAt] : 0) + ")");
        for (int i = 0; i < host.length; i++) {
            if (Float.isNaN(host[i]) || Float.isInfinite(host[i])) {
                org.junit.Assert.assertEquals(what + ": special value at " + i, host[i], device[i], 0.0f);
                continue;
            }
            org.junit.Assert.assertTrue(what + ": value " + i + " differs by "
                    + Math.abs(host[i] - device[i]) + " (host " + host[i] + ", device " + device[i] + ")",
                    Math.abs(host[i] - device[i]) <= tol);
        }
    }
}

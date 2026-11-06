package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticDflash;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.Closeables;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The DFlash stack on the GPU ({@link GpuDflashBackbone} over a {@link GpuDflashKvCache}) against the host stack
 * ({@link DflashBackbone} over a {@link DflashKvCache}) on a synthetic draft over the real OpenCL backend: the same
 * injected target features and noise blocks give the same logits and {@code t_embd} rows, the device greedy ids are
 * the host argmax of those logits, and both caches forget the same positions on a trim, for bidirectional, causal and
 * sliding-window drafts and an F16 cache. Skipped when OpenCL is unavailable.
 */
public class GpuDflashBackboneTest {

    private static final float TOL = 2e-3f;
    private static final float F16_TOL = 2e-2f;
    private static final int N_CTX = 1024;
    private static final int BLOCK = 4;
    private static final int[] TARGET_LAYERS = {1, 2};

    private final List<Path> files = new ArrayList<>();
    private GGUFModelGPU target;

    @BeforeClass
    public static void requireGpu() {
        GpuTestUtil.assumeOpenCLAvailable();
    }

    @Before
    public void openTarget() throws IOException {
        Path t = write(SyntheticModels.llama());
        target = new GGUFModelGPU(t.toString(), null, EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL).build());
    }

    @After
    public void closeTarget() throws IOException {
        if (target != null) {
            target.close();
        }
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
    }

    @Test
    public void bidirectionalBlocksMatchTheHostStackAcrossTrims() throws IOException {
        try (Stacks s = new Stacks(SyntheticDflash.Options.of(BLOCK, TARGET_LAYERS), KvDType.F32)) {
            int[] block = {5, 319, 319, 319};
            s.injectFeatures(20, 0, 1);
            float[] before = s.assertBlock(block, 20, TOL);
            s.assertBlock(new int[]{9, 319, 319, 319}, 20, TOL);
            s.injectFeatures(8, 12, 2);
            float[] after = s.assertBlock(block, 20, TOL);
            assertTrue("the block reads the re-injected rows", maxDiff(before, after) > 1e-3f);
        }
    }

    @Test
    public void causalBlocksMatchTheHostStack() throws IOException {
        try (Stacks s = new Stacks(SyntheticDflash.Options.of(BLOCK, TARGET_LAYERS).causal(), KvDType.F32)) {
            s.injectFeatures(11, 0, 3);
            s.assertBlock(new int[]{40, 319, 319, 319}, 11, TOL);
            s.assertBlock(new int[]{41, 319}, 11, TOL);
        }
    }

    /** One injection longer than the ring and a chunk: the window sees only its last positions on both sides. */
    @Test
    public void slidingWindowRingsMatchTheHostStackPastTheirLength() throws IOException {
        SyntheticDflash.Options o = SyntheticDflash.Options.of(BLOCK, TARGET_LAYERS).slidingWindow(8);
        try (Stacks s = new Stacks(o, KvDType.F32)) {
            assertTrue(s.device.window(0).windowed());
            s.injectFeatures(600, 0, 4);
            s.assertBlock(new int[]{12, 319, 319, 319}, 600, TOL);
            float[] before = s.assertBlock(new int[]{13, 319, 319, 319}, 600, TOL);
            s.injectFeatures(10, 590, 5);
            float[] after = s.assertBlock(new int[]{13, 319, 319, 319}, 600, TOL);
            assertTrue("the window reads the re-injected rows", maxDiff(before, after) > 1e-3f);
        }
    }

    @Test
    public void anF16CacheMatchesTheHostStack() throws IOException {
        try (Stacks s = new Stacks(SyntheticDflash.Options.of(BLOCK, TARGET_LAYERS), KvDType.F16)) {
            s.injectFeatures(16, 0, 6);
            s.assertBlock(new int[]{3, 319, 319, 319}, 16, F16_TOL);
        }
    }

    /** Fused host rows written by {@code inject} read the same as the device encoder's own fusion. */
    @Test
    public void hostFusedRowsInjectLikeTheDeviceEncoder() throws IOException {
        try (Stacks s = new Stacks(SyntheticDflash.Options.of(BLOCK, TARGET_LAYERS), KvDType.F32)) {
            float[] feats = s.features(6, 7);
            float[] g = s.fuse(feats, 6);
            s.trim(0);
            s.host.inject(g, 6, 0);
            s.dev.inject(g, 6, 0);
            s.assertBlock(new int[]{8, 319, 319, 319}, 6, TOL);
        }
    }

    @Test
    public void quantizedCachesRunOnTheDeviceWhenTheHeadSizeFits() throws IOException {
        try (Stacks s = new Stacks(SyntheticDflash.Options.of(BLOCK, TARGET_LAYERS), KvDType.F32)) {
            assertNull(DflashDevice.refusal(s.spec, KvDType.F16));
            // the synthetic head size of 16 is not a multiple of 32: the refusal names it
            assertTrue(DflashDevice.refusal(s.spec, KvDType.Q8_0).contains("multiple of 32"));
        }
        GgufTestWriter wideHead = SyntheticDflash.dflash(
                SyntheticModels.Config.llama().toBuilder().headDim(32).nRot(32).build(),
                SyntheticDflash.Options.of(BLOCK, TARGET_LAYERS));
        try (ModelSource dp = ModelSources.open(write(wideHead).toString())) {
            DflashSpec spec = DflashDrafter.readSpec(dp);
            assertNull(DflashDevice.refusal(spec, KvDType.Q8_0));
            assertNull(DflashDevice.refusal(spec, KvDType.Q4_0));
        }
        GgufTestWriter conv = SyntheticDflash.dflash(SyntheticModels.Config.llama(),
                SyntheticDflash.Options.of(BLOCK, TARGET_LAYERS).dflash2(2, 4, 8, 4));
        try (ModelSource dp = ModelSources.open(write(conv).toString())) {
            DflashSpec spec = DflashDrafter.readSpec(dp);
            assertTrue(spec.conv());
            assertTrue(spec.dflash2());
            assertNull("a DFlash2 draft drafts on the device", DflashDevice.refusal(spec, KvDType.F16));
        }
    }

    @Test
    public void theDeviceReadsTheStackMatricesOnly() {
        assertTrue(DflashDevice.reads("fc.weight"));
        assertTrue(DflashDevice.reads("output.weight"));
        assertTrue(DflashDevice.reads("blk.3.attn_q.weight"));
        assertTrue(DflashDevice.reads("blk.0.ffn_down.weight"));
        assertTrue(DflashDevice.reads("blk.0.attn_conv_proj.weight"));
        assertTrue(DflashDevice.reads("blk.1.ffn_conv_proj.weight"));
        assertTrue(!DflashDevice.reads("blk.0.attn_conv_base"));
        assertTrue(!DflashDevice.reads("blk.0.attn_norm.weight"));
        assertTrue(!DflashDevice.reads("token_embd.weight"));
        assertTrue(!DflashDevice.reads("d2t"));
        assertTrue(!DflashDevice.reads("markov_w1.weight"));
        assertTrue(DflashDevice.reads("selector_predecessor.weight"));
        assertTrue(DflashDevice.reads("selector_successor.weight"));
        assertTrue(DflashDevice.reads("selector_hidden.weight"));
    }

    private Path write(GgufTestWriter w) throws IOException {
        Path p = w.writeTempFile();
        files.add(p);
        return p;
    }

    /** The host and the device stack of one draft, fed the same rows. */
    private final class Stacks implements AutoCloseable {
        final DflashSpec spec;
        final DflashBackbone host;
        final DflashKvCache hostCache;
        final GpuDflashBackbone dev;
        final GpuDflashKvCache device;
        private final List<AutoCloseable> owned = new ArrayList<>();
        private final DflashMatrix fc;
        private final float[] encNorm;

        Stacks(SyntheticDflash.Options o, KvDType kvType) throws IOException {
            Path draft = write(SyntheticDflash.dflash(SyntheticModels.Config.llama(), o));
            ModelSource dp = ModelSources.open(draft.toString());
            owned.add(dp);
            GgufKv kv = GgufKv.of(dp);
            TensorCatalog catalog = TensorCatalog.of(dp);
            ModelHparams hp = HparamsLoader.load(kv);
            CpuWeights weights = CpuWeights.open(dp);
            owned.add(weights);
            spec = DflashSpec.of(hp, kv, weights);
            fc = DflashMatrix.require(weights, TensorNames.weight(TensorNames.FC));
            encNorm = weights.floats(TensorNames.weight(TensorNames.ENC_OUTPUT_NORM));
            DflashMatrix tokEmbd = DflashMatrix.require(weights, TensorNames.weight(TensorNames.TOKEN_EMBD));
            DflashMatrix output = DflashMatrix.require(weights, TensorNames.weight(TensorNames.OUTPUT));

            boolean[] swa = new boolean[spec.nLayer()];
            int[] heads = new int[spec.nLayer()];
            for (int il = 0; il < swa.length; il++) {
                swa[il] = spec.isSwa(il);
                heads[il] = spec.nHeadKv(il);
            }
            int capacity = N_CTX + spec.blockSize();
            hostCache = new DflashKvCache(capacity, spec.blockSize(), spec.nSwa(), swa, spec.causal(), false, heads,
                    spec.headDim(), kvType, true);
            host = new DflashBackbone(spec, hp, weights, tokEmbd, output, hostCache, N_CTX);

            GpuForwardContext ctx = target.forwardContext();
            WeightStore store = dp.createMappedStore(StoreUse.RESIDENT_UPLOAD);
            owned.add(store);
            GpuWeightUploader uploader = new GpuWeightUploader(target.backend(), ctx.queue(), dp.getTensors(), store,
                    "dflash", GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(ctx.kernels()))
                    .transforms(InputTransformLoader.load(dp, kv, catalog, ActivationQuant.WEIGHT_ONLY)).build());
            owned.add(uploader);
            uploader.uploadResident(t -> !DflashDevice.reads(t.name()));
            DflashDevice d = new DflashDevice(ctx, uploader);
            device = new GpuDflashKvCache(ctx.backend(), ctx.ops(), capacity, spec.blockSize(), spec.nSwa(), swa,
                    spec.causal(), heads, spec.headDim(), kvType, true,
                    Math.max(DflashDrafter.INJECT_CHUNK, spec.blockSize()));
            owned.add(device);
            dev = new GpuDflashBackbone(spec, hp, ctx, uploader, weights, tokEmbd, d.output(), device, fc.cols(),
                    N_CTX, 0);
            owned.add(dev);
        }

        float[] features(int nRows, long seed) {
            Random r = new Random(seed);
            float[] f = new float[nRows * fc.cols()];
            for (int i = 0; i < f.length; i++) {
                f[i] = (float) r.nextGaussian();
            }
            return f;
        }

        /** The host encoder of the drafter: {@code rms(fc . features) * enc.output_norm}. */
        float[] fuse(float[] feats, int nRows) {
            float[] g = new float[nRows * spec.nEmbd()];
            fc.matmul(feats, 0, g, 0, nRows);
            CpuOps.rmsNorm(g, 0, g, 0, nRows, spec.nEmbd(), encNorm, spec.rmsEps());
            return g;
        }

        void trim(int pos) {
            hostCache.trim(pos);
            device.trim(pos);
        }

        /** {@code nRows} random target rows at {@code pos0}: fused on the host in drafter chunks, on the device. */
        void injectFeatures(int nRows, int pos0, long seed) {
            float[] feats = features(nRows, seed);
            trim(pos0);
            for (int off = 0; off < nRows; off += DflashDrafter.INJECT_CHUNK) {
                int n = Math.min(DflashDrafter.INJECT_CHUNK, nRows - off);
                float[] chunk = java.util.Arrays.copyOfRange(feats, off * fc.cols(), (off + n) * fc.cols());
                host.inject(fuse(chunk, n), n, pos0 + off);
            }
            dev.injectFeatures(feats, fc.cols(), nRows, pos0, null);
        }

        /**
         * The block at {@code pos0} on both stacks, as the drafter runs it: trimmed before and after.
         *
         * @return the host logits
         */
        float[] assertBlock(int[] tokens, int pos0, float tol) {
            trim(pos0);
            DflashStack.Block h = host.forward(tokens, pos0);
            trim(pos0);
            DflashStack.Block g = dev.forward(tokens, pos0);
            trim(pos0);
            assertClose("t_embd", h.tEmbd(), g.tEmbd(), tol);
            assertClose("logits", h.logits(), g.logits(), tol);
            int[] ids = dev.forwardGreedy(tokens, pos0);
            trim(pos0);
            int[] expected = new int[tokens.length];
            for (int t = 0; t < tokens.length; t++) {
                expected[t] = CpuOps.argmax(h.logits(), t * h.vocab(), h.vocab());
            }
            assertArrayEquals("device greedy ids", expected, ids);
            return h.logits();
        }

        @Override
        public void close() {
            RuntimeException f = null;
            for (int i = owned.size() - 1; i >= 0; i--) {
                f = Closeables.close(owned.get(i), f);
            }
            if (f != null) {
                throw f;
            }
        }
    }

    private static void assertClose(String what, float[] expected, float[] actual, float tol) {
        assertEquals(what + ": length", expected.length, actual.length);
        float max = maxDiff(expected, actual);
        assertTrue(what + ": max abs diff " + max, max <= tol);
    }

    private static float maxDiff(float[] a, float[] b) {
        float max = 0;
        for (int i = 0; i < a.length; i++) {
            max = Math.max(max, Math.abs(a[i] - b[i]));
        }
        return max;
    }
}

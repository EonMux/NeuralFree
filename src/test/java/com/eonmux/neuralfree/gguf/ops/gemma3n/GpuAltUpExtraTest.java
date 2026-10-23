package com.eonmux.neuralfree.gguf.ops.gemma3n;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Dispatch;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Dispatch sequences of the Gemma3n AltUp and Laurel extras on a fake backend with strict LLM argument rules, for a
 * two-layer gemma3n model (layer 0 builds the streams and per-layer inputs, layer 1 merges the streams) at a decode
 * step of one token and a prefill step of five. Asserts the order of the family's kernels, the per-step token counts
 * and global sizes, and the buffers and offsets each prepared dispatch binds.
 */
public class GpuAltUpExtraTest {

    private static final int UBATCH = 8;
    private static final int N_EMBD = 64;
    private static final int N_ALTUP = 4;
    private static final int ACT = 0;
    private static final int WIDTH = 16;
    private static final int RANK = 8;
    private static final int N_LAYER = 2;
    private static final int VOCAB = 32;
    private static final float EPS = 1e-6f;

    private static final Set<String> LANDMARKS = new HashSet<>(Arrays.asList(GpuOps.ROW_NORM, GpuOps.GLU,
            Gemma3nKernels.ALTUP_PREDICT, Gemma3nKernels.ALTUP_CORRECT, Gemma3nKernels.MAGNITUDE_MATCH,
            Gemma3nKernels.STREAMS_MEAN, NormKernels.MUL_ROW_SCALE));

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private final List<AutoCloseable> closeables = new ArrayList<>();

    @After
    public void tearDown() throws Exception {
        for (int i = closeables.size() - 1; i >= 0; i--) {
            closeables.get(i).close();
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

    @Test
    public void familyExposesFourExtrasAndNoBlocks() {
        assertTrue(Gemma3nBlocks.gpuMixers().isEmpty());
        assertTrue(Gemma3nBlocks.cpuMixers().isEmpty());
        assertTrue(Gemma3nBlocks.gpuFfns().isEmpty());
        assertTrue(Gemma3nBlocks.cpuFfns().isEmpty());
        List<String> ids = Arrays.asList(LayerSpec.Extras.ALTUP, LayerSpec.Extras.LAUREL, LayerSpec.Extras.PLE,
                LayerSpec.Extras.NGRAM_PLE);
        for (int i = 0; i < ids.size(); i++) {
            assertEquals(ids.get(i), Gemma3nBlocks.gpuExtras().get(i).featureId());
            assertEquals(ids.get(i), Gemma3nBlocks.cpuExtras().get(i).featureId());
        }
    }

    @Test
    public void firstLayerBuildsStreamsThenPredicts() {
        ModelSpec spec = gemma3n();
        GpuForwardContext ctx = context(spec);
        GpuLayerExtra altup = bind(ctx, spec.layer(0), LayerSpec.Extras.ALTUP);
        ScratchSlot x = scratch.slot(ScratchSlots.X);
        ScratchSlot streams = scratch.temp(GpuAltUpExtra.TEMP_STREAMS, (long) N_ALTUP * UBATCH * N_EMBD);
        ScratchSlot inputs = scratch.temp(GpuPleExtra.TEMP_INPUTS, (long) UBATCH * N_LAYER * WIDTH);
        int stride = UBATCH * N_EMBD;
        int rowLocal = Gemma3nKernels.rowLocal(ops.local());

        for (int t : new int[]{1, 5}) {
            List<Dispatch> ds = run(() -> altup.beforeBlock(0, t, x));
            assertEquals(Arrays.asList(GpuOps.ROW_NORM, Gemma3nKernels.MAGNITUDE_MATCH, GpuOps.ROW_NORM,
                    Gemma3nKernels.ALTUP_PREDICT), landmarks(ds));

            Dispatch pliNorm = ds.get(index(ds, GpuOps.ROW_NORM));
            assertEquals(t * N_LAYER, pliNorm.intArg(NormKernels.ARG_ROW_NORM_N_ROWS));
            assertEquals(WIDTH, pliNorm.intArg(NormKernels.ARG_ROW_NORM_ROW_LEN));
            assertSame(inputs.buffer(), pliNorm.bufferArg(NormKernels.ARG_ROW_NORM_Y));

            Dispatch mag = only(ds, Gemma3nKernels.MAGNITUDE_MATCH);
            assertTrue(mag.isPreBound());
            assertEquals(t, mag.intArg(Gemma3nKernels.ARG_N_TOKENS));
            assertEquals(N_ALTUP - 1, mag.intArg(2));
            assertEquals(stride, mag.intArg(3));
            assertSame(x.buffer(), mag.bufferArg(6));
            assertSame(streams.buffer(), mag.bufferArg(7));
            assertEquals(streams.offset() + stride, mag.intArg(5));
            assertEquals(Gemma3nKernels.rowGlobal((long) (N_ALTUP - 1) * t, rowLocal)[0], mag.global()[0]);

            Dispatch routerNorm = ds.get(lastIndex(ds, GpuOps.ROW_NORM));
            assertSame(streams.buffer(), routerNorm.bufferArg(NormKernels.ARG_ROW_NORM_X));
            assertEquals(streams.offset() + ACT * stride, routerNorm.intArg(NormKernels.ARG_ROW_NORM_X_OFF));
            assertEquals(t, routerNorm.intArg(NormKernels.ARG_ROW_NORM_N_ROWS));

            Dispatch predict = only(ds, Gemma3nKernels.ALTUP_PREDICT);
            assertTrue(predict.isPreBound());
            assertEquals(t, predict.intArg(Gemma3nKernels.ARG_N_TOKENS));
            assertEquals(N_ALTUP, predict.intArg(2));
            assertEquals(stride, predict.intArg(3));
            assertSame(streams.buffer(), predict.bufferArg(8));
            assertSame(streams.buffer(), predict.bufferArg(9));
            assertEquals(Gemma3nKernels.elementwiseGlobal(t, N_EMBD, ops.local())[0], predict.global()[0]);

            Dispatch last = ds.get(ds.size() - 1);
            assertEquals(GpuOps.COPY2D, last.kernelName());
            assertSame(x.buffer(), last.bufferArg(last.argCount() - 1));
        }
    }

    @Test
    public void laurelFoldsTheAttentionBranch() {
        ModelSpec spec = gemma3n();
        GpuForwardContext ctx = context(spec);
        GpuLayerExtra laurel = bind(ctx, spec.layer(0), LayerSpec.Extras.LAUREL);
        ScratchSlot x = scratch.slot(ScratchSlots.X);
        ScratchSlot branch = scratch.slot(ScratchSlots.BRANCH);
        ScratchSlot normed = scratch.slot(ScratchSlots.NORMED);

        for (int t : new int[]{1, 5}) {
            boolean[] consumed = new boolean[1];
            List<Dispatch> ds = run(() -> consumed[0] = laurel.afterMixer(0, t, x, branch));
            assertTrue(consumed[0]);
            assertEquals(Arrays.asList(GpuOps.ROW_NORM), landmarks(ds));
            Dispatch first = ds.get(0);
            assertTrue("the laurel_l projection reads the attn_norm output",
                    first.bufferArgs().contains(normed.buffer()));
            Dispatch norm = only(ds, GpuOps.ROW_NORM);
            assertEquals(t, norm.intArg(NormKernels.ARG_ROW_NORM_N_ROWS));
            assertEquals(N_EMBD, norm.intArg(NormKernels.ARG_ROW_NORM_ROW_LEN));
            Dispatch scale = ds.get(ds.size() - 1);
            assertEquals(GpuOps.AXPBY, scale.kernelName());
            assertSame(x.buffer(), scale.bufferArg(scale.argCount() - 1));
            int axpby = 0;
            for (Dispatch d : ds) {
                if (GpuOps.AXPBY.equals(d.kernelName())) {
                    axpby++;
                }
            }
            assertEquals("laurel + n, branch + x, + laurel, scale", 4, axpby);
        }
    }

    @Test
    public void correctionAddsPerLayerEmbeddingAndLastLayerMerges() {
        ModelSpec spec = gemma3n();
        GpuForwardContext ctx = context(spec);
        GpuLayerExtra first = bind(ctx, spec.layer(0), LayerSpec.Extras.ALTUP);
        GpuLayerExtra second = bind(ctx, spec.layer(1), LayerSpec.Extras.ALTUP);
        ScratchSlot x = scratch.slot(ScratchSlots.X);
        ScratchSlot streams = scratch.temp(GpuAltUpExtra.TEMP_STREAMS, (long) N_ALTUP * UBATCH * N_EMBD);
        ScratchSlot inputs = scratch.temp(GpuPleExtra.TEMP_INPUTS, (long) UBATCH * N_LAYER * WIDTH);
        ScratchSlot merge = scratch.temp(GpuAltUpExtra.TEMP_MERGE, (long) N_ALTUP * UBATCH * N_EMBD);
        int stride = UBATCH * N_EMBD;

        for (int t : new int[]{1, 5}) {
            List<Dispatch> mid = run(() -> first.afterBlock(0, t, x));
            assertEquals(Arrays.asList(GpuOps.ROW_NORM, Gemma3nKernels.ALTUP_CORRECT, NormKernels.MUL_ROW_SCALE,
                    GpuOps.GLU, GpuOps.ROW_NORM), landmarks(mid));
            Dispatch correct = only(mid, Gemma3nKernels.ALTUP_CORRECT);
            assertTrue(correct.isPreBound());
            assertEquals(t, correct.intArg(Gemma3nKernels.ARG_N_TOKENS));
            assertEquals(ACT, correct.intArg(3));
            assertSame(x.buffer(), correct.bufferArg(10));
            assertEquals(x.offset(), correct.intArg(6));
            assertSame(streams.buffer(), correct.bufferArg(12));
            assertTrue("altup_correct_scale scales the active stream before per_layer_inp_gate",
                    index(mid, NormKernels.MUL_ROW_SCALE) < index(mid, GpuOps.GLU));
            Dispatch scale = only(mid, NormKernels.MUL_ROW_SCALE);
            assertEquals(t, scale.intArg(NormKernels.ARG_ROW_SCALE_N_ROWS));
            assertEquals(N_EMBD, scale.intArg(NormKernels.ARG_ROW_SCALE_SCALE_LEN));
            Dispatch glu = only(mid, GpuOps.GLU);
            assertTrue("the gate multiplies layer 0's per-layer inputs", glu.bufferArgs().contains(inputs.buffer()));
            Dispatch tail = mid.get(mid.size() - 1);
            assertEquals(GpuOps.COPY2D, tail.kernelName());
            assertSame(x.buffer(), tail.bufferArg(tail.argCount() - 1));

            List<Dispatch> end = run(() -> second.afterBlock(0, t, x));
            assertEquals(Arrays.asList(GpuOps.ROW_NORM, Gemma3nKernels.ALTUP_CORRECT, NormKernels.MUL_ROW_SCALE,
                    GpuOps.GLU, GpuOps.ROW_NORM, Gemma3nKernels.MAGNITUDE_MATCH, Gemma3nKernels.STREAMS_MEAN),
                    landmarks(end));
            Dispatch glu1 = only(end, GpuOps.GLU);
            assertTrue(glu1.bufferArgs().contains(inputs.buffer()));
            Dispatch mag = only(end, Gemma3nKernels.MAGNITUDE_MATCH);
            assertEquals(t, mag.intArg(Gemma3nKernels.ARG_N_TOKENS));
            assertSame(streams.buffer(), mag.bufferArg(6));
            assertEquals(streams.offset() + ACT * stride, mag.intArg(4));
            assertSame(merge.buffer(), mag.bufferArg(7));
            assertEquals(merge.offset() + stride, mag.intArg(5));
            Dispatch mean = only(end, Gemma3nKernels.STREAMS_MEAN);
            assertEquals(t, mean.intArg(Gemma3nKernels.ARG_N_TOKENS));
            assertEquals(N_ALTUP, mean.intArg(2));
            assertSame(merge.buffer(), mean.bufferArg(6));
            assertSame(x.buffer(), mean.bufferArg(7));
            assertEquals(x.offset(), mean.intArg(5));
            assertEquals(mean, end.get(end.size() - 1));
        }
    }

    @Test
    public void pleExtraRejectsAltUpLayers() {
        ModelSpec spec = gemma3n();
        GpuForwardContext ctx = context(spec);
        GpuLayerExtra ple = factory(LayerSpec.Extras.PLE).create(spec.layer(0));
        closeables.add(ple);
        assertThrows(IllegalStateException.class, () -> ple.bind(ctx, spec.layer(0)));
    }

    // ------------------------------------------------------------------ fixtures

    private GpuLayerExtra bind(GpuForwardContext ctx, LayerSpec layer, String id) {
        GpuLayerExtra e = factory(id).create(layer);
        closeables.add(e);
        e.bind(ctx, layer);
        return e;
    }

    private static GpuLayerExtraFactory factory(String id) {
        for (GpuLayerExtraFactory f : Gemma3nBlocks.gpuExtras()) {
            if (f.featureId().equals(id)) {
                return f;
            }
        }
        throw new AssertionError("no factory for " + id);
    }

    private static ModelSpec gemma3n() {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(Arch.GEMMA3N)).layerCounts(N_LAYER, 0)
                .nEmbd(N_EMBD).normRmsEps(EPS).nAltup(N_ALTUP).iAltupAct(ACT).laurelRank(RANK).nEmbdAltup(WIDTH)
                .build();
        Map<String, long[]> t = new HashMap<>();
        long all = (long) N_LAYER * WIDTH;
        t.put(TensorNames.weight(TensorNames.PER_LAYER_TOKEN_EMBD), new long[]{all, VOCAB});
        t.put(TensorNames.weight(TensorNames.PER_LAYER_MODEL_PROJ), new long[]{N_EMBD, all});
        t.put(TensorNames.weight(TensorNames.PER_LAYER_PROJ_NORM), new long[]{WIDTH});
        t.put(TensorNames.weight(TensorNames.ALTUP_PROJ), new long[]{N_EMBD, N_EMBD, N_ALTUP - 1});
        t.put(TensorNames.weight(TensorNames.ALTUP_UNEMBD_PROJ), new long[]{N_EMBD, N_EMBD, N_ALTUP - 1});
        for (int il = 0; il < N_LAYER; il++) {
            put(t, il, TensorNames.PER_LAYER_INP_GATE, N_EMBD, WIDTH);
            put(t, il, TensorNames.PER_LAYER_PROJ, WIDTH, N_EMBD);
            put(t, il, TensorNames.PER_LAYER_POST_NORM, N_EMBD);
            put(t, il, TensorNames.ALTUP_CORRECT_COEF, N_ALTUP, N_ALTUP);
            put(t, il, TensorNames.ALTUP_CORRECT_SCALE, N_EMBD);
            put(t, il, TensorNames.ALTUP_PREDICT_COEF, N_ALTUP, N_ALTUP * N_ALTUP);
            put(t, il, TensorNames.ALTUP_ROUTER, N_EMBD, N_ALTUP);
            put(t, il, TensorNames.ALTUP_ROUTER_NORM, N_EMBD);
            put(t, il, TensorNames.LAUREL_L, N_EMBD, RANK);
            put(t, il, TensorNames.LAUREL_R, RANK, N_EMBD);
            put(t, il, TensorNames.LAUREL_POST_NORM, N_EMBD);
        }
        ModelSpec.Builder b = ModelSpec.builder(hp, catalog(t))
                .schedule(LayerSchedule.linear(N_LAYER))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build());
        for (int il = 0; il < N_LAYER; il++) {
            b.layer(LayerSpec.builder(il, MixerKind.NONE, FfnKind.NONE)
                    .norm(NormSpec.builder(NormKind.RMS, EPS).build())
                    .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                    .extra(LayerSpec.Extras.ALTUP)
                    .extra(LayerSpec.Extras.LAUREL)
                    .build());
        }
        return b.build();
    }

    private static void put(Map<String, long[]> t, int il, String base, long... ne) {
        t.put(TensorNames.layerName(il, TensorNames.weight(base)), ne);
    }

    private static TensorCatalog catalog(Map<String, long[]> tensors) {
        List<GGUFTensorInfo> infos = new ArrayList<>();
        for (Map.Entry<String, long[]> e : tensors.entrySet()) {
            infos.add(new GGUFTensorInfo(e.getKey(), e.getValue(), GGMLType.F32, 0));
        }
        return TensorCatalog.of(Arch.GEMMA3N.ggufName(), infos);
    }

    /** Opens the fake device with the residual, norm and token-id slots and resident F32 weights. */
    private GpuForwardContext context(ModelSpec spec) {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.X, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.NORMED, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.TOKEN_IDS, 2L * UBATCH)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        LlmKernels kernels = LlmKernels.open(backend);
        ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
        Map<String, GpuWeight> resident = new HashMap<>();
        for (String name : spec.tensors().names()) {
            long[] ne = spec.tensors().get(name).ne();
            long n = 1;
            for (long d : ne) {
                n *= d;
            }
            GpuBuffer buf = backend.allocateBuffer(n * Float.BYTES);
            resident.put(name, GpuWeight.of(name, GGMLType.F32, ne, buf));
        }
        GpuWeights weights = resident::get;
        return GpuForwardContext.builder(kernels, ops, weights, spec).build();
    }

    private List<Dispatch> run(Runnable r) {
        backend.clearLog();
        r.run();
        return new ArrayList<>(backend.dispatches());
    }

    private static List<String> landmarks(List<Dispatch> ds) {
        List<String> out = new ArrayList<>();
        for (Dispatch d : ds) {
            if (LANDMARKS.contains(d.kernelName())) {
                out.add(d.kernelName());
            }
        }
        return out;
    }

    private static int index(List<Dispatch> ds, String kernel) {
        for (int i = 0; i < ds.size(); i++) {
            if (ds.get(i).kernelName().equals(kernel)) {
                return i;
            }
        }
        throw new AssertionError(kernel + " was not dispatched: " + ds);
    }

    private static int lastIndex(List<Dispatch> ds, String kernel) {
        for (int i = ds.size() - 1; i >= 0; i--) {
            if (ds.get(i).kernelName().equals(kernel)) {
                return i;
            }
        }
        throw new AssertionError(kernel + " was not dispatched: " + ds);
    }

    private static Dispatch only(List<Dispatch> ds, String kernel) {
        Dispatch found = null;
        for (Dispatch d : ds) {
            if (d.kernelName().equals(kernel)) {
                if (found != null) {
                    throw new AssertionError(kernel + " was dispatched more than once: " + ds);
                }
                found = d;
            }
        }
        if (found == null) {
            throw new AssertionError(kernel + " was not dispatched: " + ds);
        }
        return found;
    }
}

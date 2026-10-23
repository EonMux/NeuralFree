package com.eonmux.neuralfree.gguf.ops.hc;

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
import com.eonmux.neuralfree.gguf.gpu.head.GpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
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
 * Dispatch sequences of the hyper-connection extras on a fake backend with strict LLM argument rules: deepseek4-style
 * mHC sub-blocks and HC head, the qwen4exp gated collapse, the glm5-next mean head and the kimi-k3 attention-residual
 * stack. Asserts the HC kernels' order, token counts, global sizes and the buffers / offsets bound per step.
 */
public class GpuHcExtraTest {

    private static final int UBATCH = 8;
    private static final int N_EMBD = 64;
    private static final int HC = 4;
    private static final int STREAM = HC * N_EMBD;
    private static final int LOW_RANK = 16;
    private static final float EPS = 1e-6f;

    private static final Set<String> LANDMARKS = new HashSet<>(Arrays.asList(GpuOps.ROW_NORM,
            HcKernels.HC_PRE_POST_COMB, HcKernels.HC_PRE, HcKernels.HC_POST, HcKernels.RES_STACK_SCORE,
            HcKernels.RES_STACK_MIX));

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

    // ------------------------------------------------------------------ deepseek4 mHC sub-blocks

    @Test
    public void mhcLayerDecodeAndPrefillSequence() {
        ModelSpec spec = hcModel(Arch.DEEPSEEK4, mhcLayerTensors(0), true);
        GpuForwardContext ctx = context(spec, STREAM);
        GpuHcExtra extra = bindHc(ctx, spec.layers().get(0));
        ScratchSlot x = scratch.slot(ScratchSlots.X);
        ScratchSlot branch = scratch.slot(ScratchSlots.BRANCH);
        ScratchSlot stream = scratch.temp(HcSubBlock.TEMP_STREAM, (long) UBATCH * STREAM);

        for (int t : new int[]{1, 5}) {
            List<Dispatch> before = run(() -> extra.beforeBlock(0, t, x));
            assertEquals(GpuOps.COPY2D, before.get(0).kernelName());
            assertSame(x.buffer(), before.get(0).bufferArg(before.get(0).argCount() - 2));
            assertEquals(Arrays.asList(GpuOps.ROW_NORM, HcKernels.HC_PRE_POST_COMB, HcKernels.HC_PRE),
                    landmarks(before));
            assertTrue("the fn projection runs between the norm and the coefficients",
                    index(before, HcKernels.HC_PRE_POST_COMB) - index(before, GpuOps.ROW_NORM) >= 2);
            Dispatch coef = only(before, HcKernels.HC_PRE_POST_COMB);
            assertEquals(t, coef.intArg(HcKernels.COEF_ARG_N_TOKENS));
            assertEquals(HcKernels.OUT_PRE_POST_COMB, coef.intArg(HcKernels.COEF_ARG_N_OUT));
            assertEquals(HcKernels.perTokenGlobal(t, ops.local()), coef.global()[0]);
            Dispatch pre = only(before, HcKernels.HC_PRE);
            assertEquals(t, pre.intArg(HcKernels.PRE_ARG_N_TOKENS));
            assertEquals(HcKernels.PRE_MODE_WEIGHTS, pre.intArg(HcKernels.PRE_ARG_MODE));
            assertSame(stream.buffer(), pre.bufferArg(HcKernels.PRE_ARG_X));
            assertEquals(stream.offset(), pre.intArg(HcKernels.PRE_ARG_X_OFF));
            assertSame(coef.bufferArg(HcKernels.COEF_ARG_PRE), pre.bufferArg(HcKernels.PRE_ARG_W_OR_GATE));
            assertEquals(coef.intArg(HcKernels.COEF_ARG_PRE_OFF), pre.intArg(HcKernels.PRE_ARG_W_OFF));
            assertSame(x.buffer(), pre.bufferArg(HcKernels.PRE_ARG_OUT));
            assertEquals(x.offset(), pre.intArg(HcKernels.PRE_ARG_OUT_OFF));
            assertEquals(HcKernels.elementGlobal(t, N_EMBD, ops.local()), pre.global()[0]);

            List<Dispatch> mixer = new ArrayList<>();
            boolean[] skipped = new boolean[1];
            mixer.addAll(run(() -> skipped[0] = extra.afterMixer(0, t, x, branch)));
            assertTrue(skipped[0]);
            assertEquals(Arrays.asList(HcKernels.HC_POST, GpuOps.ROW_NORM, HcKernels.HC_PRE_POST_COMB,
                    HcKernels.HC_PRE), landmarks(mixer));
            Dispatch post = mixer.get(0);
            assertEquals(t, post.intArg(HcKernels.POST_ARG_N_TOKENS));
            assertEquals(1, post.intArg(HcKernels.POST_ARG_HAS_COMB));
            assertSame(branch.buffer(), post.bufferArg(HcKernels.POST_ARG_Y));
            assertEquals(branch.offset(), post.intArg(HcKernels.POST_ARG_Y_OFF));
            assertSame(stream.buffer(), post.bufferArg(HcKernels.POST_ARG_X));
            assertSame(stream.buffer(), post.bufferArg(HcKernels.POST_ARG_XOUT));
            assertEquals(post.intArg(HcKernels.POST_ARG_X_OFF), post.intArg(HcKernels.POST_ARG_XOUT_OFF));
            assertSame(coef.bufferArg(HcKernels.COEF_ARG_POST), post.bufferArg(HcKernels.POST_ARG_POST));
            assertSame(coef.bufferArg(HcKernels.COEF_ARG_COMB), post.bufferArg(HcKernels.POST_ARG_COMB));

            List<Dispatch> ffn = run(() -> skipped[0] = extra.afterFfn(0, t, x, branch));
            assertTrue(skipped[0]);
            assertEquals(1, ffn.size());
            Dispatch out = ffn.get(0);
            assertEquals(HcKernels.HC_POST, out.kernelName());
            assertSame(stream.buffer(), out.bufferArg(HcKernels.POST_ARG_X));
            assertSame(x.buffer(), out.bufferArg(HcKernels.POST_ARG_XOUT));
            assertEquals(x.offset(), out.intArg(HcKernels.POST_ARG_XOUT_OFF));
            assertEquals(HcKernels.elementGlobal(t, N_EMBD, ops.local()), out.global()[0]);
        }
    }

    @Test
    public void hcHeadCollapsesWithPreOnlyCoefficients() {
        Map<String, long[]> tensors = mhcLayerTensors(0);
        tensors.put(TensorNames.weight(TensorNames.HC_HEAD_FN), new long[]{STREAM, HC});
        tensors.put(TensorNames.weight(TensorNames.HC_HEAD_SCALE), new long[]{1});
        tensors.put(TensorNames.weight(TensorNames.HC_HEAD_BASE), new long[]{HC});
        ModelSpec spec = hcModel(Arch.DEEPSEEK4, tensors, true);
        GpuForwardContext ctx = context(spec, STREAM);
        GpuHeadPreprocessor head = HcBlocks.gpuHeadPreprocessors().get(0);
        closeables.add(head);
        assertTrue(head.appliesTo(spec));
        head.bind(ctx);
        ScratchSlot hidden = scratch.slot(ScratchSlots.X);

        List<Dispatch> ds = run(() -> head.apply(3, hidden));
        assertEquals(Arrays.asList(GpuOps.ROW_NORM, HcKernels.HC_PRE_POST_COMB, HcKernels.HC_PRE), landmarks(ds));
        Dispatch coef = only(ds, HcKernels.HC_PRE_POST_COMB);
        assertEquals(HcKernels.OUT_PRE, coef.intArg(HcKernels.COEF_ARG_N_OUT));
        assertSame(placeholders.scratchFloats(), coef.bufferArg(HcKernels.COEF_ARG_POST));
        assertSame(placeholders.scratchFloats(), coef.bufferArg(HcKernels.COEF_ARG_COMB));
        Dispatch last = ds.get(ds.size() - 1);
        assertEquals(GpuOps.COPY2D, last.kernelName());
        assertSame(hidden.buffer(), last.bufferArg(last.argCount() - 1));
    }

    // ------------------------------------------------------------------ glm5-next mean head

    @Test
    public void meanHeadIsOneCollapseAndCopy() {
        ModelSpec spec = hcModel(Arch.GLM5_NEXT, mhcLayerTensors(0), false);
        GpuForwardContext ctx = context(spec, STREAM);
        GpuHcHeadCollapse head = new GpuHcHeadCollapse();
        closeables.add(head);
        head.bind(ctx);
        ScratchSlot hidden = scratch.slot(ScratchSlots.X);

        List<Dispatch> ds = run(() -> head.apply(2, hidden));
        assertEquals(2, ds.size());
        Dispatch pre = ds.get(0);
        assertEquals(HcKernels.HC_PRE, pre.kernelName());
        assertEquals(HcKernels.PRE_MODE_MEAN, pre.intArg(HcKernels.PRE_ARG_MODE));
        assertEquals(2, pre.intArg(HcKernels.PRE_ARG_N_TOKENS));
        assertSame(hidden.buffer(), pre.bufferArg(HcKernels.PRE_ARG_X));
        assertSame(placeholders.readOnlyFloats(), pre.bufferArg(HcKernels.PRE_ARG_W_OR_GATE));
        assertEquals(GpuOps.COPY2D, ds.get(1).kernelName());
    }

    // ------------------------------------------------------------------ qwen4exp gated collapse

    @Test
    public void gatedCollapseRunsLowRankGateAndInject() {
        assertGatedCollapse(new long[]{N_EMBD, HC});
    }

    /** llama.cpp loads the grouped-norm gammas with TENSOR_ALLOW_RESHAPE, so a flattened tensor is valid. */
    @Test
    public void gatedCollapseAcceptsFlattenedGammas() {
        assertGatedCollapse(new long[]{STREAM});
    }

    private void assertGatedCollapse(long[] normShape) {
        Map<String, long[]> tensors = new HashMap<>();
        for (String site : new String[]{"attn", "ffn"}) {
            tensors.put("blk.0.hc_" + site + "_norm.weight", normShape);
            tensors.put("blk.0.hc_" + site + "_down.weight", new long[]{STREAM, LOW_RANK});
            tensors.put("blk.0.hc_" + site + "_up.weight", new long[]{LOW_RANK, STREAM});
            tensors.put("blk.0.hc_" + site + "_inject.weight", new long[]{STREAM, HC});
        }
        ModelSpec spec = hcModel(Arch.QWEN4EXP, tensors, false);
        GpuForwardContext ctx = context(spec, STREAM);
        GpuHcExtra extra = bindHc(ctx, spec.layers().get(0));
        ScratchSlot x = scratch.slot(ScratchSlots.X);

        List<Dispatch> ds = run(() -> extra.beforeBlock(0, 2, x));
        assertEquals(Arrays.asList(GpuOps.ROW_NORM, HcKernels.HC_PRE), landmarks(ds));
        Dispatch norm = only(ds, GpuOps.ROW_NORM);
        assertEquals(2 * HC, norm.intArg(NormKernels.ARG_ROW_NORM_N_ROWS));
        assertEquals(N_EMBD, norm.intArg(NormKernels.ARG_ROW_NORM_ROW_LEN));
        assertEquals(HC, norm.intArg(NormKernels.ARG_ROW_NORM_W_ROWS));
        assertEquals(NormKernels.FLAG_HAS_W, norm.intArg(NormKernels.ARG_ROW_NORM_FLAGS));
        assertEquals(norm.global()[0], (long) 2 * HC * norm.local()[0]);
        int iNorm = index(ds, GpuOps.ROW_NORM);
        int iPre = index(ds, HcKernels.HC_PRE);
        assertTrue("scale and SiLU of the low-rank projection precede the collapse",
                ds.subList(iNorm, iPre).stream().anyMatch(d -> d.kernelName().equals(GpuOps.UNARY)));
        Dispatch pre = ds.get(iPre);
        assertEquals(HcKernels.PRE_MODE_GATED, pre.intArg(HcKernels.PRE_ARG_MODE));
        assertEquals(1.0f / HC, pre.floatArg(HcKernels.PRE_ARG_GSCALE), 0f);
        assertTrue("the inject projection follows the collapse", ds.size() > iPre + 1);

        ScratchSlot branch = scratch.slot(ScratchSlots.BRANCH);
        List<Dispatch> ffn = run(() -> extra.afterFfn(0, 2, x, branch));
        Dispatch post = ffn.get(0);
        assertEquals(HcKernels.HC_POST, post.kernelName());
        assertEquals(0, post.intArg(HcKernels.POST_ARG_HAS_COMB));
        assertEquals(HcKernels.POST_ACT_SIGMOID, post.intArg(HcKernels.POST_ARG_POST_ACT));
        assertSame(placeholders.readOnlyFloats(), post.bufferArg(HcKernels.POST_ARG_COMB));
    }

    // ------------------------------------------------------------------ kimi-k3 attention residuals

    @Test
    public void attentionResidualStackBanksAndMixes() {
        int nLayer = 4;
        int blockSize = 2;
        ModelSpec spec = kimiModel(nLayer, blockSize);
        GpuForwardContext ctx = context(spec, N_EMBD);
        List<GpuLayerExtra> extras = new ArrayList<>();
        GpuLayerExtraFactory factory = factory(HcBlocks.gpuExtras(), LayerSpec.Extras.ATTN_RES);
        for (LayerSpec l : spec.layers()) {
            GpuLayerExtra e = factory.create(l);
            closeables.add(e);
            e.bind(ctx, l);
            extras.add(e);
        }
        ScratchSlot x = scratch.slot(ScratchSlots.X);
        ScratchSlot branch = scratch.slot(ScratchSlots.BRANCH);
        ScratchSlot stack = scratch.temp(HcSubBlock.TEMP_RES_STACK,
                HcKernels.stackFloats(HcKernels.checkpointCount(nLayer, blockSize), UBATCH, N_EMBD));
        int t = 3;
        int scoreLocal = HcKernels.scoreLocal(ops.local());

        List<Dispatch> l0 = run(() -> extras.get(0).beforeBlock(0, t, x));
        assertEquals(1, l0.size());
        assertEquals(GpuOps.COPY2D, l0.get(0).kernelName());
        assertSame(stack.buffer(), l0.get(0).bufferArg(l0.get(0).argCount() - 1));

        List<Dispatch> l0m = run(() -> extras.get(0).afterMixer(0, t, x, branch));
        assertEquals(Arrays.asList(HcKernels.RES_STACK_SCORE, HcKernels.RES_STACK_MIX), landmarks(l0m));
        Dispatch score = only(l0m, HcKernels.RES_STACK_SCORE);
        assertEquals(1, score.intArg(HcKernels.SCORE_ARG_N_CKPT));
        assertEquals(t, score.intArg(HcKernels.SCORE_ARG_N_TOKENS));
        assertEquals(scoreLocal, score.local()[0]);
        assertEquals(HcKernels.scoreGlobal(t, scoreLocal), score.global()[0]);
        Dispatch mix = only(l0m, HcKernels.RES_STACK_MIX);
        assertSame(x.buffer(), mix.bufferArg(HcKernels.MIX_ARG_OUT));
        assertEquals(x.offset(), mix.intArg(HcKernels.MIX_ARG_OUT_OFF));

        List<Dispatch> l1 = run(() -> extras.get(1).beforeBlock(0, t, x));
        assertEquals(Arrays.asList(HcKernels.RES_STACK_SCORE, HcKernels.RES_STACK_MIX), landmarks(l1));
        assertEquals(1, only(l1, HcKernels.RES_STACK_SCORE).intArg(HcKernels.SCORE_ARG_N_CKPT));
        assertSame(x.buffer(), only(l1, HcKernels.RES_STACK_MIX).bufferArg(HcKernels.MIX_ARG_CUR));

        List<Dispatch> l2 = run(() -> extras.get(2).beforeBlock(0, t, x));
        Dispatch bank = l2.get(0);
        assertEquals(GpuOps.COPY2D, bank.kernelName());
        assertEquals(1, only(l2, HcKernels.RES_STACK_SCORE).intArg(HcKernels.SCORE_ARG_N_CKPT));
        List<Dispatch> l2m = run(() -> extras.get(2).afterMixer(0, t, x, branch));
        assertEquals(2, only(l2m, HcKernels.RES_STACK_SCORE).intArg(HcKernels.SCORE_ARG_N_CKPT));

        List<Dispatch> mid = run(() -> extras.get(2).afterBlock(0, t, x));
        assertTrue("only the last trunk layer mixes for the head", mid.isEmpty());
        List<Dispatch> fin = run(() -> extras.get(nLayer - 1).afterBlock(0, t, x));
        assertEquals(Arrays.asList(HcKernels.RES_STACK_SCORE, HcKernels.RES_STACK_MIX), landmarks(fin));
        assertEquals(2, only(fin, HcKernels.RES_STACK_SCORE).intArg(HcKernels.SCORE_ARG_N_CKPT));
    }

    // ------------------------------------------------------------------ factories

    @Test
    public void factoriesExposeExtrasAndRejectMismatchedResiduals() {
        assertTrue(HcBlocks.gpuMixers().isEmpty());
        assertTrue(HcBlocks.gpuFfns().isEmpty());
        assertTrue(HcBlocks.cpuMixers().isEmpty());
        assertTrue(HcBlocks.cpuFfns().isEmpty());
        assertEquals(2, HcBlocks.gpuExtras().size());
        assertEquals(2, HcBlocks.cpuExtras().size());
        LayerSpec standard = layer(0, ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build(),
                LayerSpec.Extras.HC);
        assertThrows(UnsupportedOperationException.class,
                () -> factory(HcBlocks.gpuExtras(), LayerSpec.Extras.HC).create(standard));
    }

    // ------------------------------------------------------------------ fixtures

    private GpuHcExtra bindHc(GpuForwardContext ctx, LayerSpec layer) {
        GpuLayerExtra e = factory(HcBlocks.gpuExtras(), LayerSpec.Extras.HC).create(layer);
        closeables.add(e);
        e.bind(ctx, layer);
        return (GpuHcExtra) e;
    }

    private static GpuLayerExtraFactory factory(List<GpuLayerExtraFactory> all, String id) {
        for (GpuLayerExtraFactory f : all) {
            if (f.featureId().equals(id)) {
                return f;
            }
        }
        throw new AssertionError("no factory for " + id);
    }

    private static Map<String, long[]> mhcLayerTensors(int il) {
        int mix = HcKernels.mixDim(HC, HcKernels.OUT_PRE_POST_COMB);
        Map<String, long[]> m = new HashMap<>();
        for (String site : new String[]{"attn", "ffn"}) {
            m.put("blk." + il + ".hc_" + site + "_fn.weight", new long[]{STREAM, mix});
            m.put("blk." + il + ".hc_" + site + "_scale.weight", new long[]{HcKernels.OUT_PRE_POST_COMB});
            m.put("blk." + il + ".hc_" + site + "_base.weight", new long[]{mix});
        }
        return m;
    }

    private static LayerSpec layer(int il, ResidualSpec residual, String extra) {
        return LayerSpec.builder(il, MixerKind.NONE, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, EPS).build())
                .residual(residual)
                .extra(extra)
                .build();
    }

    private static ModelSpec hcModel(Arch arch, Map<String, long[]> tensors, boolean hcHead) {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(1, 0).nEmbd(N_EMBD)
                .normRmsEps(EPS).dsv4HcMult(HC).dsv4HcSinkhornIters(20).dsv4HcEps(EPS).hcLowRank(LOW_RANK)
                .build();
        ResidualSpec residual = ResidualSpec.builder(ResidualKind.HYPER_CONNECTIONS, N_EMBD)
                .hyperConnections(HC, true).build();
        return ModelSpec.builder(hp, catalog(arch, tensors))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().hcHead(hcHead).build())
                .layer(layer(0, residual, LayerSpec.Extras.HC))
                .build();
    }

    private static ModelSpec kimiModel(int nLayer, int blockSize) {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(Arch.KIMI_K3)).layerCounts(nLayer, 0)
                .nEmbd(N_EMBD).normRmsEps(EPS).attnResBlockSize(blockSize).build();
        Map<String, long[]> tensors = new HashMap<>();
        for (int il = 0; il < nLayer; il++) {
            tensors.put("blk." + il + ".attn_res_score.weight", new long[]{N_EMBD});
            tensors.put("blk." + il + ".ffn_res_score.weight", new long[]{N_EMBD});
        }
        tensors.put("output_res_score.weight", new long[]{N_EMBD});
        ModelSpec.Builder b = ModelSpec.builder(hp, catalog(Arch.KIMI_K3, tensors))
                .schedule(LayerSchedule.linear(nLayer))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build());
        for (int il = 0; il < nLayer; il++) {
            b.layer(layer(il, ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build(),
                    LayerSpec.Extras.ATTN_RES));
        }
        return b.build();
    }

    private static TensorCatalog catalog(Arch arch, Map<String, long[]> tensors) {
        List<GGUFTensorInfo> infos = new ArrayList<>();
        for (Map.Entry<String, long[]> e : tensors.entrySet()) {
            infos.add(new GGUFTensorInfo(e.getKey(), e.getValue(), GGMLType.F32, 0));
        }
        return TensorCatalog.of(arch.ggufName(), infos);
    }

    /** Opens the fake device with X and BRANCH slots of {@code xWidth} floats per token and resident F32 weights. */
    private GpuForwardContext context(ModelSpec spec, int xWidth) {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.X, (long) UBATCH * xWidth)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * N_EMBD)
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
        throw new AssertionError(kernel + " not dispatched");
    }

    private static Dispatch only(List<Dispatch> ds, String kernel) {
        Dispatch found = null;
        for (Dispatch d : ds) {
            if (d.kernelName().equals(kernel)) {
                if (found != null) {
                    throw new AssertionError(kernel + " dispatched more than once");
                }
                found = d;
            }
        }
        if (found == null) {
            throw new AssertionError(kernel + " not dispatched");
        }
        return found;
    }
}

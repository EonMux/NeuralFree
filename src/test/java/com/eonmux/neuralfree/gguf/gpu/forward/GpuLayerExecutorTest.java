package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GdnHeadMap;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
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
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.resolve.NormResolver;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Dispatch;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Op;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.OpKind;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Dispatch traces of the GPU forward path against the host-only fake backend with strict LLM argument rules. Mixers
 * and FFN blocks are recording stand-ins (the per-family blocks have their own traces), so a trace shows exactly
 * what the executor, the embedding stage and the head enqueue around them: the executor's step order for a dense
 * llama, a gemma2 sandwich layer with the final logit softcap, and a qwen35 hybrid whose GDN and attention mixers
 * come from the registry per layer; a full decode step synchronizes only for its single read.
 */
public class GpuLayerExecutorTest {

    private static final int UBATCH = 4;
    private static final int N_EMBD = 64;
    private static final int N_FF = 128;
    private static final int VOCAB = 32;
    private static final int N_HEAD = 2;
    private static final int HEAD_DIM = 32;
    private static final int N_CTX = 64;
    private static final float RMS_EPS = 1e-6f;
    private static final float FINAL_SOFTCAP = 30.0f;

    private static final String NORM = NormKernels.ROW_NORM;
    private static final String MIX = NormKernels.RESIDUAL_MIX;

    /** A recording block call and the number of dispatches enqueued before it. */
    private static final class Event {
        final String label;
        final int at;

        Event(String label, int at) {
            this.label = label;
            this.at = at;
        }
    }

    private final List<Event> events = new ArrayList<>();
    private FakeComputeBackend backend;
    private LlmKernels kernels;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private Map<String, GpuWeight> weightMap;
    private DeviceModelMemory memory;
    private final List<AutoCloseable> closeables = new ArrayList<>();

    @After
    public void tearDown() throws Exception {
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

    // ------------------------------------------------------------------ recording blocks

    private void record(String label) {
        events.add(new Event(label, backend.dispatches().size()));
    }

    private final class RecordingMixer implements GpuMixer {
        private String name;

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            this.name = layer.mixerKind() + " " + layer.il();
        }

        @Override
        public void decode(int pos, ScratchSlot in, ScratchSlot out) {
            record("mixer " + name + " decode " + in.name() + "->" + out.name());
        }

        @Override
        public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
            record("mixer " + name + " prefill " + nTokens + " " + in.name() + "->" + out.name());
        }
    }

    private final class RecordingFfn implements GpuFfnBlock {
        private int il;

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            this.il = layer.il();
        }

        @Override
        public void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
            record("ffn " + il + " " + in.name() + "->" + out.name());
        }
    }

    private GpuMixerFactory mixerFactory(MixerKind kind) {
        return new GpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return kind;
            }

            @Override
            public GpuMixer create(LayerSpec layer) {
                return new RecordingMixer();
            }
        };
    }

    private GpuBlockRegistry registry() {
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
        return new GpuBlockRegistry(Arrays.asList(mixerFactory(MixerKind.ATTENTION), mixerFactory(MixerKind.GDN)),
                Collections.singletonList(ffn), BuiltInGpuExtras.factories());
    }

    // ------------------------------------------------------------------ traces

    private void startTrace() {
        backend.clearLog();
        events.clear();
    }

    /** Dispatched kernel names with the recording block calls inserted where they happened. */
    private List<String> trace() {
        List<Dispatch> ds = backend.dispatches();
        List<String> out = new ArrayList<>();
        int e = 0;
        for (int i = 0; i <= ds.size(); i++) {
            while (e < events.size() && events.get(e).at == i) {
                out.add(events.get(e).label);
                e++;
            }
            if (i < ds.size()) {
                out.add(ds.get(i).kernelName());
            }
        }
        return out;
    }

    private static List<String> preNormLayer(String mixer, int il) {
        return Arrays.asList(NORM, "mixer " + mixer + " " + il + " decode NORMED->BRANCH", MIX,
                NORM, "ffn " + il + " NORMED->BRANCH", MIX);
    }

    // ------------------------------------------------------------------ fixtures

    private static String blk(int il, String base) {
        return TensorNames.layerName(TensorNames.STACK_BLK, il, TensorNames.weight(base));
    }

    private static GGUFTensorInfo info(String name, long... dims) {
        return new GGUFTensorInfo(name, dims, GGMLType.F32, 0);
    }

    /** Embedding, output norm and head, and per layer the given norms and a gated FFN. */
    private static List<GGUFTensorInfo> tensors(int nLayer, String... layerNorms) {
        List<GGUFTensorInfo> t = new ArrayList<>();
        t.add(info(TensorNames.weight(TensorNames.TOKEN_EMBD), N_EMBD, VOCAB));
        t.add(info(TensorNames.weight(TensorNames.OUTPUT_NORM), N_EMBD));
        t.add(info(TensorNames.weight(TensorNames.OUTPUT), N_EMBD, VOCAB));
        for (int il = 0; il < nLayer; il++) {
            for (String n : layerNorms) {
                t.add(info(blk(il, n), N_EMBD));
            }
            t.add(info(blk(il, TensorNames.FFN_GATE), N_EMBD, N_FF));
            t.add(info(blk(il, TensorNames.FFN_UP), N_EMBD, N_FF));
            t.add(info(blk(il, TensorNames.FFN_DOWN), N_FF, N_EMBD));
        }
        return t;
    }

    private static ModelHparamsBuilder hparams(Arch arch, int nLayer) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(nLayer, 0)
                .nCtxTrain(4096).nEmbd(N_EMBD).nVocab(VOCAB).normRmsEps(RMS_EPS).normEps(1e-5f)
                .headKFull(HEAD_DIM).headVFull(HEAD_DIM);
        for (int il = 0; il < nLayer; il++) {
            hb.nHead(il, N_HEAD).nHeadKv(il, N_HEAD);
        }
        return hb;
    }

    private static AttentionSpec attention(int il) {
        return AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD).headDimK(HEAD_DIM).headDimV(HEAD_DIM)
                .noRope(true).nRot(0).kqScale(1.0f / (float) Math.sqrt(HEAD_DIM)).kvOwnerLayer(il).build();
    }

    private static FfnSpec denseFfn() {
        return FfnSpec.builder(FfnSpec.Role.DENSE).nFf(N_FF).build();
    }

    private static ResidualSpec standardResidual() {
        return ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build();
    }

    /** Per-layer pieces a fixture overrides. */
    private interface LayerShape {
        LayerSpec layer(int il, NormSpec resolvedNorm);
    }

    private static ModelSpec model(ModelHparams hp, List<GGUFTensorInfo> tensors, int nLayer, LayerShape shape,
                                   float embdScale, float finalSoftcap, MemoryKind memoryKind) {
        TensorCatalog tc = TensorCatalog.of(hp.arch().ggufName(), tensors);
        NormResolver norms = new NormResolver(hp, tc);
        HeadSpec.Builder head = HeadSpec.builder()
                .outputNorm(tc.get(TensorNames.weight(TensorNames.OUTPUT_NORM)), null)
                .output(tc.get(TensorNames.weight(TensorNames.OUTPUT)), false);
        if (finalSoftcap > 0f) {
            head.finalSoftcap(finalSoftcap);
        }
        ModelSpec.Builder b = ModelSpec.builder(hp, tc)
                .schedule(LayerSchedule.linear(nLayer))
                .embedding(EmbeddingSpec.builder(tc.get(TensorNames.weight(TensorNames.TOKEN_EMBD)), N_EMBD)
                        .embdScale(embdScale).build())
                .head(head.build());
        for (int il = 0; il < nLayer; il++) {
            b.layer(shape.layer(il, norms.resolve(il)));
        }
        if (memoryKind != null) {
            b.memoryKind(memoryKind);
        }
        return b.build();
    }

    /** Dense llama: attn_norm, attention, ffn_norm, gated FFN. */
    private static ModelSpec llama(int nLayer, ResidualSpec residual, NormSpec norm, MemoryKind memoryKind) {
        ModelHparams hp = hparams(Arch.LLAMA, nLayer).build();
        return model(hp, tensors(nLayer, TensorNames.ATTN_NORM, TensorNames.FFN_NORM), nLayer,
                (il, resolved) -> LayerSpec.builder(il, MixerKind.ATTENTION, FfnKind.DENSE)
                        .norm(norm != null ? norm : resolved)
                        .residual(residual != null ? residual : standardResidual())
                        .attention(attention(il))
                        .ffn(denseFfn())
                        .build(),
                1.0f, 0f, memoryKind);
    }

    /** gemma2: sandwich norms around both blocks, sqrt(n_embd) embedding scale, final logit softcap. */
    private static ModelSpec gemma2(int nLayer) {
        ModelHparams hp = hparams(Arch.GEMMA2, nLayer).finalSoftcap(FINAL_SOFTCAP).build();
        return model(hp, tensors(nLayer, TensorNames.ATTN_NORM, TensorNames.ATTN_POST_NORM, TensorNames.FFN_NORM,
                        TensorNames.FFN_POST_NORM), nLayer,
                (il, resolved) -> LayerSpec.builder(il, MixerKind.ATTENTION, FfnKind.DENSE)
                        .norm(resolved)
                        .residual(standardResidual())
                        .attention(attention(il))
                        .ffn(denseFfn())
                        .build(),
                (float) Math.sqrt(N_EMBD), FINAL_SOFTCAP, MemoryKind.NONE);
    }

    /** qwen35: three GDN layers, then full attention; the FFN input is post_attention_norm (PRE_FFN). */
    private static ModelSpec qwen35() {
        int nLayer = 4;
        ModelHparamsBuilder hb = hparams(Arch.QWEN35, nLayer);
        for (int il = 0; il < nLayer; il++) {
            hb.isRecurrent(il, il < 3);
        }
        return model(hb.build(), tensors(nLayer, TensorNames.ATTN_NORM, TensorNames.ATTN_POST_NORM), nLayer,
                (il, resolved) -> {
                    LayerSpec.Builder l = LayerSpec.builder(il, il < 3 ? MixerKind.GDN : MixerKind.ATTENTION,
                            FfnKind.DENSE).norm(resolved).residual(standardResidual()).ffn(denseFfn());
                    if (il < 3) {
                        l.recurrent(RecurrentSpec.builder(MixerKind.GDN).gdnHeads(2, 4, 128, 128).dConv(4)
                                .headMap(GdnHeadMap.TILED).build());
                    } else {
                        l.attention(attention(il));
                    }
                    return l.build();
                },
                1.0f, 0f, MemoryKind.NONE);
    }

    /** Opens the fake device, F32 weights for every tensor of the spec, and a context (with memory when asked). */
    private GpuForwardContext open(ModelSpec spec, boolean withMemory) {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        kernels = LlmKernels.open(backend);
        scratch = GpuScratch.forModel(backend, spec, UBATCH);
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
        weightMap = new HashMap<>();
        TensorCatalog tc = spec.tensors();
        for (String name : tc.names()) {
            GGUFTensorInfo info = tc.info(name);
            long n = info.numElements();
            weightMap.put(name, GpuWeight.of(name, GGMLType.F32, info.dims().clone(),
                    backend.allocateBuffer(n * Float.BYTES, true)));
        }
        GpuWeights weights = weightMap::get;
        GpuForwardContext.Builder b = GpuForwardContext.builder(kernels, ops, weights, spec);
        if (withMemory) {
            MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16).ubatch(UBATCH).build();
            MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX);
            memory = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan,
                    spec.hparams().mropeTextPositions(), null);
            b.memory(memory);
        }
        return b.build();
    }

    private GpuLayerExecutor executor(GpuForwardContext ctx) {
        GpuLayerExecutor exec = GpuLayerExecutor.forTrunk(ctx, registry());
        closeables.add(exec);
        return exec;
    }

    private static boolean uses(Dispatch d, GpuWeight w) {
        return d.bufferArgs().contains(w.shard(0));
    }

    private List<Dispatch> dispatchesOf(String kernel) {
        return backend.dispatchesOf(kernel);
    }

    private static List<OpKind> kinds(List<Op> ops) {
        List<OpKind> out = new ArrayList<>();
        for (Op op : ops) {
            out.add(op.kind());
        }
        return out;
    }

    private static void assertNoHostSync(List<Op> ops) {
        for (Op op : ops) {
            assertTrue("unexpected " + op.kind() + " in an executor step",
                    op.kind() != OpKind.READ && op.kind() != OpKind.FINISH);
        }
    }

    // ------------------------------------------------------------------ llama

    @Test
    public void llamaDecodeStepRunsEachLayerInPreNormOrderWithoutHostSync() {
        GpuForwardContext ctx = open(llama(2, null, null, MemoryKind.NONE), false);
        GpuLayerExecutor exec = executor(ctx);
        startTrace();

        exec.run(5, 1, GpuLayerExecutor.StepMode.DECODE, GpuLayerExecutor.TapSink.NONE);

        List<String> expected = new ArrayList<>();
        expected.addAll(preNormLayer("ATTENTION", 0));
        expected.addAll(preNormLayer("ATTENTION", 1));
        assertEquals(expected, trace());
        assertNoHostSync(backend.ops());
        List<Dispatch> norms = dispatchesOf(NORM);
        assertTrue(uses(norms.get(0), weightMap.get(blk(0, TensorNames.ATTN_NORM))));
        assertTrue(uses(norms.get(1), weightMap.get(blk(0, TensorNames.FFN_NORM))));
        assertTrue(uses(norms.get(3), weightMap.get(blk(1, TensorNames.FFN_NORM))));
    }

    @Test
    public void prefillStepPassesTheTokenCountToEveryMixer() {
        GpuForwardContext ctx = open(llama(1, null, null, MemoryKind.NONE), false);
        GpuLayerExecutor exec = executor(ctx);
        startTrace();

        exec.run(0, 3, GpuLayerExecutor.StepMode.PREFILL, GpuLayerExecutor.TapSink.NONE);

        assertEquals(Arrays.asList(NORM, "mixer ATTENTION 0 prefill 3 NORMED->BRANCH", MIX,
                NORM, "ffn 0 NORMED->BRANCH", MIX), trace());
    }

    @Test
    public void tapsSeeTheInputOfEachLayerAndTheStackOutput() {
        GpuForwardContext ctx = open(llama(2, null, null, MemoryKind.NONE), false);
        GpuLayerExecutor exec = executor(ctx);
        GpuLayerExecutor.TapSink sink = new GpuLayerExecutor.TapSink() {
            @Override
            public boolean wants(int tap) {
                return tap != 0;
            }

            @Override
            public void capture(int tap, ScratchSlot x, int nTokens, int width) {
                record("tap " + tap + " " + x.name() + " " + nTokens + "x" + width);
            }
        };
        startTrace();

        exec.run(0, 2, GpuLayerExecutor.StepMode.PREFILL, sink);

        List<String> t = trace();
        assertEquals(2, t.stream().filter(s -> s.startsWith("tap ")).count());
        assertEquals("tap 1 X 2x" + N_EMBD, t.get(6));
        assertEquals("tap 2 X 2x" + N_EMBD, t.get(t.size() - 1));
    }

    @Test
    public void parallelResidualFeedsTheSharedNormToBothBlocksAndAddsBothBranches() {
        NormSpec shared = NormSpec.builder(NormKind.RMS, RMS_EPS).attnNorm(true).sharedAttnNormForFfn(true).build();
        ResidualSpec parallel = ResidualSpec.builder(ResidualKind.PARALLEL, N_EMBD).build();
        GpuForwardContext ctx = open(llama(1, parallel, shared, MemoryKind.NONE), false);
        GpuLayerExecutor exec = executor(ctx);
        startTrace();

        exec.run(0, 1, GpuLayerExecutor.StepMode.DECODE, GpuLayerExecutor.TapSink.NONE);

        assertEquals(Arrays.asList(NORM, "mixer ATTENTION 0 decode NORMED->" + GpuLayerExecutor.TEMP_PARALLEL,
                "ffn 0 NORMED->BRANCH", MIX, MIX), trace());
    }

    @Test
    public void falcon40bAttentionReadsAttnNorm2AndTheFfnReadsAttnNormOfTheLayerInput() {
        NormSpec norm = NormSpec.builder(NormKind.RMS, RMS_EPS).attnNorm(true).attnNorm2(true).ffnNorm(false).build();
        ResidualSpec parallel = ResidualSpec.builder(ResidualKind.PARALLEL, N_EMBD).ffnUsesAttnNorm2(true).build();
        ModelSpec spec = model(hparams(Arch.LLAMA, 1).build(),
                tensors(1, TensorNames.ATTN_NORM, TensorNames.ATTN_NORM_2), 1,
                (il, resolved) -> LayerSpec.builder(il, MixerKind.ATTENTION, FfnKind.DENSE)
                        .norm(norm).residual(parallel).attention(attention(il)).ffn(denseFfn()).build(),
                1.0f, 0f, MemoryKind.NONE);
        GpuForwardContext ctx = open(spec, false);
        GpuLayerExecutor exec = executor(ctx);
        startTrace();

        exec.run(0, 1, GpuLayerExecutor.StepMode.DECODE, GpuLayerExecutor.TapSink.NONE);

        assertEquals(Arrays.asList(NORM, "mixer ATTENTION 0 decode NORMED->" + GpuLayerExecutor.TEMP_PARALLEL,
                NORM, "ffn 0 NORMED->BRANCH", MIX, MIX), trace());
        List<Dispatch> norms = dispatchesOf(NORM);
        assertTrue(uses(norms.get(0), weightMap.get(blk(0, TensorNames.ATTN_NORM_2))));
        assertTrue(uses(norms.get(1), weightMap.get(blk(0, TensorNames.ATTN_NORM))));
    }

    @Test
    public void jinaBertReaddsTheLayerInputBeforeAttnNorm2() {
        List<GGUFTensorInfo> t = tensors(1, TensorNames.ATTN_OUT_NORM, TensorNames.ATTN_NORM_2,
                TensorNames.LAYER_OUT_NORM);
        ModelSpec spec = model(hparams(Arch.JINA_BERT_V2, 1).build(), t, 1,
                (il, resolved) -> LayerSpec.builder(il, MixerKind.ATTENTION, FfnKind.DENSE)
                        .norm(resolved).residual(standardResidual()).attention(attention(il)).ffn(denseFfn())
                        .build(),
                1.0f, 0f, MemoryKind.NONE);
        GpuForwardContext ctx = open(spec, false);
        GpuLayerExecutor exec = executor(ctx);
        startTrace();

        exec.run(0, 2, GpuLayerExecutor.StepMode.PREFILL, GpuLayerExecutor.TapSink.NONE);

        assertEquals(Arrays.asList(GpuOps.COPY2D, "mixer ATTENTION 0 prefill 2 X->BRANCH", MIX, NORM, MIX, NORM,
                "ffn 0 X->BRANCH", MIX, NORM), trace());
        List<Dispatch> norms = dispatchesOf(NORM);
        assertTrue(uses(norms.get(0), weightMap.get(blk(0, TensorNames.ATTN_OUT_NORM))));
        assertTrue(uses(norms.get(1), weightMap.get(blk(0, TensorNames.ATTN_NORM_2))));
        assertTrue(uses(norms.get(2), weightMap.get(blk(0, TensorNames.LAYER_OUT_NORM))));
        List<Dispatch> mixes = dispatchesOf(MIX);
        assertTrue(mixes.get(1).bufferArgs().contains(
                ctx.scratch().temp(GpuLayerExecutor.TEMP_LAYER_INPUT, (long) UBATCH * N_EMBD).buffer()));
    }

    @Test
    public void discardedLayerOutputRestoresTheLayerInput() {
        ResidualSpec discard = ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).discardLayerOutput(true).build();
        GpuForwardContext ctx = open(llama(1, discard, null, MemoryKind.NONE), false);
        GpuLayerExecutor exec = executor(ctx);
        startTrace();

        exec.run(0, 1, GpuLayerExecutor.StepMode.DECODE, GpuLayerExecutor.TapSink.NONE);

        List<String> expected = new ArrayList<>();
        expected.add(GpuOps.COPY2D);
        expected.addAll(preNormLayer("ATTENTION", 0));
        expected.add(GpuOps.COPY2D);
        assertEquals(expected, trace());
    }

    @Test
    public void scaledResidualPassesItsScaleToTheMix() {
        ResidualSpec scaled = ResidualSpec.builder(ResidualKind.SCALED, N_EMBD).residualScale(0.25f).build();
        GpuForwardContext ctx = open(llama(1, scaled, null, MemoryKind.NONE), false);
        GpuLayerExecutor exec = executor(ctx);
        startTrace();

        exec.run(0, 1, GpuLayerExecutor.StepMode.DECODE, GpuLayerExecutor.TapSink.NONE);

        assertEquals(preNormLayer("ATTENTION", 0), trace());
        for (Dispatch d : dispatchesOf(MIX)) {
            assertTrue(d.args().contains(NormKernels.MixMode.SCALED.code()));
            assertTrue(d.args().contains(0.25f));
        }
    }

    @Test
    public void alphaBetaResidualWithoutItsExtraIsRejectedAtBind() {
        ModelSpec base = llama(1, null, null, MemoryKind.NONE);
        TensorCatalog tc = base.tensors();
        ResidualSpec ab = ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD)
                .attnAlphaBeta(tc.get(blk(0, TensorNames.ATTN_NORM)), tc.get(blk(0, TensorNames.FFN_NORM))).build();
        GpuForwardContext ctx = open(llama(1, ab, null, MemoryKind.NONE), false);

        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> GpuLayerExecutor.forTrunk(ctx, registry()));
        assertTrue(e.getMessage().contains(LayerSpec.Extras.RESID_ALPHA_BETA));
    }

    @Test
    public void stepsOutsideTheUbatchOrMultiTokenDecodesAreRejected() {
        GpuForwardContext ctx = open(llama(1, null, null, MemoryKind.NONE), false);
        GpuLayerExecutor exec = executor(ctx);

        assertThrows(IllegalArgumentException.class, () -> exec.run(0, UBATCH + 1,
                GpuLayerExecutor.StepMode.PREFILL, GpuLayerExecutor.TapSink.NONE));
        assertThrows(IllegalArgumentException.class, () -> exec.run(0, 2,
                GpuLayerExecutor.StepMode.DECODE, GpuLayerExecutor.TapSink.NONE));
    }

    // ------------------------------------------------------------------ gemma2

    @Test
    public void gemma2SandwichNormsWrapBothBranchesBeforeTheirResiduals() {
        GpuForwardContext ctx = open(gemma2(2), false);
        GpuLayerExecutor exec = executor(ctx);
        startTrace();

        exec.run(3, 1, GpuLayerExecutor.StepMode.DECODE, GpuLayerExecutor.TapSink.NONE);

        List<String> expected = new ArrayList<>();
        for (int il = 0; il < 2; il++) {
            expected.addAll(Arrays.asList(NORM, "mixer ATTENTION " + il + " decode NORMED->BRANCH", NORM, MIX,
                    NORM, "ffn " + il + " NORMED->BRANCH", NORM, MIX));
        }
        assertEquals(expected, trace());
        assertNoHostSync(backend.ops());
        List<Dispatch> norms = dispatchesOf(NORM);
        assertTrue(uses(norms.get(0), weightMap.get(blk(0, TensorNames.ATTN_NORM))));
        assertTrue(uses(norms.get(1), weightMap.get(blk(0, TensorNames.ATTN_POST_NORM))));
        assertTrue(uses(norms.get(2), weightMap.get(blk(0, TensorNames.FFN_NORM))));
        assertTrue(uses(norms.get(3), weightMap.get(blk(0, TensorNames.FFN_POST_NORM))));
    }

    @Test
    public void gemma2HeadNormsProjectsSoftcapsAndSelectsOnTheDevice() {
        GpuForwardContext ctx = open(gemma2(1), false);
        GpuLogitsHead head = new GpuLogitsHead(ctx, Collections.emptyList());
        closeables.add(head);
        startTrace();

        ScratchSlot hidden = head.hidden(1, new int[]{0}, 1);
        head.logits(1, hidden);
        head.argmax(1);

        List<String> names = backend.dispatchedKernelNames();
        assertEquals(NORM, names.get(0));
        assertTrue(uses(backend.dispatches().get(0), weightMap.get(TensorNames.weight(TensorNames.OUTPUT_NORM))));
        assertEquals(GpuLogitsHead.ARGMAX, names.get(names.size() - 1));
        assertEquals(GpuOps.UNARY, names.get(names.size() - 2));
        List<Dispatch> unary = dispatchesOf(GpuOps.UNARY);
        assertEquals(1, unary.size());
        assertEquals(ActOp.SOFTCAP.code(), unary.get(0).intArg(ActKernels.ARG_UNARY_OP));
        assertNoHostSync(backend.ops());
    }

    // ------------------------------------------------------------------ qwen35

    @Test
    public void qwen35TakesEachLayersMixerFromTheRegistryAndNormsTheFfnWithPostAttentionNorm() {
        GpuForwardContext ctx = open(qwen35(), false);
        GpuLayerExecutor exec = executor(ctx);
        startTrace();

        exec.run(0, 3, GpuLayerExecutor.StepMode.PREFILL, GpuLayerExecutor.TapSink.NONE);

        List<String> expected = new ArrayList<>();
        for (int il = 0; il < 4; il++) {
            String kind = il < 3 ? "GDN" : "ATTENTION";
            expected.addAll(Arrays.asList(NORM, "mixer " + kind + " " + il + " prefill 3 NORMED->BRANCH", MIX,
                    NORM, "ffn " + il + " NORMED->BRANCH", MIX));
        }
        assertEquals(expected, trace());
        assertNoHostSync(backend.ops());
        List<Dispatch> norms = dispatchesOf(NORM);
        for (int il = 0; il < 4; il++) {
            assertTrue(uses(norms.get(2 * il), weightMap.get(blk(il, TensorNames.ATTN_NORM))));
            assertTrue(uses(norms.get(2 * il + 1), weightMap.get(blk(il, TensorNames.ATTN_POST_NORM))));
        }
    }

    // ------------------------------------------------------------------ forward backend

    @Test
    public void greedyDecodeStepSynchronizesOnlyForItsTokenId() {
        GpuForwardContext ctx = open(llama(2, null, null, null), true);
        GpuForwardBackend fb = new GpuForwardBackend(ctx, registry(), Collections.emptyList());
        closeables.add(fb);
        startTrace();

        fb.decodeGreedyOnDevice(7, 0);

        List<Op> log = backend.ops();
        List<OpKind> k = kinds(log);
        assertEquals(OpKind.FINISH, k.get(k.size() - 1));
        assertEquals(OpKind.READ, k.get(k.size() - 2));
        assertEquals(1, Collections.frequency(k, OpKind.FINISH));
        assertEquals(1, Collections.frequency(k, OpKind.READ));
        assertEquals(Integer.BYTES, log.get(log.size() - 2).bytes());
        List<String> names = backend.dispatchedKernelNames();
        assertEquals(GpuLogitsHead.ARGMAX, names.get(names.size() - 1));
        List<String> t = trace();
        assertTrue(t.contains("mixer ATTENTION 0 decode NORMED->BRANCH"));
        assertTrue(t.contains("mixer ATTENTION 1 decode NORMED->BRANCH"));
        assertEquals(1, memory.sequence().nPast());
    }

    @Test
    public void prefillReadsOnlyTheLastRowsLogitsAfterEveryUbatch() {
        GpuForwardContext ctx = open(llama(1, null, null, null), true);
        GpuForwardBackend fb = new GpuForwardBackend(ctx, registry(), Collections.emptyList());
        closeables.add(fb);
        startTrace();

        float[] logits = fb.prefill(new int[]{1, 2, 3, 4, 5, 6}, 0);

        assertEquals(VOCAB, logits.length);
        List<OpKind> k = kinds(backend.ops());
        assertEquals(1, Collections.frequency(k, OpKind.FINISH));
        assertEquals(1, Collections.frequency(k, OpKind.READ));
        assertEquals(OpKind.FINISH, k.get(k.size() - 1));
        List<String> t = trace();
        assertTrue(t.contains("mixer ATTENTION 0 prefill " + UBATCH + " NORMED->BRANCH"));
        assertTrue(t.contains("mixer ATTENTION 0 prefill 2 NORMED->BRANCH"));
        assertEquals(6, memory.sequence().nPast());
    }

    @Test
    public void forwardAllEvaluatesMemorylessModelsInOneStep() {
        GpuForwardContext ctx = open(llama(1, null, null, MemoryKind.NONE), false);
        GpuForwardBackend fb = new GpuForwardBackend(ctx, registry(), Collections.emptyList());
        closeables.add(fb);
        startTrace();

        float[][] rows = fb.forwardAll(new int[]{1, 2, 3}, new int[]{2, 0});

        assertEquals(2, rows.length);
        assertEquals(VOCAB, rows[0].length);
        List<OpKind> k = kinds(backend.ops());
        assertEquals(1, Collections.frequency(k, OpKind.FINISH));
        assertEquals(1, Collections.frequency(k, OpKind.READ));
        assertTrue(trace().contains("mixer ATTENTION 0 prefill 3 NORMED->BRANCH"));
        assertThrows(IllegalStateException.class, () -> fb.decode(1, 0));
    }

    // ------------------------------------------------------------------ built-in extras

    @Test
    public void builtInExtrasServeTheirFeatureIds() {
        List<String> ids = new ArrayList<>();
        for (GpuLayerExtraFactory f : BuiltInGpuExtras.factories()) {
            ids.add(f.featureId());
        }
        assertEquals(Arrays.asList(LayerSpec.Extras.EMBD_SKIP, LayerSpec.Extras.RESID_ALPHA_BETA,
                LayerSpec.Extras.LOOP_NORM), ids);
        assertSame(BuiltInGpuExtras.factories(), BuiltInGpuExtras.factories());
    }
}

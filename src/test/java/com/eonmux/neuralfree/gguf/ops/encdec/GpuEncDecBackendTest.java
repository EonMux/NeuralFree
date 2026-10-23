package com.eonmux.neuralfree.gguf.ops.encdec;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.BuiltInGpuExtras;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec.BiasMode;
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
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Dispatch;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Op;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.OpKind;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;
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
 * Traces of the T5 encoder-decoder flow on the GPU ({@link GpuEncDecBackend}) against the host-only fake backend with
 * strict LLM argument rules: a prompt is encoded (the {@code enc.blk} stack, {@code enc.output_norm}, the cross K/V
 * build of every decoder layer), the decoder start token runs at position 0, and two decode steps follow at positions
 * 1 and 2 over the cross caches. Self-attention mixers and FFN blocks are recording stand-ins; the cross-attention
 * mixer is the real {@link GpuCrossAttentionMixer}.
 */
public class GpuEncDecBackendTest {

    private static final int WIDTH = 128;
    private static final int N_HEAD = 2;
    private static final int HEAD_DIM = 64;
    private static final int N_FF = 256;
    private static final int VOCAB = 32;
    private static final int BUCKETS = 32;
    private static final int MAX_DISTANCE = 128;
    private static final int N_ENC_LAYER = 2;
    private static final int N_DEC_LAYER = 2;
    private static final int UBATCH = 8;
    private static final int N_CTX = 64;
    private static final float RMS_EPS = 1e-6f;
    private static final int DECLARED_START = 0;
    private static final int VOCAB_BOS = 3;
    private static final int[] SOURCE = {5, 9, 11, 2, 1};

    private final List<String> events = new ArrayList<>();
    private final List<Integer> eventAt = new ArrayList<>();
    private FakeComputeBackend backend;
    private LlmKernels kernels;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private DeviceModelMemory memory;
    private Map<String, GpuWeight> weights;
    private GpuEncDecBackend fb;

    @After
    public void tearDown() {
        if (fb != null) {
            fb.close();
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
        events.add(label);
        eventAt.add(backend.dispatches().size());
    }

    private final class RecordingMixer implements GpuMixer {
        private String name;

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            this.name = layer.stack() + " " + layer.il();
        }

        @Override
        public void decode(int pos, ScratchSlot in, ScratchSlot out) {
            record("attn " + name + " decode " + pos);
        }

        @Override
        public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
            record("attn " + name + " prefill " + pos0 + "+" + nTokens);
        }
    }

    private final class RecordingFfn implements GpuFfnBlock {
        private String name;

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            this.name = layer.stack() + " " + layer.il();
        }

        @Override
        public void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
            record("ffn " + name + " " + nTokens);
        }
    }

    private GpuBlockRegistry registry() {
        GpuMixerFactory attention = new GpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return MixerKind.ATTENTION;
            }

            @Override
            public GpuMixer create(LayerSpec layer) {
                return new RecordingMixer();
            }
        };
        GpuFfnFactory dense = new GpuFfnFactory() {
            @Override
            public FfnKind kind() {
                return FfnKind.DENSE;
            }

            @Override
            public GpuFfnBlock create(LayerSpec layer) {
                return new RecordingFfn();
            }
        };
        List<GpuMixerFactory> mixers = new ArrayList<>();
        mixers.add(attention);
        mixers.addAll(EncDecBlocks.gpuMixers());
        return new GpuBlockRegistry(mixers, Collections.singletonList(dense), BuiltInGpuExtras.factories());
    }

    // ------------------------------------------------------------------ fixtures

    private static String layer(String stack, int il, String base) {
        return TensorNames.layerName(stack, il, TensorNames.weight(base));
    }

    private static GGUFTensorInfo info(String name, long... dims) {
        return new GGUFTensorInfo(name, dims, GGMLType.F32, 0);
    }

    private static List<GGUFTensorInfo> tensors() {
        int dim = N_HEAD * HEAD_DIM;
        List<GGUFTensorInfo> t = new ArrayList<>();
        t.add(info(TensorNames.weight(TensorNames.TOKEN_EMBD), WIDTH, VOCAB));
        t.add(info(TensorNames.weight(TensorNames.ENC_OUTPUT_NORM), WIDTH));
        t.add(info(TensorNames.weight(TensorNames.DEC_OUTPUT_NORM), WIDTH));
        t.add(info(TensorNames.weight(TensorNames.OUTPUT), WIDTH, VOCAB));
        for (int il = 0; il < N_ENC_LAYER; il++) {
            t.add(info(layer(TensorNames.STACK_ENC, il, TensorNames.ATTN_NORM), WIDTH));
            t.add(info(layer(TensorNames.STACK_ENC, il, TensorNames.FFN_NORM), WIDTH));
        }
        for (int il = 0; il < N_DEC_LAYER; il++) {
            t.add(info(layer(TensorNames.STACK_DEC, il, TensorNames.ATTN_NORM), WIDTH));
            t.add(info(layer(TensorNames.STACK_DEC, il, TensorNames.DEC_CROSS_ATTN_NORM), WIDTH));
            t.add(info(layer(TensorNames.STACK_DEC, il, TensorNames.FFN_NORM), WIDTH));
            t.add(info(layer(TensorNames.STACK_DEC, il, TensorNames.DEC_CROSS_ATTN_Q), WIDTH, dim));
            t.add(info(layer(TensorNames.STACK_DEC, il, TensorNames.DEC_CROSS_ATTN_K), WIDTH, dim));
            t.add(info(layer(TensorNames.STACK_DEC, il, TensorNames.DEC_CROSS_ATTN_V), WIDTH, dim));
            t.add(info(layer(TensorNames.STACK_DEC, il, TensorNames.DEC_CROSS_ATTN_OUT), dim, WIDTH));
        }
        return t;
    }

    private static AttentionSpec.Builder attention(int il) {
        return AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD).headDimK(HEAD_DIM).headDimV(HEAD_DIM)
                .noRope(true).nRot(0).kqScale(1.0f).kvOwnerLayer(il);
    }

    private static LayerSpec.Builder block(int il, MixerKind mixer, String stack) {
        return LayerSpec.builder(il, mixer, FfnKind.DENSE)
                .stack(stack)
                .norm(NormSpec.builder(NormKind.RMS, RMS_EPS).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, WIDTH).build())
                .ffn(FfnSpec.builder(FfnSpec.Role.DENSE).nFf(N_FF).build());
    }

    /** T5 with a bidirectional, memoryless encoder and a decoder of causal self- plus cross-attention layers. */
    private static ModelSpec t5(int decoderStart) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.T5)).layerCounts(N_ENC_LAYER, 0)
                .decNLayer(N_DEC_LAYER).nCtxTrain(512).nEmbd(WIDTH).nVocab(VOCAB).normRmsEps(RMS_EPS)
                .headKFull(HEAD_DIM).headVFull(HEAD_DIM).nRelAttnBuckets(BUCKETS).decStartTokenId(decoderStart);
        for (int il = 0; il < N_ENC_LAYER; il++) {
            hb.nHead(il, N_HEAD).nHeadKv(il, N_HEAD);
        }
        TensorCatalog tc = TensorCatalog.of(Arch.T5.ggufName(), tensors());
        ModelSpec.Builder b = ModelSpec.builder(hb.build(), tc)
                .schedule(LayerSchedule.linear(N_DEC_LAYER))
                .embedding(EmbeddingSpec.builder(tc.get(TensorNames.weight(TensorNames.TOKEN_EMBD)), WIDTH).build())
                .head(HeadSpec.builder()
                        .outputNorm(tc.get(TensorNames.weight(TensorNames.DEC_OUTPUT_NORM)), null)
                        .output(tc.get(TensorNames.weight(TensorNames.OUTPUT)), false)
                        .build());
        for (int il = 0; il < N_ENC_LAYER; il++) {
            b.encoderLayer(block(il, MixerKind.ATTENTION, TensorNames.STACK_ENC)
                    .attention(attention(il).causal(false).usesMemory(false).biasMode(BiasMode.T5_BIDIR)
                            .t5Buckets(BUCKETS).t5MaxDistance(MAX_DISTANCE).t5BiasLayer(0).build())
                    .build());
        }
        for (int il = 0; il < N_DEC_LAYER; il++) {
            b.layer(block(il, MixerKind.CROSS_ATTENTION_T5, TensorNames.STACK_DEC)
                    .attention(attention(il).causal(true).biasMode(BiasMode.T5_CAUSAL).t5Buckets(BUCKETS)
                            .t5MaxDistance(MAX_DISTANCE).t5BiasLayer(0).build())
                    .crossAttention(attention(il).causal(false).build())
                    .build());
        }
        return b.build();
    }

    /** Opens the fake device, F32 weights for every tensor of the spec, the decoder memory and the context. */
    private GpuForwardContext open(ModelSpec spec) {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        kernels = LlmKernels.open(backend);
        scratch = GpuScratch.forModel(backend, spec, UBATCH);
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
        weights = new HashMap<>();
        TensorCatalog tc = spec.tensors();
        for (String name : tc.names()) {
            GGUFTensorInfo info = tc.info(name);
            weights.put(name, GpuWeight.of(name, GGMLType.F32, info.dims().clone(),
                    backend.allocateBuffer(info.numElements() * Float.BYTES, true)));
        }
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16).ubatch(UBATCH).build();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX);
        memory = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan,
                spec.hparams().mropeTextPositions(), null);
        return GpuForwardContext.builder(kernels, ops, weights::get, spec).memory(memory).build();
    }

    private GpuEncDecBackend backend(int decoderStart, int vocabBos) {
        fb = new GpuEncDecBackend(open(t5(decoderStart)), registry(), Collections.emptyList(), vocabBos);
        return fb;
    }

    private void startTrace() {
        backend.clearLog();
        events.clear();
        eventAt.clear();
    }

    /** Dispatched kernel names with the recording block calls inserted where they happened. */
    private List<String> trace() {
        List<Dispatch> ds = backend.dispatches();
        List<String> out = new ArrayList<>();
        int e = 0;
        for (int i = 0; i <= ds.size(); i++) {
            while (e < events.size() && eventAt.get(e) == i) {
                out.add(events.get(e));
                e++;
            }
            if (i < ds.size()) {
                out.add(ds.get(i).kernelName());
            }
        }
        return out;
    }

    private static List<OpKind> kinds(List<Op> log) {
        List<OpKind> out = new ArrayList<>();
        for (Op op : log) {
            out.add(op.kind());
        }
        return out;
    }

    private int firstDispatchUsing(GpuWeight w) {
        List<Dispatch> ds = backend.dispatches();
        for (int i = 0; i < ds.size(); i++) {
            if (ds.get(i).bufferArgs().contains(w.shard(0))) {
                return i;
            }
        }
        throw new AssertionError(w.name() + " is not used by any dispatch");
    }

    private int eventDispatchIndex(String label) {
        int at = events.indexOf(label);
        assertTrue("missing block call '" + label + "' in " + events, at >= 0);
        return eventAt.get(at);
    }

    // ------------------------------------------------------------------ encode + decode

    @Test
    public void promptIsEncodedOnceThenTheDecoderStartsAtZeroAndDecodesTwoSteps() {
        GpuEncDecBackend enc = backend(DECLARED_START, VOCAB_BOS);
        assertEquals(DECLARED_START, enc.decoderStartToken());
        assertEquals(UBATCH, enc.maxSourceTokens());
        startTrace();

        float[] start = enc.prefill(SOURCE, 0);

        assertEquals(VOCAB, start.length);
        assertTrue(enc.encoded());
        assertEquals(1, memory.sequence().nPast());
        assertArrayEquals(new int[]{DECLARED_START}, memory.sequence().tokens());
        List<OpKind> k = kinds(backend.ops());
        assertEquals(1, Collections.frequency(k, OpKind.FINISH));
        assertEquals(OpKind.FINISH, k.get(k.size() - 1));

        int n = SOURCE.length;
        int enc0 = eventDispatchIndex("attn enc.blk 0 prefill 0+" + n);
        int enc1 = eventDispatchIndex("attn enc.blk 1 prefill 0+" + n);
        assertTrue(events.contains("ffn enc.blk 0 " + n));
        assertTrue(events.contains("ffn enc.blk 1 " + n));
        int encNorm = firstDispatchUsing(weights.get(TensorNames.weight(TensorNames.ENC_OUTPUT_NORM)));
        assertEquals(NormKernels.ROW_NORM, backend.dispatches().get(encNorm).kernelName());
        List<Dispatch> stores = backend.dispatchesOf(RopeKernels.kvStore(KvDType.F16));
        assertEquals(2 * N_DEC_LAYER, stores.size());
        for (Dispatch s : stores) {
            assertEquals(n, s.intArg(RopeKernels.ARG_N_TOKENS));
        }
        int firstStore = backend.dispatches().indexOf(stores.get(0));
        int lastStore = backend.dispatches().indexOf(stores.get(stores.size() - 1));
        int decoderStart = events.indexOf("attn dec.blk 0 decode 0") >= 0
                ? eventDispatchIndex("attn dec.blk 0 decode 0") : eventDispatchIndex("attn dec.blk 0 prefill 0+1");
        assertTrue(enc0 <= enc1);
        assertTrue(enc1 < encNorm);
        assertTrue(encNorm < firstStore);
        assertTrue(lastStore < decoderStart);
        assertEquals(N_DEC_LAYER, backend.dispatchesOf(AttnKernels.flashDecode(KvDType.F16)).size()
                + backend.dispatchesOf(AttnKernels.flashAttn(KvDType.F16)).size());

        for (int step = 1; step <= 2; step++) {
            int token = 6 + step;
            startTrace();
            float[] logits = enc.decode(token, step);
            assertEquals(VOCAB, logits.length);
            assertEquals(step + 1, memory.sequence().nPast());
            List<String> t = trace();
            assertTrue(t.contains("attn dec.blk 0 decode " + step));
            assertTrue(t.contains("attn dec.blk 1 decode " + step));
            assertFalse(t.contains("attn enc.blk 0 prefill 0+" + n));
            assertTrue(backend.dispatchesOf(RopeKernels.kvStore(KvDType.F16)).isEmpty());
            List<Dispatch> cross = backend.dispatchesOf(AttnKernels.flashDecode(KvDType.F16));
            assertEquals(N_DEC_LAYER, cross.size());
            for (Dispatch d : cross) {
                assertEquals(0, d.intArg(AttnKernels.ARG_FD_KV_START));
                assertEquals(n, d.intArg(AttnKernels.ARG_FD_KV_END));
                assertEquals(step, d.intArg(AttnKernels.ARG_FD_Q_POS));
            }
        }
        assertArrayEquals(new int[]{DECLARED_START, 7, 8}, memory.sequence().tokens());
    }

    @Test
    public void reevaluatingADecoderPositionRollsBackAndKeepsTheEncoding() {
        GpuEncDecBackend enc = backend(DECLARED_START, VOCAB_BOS);
        enc.prefill(SOURCE, 0);
        enc.decode(7, 1);
        enc.decode(8, 1);
        assertTrue(enc.encoded());
        assertArrayEquals(new int[]{DECLARED_START, 8}, memory.sequence().tokens());
        assertTrue(enc.decodeGreedyOnDevice(9, 2) >= 0);
        assertEquals(3, memory.sequence().nPast());
    }

    // ------------------------------------------------------------------ start token

    @Test
    public void startTokenFallsBackToTheVocabularyBos() {
        assertEquals(VOCAB_BOS, backend(-1, VOCAB_BOS).decoderStartToken());
    }

    @Test
    public void missingStartTokenIsRejectedAtConstruction() {
        GpuForwardContext ctx = open(t5(-1));
        assertThrows(IllegalStateException.class,
                () -> new GpuEncDecBackend(ctx, registry(), Collections.emptyList(), SpecialTokens.NULL));
    }

    @Test
    public void startTokenOutsideTheVocabularyIsRejectedAtConstruction() {
        GpuForwardContext ctx = open(t5(VOCAB));
        assertThrows(IllegalStateException.class,
                () -> new GpuEncDecBackend(ctx, registry(), Collections.emptyList(), VOCAB_BOS));
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void sourceLongerThanOneUbatchIsRefusedBeforeAnythingChanges() {
        GpuEncDecBackend enc = backend(DECLARED_START, VOCAB_BOS);
        enc.prefill(SOURCE, 0);
        int[] tooLong = new int[UBATCH + 1];
        Arrays.fill(tooLong, 4);
        ContextFullException e = assertThrows(ContextFullException.class, () -> enc.prefill(tooLong, 0));
        assertTrue(e.getCause() instanceof IllegalArgumentException);
        assertTrue(enc.encoded());
        assertEquals(1, memory.sequence().nPast());
        assertThrows(IllegalArgumentException.class, () -> enc.prefill(SOURCE, 1));
        assertThrows(IllegalArgumentException.class, () -> enc.prefill(new int[]{VOCAB}, 0));
        assertThrows(IllegalArgumentException.class, () -> enc.prefill(new int[0], 0));
        assertEquals(1, memory.sequence().nPast());
    }

    @Test
    public void decoderCallsNeedACurrentEncoding() {
        GpuEncDecBackend enc = backend(DECLARED_START, VOCAB_BOS);
        assertThrows(IllegalStateException.class, () -> enc.decode(1, 0));
        enc.prefill(SOURCE, 0);
        enc.reset();
        assertFalse(enc.encoded());
        assertEquals(0, memory.sequence().nPast());
        assertThrows(IllegalStateException.class, () -> enc.decode(1, 0));
        assertThrows(IllegalStateException.class, () -> enc.forwardRows(new int[]{1}, 0, null));
        assertThrows(UnsupportedOperationException.class, () -> enc.forwardAll(SOURCE, null));
    }

    // ------------------------------------------------------------------ encoder output

    @Test
    public void hiddenRowsReadTheNormedEncoderOutputWithoutTouchingTheDecoder() {
        GpuEncDecBackend enc = backend(DECLARED_START, VOCAB_BOS);
        enc.prefill(SOURCE, 0);
        startTrace();

        float[][] rows = enc.hiddenRows(new int[]{1, 2, 3}, null);

        assertEquals(3, rows.length);
        assertEquals(WIDTH, rows[0].length);
        assertTrue(events.contains("attn enc.blk 1 prefill 0+3"));
        for (String e : events) {
            assertFalse(e, e.contains(TensorNames.STACK_DEC));
        }
        assertTrue(backend.dispatchesOf(RopeKernels.kvStore(KvDType.F16)).isEmpty());
        List<Op> log = backend.ops();
        List<OpKind> k = kinds(log);
        assertEquals(1, Collections.frequency(k, OpKind.READ));
        assertEquals((long) 3 * WIDTH * Float.BYTES, log.get(k.lastIndexOf(OpKind.READ)).bytes());
        assertEquals(1, memory.sequence().nPast());
        assertTrue(enc.encoded());
        assertThrows(IllegalArgumentException.class, () -> enc.hiddenRows(new int[UBATCH + 1], null));
    }
}

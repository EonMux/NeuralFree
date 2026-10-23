package com.eonmux.neuralfree.gguf.ops.act;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
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
import com.eonmux.neuralfree.gguf.ops.gemma3n.Gemma3nKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
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
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
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
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

/**
 * Dispatch traces of {@link GpuDenseFfn} on the host-only fake backend with strict LLM argument rules: one test per
 * supported FFN form (llama SwiGLU, phi3 fused up, gpt2 ungated GELU with biases, gemma GEGLU, gemma3n
 * activation sparsity, nemotron ReLU², apertus xIELU, bitnet sub-norm, mpt activation scales), the bias-then-scale
 * order of {@code w_s}, the tensor-scale path into an offset output, and the binding errors.
 */
public class GpuDenseFfnTest {

    private static final int N_EMBD = 64;
    private static final int N_FF = 128;
    private static final int UBATCH = 8;
    private static final String MATVEC = "Float_MatVec";
    private static final String BATCH_2D = "LLM_Batch_Float_MatMul_2D";

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private int local;
    private final Map<String, GpuWeight> tensors = new HashMap<>();
    private GpuDenseFfn block;

    private void open() {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.NORMED, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.BRANCH, 2L * UBATCH * N_EMBD)
                .slot(ScratchSlots.FFN_GATE_UP, 4L * UBATCH * N_FF)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
        local = ops.local();
    }

    @After
    public void tearDown() {
        if (block != null) {
            block.close();
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

    // ------------------------------------------------------------------ fixtures

    private GpuWeight matrix(String suffix, int cols, int rows) {
        return put(suffix, new long[]{cols, rows});
    }

    private GpuWeight vector(String suffix, int n) {
        return put(suffix, new long[]{n});
    }

    private GpuWeight put(String suffix, long[] ne) {
        String name = TensorNames.layerName(0, suffix);
        long n = 1;
        for (long d : ne) {
            n *= d;
        }
        GpuWeight w = GpuWeight.of(name, GGMLType.F32, ne, backend.allocateBuffer(n * Float.BYTES, true));
        tensors.put(name, w);
        return w;
    }

    private void putWeight(GpuWeight w) {
        tensors.put(w.name(), w);
    }

    private static FfnSpec.Builder ffn() {
        return FfnSpec.builder(FfnSpec.Role.DENSE).nFf(N_FF);
    }

    private static LayerSpec layer(FfnSpec spec, NormSpec norm) {
        return LayerSpec.builder(0, MixerKind.NONE, FfnKind.DENSE)
                .norm(norm)
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .ffn(spec)
                .build();
    }

    private static LayerSpec layer(FfnSpec spec) {
        return layer(spec, NormSpec.builder(NormKind.RMS, 1e-5f).build());
    }

    private GpuForwardContext context(Arch arch, LayerSpec l) {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(1, 0).nCtxTrain(4096)
                .nEmbd(N_EMBD).build();
        ModelSpec spec = ModelSpec.builder(hp, TensorCatalog.of(arch.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build())
                .layer(l)
                .build();
        GpuWeights weights = new GpuWeights() {
            @Override
            public GpuWeight find(String name) {
                return tensors.get(name);
            }
        };
        return GpuForwardContext.builder(ops.kernels(), ops, weights, spec).build();
    }

    private GpuDenseFfn bind(Arch arch, LayerSpec l) {
        GpuFfnFactory f = FfnBlocks.gpuFfns().get(0);
        block = (GpuDenseFfn) f.create(l);
        block.bind(context(arch, l), l);
        return block;
    }

    private List<Dispatch> run(int nTokens, ScratchSlot out) {
        backend.clearLog();
        block.forward(nTokens, scratch.slot(ScratchSlots.NORMED), out);
        return backend.dispatches();
    }

    private List<Dispatch> run(int nTokens) {
        return run(nTokens, scratch.slot(ScratchSlots.BRANCH));
    }

    private GpuBuffer gateUp() {
        return scratch.slot(ScratchSlots.FFN_GATE_UP).buffer();
    }

    private GpuBuffer normed() {
        return scratch.slot(ScratchSlots.NORMED).buffer();
    }

    private GpuBuffer branch() {
        return scratch.slot(ScratchSlots.BRANCH).buffer();
    }

    private static List<String> names(List<Dispatch> ds) {
        List<String> out = new ArrayList<>();
        for (Dispatch d : ds) {
            out.add(d.kernelName());
        }
        return out;
    }

    private static int[] ints(Dispatch d, int from, int to) {
        int[] v = new int[to - from];
        for (int i = from; i < to; i++) {
            v[i - from] = d.intArg(i);
        }
        return v;
    }

    private long roundUp(long n) {
        return (n + local - 1) / local * local;
    }

    private void assertMatvec(Dispatch d, String kernel, GpuWeight w, GpuBuffer in, GpuBuffer out) {
        assertEquals(kernel, d.kernelName());
        assertSame(w.shard(0), d.bufferArg(MATVEC.equals(kernel) ? 2 : 3));
        assertSame(in, d.bufferArg(MATVEC.equals(kernel) ? 3 : 4));
        assertSame(out, d.bufferArg(MATVEC.equals(kernel) ? 4 : 5));
    }

    // ------------------------------------------------------------------ llama: SwiGLU, separate gate

    @Test
    public void llamaDecodeProjectsGateAndUpThenSwigluThenDown() {
        open();
        GpuWeight gate = matrix("ffn_gate.weight", N_EMBD, N_FF);
        GpuWeight upW = matrix("ffn_up.weight", N_EMBD, N_FF);
        GpuWeight down = matrix("ffn_down.weight", N_FF, N_EMBD);
        bind(Arch.LLAMA, layer(ffn().glu(FfnGating.PAR, GluOp.SWIGLU).build()));
        List<Dispatch> ds = run(1);

        assertEquals(Arrays.asList(MATVEC, MATVEC, ActKernels.GLU, MATVEC), names(ds));
        assertMatvec(ds.get(0), MATVEC, gate, normed(), gateUp());
        GpuBuffer upTemp = ds.get(1).bufferArg(4);
        assertMatvec(ds.get(1), MATVEC, upW, normed(), upTemp);
        assertNotSame(gateUp(), upTemp);
        Dispatch glu = ds.get(2);
        assertArrayEquals(new int[]{N_FF, 1, GluOp.SWIGLU.code()}, ints(glu, 0, 3));
        assertArrayEquals(new int[]{N_FF, 0, N_FF, 0, N_FF, 0}, ints(glu, ActKernels.ARG_GLU_GATE_STRIDE,
                ActKernels.ARG_GLU_OUT_OFF + 1));
        assertSame(gateUp(), glu.bufferArg(ActKernels.ARG_GLU_GATE));
        assertSame(upTemp, glu.bufferArg(ActKernels.ARG_GLU_UP));
        assertSame(gateUp(), glu.bufferArg(ActKernels.ARG_GLU_OUT));
        assertArrayEquals(new long[]{roundUp(N_FF)}, glu.global());
        assertArrayEquals(new long[]{local}, glu.local());
        assertMatvec(ds.get(3), MATVEC, down, gateUp(), branch());
    }

    @Test
    public void llamaPrefillReSetsRowCountsOnly() {
        open();
        matrix("ffn_gate.weight", N_EMBD, N_FF);
        matrix("ffn_up.weight", N_EMBD, N_FF);
        matrix("ffn_down.weight", N_FF, N_EMBD);
        bind(Arch.LLAMA, layer(ffn().glu(FfnGating.PAR, GluOp.SWIGLU).build()));
        List<Dispatch> ds = run(4);

        assertEquals(Arrays.asList(BATCH_2D, BATCH_2D, ActKernels.GLU, BATCH_2D), names(ds));
        Dispatch glu = ds.get(2);
        assertEquals(4, glu.intArg(ActKernels.ARG_GLU_N_ROWS));
        assertArrayEquals(new long[]{roundUp(4L * N_FF)}, glu.global());
        assertArrayEquals(new int[]{N_EMBD, N_FF, 4}, ints(ds.get(3), 0, 3));
    }

    // ------------------------------------------------------------------ phi3: fused gate+up

    @Test
    public void phi3FusedUpSplitsHalvesIntoTheActivationTemporary() {
        open();
        GpuWeight upW = matrix("ffn_up.weight", N_EMBD, 2 * N_FF);
        GpuWeight down = matrix("ffn_down.weight", N_FF, N_EMBD);
        bind(Arch.PHI3, layer(ffn().glu(FfnGating.FUSED_UP, GluOp.SWIGLU).build()));
        List<Dispatch> ds = run(2);

        assertEquals(Arrays.asList(BATCH_2D, ActKernels.GLU, BATCH_2D), names(ds));
        assertMatvec(ds.get(0), BATCH_2D, upW, normed(), gateUp());
        Dispatch glu = ds.get(1);
        assertArrayEquals(new int[]{N_FF, 2, GluOp.SWIGLU.code()}, ints(glu, 0, 3));
        assertArrayEquals(new int[]{2 * N_FF, 0, 2 * N_FF, N_FF, N_FF, 0}, ints(glu,
                ActKernels.ARG_GLU_GATE_STRIDE, ActKernels.ARG_GLU_OUT_OFF + 1));
        assertSame(gateUp(), glu.bufferArg(ActKernels.ARG_GLU_GATE));
        assertSame(gateUp(), glu.bufferArg(ActKernels.ARG_GLU_UP));
        GpuBuffer act = glu.bufferArg(ActKernels.ARG_GLU_OUT);
        assertNotSame(gateUp(), act);
        assertMatvec(ds.get(2), BATCH_2D, down, act, branch());
    }

    // ------------------------------------------------------------------ gpt2: ungated GELU with biases

    @Test
    public void gpt2UngatedGeluAddsBiasesAfterEachProjection() {
        open();
        matrix("ffn_up.weight", N_EMBD, N_FF);
        GpuWeight upB = vector("ffn_up.bias", N_FF);
        matrix("ffn_down.weight", N_FF, N_EMBD);
        GpuWeight downB = vector("ffn_down.bias", N_EMBD);
        bind(Arch.GPT2, layer(ffn().act(false, ActOp.GELU_TANH).build()));
        List<Dispatch> ds = run(3);

        assertEquals(Arrays.asList(BATCH_2D, GpuDenseFfn.BATCH_BIAS_ADD, ActKernels.UNARY, BATCH_2D,
                GpuDenseFfn.BATCH_BIAS_ADD), names(ds));
        Dispatch b1 = ds.get(1);
        assertArrayEquals(new int[]{N_FF, 3}, ints(b1, 0, 2));
        assertSame(upB.shard(0), b1.bufferArg(GpuDenseFfn.ARG_BIAS_BIAS));
        assertSame(gateUp(), b1.bufferArg(GpuDenseFfn.ARG_BIAS_DATA));
        assertArrayEquals(new long[]{roundUp(N_FF)}, b1.global());
        Dispatch act = ds.get(2);
        assertArrayEquals(new int[]{3 * N_FF, ActOp.GELU_TANH.code()}, ints(act, 0, 2));
        assertSame(gateUp(), act.bufferArg(ActKernels.ARG_UNARY_IN));
        assertSame(gateUp(), act.bufferArg(ActKernels.ARG_UNARY_OUT));
        Dispatch b2 = ds.get(4);
        assertArrayEquals(new int[]{N_EMBD, 3}, ints(b2, 0, 2));
        assertSame(downB.shard(0), b2.bufferArg(GpuDenseFfn.ARG_BIAS_BIAS));
        assertSame(branch(), b2.bufferArg(GpuDenseFfn.ARG_BIAS_DATA));
    }

    @Test
    public void scaledBiasedProjectionAddsTheBiasBeforeTheScale() {
        open();
        putWeight(matrix("ffn_up.weight", N_EMBD, N_FF).withScale(0.5f));
        vector("ffn_up.bias", N_FF);
        matrix("ffn_down.weight", N_FF, N_EMBD);
        bind(Arch.GPT2, layer(ffn().act(false, ActOp.GELU_TANH).build()));
        List<Dispatch> ds = run(1);

        assertEquals(Arrays.asList(MATVEC, GpuDenseFfn.BATCH_BIAS_ADD, GpuOps.AXPBY, ActKernels.UNARY, MATVEC),
                names(ds));
        Dispatch scale = ds.get(2);
        assertEquals(N_FF, scale.intArg(0));
        assertEquals(0.0f, scale.floatArg(1), 0.0f);
        assertEquals(0.5f, scale.floatArg(2), 0.0f);
        assertSame(gateUp(), scale.bufferArg(8));
    }

    @Test
    public void tensorScaledDownIntoAnOffsetOutputGoesThroughTheDownTemporary() {
        open();
        matrix("ffn_up.weight", N_EMBD, N_FF);
        matrix("ffn_down.weight", N_FF, N_EMBD);
        GpuWeight downS = vector("ffn_down.scale", 1);
        bind(Arch.NEMOTRON, layer(ffn().act(false, ActOp.RELU_SQR).build()));
        ScratchSlot out = scratch.slot(ScratchSlots.BRANCH).slice(N_EMBD, (long) UBATCH * N_EMBD);
        List<Dispatch> ds = run(1, out);

        assertEquals(Arrays.asList(MATVEC, ActKernels.UNARY, MATVEC, NormKernels.MUL_ROW_SCALE, GpuOps.COPY2D),
                names(ds));
        assertEquals(ActOp.RELU_SQR.code(), ds.get(1).intArg(ActKernels.ARG_UNARY_OP));
        GpuBuffer temp = ds.get(2).bufferArg(4);
        assertNotSame(branch(), temp);
        Dispatch rs = ds.get(3);
        assertArrayEquals(new int[]{1, N_EMBD, 1}, ints(rs, 0, 3));
        assertSame(downS.shard(0), rs.bufferArg(NormKernels.ARG_ROW_SCALE_SCALE));
        assertSame(temp, rs.bufferArg(NormKernels.ARG_ROW_SCALE_X));
        Dispatch cp = ds.get(4);
        assertArrayEquals(new int[]{1, N_EMBD, N_EMBD, 0, N_EMBD, N_EMBD}, ints(cp, 0, 6));
        assertSame(temp, cp.bufferArg(6));
        assertSame(branch(), cp.bufferArg(7));
    }

    // ------------------------------------------------------------------ gemma / gemma3n: GEGLU, sparsity

    @Test
    public void gemmaUsesTanhGeglu() {
        open();
        matrix("ffn_gate.weight", N_EMBD, N_FF);
        matrix("ffn_up.weight", N_EMBD, N_FF);
        matrix("ffn_down.weight", N_FF, N_EMBD);
        bind(Arch.GEMMA, layer(ffn().glu(FfnGating.PAR, GluOp.GEGLU_TANH).build()));
        List<Dispatch> ds = run(1);

        assertEquals(Arrays.asList(MATVEC, MATVEC, ActKernels.GLU, MATVEC), names(ds));
        assertEquals(GluOp.GEGLU_TANH.code(), ds.get(2).intArg(ActKernels.ARG_GLU_OP));
    }

    @Test
    public void gemma3nAppliesGaussianTopKToTheGateRowsBeforeTheGlu() {
        open();
        matrix("ffn_gate.weight", N_EMBD, N_FF);
        matrix("ffn_up.weight", N_EMBD, N_FF);
        matrix("ffn_down.weight", N_FF, N_EMBD);
        float k = 1.6448533535003662f;
        bind(Arch.GEMMA3N, layer(ffn().glu(FfnGating.PAR, GluOp.GEGLU_TANH).activationSparsityStdMul(k).build()));
        List<Dispatch> ds = run(2);

        assertEquals(Arrays.asList(BATCH_2D, Gemma3nKernels.GAUSSIAN_TOPK, BATCH_2D, ActKernels.GLU, BATCH_2D),
                names(ds));
        Dispatch topK = ds.get(1);
        assertArrayEquals(new int[]{2, N_FF, N_FF, 0}, ints(topK, 0, 4));
        assertEquals(k, topK.floatArg(Gemma3nKernels.ARG_TOPK_K_STD), 0.0f);
        assertSame(gateUp(), topK.bufferArg(5));
        int rowLocal = Gemma3nKernels.rowLocal(local);
        assertArrayEquals(new long[]{2L * rowLocal}, topK.global());
        assertArrayEquals(new long[]{rowLocal}, topK.local());
    }

    // ------------------------------------------------------------------ nemotron, apertus: unary activations

    @Test
    public void apertusPassesPreparedXieluParameters() {
        open();
        matrix("ffn_up.weight", N_EMBD, N_FF);
        matrix("ffn_down.weight", N_FF, N_EMBD);
        ActParams p = ActParams.builder().xieluRaw(0.8f, 0.8f, 0.5f, -1e-6f).build();
        bind(Arch.APERTUS, layer(ffn().act(false, ActOp.XIELU).params(p).build()));
        List<Dispatch> ds = run(1);

        assertEquals(Arrays.asList(MATVEC, ActKernels.UNARY, MATVEC), names(ds));
        Dispatch act = ds.get(1);
        assertEquals(ActOp.XIELU.code(), act.intArg(ActKernels.ARG_UNARY_OP));
        assertEquals(p.xieluAlphaN(), act.floatArg(ActKernels.ARG_UNARY_P0), 0.0f);
        assertEquals(p.xieluAlphaP(), act.floatArg(ActKernels.ARG_UNARY_P1), 0.0f);
        assertEquals(p.xieluBeta(), act.floatArg(ActKernels.ARG_UNARY_P2), 0.0f);
        assertEquals(p.xieluEps(), act.floatArg(ActKernels.ARG_UNARY_P3), 0.0f);
    }

    // ------------------------------------------------------------------ bitnet: sub-norm

    @Test
    public void bitnetNormalizesTheActivatedRowsBeforeTheDownProjection() {
        open();
        matrix("ffn_gate.weight", N_EMBD, N_FF);
        matrix("ffn_up.weight", N_EMBD, N_FF);
        GpuWeight down = matrix("ffn_down.weight", N_FF, N_EMBD);
        GpuWeight sub = vector("ffn_sub_norm.weight", N_FF);
        NormSpec norm = NormSpec.builder(NormKind.RMS, 1e-5f).ffnSubNorm(true).build();
        bind(Arch.BITNET, layer(ffn().glu(FfnGating.PAR, GluOp.SWIGLU).build(), norm));
        List<Dispatch> ds = run(3);

        assertEquals(Arrays.asList(BATCH_2D, BATCH_2D, ActKernels.GLU, NormKernels.ROW_NORM, BATCH_2D), names(ds));
        Dispatch rn = ds.get(3);
        assertArrayEquals(new int[]{3, N_FF, N_FF, NormKind.RMS.rowNormMode()}, ints(rn, 0, 4));
        assertEquals(1e-5f, rn.floatArg(NormKernels.ARG_ROW_NORM_EPS), 0.0f);
        assertEquals(NormKernels.FLAG_HAS_W, rn.intArg(NormKernels.ARG_ROW_NORM_FLAGS));
        assertSame(gateUp(), rn.bufferArg(NormKernels.ARG_ROW_NORM_X));
        assertSame(gateUp(), rn.bufferArg(NormKernels.ARG_ROW_NORM_Y));
        assertSame(sub.shard(0), rn.bufferArg(NormKernels.ARG_ROW_NORM_W));
        assertArrayEquals(new long[]{3L * local}, rn.global());
        assertMatvec(ds.get(4), BATCH_2D, down, gateUp(), branch());
    }

    // ------------------------------------------------------------------ mpt: activation scales

    @Test
    public void mptDividesTheGeluOutputByTheActivationScales() {
        open();
        matrix("ffn_up.weight", N_EMBD, N_FF);
        matrix("ffn_down.weight", N_FF, N_EMBD);
        GpuWeight scales = put("ffn.act.scales", new long[]{N_FF});
        float[] values = new float[N_FF];
        for (int i = 0; i < N_FF; i++) {
            values[i] = 0.25f * (i + 1);
        }
        ((FakeGpuBuffer) scales.shard(0)).putFloats(0, values);
        bind(Arch.MPT, layer(ffn().act(false, ActOp.GELU_TANH).build()));
        List<Dispatch> ds = run(2);

        assertEquals(Arrays.asList(BATCH_2D, ActKernels.UNARY, NormKernels.MUL_ROW_SCALE, BATCH_2D), names(ds));
        Dispatch rs = ds.get(2);
        assertArrayEquals(new int[]{2, N_FF, N_FF}, ints(rs, 0, 3));
        assertSame(gateUp(), rs.bufferArg(NormKernels.ARG_ROW_SCALE_X));
        float[] reciprocals = rs.bufferArg(NormKernels.ARG_ROW_SCALE_SCALE).getFloats(0, N_FF);
        for (int i = 0; i < N_FF; i++) {
            assertEquals(1.0f / values[i], reciprocals[i], 0.0f);
        }
    }

    // ------------------------------------------------------------------ binding errors

    @Test
    public void gateTensorOnAnUngatedSpecIsRejected() {
        open();
        matrix("ffn_gate.weight", N_EMBD, N_FF);
        matrix("ffn_up.weight", N_EMBD, N_FF);
        matrix("ffn_down.weight", N_FF, N_EMBD);
        LayerSpec l = layer(ffn().act(false, ActOp.GELU_TANH).build());
        assertThrows(IllegalStateException.class, () -> bind(Arch.GPT2, l));
    }

    @Test
    public void fusedUpOfTheWrongHeightIsRejected() {
        open();
        matrix("ffn_up.weight", N_EMBD, N_FF);
        matrix("ffn_down.weight", N_FF, N_EMBD);
        LayerSpec l = layer(ffn().glu(FfnGating.FUSED_UP, GluOp.SWIGLU).build());
        assertThrows(IllegalStateException.class, () -> bind(Arch.PHI3, l));
    }

    @Test
    public void outputSharingTheWorkingRowsIsRejected() {
        open();
        matrix("ffn_up.weight", N_EMBD, N_FF);
        matrix("ffn_down.weight", N_FF, N_EMBD);
        bind(Arch.NEMOTRON, layer(ffn().act(false, ActOp.RELU_SQR).build()));
        assertThrows(IllegalArgumentException.class,
                () -> block.forward(1, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.FFN_GATE_UP)));
    }

    @Test
    public void factoryRejectsLayersWithoutADenseFfn() {
        LayerSpec none = LayerSpec.builder(0, MixerKind.NONE, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-5f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .build();
        assertEquals(FfnKind.DENSE, FfnBlocks.gpuFfns().get(0).kind());
        assertEquals(FfnKind.DENSE, FfnBlocks.cpuFfns().get(0).kind());
        assertEquals(0, FfnBlocks.gpuMixers().size() + FfnBlocks.cpuMixers().size() + FfnBlocks.gpuExtras().size()
                + FfnBlocks.cpuExtras().size());
        assertThrows(IllegalArgumentException.class, () -> FfnBlocks.gpuFfns().get(0).create(none));
        assertThrows(IllegalArgumentException.class, () -> FfnBlocks.cpuFfns().get(0).create(none));
    }
}

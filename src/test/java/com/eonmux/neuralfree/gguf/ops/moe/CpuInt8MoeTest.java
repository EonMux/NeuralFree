package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnBlock;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuInt8Fusion;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The fused int8 expert pass ({@link CpuInt8Moe}) against the generic per-pair path over the same weights: a
 * one-layer qwen3next file whose projections are {@code Q8_0} int8 blocks runs the fused pass and returns the
 * generic results bit for bit, with gated and ungated routed experts, an {@code F32} file keeps the generic path,
 * and the fused output matches an independent per-pair reference of {@code build_moe_ffn}.
 */
public class CpuInt8MoeTest {

    private static final int UBATCH = 8;
    private static final int N_EXPERT = 4;
    private static final int K = 2;
    private static final int N_FF = 32;

    /** A one-layer qwen3next MoE file in {@code weightType}, its weights and a spec whose only layer is the MoE. */
    private static final class Fixture implements AutoCloseable {
        final Path path;
        final GGUFFileParser parser;
        final CpuWeights weights;
        final LayerSpec layer;
        final CpuForwardContext ctx;
        final int nEmbd;

        Fixture(GGMLType weightType) throws IOException {
            this(weightType, true, false);
        }

        Fixture(GGMLType weightType, boolean gated) throws IOException {
            this(weightType, gated, false);
        }

        /**
         * @param gated   whether the routed experts use a GLU ({@code ffn_gate_exps}) or an activation alone
         * @param biases  whether the file carries the per-expert {@code ffn_{gate,up,down}_exps.bias} rows
         */
        Fixture(GGMLType weightType, boolean gated, boolean biases) throws IOException {
            SyntheticModels.Config config = SyntheticModels.Config.qwen3next().toBuilder().nLayer(1)
                    .weightType(weightType).expertBiases(biases).build();
            this.path = SyntheticModels.qwen3next(config).writeTempFile();
            this.parser = new GGUFFileParser(path.toString());
            TensorCatalog tc = TensorCatalog.of(parser);
            ModelHparams hp = HparamsLoader.load(GgufKv.of(parser));
            this.nEmbd = hp.nEmbd();
            FfnSpec shexp = FfnSpec.builder(FfnSpec.Role.SHARED_EXPERT).nFf(N_FF).glu(FfnGating.PAR, GluOp.SWIGLU)
                    .build();
            MoeSpec.Builder routed = MoeSpec.builder().experts(N_EXPERT, K, N_FF).gating(GatingFunc.SOFTMAX)
                    .normWeights(true);
            MoeSpec moe = (gated ? routed.glu(GluOp.SWIGLU) : routed.act(ActOp.SILU))
                    .sharedExpert(shexp, true).sharedCombineScale(1.0f).expertBiases(biases).build();
            this.layer = LayerSpec.builder(0, MixerKind.NONE, FfnKind.MOE)
                    .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                    .residual(ResidualSpec.builder(ResidualKind.STANDARD, nEmbd).build())
                    .moe(moe)
                    .build();
            ModelSpec spec = ModelSpec.builder(hp, tc)
                    .layer(layer)
                    .schedule(LayerSchedule.linear(1))
                    .embedding(EmbeddingSpec.builder(null, nEmbd).build())
                    .head(HeadSpec.builder().build())
                    .memoryKind(MemoryKind.NONE)
                    .build();
            this.weights = CpuWeights.open(parser);
            this.ctx = CpuForwardContext.builder(spec, weights, CpuScratch.forModel(spec, UBATCH)).build();
        }

        CpuTensorRef tensor(String suffix) {
            return weights.layer(layer, TensorNames.weight(suffix));
        }

        @Override
        public void close() throws IOException {
            try {
                weights.close();
            } finally {
                try {
                    parser.close();
                } finally {
                    Files.deleteIfExists(path);
                }
            }
        }
    }

    private static float[] input(int nEmbd, long seed) {
        Random rnd = new Random(seed);
        float[] x = new float[UBATCH * nEmbd];
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }

    /** Runs the routed MoE block of {@code f} with {@link CpuInt8Fusion#PROP_INT8} set to {@code fused}. */
    private static float[] run(Fixture f, int n, float[] in, boolean fused) throws IOException {
        return run(f, n, in, fused, fused);
    }

    /** {@link #run(Fixture, int, float[], boolean)} whose block must (or must not) select the fused pass. */
    private static float[] run(Fixture f, int n, float[] in, boolean fused, boolean expectFused) throws IOException {
        String previous = System.getProperty(CpuInt8Fusion.PROP_INT8);
        System.setProperty(CpuInt8Fusion.PROP_INT8, Boolean.toString(fused));
        CpuFfnBlock block = null;
        try {
            block = new CpuMoeFfn(FfnKind.MOE, f.layer);
            block.bind(f.ctx, f.layer);
            assertEquals("fused int8 pass selection of a " + f.tensor(TensorNames.FFN_UP_EXPS).type() + " file",
                    expectFused, ((CpuMoeFfn) block).fusedExperts());
            float[] out = new float[UBATCH * f.nEmbd];
            block.forward(n, in, out);
            return out;
        } finally {
            if (block != null) {
                block.close();
            }
            if (previous == null) {
                System.clearProperty(CpuInt8Fusion.PROP_INT8);
            } else {
                System.setProperty(CpuInt8Fusion.PROP_INT8, previous);
            }
        }
    }

    private static float silu(float g) {
        return (float) (g / (1.0 + Math.exp(-g)));
    }

    /** Adds expert {@code ex}'s row of a flat {@code [dim, n_expert]} expert bias; none without a bias. */
    private static void addBiasRow(float[] y, float[] bias, int ex, int dim) {
        if (bias == null) {
            return;
        }
        for (int i = 0; i < dim; i++) {
            y[i] += bias[ex * dim + i];
        }
    }

    /** Per token: top-2 renormalized experts, each its own SwiGLU FFN, plus the sigmoid-gated shared FFN. */
    private static float[] reference(Fixture f, int n, float[] in) {
        CpuMatVec mv = f.ctx.matVec();
        int e = f.nEmbd;
        float[] logits = new float[n * N_EXPERT];
        mv.matmul(f.tensor(TensorNames.FFN_GATE_INP), in, 0, logits, 0, n);
        MoeRouting.Selection sel = MoeRouting.route(MoeRouting.Params.forExperts(f.layer.moe()), logits, n, null);
        CpuTensorRef gate = f.tensor(TensorNames.FFN_GATE_EXPS);
        CpuTensorRef up = f.tensor(TensorNames.FFN_UP_EXPS);
        CpuTensorRef down = f.tensor(TensorNames.FFN_DOWN_EXPS);
        float[] gateBias = f.weights.layerFloatsOrNull(f.layer, TensorNames.bias(TensorNames.FFN_GATE_EXPS));
        float[] upBias = f.weights.layerFloatsOrNull(f.layer, TensorNames.bias(TensorNames.FFN_UP_EXPS));
        float[] downBias = f.weights.layerFloatsOrNull(f.layer, TensorNames.bias(TensorNames.FFN_DOWN_EXPS));
        float[] out = new float[UBATCH * e];
        float[] g = new float[N_FF];
        float[] u = new float[N_FF];
        float[] d = new float[e];
        for (int t = 0; t < n; t++) {
            for (int j = 0; j < K; j++) {
                int ex = sel.id(t, j);
                mv.matvecExpert(gate, ex, N_EXPERT, in, t * e, g, 0);
                mv.matvecExpert(up, ex, N_EXPERT, in, t * e, u, 0);
                addBiasRow(g, gateBias, ex, N_FF);
                addBiasRow(u, upBias, ex, N_FF);
                for (int i = 0; i < N_FF; i++) {
                    u[i] = silu(g[i]) * u[i];
                }
                mv.matvecExpert(down, ex, N_EXPERT, u, 0, d, 0);
                addBiasRow(d, downBias, ex, e);
                for (int i = 0; i < e; i++) {
                    out[t * e + i] += sel.weight(t, j) * d[i];
                }
            }
            float[] gs = new float[1];
            mv.matvec(f.tensor(TensorNames.FFN_GATE_INP_SHEXP), in, t * e, gs, 0);
            float gate1 = (float) (1.0 / (1.0 + Math.exp(-gs[0])));
            mv.matvec(f.tensor(TensorNames.FFN_GATE_SHEXP), in, t * e, g, 0);
            mv.matvec(f.tensor(TensorNames.FFN_UP_SHEXP), in, t * e, u, 0);
            for (int i = 0; i < N_FF; i++) {
                u[i] = silu(g[i]) * u[i];
            }
            mv.matvec(f.tensor(TensorNames.FFN_DOWN_SHEXP), u, 0, d, 0);
            for (int i = 0; i < e; i++) {
                out[t * e + i] += gate1 * d[i];
            }
        }
        return out;
    }

    @Test
    public void anInt8FileRunsTheFusedPassAndMatchesTheGenericPath() throws IOException {
        try (Fixture f = new Fixture(GGMLType.Q8_0)) {
            float[] in = input(f.nEmbd, 1L);
            assertArrayEquals("decode", run(f, 1, in, false), run(f, 1, in, true), 0f);
            float[] rows = input(f.nEmbd, 2L);
            assertArrayEquals("eight tokens over four experts", run(f, UBATCH, rows, false),
                    run(f, UBATCH, rows, true), 0f);
        }
    }

    @Test
    public void anInt8SetWithExpertBiasesMatchesTheGenericPath() throws IOException {
        try (Fixture f = new Fixture(GGMLType.Q8_0, true, true)) {
            float[] in = input(f.nEmbd, 7L);
            assertArrayEquals("biased decode", run(f, 1, in, false), run(f, 1, in, true), 0f);
            float[] rows = input(f.nEmbd, 8L);
            float[] fused = run(f, UBATCH, rows, true);
            assertArrayEquals("biased eight tokens over four experts", run(f, UBATCH, rows, false), fused, 0f);
            float[] expected = reference(f, UBATCH, rows);
            for (int i = 0; i < UBATCH * f.nEmbd; i++) {
                assertEquals("[" + i + "]", expected[i], fused[i], 1e-5f * Math.max(1f, Math.abs(expected[i])));
            }
        }
    }

    @Test
    public void anUngatedInt8SetMatchesTheGenericPath() throws IOException {
        try (Fixture f = new Fixture(GGMLType.Q8_0, false)) {
            float[] in = input(f.nEmbd, 5L);
            assertArrayEquals("ungated decode", run(f, 1, in, false), run(f, 1, in, true), 0f);
            float[] rows = input(f.nEmbd, 6L);
            assertArrayEquals("ungated eight tokens over four experts", run(f, UBATCH, rows, false),
                    run(f, UBATCH, rows, true), 0f);
        }
    }

    @Test
    public void theFusedOutputMatchesThePerPairReference() throws IOException {
        try (Fixture f = new Fixture(GGMLType.Q8_0)) {
            float[] in = input(f.nEmbd, 3L);
            float[] fused = run(f, UBATCH, in, true);
            float[] expected = reference(f, UBATCH, in);
            for (int i = 0; i < UBATCH * f.nEmbd; i++) {
                assertEquals("[" + i + "]", expected[i], fused[i], 1e-5f * Math.max(1f, Math.abs(expected[i])));
            }
        }
    }

    @Test
    public void aFloatFileKeepsTheGenericPath() throws IOException {
        try (Fixture f = new Fixture(GGMLType.F32)) {
            assertFalse(CpuInt8Moe.isInt8(f.tensor(TensorNames.FFN_UP_EXPS)));
            float[] in = input(f.nEmbd, 4L);
            float[] out = run(f, UBATCH, in, true, false);
            assertTrue("a float file does not select the fused pass", out.length > 0);
        }
    }

    @Test
    public void theFusedSelectionNeedsBothProjectionsInInt8() {
        assertTrue(CpuInt8Moe.isInt8(int8Tensor(GGMLType.Q8_0)));
        assertTrue(CpuInt8Moe.isInt8(int8Tensor(GGMLType.I8_B128)));
        assertFalse(CpuInt8Moe.isInt8(int8Tensor(GGMLType.Q4_K)));
        assertFalse(CpuInt8Moe.isInt8(null));
    }

    private static CpuTensorRef int8Tensor(GGMLType type) {
        int cols = type == GGMLType.Q4_K ? 256 : type == GGMLType.I8_B128 ? 128 : 32;
        int rowBytes = type.getTypeSize();
        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(rowBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        return new CpuTensorRef(type, new java.nio.ByteBuffer[]{buffer}, new long[]{0}, 1, cols, rowBytes);
    }
}

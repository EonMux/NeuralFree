package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
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

import static org.junit.Assert.assertEquals;

/**
 * {@link CpuMoeFfn} on a one-layer synthetic qwen3next file (4 experts, 2 used, a sigmoid-gated shared expert):
 * the block output against a per-pair reference of {@code build_moe_ffn} plus the shared expert
 * (llama-graph.cpp:2189-2580, qwen3next.cpp), over decode and a several-token step whose experts take more than
 * one pair each; and the shared combine scale, which multiplies the routed + shared sum
 * (cohere2moe.cpp:241-243).
 */
public class CpuMoeFfnTest {

    private static final int UBATCH = 8;
    private static final int N_EXPERT = 4;
    private static final int K = 2;
    private static final int N_FF = 32;

    /** A one-layer qwen3next file, its weights, and a context whose only layer is the given MoE spec. */
    private static final class Fixture implements AutoCloseable {
        final Path path;
        final GGUFFileParser parser;
        final CpuWeights weights;
        final LayerSpec layer;
        final CpuForwardContext ctx;
        final int nEmbd;

        Fixture(float combineScale) throws IOException {
            SyntheticModels.Config config = SyntheticModels.Config.qwen3next().toBuilder().nLayer(1).build();
            this.path = SyntheticModels.qwen3next(config).writeTempFile();
            this.parser = new GGUFFileParser(path.toString());
            TensorCatalog tc = TensorCatalog.of(parser);
            ModelHparams hp = HparamsLoader.load(GgufKv.of(parser));
            this.nEmbd = hp.nEmbd();
            FfnSpec shexp = FfnSpec.builder(FfnSpec.Role.SHARED_EXPERT).nFf(N_FF).glu(FfnGating.PAR, GluOp.SWIGLU)
                    .build();
            MoeSpec moe = MoeSpec.builder().experts(N_EXPERT, K, N_FF).gating(GatingFunc.SOFTMAX).normWeights(true)
                    .glu(GluOp.SWIGLU).sharedExpert(shexp, true).sharedCombineScale(combineScale).build();
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

        float[] run(int n, float[] in) {
            CpuFfnBlock block = new CpuMoeFfn(FfnKind.MOE, layer);
            block.bind(ctx, layer);
            float[] out = new float[UBATCH * nEmbd];
            block.forward(n, in, out);
            return out;
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

    private static float silu(float g) {
        return (float) (g / (1.0 + Math.exp(-g)));
    }

    /** Per token: softmax top-2 renormalized experts, each its own SwiGLU FFN, plus the sigmoid-gated shared FFN. */
    private static float[] reference(Fixture f, int n, float[] in) {
        CpuMatVec mv = f.ctx.matVec();
        int e = f.nEmbd;
        CpuTensorRef gateInp = f.tensor(TensorNames.FFN_GATE_INP);
        float[] logits = new float[n * N_EXPERT];
        mv.matmul(gateInp, in, 0, logits, 0, n);
        MoeRouting.Selection sel = MoeRouting.route(MoeRouting.Params.forExperts(f.layer.moe()), logits, n, null);
        CpuTensorRef gate = f.tensor(TensorNames.FFN_GATE_EXPS);
        CpuTensorRef up = f.tensor(TensorNames.FFN_UP_EXPS);
        CpuTensorRef down = f.tensor(TensorNames.FFN_DOWN_EXPS);
        float[] out = new float[UBATCH * e];
        float[] g = new float[N_FF];
        float[] u = new float[N_FF];
        float[] d = new float[e];
        for (int t = 0; t < n; t++) {
            for (int j = 0; j < K; j++) {
                int ex = sel.id(t, j);
                mv.matvecExpert(gate, ex, N_EXPERT, in, t * e, g, 0);
                mv.matvecExpert(up, ex, N_EXPERT, in, t * e, u, 0);
                for (int i = 0; i < N_FF; i++) {
                    u[i] = silu(g[i]) * u[i];
                }
                mv.matvecExpert(down, ex, N_EXPERT, u, 0, d, 0);
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

    private static void assertNear(float[] expected, float[] actual, int count) {
        for (int i = 0; i < count; i++) {
            assertEquals("[" + i + "]", expected[i], actual[i], 1e-5f * Math.max(1f, Math.abs(expected[i])));
        }
    }

    @Test
    public void decodeMatchesThePerPairReference() throws IOException {
        try (Fixture f = new Fixture(1.0f)) {
            float[] in = input(f.nEmbd, 1L);
            assertNear(reference(f, 1, in), f.run(1, in), f.nEmbd);
        }
    }

    /** Eight tokens over four experts: every selected expert serves several pairs in one grouped product. */
    @Test
    public void groupedStepMatchesThePerPairReference() throws IOException {
        try (Fixture f = new Fixture(1.0f)) {
            float[] in = input(f.nEmbd, 2L);
            assertNear(reference(f, UBATCH, in), f.run(UBATCH, in), UBATCH * f.nEmbd);
        }
    }

    @Test
    public void sharedCombineScaleHalvesTheRoutedPlusSharedSum() throws IOException {
        int n = 3;
        float[] plain;
        int nx;
        try (Fixture f = new Fixture(1.0f)) {
            nx = n * f.nEmbd;
            plain = f.run(n, input(f.nEmbd, 3L));
        }
        try (Fixture f = new Fixture(0.5f)) {
            float[] halved = f.run(n, input(f.nEmbd, 3L));
            for (int i = 0; i < nx; i++) {
                assertEquals("[" + i + "]", 0.5f * plain[i], halved[i], 0f);
            }
        }
    }
}

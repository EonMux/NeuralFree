package com.eonmux.neuralfree.gguf.ops.deltanet;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GdnHeadMap;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.SplittableRandom;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * The qwen4exp merged linear-attention projections on the CPU: {@code attn_qkv_gate} ({@code [attn_qkv |
 * attn_gate]} rows, qwen4exp.cpp:682-695) and {@code ssm_beta_alpha} ({@code [ssm_beta | ssm_alpha]} rows,
 * qwen4exp.cpp:1131-1138) must give the {@link CpuGdnMixer} output of the separate weights.
 */
public class CpuGdnMixerMergedTest {

    private static final int UBATCH = 4;
    private static final int N_EMBD = 32;
    private static final int NK = 2;
    private static final int NV = 4;
    private static final int HK = 8;
    private static final int HV = 8;
    private static final int VAL = NV * HV;
    private static final int CONV_CH = 2 * NK * HK + VAL;
    private static final int D_CONV = 4;
    private static final float EPS = 1e-6f;

    private final float[] wQkv = random(1, CONV_CH * N_EMBD, -0.3f, 0.3f);
    private final float[] wGate = random(2, VAL * N_EMBD, -0.3f, 0.3f);
    private final float[] wBeta = random(3, NV * N_EMBD, -0.5f, 0.5f);
    private final float[] wAlpha = random(4, NV * N_EMBD, -0.5f, 0.5f);
    private final float[] conv1d = random(5, D_CONV * CONV_CH, -0.5f, 0.5f);
    private final float[] dtBias = random(6, NV, -0.5f, 0.5f);
    private final float[] ssmA = random(7, NV, -1.5f, -0.2f);
    private final float[] ssmNorm = random(8, HV, 0.5f, 1.5f);
    private final float[] wOut = random(9, N_EMBD * VAL, -0.3f, 0.3f);

    @Test
    public void theMergedProjectionsReproduceTheSeparateOnes() throws IOException {
        int n = 3;
        float[] in = random(10, n * N_EMBD, -1f, 1f);
        float[] separate = run(false, n, in);
        float[] merged = run(true, n, in);
        assertArrayEquals(separate, merged, 1e-5f);
    }

    private float[] run(boolean merged, int n, float[] in) throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture(Arch.QWEN4EXP.ggufName());
        if (merged) {
            put(w, "attn_qkv_gate.weight", concat(wQkv, wGate), N_EMBD, CONV_CH + VAL);
            put(w, "ssm_beta_alpha.weight", concat(wBeta, wAlpha), N_EMBD, 2 * NV);
        } else {
            put(w, "attn_qkv.weight", wQkv, N_EMBD, CONV_CH);
            put(w, "attn_gate.weight", wGate, N_EMBD, VAL);
            put(w, "ssm_beta.weight", wBeta, N_EMBD, NV);
            put(w, "ssm_alpha.weight", wAlpha, N_EMBD, NV);
        }
        put(w, "ssm_conv1d.weight", conv1d, D_CONV, CONV_CH);
        put(w, "ssm_dt.bias", dtBias, NV);
        put(w, "ssm_a", ssmA, NV);
        put(w, "ssm_norm.weight", ssmNorm, HV);
        put(w, "ssm_out.weight", wOut, VAL, N_EMBD);
        Path path = w.writeTempFile();
        try (GGUFFileParser parser = new GGUFFileParser(path.toString());
             CpuWeights weights = CpuWeights.open(parser)) {
            RecurrentSpec r = RecurrentSpec.builder(MixerKind.GDN).gdnHeads(NK, NV, HK, HV).dConv(D_CONV)
                    .headMap(GdnHeadMap.TILED).mergedQkvGate(merged).concatBetaAlpha(merged).build();
            LayerSpec layer = LayerSpec.builder(0, MixerKind.GDN, FfnKind.NONE)
                    .norm(NormSpec.builder(NormKind.RMS, EPS).build())
                    .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                    .recurrent(r)
                    .build();
            ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.QWEN4EXP)).layerCounts(1, 0)
                    .nCtxTrain(4096).nEmbd(N_EMBD).normRmsEps(EPS);
            hb.isRecurrent(0, true);
            ModelSpec spec = ModelSpec.builder(hb.build(), TensorCatalog.of(parser))
                    .schedule(LayerSchedule.linear(1))
                    .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                    .head(HeadSpec.builder().build())
                    .layer(layer)
                    .build();
            CpuForwardContext ctx = CpuForwardContext.builder(spec, weights, CpuScratch.forModel(spec, UBATCH))
                    .build();
            try (CpuMixer mixer = DeltaNetBlocks.cpuMixers().get(0).create(layer)) {
                mixer.bind(ctx, layer);
                if (merged) {
                    assertEquals("CpuGdnMixer{2k/4v heads, 8x8, attn_qkv_gate, ssm_beta_alpha}", mixer.toString());
                }
                float[] out = new float[n * N_EMBD];
                mixer.prefill(0, n, in, out);
                return out;
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }

    private static void put(GgufTestWriter w, String suffix, float[] values, long... ne) {
        w.tensorFromFloats("blk.0." + suffix, GGMLType.F32, ne, values);
    }

    private static float[] concat(float[] a, float[] b) {
        float[] out = new float[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private static float[] random(long seed, int n, float lo, float hi) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = lo + (hi - lo) * (float) r.nextDouble();
        }
        return out;
    }
}

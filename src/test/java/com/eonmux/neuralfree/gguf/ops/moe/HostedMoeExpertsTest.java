package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
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
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * The host-computed expert outputs of {@link HostedMoeExperts} against the naive per-pair reference (the SwiGLU
 * chain of the routed experts over the synthetic qwen3next file, as {@code CpuMoeFfn} computes it): scattered
 * miss pairs over several token rows, the whole set of pairs, pair-row sources, and no misses.
 */
public class HostedMoeExpertsTest {

    private static final int UBATCH = 6;
    private static final int N_EXPERT = 4;
    private static final int K = 2;
    private static final int N_FF = 32;

    private Path path;
    private GGUFFileParser parser;
    private CpuWeights weights;
    private LayerSpec layer;
    private HostedMoeExperts hosted;
    private CpuMatVec mv;
    private int nEmbd;

    @Before
    public void open() throws IOException {
        SyntheticModels.Config config = SyntheticModels.Config.qwen3next().toBuilder().nLayer(1).build();
        this.path = SyntheticModels.qwen3next(config).writeTempFile();
        this.parser = new GGUFFileParser(path.toString());
        TensorCatalog tc = TensorCatalog.of(parser);
        ModelHparams hp = HparamsLoader.load(GgufKv.of(parser));
        this.nEmbd = hp.nEmbd();
        MoeSpec moe = MoeSpec.builder().experts(N_EXPERT, K, N_FF).gating(GatingFunc.SOFTMAX).normWeights(true)
                .glu(GluOp.SWIGLU).build();
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
        this.hosted = new HostedMoeExperts(MoeExpertWeights.routed(moe, nEmbd), weights, K, UBATCH, layer);
        this.mv = weights.matVec();
    }

    @After
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

    private static float silu(float g) {
        return (float) (g / (1.0 + Math.exp(-g)));
    }

    /**
     * The routed FFN rows of the miss pairs: row {@code q} is the chain of expert {@code ids[missPairs[q]]} over
     * source row {@code missPairs[q] / pairsPerRow} of {@code x}.
     */
    private float[][] referenceRows(int[] ids, int[] missPairs, float[] x, int pairsPerRow) {
        CpuTensorRef gate = weights.layer(layer, TensorNames.weight(TensorNames.FFN_GATE_EXPS));
        CpuTensorRef up = weights.layer(layer, TensorNames.weight(TensorNames.FFN_UP_EXPS));
        CpuTensorRef down = weights.layer(layer, TensorNames.weight(TensorNames.FFN_DOWN_EXPS));
        float[][] out = new float[missPairs.length][nEmbd];
        float[] g = new float[N_FF];
        float[] u = new float[N_FF];
        float[] d = new float[nEmbd];
        for (int q = 0; q < missPairs.length; q++) {
            int p = missPairs[q];
            int srcRow = p / pairsPerRow;
            int e = ids[p];
            mv.matvecExpert(gate, e, N_EXPERT, x, srcRow * nEmbd, g, 0);
            mv.matvecExpert(up, e, N_EXPERT, x, srcRow * nEmbd, u, 0);
            for (int i = 0; i < N_FF; i++) {
                u[i] = silu(g[i]) * u[i];
            }
            mv.matvecExpert(down, e, N_EXPERT, u, 0, d, 0);
            System.arraycopy(d, 0, out[q], 0, nEmbd);
        }
        return out;
    }

    private static float[] input(int n, long seed) {
        Random rnd = new Random(seed);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }

    @Test
    public void scatteredMissPairsMatchTheReference() {
        int n = 3;
        float[] x = input(n * nEmbd, 42L);
        int[] ids = {0, 2, 1, 3, 2, 0};
        int[] missPairs = {1, 4, 5};
        float[] rows = hosted.run(ids, missPairs, missPairs.length, x, K);

        assertEquals(missPairs.length * nEmbd, rows.length);
        float[][] expected = referenceRows(ids, missPairs, x, K);
        for (int q = 0; q < missPairs.length; q++) {
            assertArrayEquals("pair " + missPairs[q], expected[q],
                    Arrays.copyOfRange(rows, q * nEmbd, (q + 1) * nEmbd), 1e-4f);
        }
    }

    @Test
    public void everyPairMatchesTheReference() {
        int n = 2;
        float[] x = input(n * nEmbd, 7L);
        int[] ids = {3, 1, 0, 2};
        int[] missPairs = {0, 1, 2, 3};
        float[] rows = hosted.run(ids, missPairs, missPairs.length, x, K);

        float[][] expected = referenceRows(ids, missPairs, x, K);
        for (int q = 0; q < missPairs.length; q++) {
            assertArrayEquals(expected[q], Arrays.copyOfRange(rows, q * nEmbd, (q + 1) * nEmbd), 1e-4f);
        }
    }

    @Test
    public void pairRowSourcesMatchTheReference() {
        int n = 2;
        float[] tokens = input(n * nEmbd, 11L);
        // the weighted pair rows of a weight-before-FFN step: one row per pair, a scaled copy of its token row
        float[] pairRows = new float[n * K * nEmbd];
        for (int p = 0; p < n * K; p++) {
            for (int i = 0; i < nEmbd; i++) {
                pairRows[p * nEmbd + i] = tokens[(p / K) * nEmbd + i] * (0.5f + p);
            }
        }
        int[] ids = {2, 2, 1, 0};
        int[] missPairs = {1, 3};
        float[] rows = hosted.run(ids, missPairs, missPairs.length, pairRows, 1);

        float[][] expected = referenceRows(ids, missPairs, pairRows, 1);
        for (int q = 0; q < missPairs.length; q++) {
            assertArrayEquals(expected[q], Arrays.copyOfRange(rows, q * nEmbd, (q + 1) * nEmbd), 1e-4f);
        }
    }

    @Test
    public void noMissesProduceNoRows() {
        float[] x = input(nEmbd, 3L);
        float[] rows = hosted.run(new int[]{1}, new int[0], 0, x, K);
        assertEquals(0, rows.length);
    }

    @Test
    public void aNonPositiveTopKIsRefusedByName() {
        assertThrows(IllegalArgumentException.class,
                () -> new HostedMoeExperts(MoeExpertWeights.routed(layer.moe(), nEmbd), weights, 0, UBATCH, layer));
    }

    @Test
    public void outOfRangeArgumentsAreRefusedByName() {
        float[] x = input(nEmbd, 5L);
        assertThrows(IllegalArgumentException.class, () -> hosted.run(new int[]{0}, new int[]{0}, 2, x, K));
        assertThrows(IllegalArgumentException.class,
                () -> hosted.run(new int[]{N_EXPERT}, new int[]{0}, 1, x, K));
        assertThrows(IllegalArgumentException.class, () -> hosted.run(new int[]{0}, new int[]{3}, 1, x, K));
        assertThrows(IllegalArgumentException.class, () -> hosted.run(new int[]{0}, new int[]{0}, 1, x, 0));
    }
}

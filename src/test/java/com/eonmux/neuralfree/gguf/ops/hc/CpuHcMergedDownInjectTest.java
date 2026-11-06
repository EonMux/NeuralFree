package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;

/**
 * The qwen4exp merged {@code hc_*_down_inject} weight on the CPU (qwen4exp.cpp:363-380): one matmul whose
 * {@code [down | inject]} rows are scaled by {@code 1/hc} before the split must reproduce the separate
 * {@code hc_*_down} / {@code hc_*_inject} pair through the collapse, the mixer expand and the FFN expand.
 */
public class CpuHcMergedDownInjectTest {

    private static final int UBATCH = 4;
    private static final int N_EMBD = 16;
    private static final int HC = 4;
    private static final int STREAM = HC * N_EMBD;
    private static final int LOW_RANK = 8;
    private static final float EPS = 1e-6f;
    private static final String[] SITES = {"attn", "ffn"};

    /** Separate and merged tensors of both sites, drawn once so both forms hold the same values. */
    private static final class Weights {
        final Map<String, float[]> norm = new LinkedHashMap<>();
        final Map<String, float[]> down = new LinkedHashMap<>();
        final Map<String, float[]> up = new LinkedHashMap<>();
        final Map<String, float[]> inject = new LinkedHashMap<>();

        Weights() {
            long seed = 1;
            for (String site : SITES) {
                norm.put(site, random(seed++, STREAM, 0.5f, 1.5f));
                down.put(site, random(seed++, LOW_RANK * STREAM, -0.3f, 0.3f));
                up.put(site, random(seed++, STREAM * LOW_RANK, -0.3f, 0.3f));
                inject.put(site, random(seed++, HC * STREAM, -0.6f, 0.6f));
            }
        }

        void write(GgufTestWriter w, boolean merged) {
            for (String site : SITES) {
                put(w, "hc_" + site + "_norm", norm.get(site), N_EMBD, HC);
                put(w, "hc_" + site + "_up", up.get(site), LOW_RANK, STREAM);
                if (merged) {
                    put(w, "hc_" + site + "_down_inject", concat(down.get(site), inject.get(site)), STREAM,
                            LOW_RANK + HC);
                } else {
                    put(w, "hc_" + site + "_down", down.get(site), STREAM, LOW_RANK);
                    put(w, "hc_" + site + "_inject", inject.get(site), STREAM, HC);
                }
            }
        }
    }

    /** One bound {@link CpuHcExtra} over a one-layer qwen4exp file. */
    private static final class Fixture implements AutoCloseable {
        final Path path;
        final GGUFFileParser parser;
        final CpuWeights weights;
        final ModelSpec spec;
        final CpuHcExtra extra = new CpuHcExtra();

        Fixture(Weights ws, boolean merged) throws IOException {
            GgufTestWriter w = new GgufTestWriter().architecture(Arch.QWEN4EXP.ggufName());
            ws.write(w, merged);
            this.path = w.writeTempFile();
            this.parser = new GGUFFileParser(path.toString());
            this.weights = CpuWeights.open(parser);
            this.spec = model(TensorCatalog.of(parser));
            CpuForwardContext ctx = CpuForwardContext.builder(spec, weights, CpuScratch.forModel(spec, UBATCH))
                    .build();
            extra.bind(ctx, spec.layers().get(0));
        }

        /** Collapse, mixer expand + FFN collapse, FFN expand; returns the three outputs in order. */
        float[][] run(int n, float[] streams, float[] mixerOut, float[] ffnOut) {
            float[] x = streams.clone();
            extra.beforeBlock(0, n, x);
            float[] attnIn = slice(x, n * N_EMBD);
            extra.afterMixer(0, n, x, mixerOut);
            float[] ffnIn = slice(x, n * N_EMBD);
            extra.afterFfn(0, n, x, ffnOut);
            return new float[][]{attnIn, ffnIn, slice(x, n * STREAM)};
        }

        @Override
        public void close() throws IOException {
            try {
                extra.close();
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

    @Test
    public void theMergedWeightReproducesTheSeparatePair() throws IOException {
        Weights ws = new Weights();
        int n = 3;
        float[] streams = random(11, n * STREAM, -1f, 1f);
        float[] mixerOut = random(12, n * N_EMBD, -1f, 1f);
        float[] ffnOut = random(13, n * N_EMBD, -1f, 1f);
        float[][] separate;
        float[][] merged;
        try (Fixture f = new Fixture(ws, false)) {
            separate = f.run(n, streams, mixerOut, ffnOut);
        }
        try (Fixture f = new Fixture(ws, true)) {
            HcSubBlock attn = HcSubBlock.forLayer(f.spec, f.spec.layers().get(0), false);
            assertEquals(HcSubBlock.Kind.GATED, attn.kind());
            assertEquals(LOW_RANK, attn.lowRank());
            assertNull(attn.down());
            assertNull(attn.inject());
            assertEquals("blk.0.hc_attn_down_inject.weight", attn.downInject().name());
            merged = f.run(n, streams, mixerOut, ffnOut);
        }
        String[] stage = {"attention collapse", "FFN collapse", "streams after the FFN"};
        for (int i = 0; i < stage.length; i++) {
            assertArrayEquals(stage[i], separate[i], merged[i], 1e-5f);
        }
    }

    @Test
    public void anUpProjectionOfAnotherRankIsRefused() throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture(Arch.QWEN4EXP.ggufName());
        put(w, "hc_attn_norm", new float[STREAM], N_EMBD, HC);
        put(w, "hc_attn_up", new float[(LOW_RANK + HC) * STREAM], LOW_RANK + HC, STREAM);
        put(w, "hc_attn_down_inject", new float[(LOW_RANK + HC) * STREAM], STREAM, LOW_RANK + HC);
        Path path = w.writeTempFile();
        try (GGUFFileParser parser = new GGUFFileParser(path.toString())) {
            ModelSpec spec = model(TensorCatalog.of(parser));
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> HcSubBlock.forLayer(spec, spec.layers().get(0), false));
            assertEquals("qwen4exp layer 0: blk.0.hc_attn_up.weight must be [" + LOW_RANK + ", " + STREAM + "], got "
                    + spec.tensors().get("blk.0.hc_attn_up.weight"), e.getMessage());
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void aMergedWeightWithoutLowRankRowsIsRefused() throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture(Arch.QWEN4EXP.ggufName());
        put(w, "hc_attn_norm", new float[STREAM], N_EMBD, HC);
        put(w, "hc_attn_up", new float[STREAM], 1, STREAM);
        put(w, "hc_attn_down_inject", new float[HC * STREAM], STREAM, HC);
        Path path = w.writeTempFile();
        try (GGUFFileParser parser = new GGUFFileParser(path.toString())) {
            ModelSpec spec = model(TensorCatalog.of(parser));
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> HcSubBlock.forLayer(spec, spec.layers().get(0), false));
            assertEquals(true, e.getMessage().contains("hc_attn_down_inject.weight must be [" + STREAM + ", rank + "
                    + HC + "] with rank >= 1"));
        } finally {
            Files.deleteIfExists(path);
        }
    }

    private static ModelSpec model(TensorCatalog catalog) {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(Arch.QWEN4EXP)).layerCounts(1, 0)
                .nEmbd(N_EMBD).normRmsEps(EPS).dsv4HcMult(HC).hcLowRank(LOW_RANK).build();
        ResidualSpec residual = ResidualSpec.builder(ResidualKind.HYPER_CONNECTIONS, N_EMBD)
                .hyperConnections(HC, true).build();
        LayerSpec layer = LayerSpec.builder(0, MixerKind.NONE, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, EPS).build())
                .residual(residual)
                .extra(LayerSpec.Extras.HC)
                .build();
        return ModelSpec.builder(hp, catalog)
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build())
                .layer(layer)
                .build();
    }

    private static void put(GgufTestWriter w, String base, float[] values, long... ne) {
        w.tensorFromFloats("blk.0." + base + ".weight", GGMLType.F32, ne, values);
    }

    private static float[] concat(float[] a, float[] b) {
        float[] out = new float[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private static float[] slice(float[] a, int n) {
        float[] out = new float[n];
        System.arraycopy(a, 0, out, 0, n);
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

package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnBlock;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticRoutedModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@link CpuMoeFfn} over a step mixing token and embedding rows on deepseek4 blocks resolved from the synthetic file
 * (block 0 hash-routed, both with {@code exp_probs_b_vl}): each row routes by its own kind, so every row of the
 * mixed step equals the same row in a step of its kind alone (rows are independent in the MoE), and an embedding
 * row does not route as any token row would (deepseek4.cpp:1290-1301).
 */
public class CpuMoeFfnMediaRowsTest {

    private static final int UBATCH = 8;
    private static final int[] MIXED = {5, -1, 9, -1, -1, 12};

    /** The resolved deepseek4 model, its weights and a context over them. */
    private static final class Fixture implements AutoCloseable {
        final Path path;
        final GGUFFileParser parser;
        final CpuWeights weights;
        final ModelSpec spec;
        final CpuForwardContext ctx;
        final int nEmbd;

        Fixture() throws IOException {
            this(true);
        }

        /** The model with block 0 hash-routed, with or without the vision routing bias. */
        Fixture(boolean visionBias) throws IOException {
            this.path = SyntheticRoutedModels.deepseek4(SyntheticRoutedModels.deepseek4Config(), 1, visionBias)
                    .writeTempFile();
            this.parser = new GGUFFileParser(path.toString());
            this.spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), MemoryConfig.defaults());
            this.weights = CpuWeights.open(parser);
            this.ctx = CpuForwardContext.builder(spec, weights, CpuScratch.forModel(spec, UBATCH)).build();
            this.nEmbd = spec.hparams().nEmbd();
        }

        /** Whether the bound block of layer {@code il} may run over a subset of a step's rows. */
        boolean rowLocal(int il) {
            LayerSpec layer = spec.layer(il);
            CpuFfnBlock block = new CpuMoeFfn(layer.ffnKind(), layer);
            block.bind(ctx, layer);
            return block.rowLocal();
        }

        /** The block of layer {@code il} over {@code rows} of {@code in} with step tokens {@code tokens}. */
        float[] run(int il, float[] in, int[] rows, int[] tokens) {
            LayerSpec layer = spec.layer(il);
            CpuFfnBlock block = new CpuMoeFfn(layer.ffnKind(), layer);
            block.bind(ctx, layer);
            int n = rows.length;
            float[] x = new float[UBATCH * nEmbd];
            for (int i = 0; i < n; i++) {
                System.arraycopy(in, rows[i] * nEmbd, x, i * nEmbd, nEmbd);
            }
            System.arraycopy(tokens, 0, ctx.scratch().ints(CpuScratch.TOKEN_IDS), 0, n);
            float[] out = new float[UBATCH * nEmbd];
            block.forward(n, x, out);
            return out;
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

    private static float[] input(int nEmbd) {
        Random rnd = new Random(7L);
        float[] x = new float[UBATCH * nEmbd];
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }

    private static int[] rowsOf(boolean media) {
        int n = 0;
        for (int t : MIXED) {
            n += (t < 0) == media ? 1 : 0;
        }
        int[] rows = new int[n];
        int w = 0;
        for (int i = 0; i < MIXED.length; i++) {
            if ((MIXED[i] < 0) == media) {
                rows[w++] = i;
            }
        }
        return rows;
    }

    private static int[] identity(int n) {
        int[] rows = new int[n];
        for (int i = 0; i < n; i++) {
            rows[i] = i;
        }
        return rows;
    }

    private static int[] tokensOf(int[] rows) {
        int[] out = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            out[i] = MIXED[rows[i]];
        }
        return out;
    }

    @Test
    public void theFixtureResolvesAHashLayerAndVisionBiases() throws IOException {
        try (Fixture f = new Fixture()) {
            assertTrue(f.spec.layer(0).moe().hashRouting());
            assertTrue(f.spec.layer(0).moe().mediaProbBias());
            assertFalse(f.spec.layer(1).moe().hashRouting());
            assertTrue(f.spec.layer(1).moe().mediaProbBias());
        }
    }

    /**
     * Hash routing reads the step's token ids by step row and the vision bias its embedding-row flags, so neither
     * layer may run over a subset of the rows (the last layer's output-row pruning); a biased layer without the
     * vision bias routes each row by its own logits and may.
     */
    @Test
    public void onlyALayerThatRoutesEachRowByItsOwnInputsIsRowLocal() throws IOException {
        try (Fixture f = new Fixture(false)) {
            assertTrue(f.spec.layer(0).moe().hashRouting());
            assertFalse("a hash-routed layer reads the step's token ids", f.rowLocal(0));
            assertTrue("a biased layer routes each row alone", f.rowLocal(1));
        }
        try (Fixture f = new Fixture()) {
            assertFalse("a media-routed layer reads the step's embedding-row flags", f.rowLocal(1));
        }
    }

    @Test
    public void everyRowOfAMixedStepRoutesByItsKindOnAHashLayer() throws IOException {
        assertRowsRouteByKind(0);
    }

    @Test
    public void everyRowOfAMixedStepRoutesByItsKindOnABiasedLayer() throws IOException {
        assertRowsRouteByKind(1);
    }

    private static void assertRowsRouteByKind(int il) throws IOException {
        try (Fixture f = new Fixture()) {
            float[] in = input(f.nEmbd);
            float[] mixed = f.run(il, in, identity(MIXED.length), MIXED);
            for (boolean media : new boolean[] {false, true}) {
                int[] rows = rowsOf(media);
                float[] alone = f.run(il, in, rows, tokensOf(rows));
                for (int i = 0; i < rows.length; i++) {
                    for (int d = 0; d < f.nEmbd; d++) {
                        assertEquals("layer " + il + (media ? " embedding" : " token") + " row " + rows[i] + "[" + d
                                + "]", alone[i * f.nEmbd + d], mixed[rows[i] * f.nEmbd + d], 1e-6f);
                    }
                }
            }
        }
    }

    @Test
    public void anEmbeddingRowDoesNotRouteAsATokenRow() throws IOException {
        for (int il = 0; il < 2; il++) {
            try (Fixture f = new Fixture()) {
                float[] in = input(f.nEmbd);
                int[] rows = rowsOf(true);
                float[] media = f.run(il, in, rows, tokensOf(rows));
                for (int token = 0; token < 4; token++) {
                    int[] asTokens = new int[rows.length];
                    Arrays.fill(asTokens, token);
                    float[] tokens = f.run(il, in, rows, asTokens);
                    double diff = 0;
                    for (int i = 0; i < rows.length * f.nEmbd; i++) {
                        diff = Math.max(diff, Math.abs(tokens[i] - media[i]));
                    }
                    assertTrue("layer " + il + ": embedding rows route otherwise than token " + token + " (max diff "
                            + diff + ")", diff > 1e-4);
                }
            }
        }
    }
}

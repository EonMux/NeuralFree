package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.gguf.testutil.SyntheticRoutedModels;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.function.Function;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Embedding rows on deepseek4's routed experts, on the CPU and the GPU backend (deepseek4.cpp:1290-1301): an
 * embedding row skips {@code ffn_gate_tid2eid} on a hash layer and routes through the router with
 * {@code exp_probs_b_vl} as its selection bias, or none without it; on a non-hash layer {@code exp_probs_b_vl}
 * replaces {@code exp_probs_b} when the file has it. The reference is a model without hash layers whose
 * {@code exp_probs_b} is the bias the rows select with ({@code exp_probs_b_vl}, zeros for none, every other tensor
 * equal): the source model's embedding rows of the token table's rows must give that model's logits for the
 * tokens. Text around embedding rows routes each call's rows by their kind on the GPU as on the CPU. A ubatch mixing
 * both kinds is not reachable here (deepseek4 runs no paged step); {@link CpuMoeFfnMediaRowsTest} and
 * {@link GpuMediaRouterTest} cover the per-row routing of one.
 */
public class Dsv4EmbeddingRowsRoutingTest {

    private static final int HASH_LAYERS = 1;
    private static final float CPU_TOL = 1e-4f;
    private static final float GPU_TOL = 2e-3f;
    private static final int[] TOKENS = {71, 120, 8, 33, 150, 201};
    private static final int[] HEAD = {71, 120, 8, 33};
    private static final int[] ROW_TOKENS = {150, 201, 17, 99};
    private static final int[] TAIL = {64, 12, 230};

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(16).typeK(KvDType.F32)
                .typeV(KvDType.F32).build()).build();
    }

    private static SyntheticModels.Config fixture() {
        return SyntheticRoutedModels.deepseek4Config();
    }

    private static GgufTestWriter source(int hashLayers, boolean visionBias) {
        return SyntheticRoutedModels.deepseek4(fixture(), hashLayers, visionBias);
    }

    /**
     * The source model's embedding-row routing as token routing: no hash layers, and each block's
     * {@code exp_probs_b} the bias an embedding row of the source selects with.
     */
    private static Path reference(int hashLayers, boolean visionBias) throws IOException {
        SyntheticModels.Config c = fixture();
        GgufTestWriter src = source(hashLayers, visionBias);
        GgufTestWriter ref = SyntheticRoutedModels.deepseek4(c, 0, false);
        for (int il = 0; il < c.nLayer(); il++) {
            String bias = blk(il, "exp_probs_b.bias");
            if (visionBias) {
                String vl = blk(il, "exp_probs_b_vl.bias");
                ref.tensor(bias, src.tensorType(vl), src.tensorShape(vl), src.tensorData(vl));
            } else if (il < hashLayers) {
                ref.tensorZeros(bias, GGMLType.F32, c.nExpert());
            }
        }
        return ref.writeTempFile();
    }

    // ------------------------------------------------------------------ CPU

    @Test
    public void theModelTakesEmbeddingRows() throws IOException {
        try (GGUFModel m = new GGUFModel(source(HASH_LAYERS, true).writeTempFile().toString(), config())) {
            assertNull("hash layers route embedding rows through the router",
                    m.forwardBackend().embeddingRowsRefusal());
        }
    }

    @Test
    public void rowsOnAHashLayerRouteThroughTheRouterWithoutABiasOnTheCpu() throws IOException {
        assertRowsMatchReference(HASH_LAYERS, false);
    }

    @Test
    public void rowsTakeTheVisionBiasOnHashAndBiasedLayersOnTheCpu() throws IOException {
        assertRowsMatchReference(HASH_LAYERS, true);
    }

    @Test
    public void theVisionBiasReplacesExpProbsBWithoutHashLayersOnTheCpu() throws IOException {
        assertRowsMatchReference(0, true);
    }

    private static void assertRowsMatchReference(int hashLayers, boolean visionBias) throws IOException {
        Path model = source(hashLayers, visionBias).writeTempFile();
        float[] rows;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            rows = m.forwardBackend().prefillEmbeddings(tableRows(model, TOKENS), 0);
        }
        float[] tokens;
        try (GGUFModel ref = new GGUFModel(reference(hashLayers, visionBias).toString(), config())) {
            tokens = lastLogits(ref.forwardBackend(), TOKENS);
        }
        assertArrayEquals("embedding rows route as the reference's tokens (hash layers " + hashLayers
                + ", vision bias " + visionBias + ")", tokens, rows, CPU_TOL);
    }

    @Test
    public void theReferenceDiffersFromTheSourcesTokenRouting() throws IOException {
        float[] source;
        try (GGUFModel m = new GGUFModel(source(HASH_LAYERS, true).writeTempFile().toString(), config())) {
            source = lastLogits(m.forwardBackend(), TOKENS);
        }
        float[] reference;
        try (GGUFModel ref = new GGUFModel(reference(HASH_LAYERS, true).toString(), config())) {
            reference = lastLogits(ref.forwardBackend(), TOKENS);
        }
        assertDiffers("hash and exp_probs_b routing of token rows against the reference's", source, reference);
    }

    // ------------------------------------------------------------------ GPU

    @Test
    public void rowsOnAHashLayerRouteThroughTheRouterWithoutABiasOnTheGpu() throws IOException {
        assertGpuRowsMatchReference(false);
    }

    @Test
    public void rowsTakeTheVisionBiasOnTheGpu() throws IOException {
        assertGpuRowsMatchReference(true);
    }

    private static void assertGpuRowsMatchReference(boolean visionBias) throws IOException {
        Path model = source(HASH_LAYERS, visionBias).writeTempFile();
        float[][] rows = tableRows(model, TOKENS);
        float[] fromRows = onGpu(be -> be.prefillEmbeddings(rows, 0), model);
        float[] fromTokens = onGpu(be -> lastLogits(be, TOKENS), reference(HASH_LAYERS, visionBias));
        assertArrayEquals("GPU embedding rows route as the reference's tokens (vision bias " + visionBias + ")",
                fromTokens, fromRows, GPU_TOL);
    }

    /** Text, embedding rows and text of one sequence: hash-routed token rows around router-routed rows. */
    @Test
    public void textAroundEmbeddingRowsOnTheGpuMatchesTheCpu() throws IOException {
        Path model = source(HASH_LAYERS, true).writeTempFile();
        float[][] rows = tableRows(model, ROW_TOKENS);
        float[] cpu;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            cpu = spliced(m.forwardBackend(), rows);
        }
        float[] gpu = onGpu(be -> spliced(be, rows), model);
        assertArrayEquals("the GPU routes each call's rows as the CPU", cpu, gpu, GPU_TOL);
    }

    private static float[] onGpu(Function<ForwardBackend, float[]> call, Path model) throws IOException {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        HWAcceleration acceleration = new HWAcceleration();
        try (ComputeBackend backend = new OclComputeBackend(acceleration.getCLDevices().get(0));
             GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config())) {
            return call.apply(gpu.forwardBackend());
        } finally {
            acceleration.shutdown();
        }
    }

    // ------------------------------------------------------------------ helpers

    private static String blk(int il, String name) {
        return "blk." + il + "." + name;
    }

    /** {@link #HEAD} as tokens, {@code rows} as embedding rows, {@link #TAIL} as tokens; the last row's logits. */
    private static float[] spliced(ForwardBackend be, float[][] rows) {
        be.prefill(HEAD, 0);
        be.prefillEmbeddings(rows, HEAD.length);
        return be.forwardRows(TAIL, HEAD.length + rows.length, new int[] {TAIL.length - 1})[0];
    }

    private static float[] lastLogits(ForwardBackend be, int[] tokens) {
        return be.forwardRows(tokens, 0, new int[] {tokens.length - 1})[0];
    }

    /** The token table's rows of {@code tokens}, read from the GGUF file. */
    private static float[][] tableRows(Path model, int[] tokens) throws IOException {
        try (GGUFFileParser parser = new GGUFFileParser(model.toString())) {
            float[] flat = parser.readTensorDataAsFloat(parser.getTensor("token_embd.weight"));
            int nEmbd = fixture().nEmbd();
            float[][] rows = new float[tokens.length][];
            for (int i = 0; i < tokens.length; i++) {
                rows[i] = Arrays.copyOfRange(flat, tokens[i] * nEmbd, (tokens[i] + 1) * nEmbd);
            }
            return rows;
        }
    }

    private static void assertDiffers(String what, float[] a, float[] b) {
        double maxDiff = 0;
        for (int i = 0; i < a.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
        }
        assertTrue(what + " changes the logits (max diff " + maxDiff + ")", maxDiff > 1e-3);
    }
}

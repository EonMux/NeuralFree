package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The W12 streaming of an F16 model — the shape the real embedding checkpoints store — against the plain
 * greedy loop on the same file: every projection's values lie on the bfloat16 grid in F16's representable
 * band, so the upload's grid rounding is an identity and both runs multiply one set of weights, and the
 * committed tokens must equal. A real-model-shaped body (hidden 1024, FFN 3072, GQA heads of 64) over two
 * layers runs the row kernels, the fused gate+up form and, over a prefill wider than the W12 matvec budget,
 * the decoded BF16 mirror, on CUDA and OpenCL. The factory load keeps the model on the device.
 */
@RunWith(Parameterized.class)
public class W12F16ParityTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26};
    private static final int GREEDY_TOKENS = 12;
    private static final int N_LAYER = 2;
    private static final int LONG_PROMPT_TOKENS = 96;
    /** The seed of the projections' on-grid values; per-tensor derivations keep them deterministic. */
    private static final long GRID_SEED = 20260125L;

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"cuda"});
    }

    private final String api;
    private final List<Path> files = new ArrayList<>();
    private Path model;

    public W12F16ParityTest(String api) {
        this.api = api;
    }

    @Before
    public void writeModel() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama().toBuilder().nLayer(N_LAYER)
                .nEmbd(1024).nHead(16).nHeadKv(8).headDim(64).nRot(64).nFf(3072)
                .weightType(GGMLType.F16).build();
        model = written(onGridProjections(SyntheticModels.llama(c)));
    }

    @After
    public void cleanUp() throws IOException {
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
        files.clear();
    }

    private Path written(GgufTestWriter w) throws IOException {
        Path p = w.writeTempFile();
        files.add(p);
        return p;
    }

    /**
     * Replaces every projection's random values with ones on the bfloat16 grid inside F16's normal band
     * (exponents {@code 2^-14 .. 2^-4}, seven mantissa bits): F16 stores them exactly, the plain run reads
     * them back unchanged, and the W12 upload's round-to-nearest-even keeps them.
     */
    private static GgufTestWriter onGridProjections(GgufTestWriter w) {
        for (String name : w.tensorNames()) {
            if (!isProjection(name)) {
                continue;
            }
            long[] ne = w.tensorShape(name);
            long n = 1;
            for (long d : ne) {
                n *= d;
            }
            Random r = new Random(GRID_SEED ^ (name.hashCode() * 0x9E3779B97F4A7C15L));
            float[] values = new float[(int) n];
            for (int i = 0; i < values.length; i++) {
                int sign = r.nextBoolean() ? 0x8000 : 0;
                int e = 113 + r.nextInt(11);
                values[i] = Float.intBitsToFloat((sign | (e << 7) | r.nextInt(128)) << 16);
            }
            w.tensorFromFloats(name, GGMLType.F16, ne, values);
        }
        return w;
    }

    /** The tensors that stream as W12: the attention and FFN projections of every block and the head. */
    private static boolean isProjection(String name) {
        if (!name.endsWith(".weight")) {
            return false;
        }
        boolean blockProjection = name.startsWith("blk.") && (name.contains(".attn_q.") || name.contains(".attn_k.")
                || name.contains(".attn_v.") || name.contains(".attn_output.") || name.contains(".ffn_gate.")
                || name.contains(".ffn_up.") || name.contains(".ffn_down."));
        return blockProjection || name.equals("output.weight");
    }

    // ------------------------------------------------------------------ configs and drivers

    private void assumeBackend() {
        if (api.equals("opencl")) {
            GpuTestUtil.assumeOpenCLAvailable();
        } else {
            GpuTestUtil.assumeCudaAvailable();
        }
    }

    private EngineConfig.Builder base() {
        MemoryConfig memory = MemoryConfig.defaults().toBuilder().nCtx(128).ubatch(128).build();
        EngineConfig.BackendMode mode = api.equals("opencl") ? EngineConfig.BackendMode.OPENCL
                : EngineConfig.BackendMode.CUDA;
        return EngineConfig.builder().backend(mode).addGpuDevice("0").memory(memory);
    }

    private static SpeculativeOptions w12(boolean head) {
        return SpeculativeOptions.defaults().toBuilder().w12(true).w12Head(head).build();
    }

    private static int[] greedy(LLMModel m, EngineConfig cfg, int[] prompt) throws IOException {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                .sampler(cfg.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
        return m.generate(prompt, request, null).tokens();
    }

    /** The plain greedy tokens of the F16 file on this backend; GGUFModelGPU refuses instead of falling back. */
    private int[] baseline(EngineConfig cfg, int[] prompt) throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            int[] tokens = greedy(m, cfg, prompt);
            m.forwardBackend().reset();
            return tokens;
        }
    }

    // ------------------------------------------------------------------ the parity

    @Test
    public void theW12WeightsOfAnF16ModelKeepTheGreedyTokens() throws IOException {
        assumeBackend();
        int[] reference = baseline(base().build(), PROMPT);
        assertArrayEquals(reference, baseline(base().speculativeOptions(w12(false)).build(), PROMPT));
    }

    @Test
    public void theW12HeadOfAnF16ModelKeepsTheGreedyTokens() throws IOException {
        assumeBackend();
        int[] reference = baseline(base().build(), PROMPT);
        assertArrayEquals(reference, baseline(base().speculativeOptions(w12(true)).build(), PROMPT));
    }

    @Test
    public void aWidePrefillRunsTheF16MirrorAndKeepsTheGreedyTokens() throws IOException {
        assumeBackend();
        int[] prompt = new int[LONG_PROMPT_TOKENS];
        for (int i = 0; i < prompt.length; i++) {
            prompt[i] = (i * 7 + 3) % 320;
        }
        int[] reference = baseline(base().build(), prompt);
        // the head too: its prefill row runs over the mirror with the projections
        assertArrayEquals(reference, baseline(base().speculativeOptions(w12(true)).build(), prompt));
    }

    @Test
    public void theF16ModelStreamsItsW12WeightsThroughTheFactory() throws IOException {
        assumeBackend();
        EngineConfig cfg = base().speculativeOptions(w12(true)).build();
        try (EngineModel m = LLMModelFactory.open(model.toString(), cfg)) {
            assertTrue("the F16 model stays on the device under the W12 flag", m instanceof GGUFModelGPU);
            assertEquals(GREEDY_TOKENS, greedy(m, cfg, PROMPT).length);
        }
    }
}

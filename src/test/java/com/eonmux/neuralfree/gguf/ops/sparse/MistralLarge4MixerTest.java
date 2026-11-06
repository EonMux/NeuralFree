package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The mistral-large4 DSA mixers on synthetic models: the direct {@code attn_q} path, the SSMax per-head query
 * temperatures ({@code attn_ssmax}, {@code indexer.ssmax}), the layer-normed indexer queries ({@code indexer.q_norm}),
 * the LeRoPE {@code rope_freqs} factors and, with {@link SyntheticModels#mistralLarge4NoPe}, a mixed rope pattern whose
 * NoPE layers zero-pad the rope dims of Q and of the latent row. The GPU backend runs the same model as the host and
 * returns the same logits.
 */
public class MistralLarge4MixerTest {

    private static final int[] PROMPT = {4, 17, 42, 5};
    private static final int[] DECODE = {77, 3};
    /** More tokens than the indexer's top-k, so the scores decide which latent rows the attention reads. */
    private static final int[] LONG_PROMPT = {4, 17, 42, 5, 9, 23, 31, 8, 60, 2, 11, 7};
    private static final int[] LONG_DECODE = {33};
    private static final float TOL = 2e-3f;

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
    }

    private static MemoryConfig longMemory() {
        return MemoryConfig.builder().nCtx(64).ubatch(16).typeK(KvDType.F32).typeV(KvDType.F32).build();
    }

    private static EngineConfig cpuConfig() {
        return EngineConfig.builder().memory(memory()).build();
    }

    private static EngineConfig gpuConfig() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(memory()).build();
    }

    /** Prefill and decode logits of the prompt on the host backend. */
    private static float[][] cpuLogits(Path model) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), cpuConfig())) {
            return logits(m.forwardBackend().prefill(PROMPT, 0), m, DECODE);
        }
    }

    /** Prefill and decode logits of the prompt on the OpenCL backend. */
    private static float[][] gpuLogits(Path model) throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, gpuConfig())) {
            return logits(m.forwardBackend().prefill(PROMPT, 0), m, DECODE);
        }
    }

    /** Prefill and decode logits of the long prompt on the host backend. */
    private static float[][] longCpuLogits(Path model) throws IOException {
        EngineConfig cfg = EngineConfig.builder().memory(longMemory()).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            return logits(m.forwardBackend().prefill(LONG_PROMPT, 0), m, LONG_PROMPT, LONG_DECODE);
        }
    }

    /** Prefill and decode logits of the long prompt on the OpenCL backend. */
    private static float[][] longGpuLogits(Path model) throws IOException {
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(longMemory()).build();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            return logits(m.forwardBackend().prefill(LONG_PROMPT, 0), m, LONG_PROMPT, LONG_DECODE);
        }
    }

    /** The prefill row followed by the decode rows, each copied before the backend reuses it. */
    private static float[][] logits(float[] prefill, GGUFModel m, int[] decode) {
        return logits(prefill, m, PROMPT, decode);
    }

    private static float[][] logits(float[] prefill, GGUFModelGPU m, int[] decode) {
        return logits(prefill, m, PROMPT, decode);
    }

    private static float[][] logits(float[] prefill, GGUFModel m, int[] prompt, int[] decode) {
        float[][] out = new float[1 + decode.length][];
        out[0] = prefill.clone();
        for (int i = 0; i < decode.length; i++) {
            out[1 + i] = m.forwardBackend().decode(decode[i], prompt.length + i).clone();
        }
        return out;
    }

    private static float[][] logits(float[] prefill, GGUFModelGPU m, int[] prompt, int[] decode) {
        float[][] out = new float[1 + decode.length][];
        out[0] = prefill.clone();
        for (int i = 0; i < decode.length; i++) {
            out[1 + i] = m.forwardBackend().decode(decode[i], prompt.length + i).clone();
        }
        return out;
    }

    private static void assertSameLogits(float[][] host, float[][] device) {
        assertEquals(host.length, device.length);
        for (int step = 0; step < host.length; step++) {
            assertEquals("step " + step + " width", host[step].length, device[step].length);
            for (int i = 0; i < host[step].length; i++) {
                assertTrue("step " + step + " logit " + i + " is not finite",
                        Float.isFinite(host[step][i]) && Float.isFinite(device[step][i]));
                assertEquals("step " + step + " logit " + i, host[step][i], device[step][i], TOL);
            }
        }
    }

    @Test
    public void repeatedPrefillIsDeterministic() throws IOException {
        Path model = SyntheticModels.mistralLarge4().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), cpuConfig())) {
            float[] first = m.forwardBackend().prefill(PROMPT, 0).clone();
            m.forwardBackend().reset();
            float[] second = m.forwardBackend().prefill(PROMPT, 0).clone();
            for (int i = 0; i < first.length; i++) {
                assertEquals("logit " + i, first[i], second[i], TOL);
            }
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void cpuRunsTheRopedModel() throws IOException {
        Path model = SyntheticModels.mistralLarge4().writeTempFile();
        try {
            float[][] logits = cpuLogits(model);
            assertEquals(1 + DECODE.length, logits.length);
            assertTrue(logits[0].length > 0);
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void cpuRunsTheMixedRopePattern() throws IOException {
        Path model = SyntheticModels.mistralLarge4NoPe(SyntheticModels.Config.mistralLarge4()).writeTempFile();
        Path ropedModel = SyntheticModels.mistralLarge4().writeTempFile();
        try {
            float[][] roped = cpuLogits(ropedModel);
            float[][] mixed = cpuLogits(model);
            assertEquals(roped.length, mixed.length);
            assertEquals(roped[0].length, mixed[0].length);
            assertTrue("the mixed rope pattern must change the logits", differs(roped[0], mixed[0]));
        } finally {
            Files.deleteIfExists(model);
            Files.deleteIfExists(ropedModel);
        }
    }

    @Test
    public void gpuMatchesTheHostOnTheRopedModel() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = SyntheticModels.mistralLarge4().writeTempFile();
        try {
            assertSameLogits(cpuLogits(model), gpuLogits(model));
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void gpuMatchesTheHostOnTheMixedRopePattern() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = SyntheticModels.mistralLarge4NoPe(SyntheticModels.Config.mistralLarge4()).writeTempFile();
        try {
            assertSameLogits(cpuLogits(model), gpuLogits(model));
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void gpuMatchesTheHostWhenTheIndexerSelectsAmongMoreRowsThanTopK() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path roped = SyntheticModels.mistralLarge4().writeTempFile();
        Path mixed = SyntheticModels.mistralLarge4NoPe(SyntheticModels.Config.mistralLarge4()).writeTempFile();
        try {
            assertSameLogits(longCpuLogits(roped), longGpuLogits(roped));
            assertSameLogits(longCpuLogits(mixed), longGpuLogits(mixed));
        } finally {
            Files.deleteIfExists(roped);
            Files.deleteIfExists(mixed);
        }
    }

    private static boolean differs(float[] a, float[] b) {
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > 1e-4f) {
                return true;
            }
        }
        return false;
    }
}

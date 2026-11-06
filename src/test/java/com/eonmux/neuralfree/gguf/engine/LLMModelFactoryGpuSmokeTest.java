package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.SequenceEncoder;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Every synthetic model family the GPU engine claims to run opens through {@link LLMModelFactory} on each backend
 * the machine has, without falling back to the CPU engine, and decodes what the CPU engine decodes. The factory's
 * fallback must stay a deliberate decision (a device that cannot serve the model), so a family whose GPU path broke
 * — a block that refuses to bind, a kernel left unbound, a weight the uploader rejects — fails here the day it
 * regresses, instead of silently opening at a fraction of the speed.
 *
 * <p>Generative families are compared on greedy tokens and on prefill/decode logits against the CPU engine; the
 * encoder and encoder-decoder families are compared on their embedding row and decoder rows. The models are the
 * tiny {@link SyntheticModels} fixtures, so the whole test runs in seconds on the normal {@code test} task and skips
 * itself per backend on machines without one.</p>
 */
public class LLMModelFactoryGpuSmokeTest {

    private static final int[] PROMPT = {2, 7, 3, 9, 4, 11};
    private static final int[] DECODE = {13, 5};
    private static final int GREEDY_TOKENS = 3;
    /** Logit tolerance of a GPU engine against the CPU engine on the same tiny model. */
    private static final float TOL = 5e-3f;
    private static final int T5_SEQ = 0;

    private final List<Path> files = new ArrayList<>();

    @After
    public void cleanUp() throws IOException {
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
        files.clear();
    }

    /** How the smoke test drives one family. */
    private enum Mode {
        /** Greedy tokens and prefill/decode logits. */
        GENERATIVE,
        /** The pooled row of one prompt ({@code encode}). */
        ENCODE,
        /** The paged decoder batch over one encoded source. */
        ENC_DEC
    }

    /** One synthetic family the smoke test loads on every present GPU backend. */
    private static final class Family {
        final String name;
        final Supplier<GgufTestWriter> writer;
        final Mode mode;

        Family(String name, Mode mode, Supplier<GgufTestWriter> writer) {
            this.name = name;
            this.mode = mode;
            this.writer = writer;
        }
    }

    private static final List<Family> FAMILIES = Arrays.asList(
            new Family("dense attention (llama)", Mode.GENERATIVE, () -> SyntheticModels.llama()),
            new Family("sliding-window attention (gemma2)", Mode.GENERATIVE, () -> SyntheticModels.gemma2()),
            new Family("routed MoE (qwen3moe)", Mode.GENERATIVE, () -> SyntheticModels.qwen3moe()),
            new Family("routed MoE with a shared expert (glm4moe)", Mode.GENERATIVE,
                    () -> SyntheticModels.glm4moe(SyntheticModels.Config.qwen3moe(), 0)),
            new Family("MLA (deepseek2)", Mode.GENERATIVE, () -> SyntheticModels.deepseek2()),
            new Family("MLA with the DSA indexer (deepseek32)", Mode.GENERATIVE, () -> SyntheticModels.deepseek32()),
            new Family("MLA with the DSA indexer and routed experts (mistralLarge4)", Mode.GENERATIVE,
                    () -> SyntheticModels.mistralLarge4(SyntheticModels.Config.mistralLarge4())),
            new Family("recurrent SSM (mamba)", Mode.GENERATIVE, () -> SyntheticModels.mamba()),
            new Family("recurrent SSM (mamba2)", Mode.GENERATIVE, () -> SyntheticModels.mamba2()),
            new Family("hybrid mamba attention (plamo2)", Mode.GENERATIVE, () -> SyntheticModels.plamo2()),
            new Family("hybrid KDA attention with MLA (kimiLinear)", Mode.GENERATIVE,
                    () -> SyntheticModels.kimiLinear()),
            new Family("hybrid GDN attention with MoE (qwen3next)", Mode.GENERATIVE,
                    () -> SyntheticModels.qwen3next()),
            new Family("hybrid short convolution (lfm2)", Mode.GENERATIVE, () -> SyntheticModels.lfm2()),
            new Family("hybrid short convolution with sliding window (lfm3)", Mode.GENERATIVE,
                    () -> SyntheticModels.lfm3()),
            new Family("lightning attention (h2oLightning)", Mode.GENERATIVE, () -> SyntheticModels.h2oLightning()),
            new Family("encoder (bert)", Mode.ENCODE, () -> SyntheticModels.bert()),
            new Family("encoder-decoder (t5)", Mode.ENC_DEC, () -> SyntheticModels.t5()));

    @Test
    public void everyFamilyOpensOnOpenClAndDecodesLikeTheCpuEngine() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        assertFamilies(EngineConfig.BackendMode.OPENCL);
    }

    @Test
    public void everyFamilyOpensOnCudaAndDecodesLikeTheCpuEngine() throws IOException {
        GpuTestUtil.assumeCudaAvailable();
        assertFamilies(EngineConfig.BackendMode.CUDA);
    }

    @Test
    public void everyFamilyOpensOnVulkanAndDecodesLikeTheCpuEngine() throws IOException {
        GpuTestUtil.assumeVulkanAvailable();
        assertFamilies(EngineConfig.BackendMode.VULKAN);
    }

    private void assertFamilies(EngineConfig.BackendMode backend) throws IOException {
        for (Family family : FAMILIES) {
            Path model = family.writer.get().writeTempFile();
            files.add(model);
            try {
                assertOneFamily(backend, family, model);
            } finally {
                Files.deleteIfExists(model);
                files.remove(model);
            }
        }
    }

    /**
     * One family on one backend: the factory hands back the GPU engine of the requested backend with no fallback,
     * and its decode matches the CPU engine's.
     */
    private void assertOneFamily(EngineConfig.BackendMode backend, Family family, Path model) throws IOException {
        try (EngineModel gpu = LLMModelFactory.open(model.toString(), gpuConfig(backend, family.mode))) {
            assertTrue(family.name + ": the factory fell back to the CPU engine (" + gpu.gpuFallbackReason() + ")",
                    gpu instanceof GGUFModelGPU);
            GGUFModelGPU gpuModel = (GGUFModelGPU) gpu;
            assertEquals(family.name + ": backend type",
                    ComputeBackend.BackendType.valueOf(backend.name()), gpuModel.backend().backendType());
            try (GGUFModel cpu = new GGUFModel(model.toString(), cpuConfig(family.mode))) {
                switch (family.mode) {
                    case GENERATIVE:
                        assertGenerativeDecodeMatchesTheCpuEngine(family.name, cpu, gpuModel);
                        break;
                    case ENCODE:
                        assertEncodeMatchesTheCpuEngine(family.name, cpu, gpuModel);
                        break;
                    case ENC_DEC:
                        assertT5DecodeMatchesTheCpuEngine(family.name, cpu, gpuModel);
                        break;
                }
            }
        }
    }

    /** Greedy tokens equal, and prefill and decode logits equal within {@link #TOL}. */
    private void assertGenerativeDecodeMatchesTheCpuEngine(String name, GGUFModel cpu, GGUFModelGPU gpu)
            throws IOException {
        int[] cpuTokens = cpu.generate(PROMPT, GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                .sampler(SamplerConfig.greedyDecoding()).build(), null).tokens();
        int[] gpuTokens = gpu.generate(PROMPT, GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                .sampler(SamplerConfig.greedyDecoding()).build(), null).tokens();
        assertArrayEquals(name + ": greedy tokens", cpuTokens, gpuTokens);

        float[][] cpuRows = prefillAndDecode(cpu.forwardBackend());
        float[][] gpuRows = prefillAndDecode(gpu.forwardBackend());
        assertEquals(name + ": steps", cpuRows.length, gpuRows.length);
        for (int i = 0; i < cpuRows.length; i++) {
            assertClose(name + ": step " + i, cpuRows[i], gpuRows[i]);
        }
    }

    private float[][] prefillAndDecode(com.eonmux.neuralfree.gguf.engine.ForwardBackend fb) {
        fb.reset();
        float[][] out = new float[1 + DECODE.length][];
        out[0] = fb.prefill(PROMPT, 0).clone();
        for (int i = 0; i < DECODE.length; i++) {
            out[1 + i] = fb.decode(DECODE[i], PROMPT.length + i).clone();
        }
        fb.reset();
        return out;
    }

    /** The encoder family: the pooled row of one prompt equals the CPU engine's. */
    private void assertEncodeMatchesTheCpuEngine(String name, GGUFModel cpu, GGUFModelGPU gpu) throws IOException {
        assertClose(name + ": embedding row", cpu.encode(PROMPT), gpu.encode(PROMPT));
    }

    /**
     * The encoder-decoder family: one source encoded per sequence, the decoder start and a few rows stepped, then a
     * decode round, everything equal to the CPU engine within {@link #TOL} (the paged path is the one a t5 serves).
     */
    private void assertT5DecodeMatchesTheCpuEngine(String name, GGUFModel cpu, GGUFModelGPU gpu) throws IOException {
        assertNotNull(name + ": the t5 load keeps a sequence encoder", gpu.forwardBackend().sequenceEncoder());
        float[][] cpuRows = t5Scenario(cpu.forwardBackend());
        float[][] gpuRows = t5Scenario(gpu.forwardBackend());
        assertEquals(name + ": steps", cpuRows.length, gpuRows.length);
        for (int i = 0; i < cpuRows.length; i++) {
            assertClose(name + ": decoder row " + i, cpuRows[i], gpuRows[i]);
        }
    }

    private float[][] t5Scenario(com.eonmux.neuralfree.gguf.engine.ForwardBackend fb) {
        SequenceEncoder enc = fb.sequenceEncoder();
        enc.encodeSequence(T5_SEQ, Arrays.copyOf(PROMPT, 4));
        ForwardBatch.Builder batch = ForwardBatch.builder(4);
        batch.add(SyntheticModels.T5_DECODER_START, 0, T5_SEQ, true);
        batch.add(DECODE[0], 1, T5_SEQ, true);
        batch.add(DECODE[1], 2, T5_SEQ, true);
        batch.add(DECODE[0], 3, T5_SEQ, true);
        float[][] first = fb.forwardBatch(batch.build());
        ForwardBatch.Builder second = ForwardBatch.builder(1);
        second.add(DECODE[1], 4, T5_SEQ, true);
        float[][] out = new float[first.length + 1][];
        for (int i = 0; i < first.length; i++) {
            out[i] = first[i].clone();
        }
        out[first.length] = fb.forwardBatch(second.build())[0].clone();
        fb.memory().sequences().remove(T5_SEQ, 0, -1);
        fb.reset();
        return out;
    }

    private void assertClose(String what, float[] expected, float[] got) {
        assertEquals(what + ": width", expected.length, got.length);
        float max = 0f;
        for (int j = 0; j < expected.length; j++) {
            if (!Float.isFinite(expected[j]) || !Float.isFinite(got[j])) {
                throw new AssertionError(what + ": channel " + j + " is not finite (cpu " + expected[j]
                        + ", gpu " + got[j] + ")");
            }
            max = Math.max(max, Math.abs(expected[j] - got[j]));
        }
        assertTrue(what + ": the GPU row differs from the CPU row by " + max + " > " + TOL, max <= TOL);
    }

    private static MemoryConfig memory(Mode mode) {
        // the ubatch covers the whole 6-token prompt: the encoder evaluates every token in one step
        MemoryConfig.Builder mc = MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(8);
        if (mode == Mode.ENC_DEC) {
            // the encoder-decoder serves through the paged batch, which keeps one encoding per sequence
            mc = mc.kvPaged(true).parallel(2);
        }
        return mc.typeK(KvDType.F32).typeV(KvDType.F32).build();
    }

    private static EngineConfig gpuConfig(EngineConfig.BackendMode backend, Mode mode) {
        return EngineConfig.builder().backend(backend).gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                .memory(memory(mode)).build();
    }

    private static EngineConfig cpuConfig(Mode mode) {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory(mode)).build();
    }
}

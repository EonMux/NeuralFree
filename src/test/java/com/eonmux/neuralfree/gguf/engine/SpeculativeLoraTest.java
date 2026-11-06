package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.LoraAdapterFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Speculative decoding over per-request adapters and a startup activated LoRA (vLLM serves LoRA with speculative
 * decoding; the drafter runs no adapters): the target's verify rows run in the request's adapter slot, the rows
 * before an aLoRA's invocation prefill in the slot before it, and greedy speculative output equals the plain
 * loop's greedy output of the same request (the {@code MATCH} verification is lossless).
 */
public class SpeculativeLoraTest {

    private static final int NEW_TOKENS = 12;
    private static final int RANK = 4;
    private static final float ALPHA = 8.0f;
    /** The aLoRA invocation tokens, early in the prompt: the adapter applies from them to nearly every row. */
    private static final int[] INVOCATION = {5, 133};
    /** Strong enough that the rank-4 factors flip greedy tokens on the repeat prompt (the differ checks). */
    private static final float SCALE = 3.0f;

    private static Path modelFile;
    private static Path draftFile;
    private static Path aloraFile;
    private static Path plainFile;

    @BeforeClass
    public static void writeModels() throws IOException {
        modelFile = SyntheticModels.llama().writeTempFile();
        draftFile = SyntheticModels.llama().writeTempFile();
        aloraFile = LoraAdapterFiles.activated(3000, RANK, ALPHA, INVOCATION);
        plainFile = LoraAdapterFiles.llama(1000, RANK, ALPHA);
    }

    @AfterClass
    public static void deleteModels() throws IOException {
        Files.deleteIfExists(modelFile);
        Files.deleteIfExists(draftFile);
        Files.deleteIfExists(aloraFile);
        Files.deleteIfExists(plainFile);
    }

    /** The CPU engine with exact F32 caches, a per-request adapter table and a short n-gram drafter. */
    private static EngineConfig.Builder base() {
        return engine().draftType(EngineConfig.DraftType.NGRAM).draftMax(4).draftMin(0)
                .speculativeOptions(SpeculativeOptions.builder().ngram(NgramDrafter.Options.builder(
                        NgramDrafter.Kind.SIMPLE).sizeN(2).sizeM(4).draftMin(0).build()).build());
    }

    /** The engine without a drafter: the caller adds a lookup type or a draft model file. */
    private static EngineConfig.Builder engine() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(128).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory)
                .loraConfig(LoraConfig.builder().maxAdapters(2).maxRank(64).cacheEntries(2).build())
                .draftMax(4).draftMin(0);
    }

    /**
     * The {@code (4, 5, v)} triples and the final {@code (4, 5)} a size-2 n-gram drafter proposes from, headed by
     * a {@code (5, 133)} pair and a 133 run: an adapter whose continuation is a 133 run (the strong scale flips
     * the greedy tokens) still finds the pattern deep in the history, so its rounds draft too.
     */
    private static int[] invokedPrompt() {
        int[] p = new int[5 + 3 * 30 + 2];
        int w = 0;
        p[w++] = 11;
        p[w++] = 5;
        p[w++] = 133;
        p[w++] = 133;
        p[w++] = 133;
        for (int v = 6; v <= 35; v++) {
            p[w++] = 4;
            p[w++] = 5;
            p[w++] = v;
        }
        p[w++] = 4;
        p[w] = 5;
        return p;
    }

    private static GenerationRequest request(LoraSelection lora) {
        return GenerationRequest.builder().maxNewTokens(NEW_TOKENS).sampler(SamplerConfig.greedyDecoding())
                .lora(lora).build();
    }

    @Test
    public void speculativeDecodingRunsTheRequestAdaptersOnTheTarget() throws IOException {
        EngineConfig cfg = base().build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg)) {
            model.loraRegistry().load("p", plainFile.toString(), SCALE);
            GenerationLoop loop = GenerationLoop.forModel(model, cfg);
            int[] adapted = loop.generate(invokedPrompt(), request(LoraSelection.of("p")), null).tokens();
            int[] bare = loop.generate(invokedPrompt(), request(null), null).tokens();
            assertTrue("the adapter must change the text", !Arrays.equals(adapted, bare));

            try (SpeculativeSession spec = LLMModelFactory.speculate(model, cfg)) {
                int[] speculated = spec.generate(invokedPrompt(), request(LoraSelection.of("p")), null).tokens();
                assertArrayEquals("greedy speculation is lossless under the request's adapter", adapted,
                        speculated);
                assertTrue("the adapted rounds drafted and verified: " + spec.lastStats(),
                        spec.lastStats().rounds() > 0 && spec.lastStats().drafted() > 0);
                spec.reset();
                assertArrayEquals("the base rows run in the bare slot", bare,
                        spec.generate(invokedPrompt(), request(null), null).tokens());
            }
        }
    }

    @Test
    public void speculativeDecodingRunsAStartupAloraFromItsInvocation() throws IOException {
        // a separate draft model proposes every round (vLLM: the drafter runs without the target's adapter), so
        // the aLoRA's verify rows run in slot 0 whatever the adapter makes the target emit
        EngineConfig cfg = engine().draftModelPath(draftFile.toString())
                .addLora(aloraFile.toString(), SCALE).build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg)) {
            String name = model.loraRegistry().list().get(0).name();
            GenerationLoop loop = GenerationLoop.forModel(model, cfg);
            int[] invoked = loop.generate(invokedPrompt(), request(null), null).tokens();
            int[] bare = loop.generate(invokedPrompt(),
                    request(LoraSelection.builder().add(name, 0f).build()), null).tokens();
            assertTrue("the invocation start must change the text", !Arrays.equals(invoked, bare));

            try (SpeculativeSession spec = LLMModelFactory.speculate(model, cfg)) {
                int[] specInvoked = spec.generate(invokedPrompt(), request(null), null).tokens();
                assertArrayEquals("the verify rows run in the startup aLoRA's slot 0", invoked, specInvoked);
                assertTrue("the draft model proposed every round: " + spec.lastStats(),
                        spec.lastStats().rounds() > 0 && spec.lastStats().drafted() > 0);
                spec.reset();
                assertArrayEquals("without the invocation the whole run is bare", bare,
                        spec.generate(invokedPrompt(),
                                request(LoraSelection.builder().add(name, 0f).build()), null).tokens());
            }
        }
    }

    @Test
    public void anUnloadedAdapterRefusesByName() throws IOException {
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(128).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .draftType(EngineConfig.DraftType.NGRAM).draftMax(4)
                .speculativeOptions(SpeculativeOptions.builder().ngram(NgramDrafter.Options.builder(
                        NgramDrafter.Kind.SIMPLE).sizeN(2).sizeM(4).draftMin(0).build()).build()).build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg);
             SpeculativeSession spec = LLMModelFactory.speculate(model, cfg)) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> spec.generate(invokedPrompt(), request(LoraSelection.of("p")), null));
            assertTrue(e.getMessage(), e.getMessage().contains("p is not loaded"));
        }
    }
}

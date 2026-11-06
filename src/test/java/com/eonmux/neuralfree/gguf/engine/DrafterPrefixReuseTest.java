package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticDflash;
import com.eonmux.neuralfree.gguf.testutil.SyntheticEagle;
import com.eonmux.neuralfree.gguf.testutil.SyntheticGemma4Assistant;
import com.eonmux.neuralfree.gguf.testutil.SyntheticGemmaModels;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Prefix reuse across a drafter's generations: the second generation of a conversation shares a prompt prefix
 * with the first, the target rolls back to it and prefills only the tail, and the drafter keeps the shared prefix
 * aligned instead of failing to place its own cache rows. Each kind keeps the plain greedy tokens of the second
 * prompt, drafts on it, and computes exactly the drafts and acceptances a fresh session over the same prompt does
 * (the shared prefix left the drafter's rows in the state a full prefill builds).
 *
 * <p>The kinds covered are EAGLE-1, EAGLE-3, a dflash host cache, the target's NextN/MTP heads, a separate draft
 * model and gemma4-assistant (which owns no cache and re-reads the target's rows on every generation).
 */
public class DrafterPrefixReuseTest {

    private static final int[] FIRST = {1, 17, 42, 5, 99, 26, 3, 64, 8, 51};
    /** Shares the first six tokens of {@link #FIRST}, then diverges. */
    private static final int[] SECOND = {1, 17, 42, 5, 99, 26, 11, 12, 13};
    private static final int GREEDY_TOKENS = 10;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static EngineConfig.Builder base() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(96).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory);
    }

    private static GenerationRequest greedy(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    /** Opens one session over a target and its drafter; closing it releases everything the opener owns. */
    private interface Opener {
        Owned open() throws IOException;
    }

    /** A session and the target it does not own, when the opener opened one. */
    private static final class Owned implements AutoCloseable {
        final SpeculativeSession session;
        private final EngineModel target;

        Owned(SpeculativeSession session, EngineModel target) {
            this.session = session;
            this.target = target;
        }

        @Override
        public void close() throws IOException {
            try {
                session.close();
            } finally {
                if (target != null) {
                    target.close();
                }
            }
        }
    }

    /** The tokens the plain target emits for {@link #SECOND}, from a model that never saw {@link #FIRST}. */
    private static int[] plainSecond(Path target) throws IOException {
        try (EngineModel m = LLMModelFactory.open(target.toString(), base().build())) {
            return m.generate(SECOND, greedy(GREEDY_TOKENS), null).tokens();
        }
    }

    /**
     * Two generations of one session against one fresh session over the same prompt: the second reuses the first's
     * stored prefix, keeps the target's greedy tokens, and leaves the drafter in exactly the state the full prefill
     * of the fresh session builds. The reused run's proposals and acceptances equal the fresh run's; a fixture whose
     * drafts the target never confirms (the random-weight eagle and NextN heads) checks that equality rather than a
     * positive count.
     */
    private static void twoGenerations(Opener opener, int[] expectedSecond, boolean requireAccepted)
            throws IOException {
        GenerationResult reused;
        SpeculativeDecoder.Stats reusedStats;
        try (Owned o = opener.open()) {
            o.session.generate(FIRST, greedy(GREEDY_TOKENS), null);
            reused = o.session.generate(SECOND, greedy(GREEDY_TOKENS), null);
            reusedStats = o.session.lastStats();
        }
        GenerationResult fresh;
        SpeculativeDecoder.Stats freshStats;
        try (Owned o = opener.open()) {
            fresh = o.session.generate(SECOND, greedy(GREEDY_TOKENS), null);
            freshStats = o.session.lastStats();
        }
        assertArrayEquals("prefix reuse keeps the target's greedy tokens", expectedSecond, reused.tokens());
        assertArrayEquals("a reusable prefix does not change the continuation", fresh.tokens(), reused.tokens());
        assertTrue("the second prompt reused its stored prefix: " + reused.timings().promptTokens(),
                reused.timings().promptTokens() < SECOND.length);
        System.out.println("reused " + reusedStats + " fresh " + freshStats);
        assertTrue("the second generation drafted: " + reusedStats, reusedStats.drafted() > 0);
        assertEquals("the reused prefix leaves the same proposals verified: " + reusedStats + " vs " + freshStats,
                freshStats.drafted(), reusedStats.drafted());
        assertEquals("the reused prefix leaves the same drafts accepted: " + reusedStats + " vs " + freshStats,
                freshStats.accepted(), reusedStats.accepted());
        if (requireAccepted) {
            assertTrue("the second generation accepted drafts: " + reusedStats, reusedStats.accepted() > 0);
        }
    }

    private Path target() throws IOException {
        return SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(4).build()).writeTempFile();
    }

    @Test
    public void eagle1KeepsTheReusedPrefix() throws IOException {
        Path target = target();
        Path draft = SyntheticEagle.eagle(SyntheticModels.Config.llama().toBuilder().nLayer(1).seed(43L).build(),
                true, 3).writeTempFile();
        EngineConfig cfg = base().draftMax(3).build();
        twoGenerations(() -> new Owned(LLMModelFactory.openSpeculative(target.toString(), draft.toString(), cfg),
                null), plainSecond(target), false);
    }

    @Test
    public void eagle3KeepsTheReusedPrefix() throws IOException {
        Path target = target();
        Path draft = SyntheticEagle.eagle3(SyntheticModels.Config.llama().toBuilder().nLayer(1).seed(44L).build(),
                new int[]{1, 2, 3}, true).writeTempFile();
        EngineConfig cfg = base().draftMax(3).build();
        twoGenerations(() -> new Owned(LLMModelFactory.openSpeculative(target.toString(), draft.toString(), cfg),
                null), plainSecond(target), false);
    }

    @Test
    public void dflashKeepsTheReusedPrefix() throws IOException {
        Path target = target();
        Path draft = SyntheticDflash.dflash(SyntheticModels.Config.llama(),
                SyntheticDflash.Options.of(4, 1, 2)).writeTempFile();
        EngineConfig cfg = base().draftModelPath(draft.toString()).draftMax(3).build();
        twoGenerations(() -> {
            EngineModel m = LLMModelFactory.open(target.toString(), base().build());
            return new Owned(LLMModelFactory.speculate(m, cfg), m);
        }, plainSecond(target), true);
    }

    @Test
    public void nextnKeepsTheReusedPrefix() throws IOException {
        Path target = SyntheticModels.glm4moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(2).build(), 1)
                .writeTempFile();
        EngineConfig cfg = base().mtp(true).draftMax(3).build();
        twoGenerations(() -> {
            EngineModel m = LLMModelFactory.open(target.toString(), base().build());
            return new Owned(LLMModelFactory.speculate(m, cfg), m);
        }, plainSecond(target), false);
    }

    @Test
    public void gemma4AssistantKeepsTheReusedPrefix() throws IOException {
        // four target layers put the assistant's blocks at layers 2 and 3, so no block reads a target layer of its
        // own index (which the drafter refuses)
        SyntheticModels.Config targetDims = SyntheticGemmaModels.gemma4Config().toBuilder().nLayer(4).build();
        Path target = SyntheticGemmaModels.gemma4(targetDims).writeTempFile();
        SyntheticModels.Config own = SyntheticModels.Config.llama().toBuilder().nEmbd(32).nLayer(2).seed(45L)
                .nVocab(targetDims.nVocab()).build();
        Path draft = SyntheticGemma4Assistant.gemma4Assistant(targetDims, own, 2).writeTempFile();
        EngineConfig cfg = base().draftMax(3).build();
        twoGenerations(() -> new Owned(LLMModelFactory.openSpeculative(target.toString(), draft.toString(), cfg),
                null), plainSecond(target), false);
    }

    @Test
    public void aSeparateDraftModelKeepsTheReusedPrefix() throws IOException {
        Path target = target();
        EngineConfig cfg = base().draftModelPath(target.toString()).draftMax(3).build();
        twoGenerations(() -> {
            EngineModel m = LLMModelFactory.open(target.toString(), base().build());
            return new Owned(LLMModelFactory.speculate(m, cfg), m);
        }, plainSecond(target), true);
    }
}

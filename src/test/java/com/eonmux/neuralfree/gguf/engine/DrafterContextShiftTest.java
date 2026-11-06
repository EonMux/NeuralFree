package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticDflash;
import com.eonmux.neuralfree.gguf.testutil.SyntheticEagle;
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
 * Hidden-fed drafters across a context shift ({@code neuralfree.context.shift}): the prompt fills the context, so
 * the target shifts before the first draft round, and the drafter rebuilds its cache over the post-shift positions
 * through {@link DraftModel#beginGeneration(int)} and the target's re-evaluated window. Each kind drafts after the
 * shift and emits exactly the target's greedy tokens, which a reference decoder with the same shift and no proposals
 * computes.
 *
 * <p>The kinds covered are EAGLE-1, EAGLE-3, the target's NextN/MTP head and a dflash draft; without the
 * re-alignment EAGLE-1, EAGLE-3 and NextN fail on their own cache's position gap.
 */
public class DrafterContextShiftTest {

    /** The prompt fills the context exactly, so the first loop iteration shifts. */
    private static final int N_CTX = 48;
    private static final int NEW_TOKENS = 10;
    private static final int BOS = 1;
    private static final int[] PROMPT = prompt();

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static int[] prompt() {
        int[] prompt = new int[N_CTX];
        prompt[0] = BOS;
        for (int i = 1; i < prompt.length; i++) {
            prompt[i] = 3 + (i * 37) % 120;
        }
        return prompt;
    }

    private static EngineConfig.Builder base() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8).typeK(KvDType.F32)
                .typeV(KvDType.F32).contextShift(true).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory);
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(NEW_TOKENS).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** Opens one speculative session over a target and its drafter; closing it releases everything it owns. */
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

    /**
     * A drafter that proposes nothing and asks the target for the same replay window the kind under test asks for:
     * its generation is the target's greedy continuation over the re-evaluated post-shift context.
     */
    private static final class SilentDrafter implements DraftModel {
        private final int replay;

        SilentDrafter(int replay) {
            this.replay = replay;
        }

        @Override
        public void bindTarget(EngineModel target) {
        }

        @Override
        public int[] draft(int lastToken, int n) {
            return new int[0];
        }

        @Override
        public void accept(int nAccepted) {
        }

        @Override
        public boolean readsTargetHidden() {
            return true;
        }

        @Override
        public int beginGeneration(int reused) {
            if (reused < 0) {
                throw new IllegalArgumentException("reused prefix length " + reused + " is negative");
            }
            return reused == 0 ? 0 : Math.min(replay, reused);
        }
    }

    /** The target's greedy tokens across the same shift without proposals. */
    private static int[] reference(Path targetPath, int replay) throws IOException {
        EngineConfig cfg = base().build();
        try (EngineModel model = LLMModelFactory.open(targetPath.toString(), cfg)) {
            SpeculativeDecoder decoder = SpeculativeDecoder.builder(model.forwardBackend(),
                    new SilentDrafter(replay), model.tokenizer())
                    .draftMax(1)
                    .contextShift(true)
                    .samplingDefaults(SamplingDefaults.forModel(model, cfg))
                    .build();
            GenerationResult r = decoder.generate(PROMPT, request(), (TokenCallback) null);
            assertEquals(StopReason.MAX_TOKENS, r.stopReason());
            return r.tokens();
        }
    }

    /**
     * The session's generation fills the context before its first draft round and shifts; it reaches the token
     * budget, drafts after the shift, and emits the reference's tokens.
     */
    private static void shiftsAndKeepsDrafting(Opener opener, Path targetPath, int replay) throws IOException {
        int[] tokens;
        SpeculativeDecoder.Stats stats;
        try (Owned o = opener.open()) {
            GenerationResult r = o.session.generate(PROMPT, request(), (TokenCallback) null);
            assertEquals("the shifted generation reaches its budget", StopReason.MAX_TOKENS, r.stopReason());
            tokens = r.tokens();
            stats = o.session.lastStats();
        }
        assertEquals(NEW_TOKENS, tokens.length);
        assertTrue("the drafter drafted after the shift: " + stats, stats.drafted() > 0);
        assertArrayEquals("drafting keeps the target's greedy tokens after the shift", reference(targetPath, replay),
                tokens);
    }

    @Test
    public void eagle1RealignsAfterTheShift() throws IOException {
        Path target = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(4).build())
                .writeTempFile();
        Path draft = SyntheticEagle.eagle(SyntheticModels.Config.llama().toBuilder().nLayer(1).seed(43L).build(),
                true, 3).writeTempFile();
        EngineConfig cfg = base().draftMax(3).build();
        shiftsAndKeepsDrafting(() -> new Owned(LLMModelFactory.openSpeculative(target.toString(),
                draft.toString(), cfg), null), target, 1);
    }

    @Test
    public void eagle3RealignsAfterTheShift() throws IOException {
        Path target = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(4).build())
                .writeTempFile();
        Path draft = SyntheticEagle.eagle3(
                SyntheticModels.Config.llama().toBuilder().nLayer(1).seed(44L).build(), new int[]{1, 2, 3}, true)
                .writeTempFile();
        EngineConfig cfg = base().draftMax(3).build();
        shiftsAndKeepsDrafting(() -> new Owned(LLMModelFactory.openSpeculative(target.toString(),
                draft.toString(), cfg), null), target, 1);
    }

    @Test
    public void nextnRealignsAfterTheShift() throws IOException {
        Path target = SyntheticModels.glm4moe(
                SyntheticModels.Config.qwen3moe().toBuilder().nLayer(2).build(), 1).writeTempFile();
        EngineConfig cfg = base().mtp(true).draftMax(3).build();
        shiftsAndKeepsDrafting(() -> {
            EngineModel m = LLMModelFactory.open(target.toString(), base().build());
            return new Owned(LLMModelFactory.speculate(m, cfg), m);
        }, target, 1);
    }

    @Test
    public void dflashRealignsAfterTheShift() throws IOException {
        Path target = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(4).build())
                .writeTempFile();
        Path draft = SyntheticDflash.dflash(SyntheticModels.Config.llama(),
                SyntheticDflash.Options.of(4, 1, 2)).writeTempFile();
        EngineConfig cfg = base().draftModelPath(draft.toString()).draftMax(3).build();
        shiftsAndKeepsDrafting(() -> {
            EngineModel m = LLMModelFactory.open(target.toString(), base().build());
            return new Owned(LLMModelFactory.speculate(m, cfg), m);
        }, target, 0);
    }
}

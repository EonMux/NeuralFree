package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The request's grammar constraint and request stages on the speculative path over the synthetic
 * llama (llama.cpp's speculative loop sampling through the same {@code common_sampler} with its grammar; vLLM's
 * structured output composed with spec decode): a drafter that keeps proposing text partly outside the grammar
 * changes nothing in a greedy constrained generation with a {@code min_tokens} floor, whichever verification mode
 * decides the rounds, and a seeded stochastic generation with repetition penalties still spells only grammar
 * sentences.
 */
public class SpeculativeConstraintModelTest {

    private static final ConstraintSpec YES_NO = ConstraintSpec.gbnf("root ::= (\"yes\" | \"no\")+");
    private static final String PARTIAL_YES_NO = "(yes|no)*(y|ye|n)?";

    private static EngineConfig config() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static int[] prompt() {
        return new int[]{11, 24, 37, 50, 63, 76};
    }

    /** Proposes the start of a fixed text every round: valid at some positions, cut by the grammar at others. */
    private static final class TextDrafter implements DraftModel {
        private final int[] tokens;
        int proposals;

        TextDrafter(int[] tokens) {
            this.tokens = tokens.clone();
        }

        @Override public void bindTarget(EngineModel target) {
        }

        @Override public int[] draft(int lastToken, int n) {
            proposals++;
            return Arrays.copyOf(tokens, Math.min(n, tokens.length));
        }

        @Override public void accept(int nAccepted) {
        }
    }

    private static GenerationRequest.Builder constrained(SamplerConfig sampler) {
        return GenerationRequest.builder().maxNewTokens(12).sampler(sampler).constraint(YES_NO);
    }

    @Test
    public void aGreedyConstrainedGenerationIsTheSameWithAndWithoutSpeculation() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            GenerationRequest request = constrained(SamplerConfig.greedyDecoding()).minTokens(4).build();
            GenerationResult plain = GenerationLoop.forModel(one, cfg).generate(prompt(), request, null);
            assertTrue("plain text was '" + plain.text() + "'", plain.text().matches(PARTIAL_YES_NO));
            int[] text = one.tokenizer().encode("nonoyesno", false, false);
            for (VerifyMode mode : new VerifyMode[]{VerifyMode.MATCH, VerifyMode.REJECTION, VerifyMode.BLOCK}) {
                TextDrafter drafter = new TextDrafter(text);
                SpeculativeDecoder.Stats[] stats = new SpeculativeDecoder.Stats[1];
                SpeculativeDecoder dec = SpeculativeDecoder.builder(one.forwardBackend(), drafter, one.tokenizer())
                        .draftMax(4).verifyMode(mode).statsSink(s -> stats[0] = s).build();
                GenerationResult spec = dec.generate(prompt(), request, null);
                assertEquals(mode + ": the speculative tokens", Arrays.toString(plain.tokens()),
                        Arrays.toString(spec.tokens()));
                assertTrue(mode + ": the drafter was asked", drafter.proposals > 0);
            }
        }
    }

    @Test
    public void aSeededStochasticConstrainedGenerationSpellsOnlyGrammarSentences() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            int[] text = one.tokenizer().encode("yesyesno", false, false);
            for (VerifyMode mode : new VerifyMode[]{VerifyMode.MATCH, VerifyMode.REJECTION, VerifyMode.BLOCK,
                    VerifyMode.FLY}) {
                for (long seed = 1; seed <= 4; seed++) {
                    SamplerConfig sampler = SamplerConfig.builder().temperature(1.5f).topK(0).topP(1f).minP(0f)
                            .repeatPenalty(1.1f).penaltyLastN(16).seed(seed).build();
                    GenerationRequest request = constrained(sampler).build();
                    SpeculativeDecoder dec = SpeculativeDecoder.builder(one.forwardBackend(),
                            new TextDrafter(text), one.tokenizer()).draftMax(4).verifyMode(mode).build();
                    String out = dec.generate(prompt(), request, null).text();
                    assertTrue(mode + " seed " + seed + " spelled '" + out + "'", out.matches(PARTIAL_YES_NO));
                }
            }
        }
    }
}

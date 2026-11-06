package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * A grammar-constrained sequence in a speculating {@link BatchEngine} (vLLM's structured output with spec decode,
 * {@code validate_tokens}): each proposal is cut before its first token the sequence's grammar refuses, as
 * {@code SpeculativeDecoder} cuts it, so the step verifies only rows the grammar could confirm. A greedy
 * constrained generation over the synthetic llama gives the same tokens as the plain single-sequence loop, and the
 * engine's spec counters show the cut: a proposal the grammar refuses outright is never verified, and a partly
 * admitted one verifies fewer tokens than were proposed.
 */
public class BatchEngineConstraintSpeculationTest {

    private static final ConstraintSpec YES_NO = ConstraintSpec.gbnf("root ::= (\"yes\" | \"no\")+");
    private static final int N_CTX = 64;

    private static EngineConfig config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(N_CTX).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32);
        if (parallel > 1) {
            mc.parallel(parallel);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    private static GenerationRequest constrained() {
        return GenerationRequest.builder().maxNewTokens(12).minTokens(4)
                .sampler(SamplerConfig.greedyDecoding()).constraint(YES_NO).build();
    }

    private static int[] prompt() {
        return new int[]{11, 24, 37, 50, 63, 76};
    }

    /** The reference: the request through the single-sequence GenerationLoop, no drafter. */
    private static GenerationResult reference(Path model, GenerationRequest request) throws IOException {
        EngineConfig cfg = config(1);
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt(), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    /** Proposes the start of a fixed token text every round and counts the tokens it proposed. */
    private static final class TextDrafter implements BatchDrafter {
        private final int[] tokens;
        int proposed;

        TextDrafter(int[] tokens) {
            this.tokens = tokens.clone();
        }

        @Override
        public int[] draft(int seq, int lastToken, int n) {
            int[] out = Arrays.copyOf(tokens, Math.min(n, tokens.length));
            proposed += out.length;
            return out;
        }

        @Override
        public void accept(int seq, int nAccepted) {
            if (nAccepted < 0) {
                throw new IllegalArgumentException("nAccepted must not be negative, got " + nAccepted);
            }
        }
    }

    /** The tokens spelling {@code text} one character each (the tokenizer's single-character pieces). */
    private static int[] charTokens(GGUFTokenizer tokenizer, String text) {
        int[] out = new int[text.length()];
        for (int c = 0; c < out.length; c++) {
            String ch = text.substring(c, c + 1);
            out[c] = -1;
            for (int id = 0; id < tokenizer.vocabSize() && out[c] < 0; id++) {
                if (ch.equals(tokenizer.decode(new int[]{id}))) {
                    out[c] = id;
                }
            }
            assertTrue("no single-character token for '" + ch + "'", out[c] >= 0);
        }
        return out;
    }

    /** Runs the constrained request through a batch engine whose drafter proposes {@code text}. */
    private static Run run(Path model, String text) throws IOException {
        EngineConfig cfg = config(2);
        try (GGUFModel two = new GGUFModel(model.toString(), cfg)) {
            TextDrafter drafter = new TextDrafter(charTokens(two.tokenizer(), text));
            try (BatchEngine engine = BatchEngine.forModel(two, cfg, drafter, new BatchSpec(4, 0))) {
                BatchSequence sequence = engine.submit(prompt(), constrained(),
                        (com.eonmux.neuralfree.gguf.TokenCallback) null);
                engine.drain();
                assertEquals(String.valueOf(sequence.error()), BatchSequence.State.DONE, sequence.state());
                return new Run(sequence.result(), engine.stats(), drafter.proposed);
            }
        }
    }

    private static final class Run {
        final GenerationResult result;
        final BatchStats stats;
        final int proposed;

        Run(GenerationResult result, BatchStats stats, int proposed) {
            this.result = result;
            this.stats = stats;
            this.proposed = proposed;
        }
    }

    @Test
    public void aProposalTheGrammarRefusesIsNeverVerified() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        GenerationResult ref = reference(model, constrained());
        assertTrue("plain text was '" + ref.text() + "'", ref.text().matches("(yes|no)*(y|ye|n)?"));

        Run run = run(model, "zzzz");

        assertArrayEquals(ref.tokens(), run.result.tokens());
        assertEquals(ref.text(), run.result.text());
        assertTrue("the drafter was asked", run.proposed > 0);
        assertEquals("no refused token reached a verify row", 0, run.stats.specDrafted());
        assertEquals(0, run.stats.specRounds());
    }

    @Test
    public void aPartlyAdmittedProposalVerifiesItsAdmittedPrefixOnly() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        GenerationResult ref = reference(model, constrained());

        // after a pending "n" the proposal "onoy" is admitted whole; after a completed word its "o" is refused
        Run run = run(model, "onoy");

        assertArrayEquals("the speculative tokens", ref.tokens(), run.result.tokens());
        assertEquals(ref.text(), run.result.text());
        assertEquals(ref.stopReason(), run.result.stopReason());
        assertTrue("some proposal was verified", run.stats.specDrafted() > 0);
        assertTrue("the grammar cut " + run.proposed + " proposed tokens to " + run.stats.specDrafted(),
                run.stats.specDrafted() < run.proposed);
        assertTrue(run.stats.specAccepted() <= run.stats.specDrafted());
    }
}

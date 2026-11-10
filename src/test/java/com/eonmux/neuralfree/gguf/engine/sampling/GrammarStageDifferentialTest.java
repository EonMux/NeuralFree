package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.ConstraintDeadEndException;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link GrammarStage} against {@link LegacyGrammarStage}, the list-based transcription of llama.cpp's grammar
 * runtime: over random grammars ({@link RandomGrammars}) and random walks through them, both stages compute the same
 * whole-row mask, the same {@code admits} answer for every token and the same completion after every accepted token,
 * including walks that leave a character pending across tokens, budget regions and saved and restored states. The
 * stage's {@link GrammarStage#keeps} answers equal its own mask, and a remembered mask ({@link GrammarMaskMemo})
 * equals the walk it replaced.
 */
public class GrammarStageDifferentialTest {

    private static final int GRAMMARS = 400;
    private static final int STEPS = 24;

    @Test
    public void masksAdmitsAndAcceptsMatchTheReferenceOverRandomWalks() {
        Random rnd = new Random(20261110L);
        int memoReuse = 0;
        int pendingSteps = 0;
        int refused = 0;
        for (int g = 0; g < GRAMMARS; g++) {
            Grammar grammar = RandomGrammars.next(rnd);
            SamplerVocab vocab = RandomGrammars.VOCAB;
            LegacyGrammarStage legacy = new LegacyGrammarStage(grammar, vocab);
            GrammarStage stage = new GrammarStage(grammar, vocab);
            Sampler legacySampler = sampler(legacy, vocab);
            Sampler sampler = sampler(stage, vocab);
            List<Object[]> saved = new ArrayList<>();
            List<boolean[]> masks = new ArrayList<>();
            for (int step = 0; step < STEPS; step++) {
                String where = "grammar " + g + " step " + step;
                boolean[] expected = mask(legacySampler, vocab);
                boolean[] actual = mask(sampler, vocab);
                assertArrayEquals(where + " mask", expected, actual);
                for (boolean[] earlier : masks) {
                    if (java.util.Arrays.equals(earlier, actual)) {
                        memoReuse++;
                        break;
                    }
                }
                masks.add(actual);
                List<Integer> admitted = new ArrayList<>();
                List<Integer> unmaskedOnly = new ArrayList<>();
                for (int id = 0; id < vocab.vocabSize(); id++) {
                    assertEquals(where + " admits " + id, legacy.admits(id), stage.admits(id));
                    assertEquals(where + " keeps " + id, actual[id], stage.keeps(id));
                    if (actual[id]) {
                        admitted.add(id);
                    } else if (legacy.admits(id)) {
                        unmaskedOnly.add(id);
                    }
                }
                if (rnd.nextInt(5) == 0) {
                    saved.add(new Object[]{legacySampler.saveState(), sampler.saveState(), actual});
                }
                List<Integer> pool = !unmaskedOnly.isEmpty() && rnd.nextInt(4) == 0 ? unmaskedOnly : admitted;
                if (pool.isEmpty()) {
                    break;
                }
                int token = pool.get(rnd.nextInt(pool.size()));
                if (token == 0) {
                    break;
                }
                String legacyRefusal = refusal(legacySampler, token);
                assertEquals(where + " accept " + token, legacyRefusal, refusal(sampler, token));
                if (legacyRefusal != null) {
                    refused++;
                    continue; // llama.cpp's mask keeps a token whose code points run into a token element
                }
                assertEquals(where + " completed", legacy.completed(), stage.completed());
                byte[] piece = vocab.grammarPiece(token);
                pendingSteps += (piece[piece.length - 1] & 0xC0) == 0x80 || (piece[piece.length - 1] & 0x80) != 0
                        ? 1 : 0;
                if (!saved.isEmpty() && rnd.nextInt(6) == 0) {
                    Object[] back = saved.get(rnd.nextInt(saved.size()));
                    legacySampler.restoreState(back[0]);
                    sampler.restoreState(back[1]);
                    assertArrayEquals(where + " restored mask", (boolean[]) back[2], mask(sampler, vocab));
                }
            }
        }
        assertTrue("some masked-in tokens are refused on accept, alike: " + refused, refused > 0);
        assertTrue("the walks revisit grammar states: " + memoReuse, memoReuse > 100);
        assertTrue("the walks leave characters pending across tokens: " + pendingSteps, pendingSteps > 100);
    }

    @Test
    public void rememberedMaskEqualsAFreshWalkInTheSameState() {
        Random rnd = new Random(7L);
        for (int g = 0; g < 100; g++) {
            Grammar grammar = RandomGrammars.next(rnd);
            SamplerVocab vocab = RandomGrammars.VOCAB;
            GrammarStage walked = new GrammarStage(grammar, vocab);
            Sampler sampler = sampler(walked, vocab);
            for (int step = 0; step < 12; step++) {
                boolean[] first = mask(sampler, vocab);
                boolean[] again = mask(sampler, vocab);
                assertArrayEquals("grammar " + g + " step " + step + " remembered", first, again);
                GrammarStage fresh = new GrammarStage(grammar, vocab);
                fresh.restoreState(walked.saveState());
                assertArrayEquals("grammar " + g + " step " + step + " fresh", first, mask(sampler(fresh, vocab),
                        vocab));
                List<Integer> admitted = new ArrayList<>();
                for (int id = 1; id < first.length; id++) {
                    if (first[id]) {
                        admitted.add(id);
                    }
                }
                if (admitted.isEmpty()) {
                    break;
                }
                sampler.accept(admitted.get(rnd.nextInt(admitted.size())));
            }
        }
    }

    /** The message of the exception accepting {@code token} throws, or null when the chain accepts it. */
    private static String refusal(Sampler sampler, int token) {
        try {
            sampler.accept(token);
            return null;
        } catch (IllegalStateException e) {
            return e.getMessage();
        }
    }

    private static Sampler sampler(SamplerStage stage, SamplerVocab vocab) {
        return Sampler.create(SamplerConfig.greedyDecoding(), new int[0], List.of(stage), vocab);
    }

    /** The tokens the chain's mask keeps over a uniform row; all false when it keeps none. */
    private static boolean[] mask(Sampler sampler, SamplerVocab vocab) {
        boolean[] kept = new boolean[vocab.vocabSize()];
        Sampler.Candidates c;
        try {
            c = sampler.apply(new float[vocab.vocabSize()]);
        } catch (ConstraintDeadEndException deadEnd) {
            return kept;
        }
        for (int i = 0; i < c.size(); i++) {
            kept[c.id(i)] = c.logit(i) != Float.NEGATIVE_INFINITY;
        }
        return kept;
    }
}

package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.GrammarTriggers;
import org.junit.Test;

import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The word-trigger window of {@link TriggeredGrammarStage}: a stage whose triggers are all literal words searches
 * only the bytes the newest piece can complete a word in, and fires exactly where a stage searching the whole
 * buffer for the same words as quoted patterns fires. Over random grammars, a grammar that takes any text, and
 * token walks through the stress vocabulary, with words that span pieces, start on a split UTF-8 sequence's
 * neighbour, or overlap, both stages wait and activate at the same tokens and then admit the same tokens.
 */
public class TriggerWordWindowTest {

    private static final List<List<String>> WORDS = List.of(List.of("ab"), List.of("é叫", "x"), List.of("bca", "c"),
            List.of("叫b", "a\""), List.of("aa", "aaa"));

    /** A grammar that takes whatever follows a trigger, so the walks fire it often. */
    private static final Grammar ANY_TEXT = GbnfBuilder.compile(ConstraintSpec.gbnf("root ::= .*")).grammar();

    @Test
    public void windowedWordsFireWhereTheWholeBufferScanFires() {
        Random rnd = new Random(31);
        SamplerVocab vocab = RandomGrammars.VOCAB;
        int activations = 0;
        for (int g = 0; g < 250; g++) {
            Grammar grammar = rnd.nextBoolean() ? RandomGrammars.next(rnd) : ANY_TEXT;
            List<String> words = WORDS.get(rnd.nextInt(WORDS.size()));
            GrammarTriggers.Builder patterns = GrammarTriggers.builder(GrammarTriggers.Mode.LAZY);
            for (String w : words) {
                patterns.pattern(Pattern.quote(w));
            }
            TriggeredGrammarStage windowed = new TriggeredGrammarStage(grammar, GrammarTriggers.lazyWords(words),
                    vocab);
            TriggeredGrammarStage scanned = new TriggeredGrammarStage(grammar, patterns.build(), vocab);
            windowed.keeps(1); // arms both stages, as the first sampled step does
            scanned.keeps(1);
            boolean wasWaiting = true;
            for (int step = 0; step < 30; step++) {
                String where = "grammar " + g + " words " + words + " step " + step;
                int token = pick(rnd, windowed, scanned, vocab);
                if (token < 0) {
                    break;
                }
                String a = accept(windowed, token);
                assertEquals(where, a, accept(scanned, token));
                if (a != null) {
                    break;
                }
                assertEquals(where, scanned.awaitingTrigger(), windowed.awaitingTrigger());
                if (!windowed.awaitingTrigger() && (wasWaiting || step % 4 == 0)) {
                    if (wasWaiting) {
                        activations++;
                        wasWaiting = false;
                    }
                    for (int id = 0; id < vocab.vocabSize(); id++) {
                        assertEquals(where + " admits " + id, scanned.admits(id), windowed.admits(id));
                    }
                }
            }
        }
        assertTrue("the walks fire triggers: " + activations, activations > 60);
    }

    /**
     * A random token both stages admit, or -1; one in six steps any non-end token, admitted or not. Half the draws
     * take a piece of one or two bytes, so the words build up across pieces and end on a piece's first byte.
     */
    private static int pick(Random rnd, TriggeredGrammarStage a, TriggeredGrammarStage b, SamplerVocab vocab) {
        if (rnd.nextInt(6) == 0) {
            return 1 + rnd.nextInt(vocab.vocabSize() - 1);
        }
        for (int tries = 0; tries < 40; tries++) {
            int id = 1 + rnd.nextInt(vocab.vocabSize() - 1);
            if (rnd.nextBoolean() && vocab.grammarPiece(id).length > 2) {
                continue;
            }
            if (a.admits(id) && b.admits(id)) {
                return id;
            }
        }
        return -1;
    }

    /** The message of the exception accepting {@code token} throws, or null. */
    private static String accept(TriggeredGrammarStage stage, int token) {
        try {
            stage.accept(token);
            return null;
        } catch (RuntimeException e) {
            return e.getClass().getSimpleName() + ": " + e.getMessage();
        }
    }
}

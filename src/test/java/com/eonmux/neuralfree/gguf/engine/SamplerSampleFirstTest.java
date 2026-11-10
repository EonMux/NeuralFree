package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.sampling.AllowedTokensStage;
import com.eonmux.neuralfree.gguf.engine.sampling.BadWordsStage;
import com.eonmux.neuralfree.gguf.engine.sampling.GrammarStage;
import com.eonmux.neuralfree.gguf.engine.sampling.MinTokensStage;
import com.eonmux.neuralfree.gguf.engine.sampling.RandomGrammars;
import com.eonmux.neuralfree.gguf.engine.sampling.ReasoningBudgetStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import com.eonmux.neuralfree.gguf.engine.sampling.TriggeredGrammarStage;
import com.eonmux.neuralfree.gguf.grammar.GbnfParser;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.GrammarTriggers;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The sample-first selection of a greedy constrained chain ({@link Sampler#sample}, {@link Sampler#sampleFirst}):
 * two identically built chains run in lockstep over random grammars and random rows, one selecting through
 * {@code sample}, the other through the whole-row mask of {@link Sampler#apply}; they pick the same token at every
 * step, or fail with the same exception, under a plain greedy chain, penalties, DRY, a logit bias with bans, the
 * request masks (allowed ids, bad words, the min-tokens ban) and a trigger-gated grammar. The sample-first path
 * decides most steps itself, so the comparison covers both of its outcomes. A token whose piece ends inside a
 * UTF-8 character gets the mask's answer from sample-first too, in a plain and a trigger-gated grammar: kept only
 * where a parse can complete the character. Sampled chains and chains with a set-dependent stage (typical-p, XTC,
 * top-n-sigma, infill, mirostat, adaptive-p, a logits processor, a reasoning budget) keep the whole-row mask.
 */
public class SamplerSampleFirstTest {

    private static final SamplerVocab VOCAB = RandomGrammars.VOCAB;

    @Test
    public void plainGreedyChainsAgreeWithTheMaskedChain() {
        lockstep(SamplerConfig.greedyDecoding(), Map.of(), grammar -> List.of(new GrammarStage(grammar, VOCAB)), 1);
    }

    @Test
    public void penaltiesDryAndBiasAgreeWithTheMaskedChain() {
        SamplerConfig config = SamplerConfig.builder().greedy(true).repeatPenalty(1.3f).presencePenalty(0.4f)
                .frequencyPenalty(0.2f).penaltyLastN(16).dryMultiplier(0.8f).dryBase(1.75f).dryAllowedLength(1)
                .dryPenaltyLastN(16).build();
        lockstep(config, Map.of(3, -1.5f, 5, 2f, 9, -100f),
                grammar -> List.of(new GrammarStage(grammar, VOCAB)), 2);
    }

    @Test
    public void requestMasksAgreeWithTheMaskedChain() {
        int[] allowed = new int[VOCAB.vocabSize() - 4];
        for (int i = 0; i < allowed.length; i++) {
            allowed[i] = i;
        }
        lockstep(SamplerConfig.greedyDecoding(), Map.of(), grammar -> List.of(new GrammarStage(grammar, VOCAB),
                new AllowedTokensStage(allowed), new BadWordsStage(List.of(new int[]{1, 2}, new int[]{5}), VOCAB),
                new MinTokensStage(3, VOCAB)), 3);
    }

    @Test
    public void triggeredGrammarsAgreeWithTheMaskedChain() {
        GrammarTriggers triggers = GrammarTriggers.lazyWords(List.of("ab", "x"));
        lockstep(SamplerConfig.greedyDecoding(), Map.of(),
                grammar -> List.of(new TriggeredGrammarStage(grammar, triggers, VOCAB, List.of("cc"), new int[]{8})),
                4);
    }

    @Test
    public void sampledAndSetDependentChainsKeepTheWholeRowMask() {
        Grammar grammar = RandomGrammars.next(new Random(1));
        assertTrue(sampler(SamplerConfig.greedyDecoding(), Map.of(), List.of(new GrammarStage(grammar, VOCAB)))
                .sampleFirstChain());
        assertFalse("a sampled chain truncates and draws over the admitted set",
                sampler(SamplerConfig.builder().temperature(0.7f).build(), Map.of(),
                        List.of(new GrammarStage(grammar, VOCAB))).sampleFirstChain());
        assertFalse("typical-p reads the whole set",
                sampler(SamplerConfig.builder().greedy(true).typicalP(0.5f).build(), Map.of(),
                        List.of(new GrammarStage(grammar, VOCAB))).sampleFirstChain());
        assertFalse("without a constraint there is nothing to check first",
                sampler(SamplerConfig.greedyDecoding(), Map.of(), List.of()).sampleFirstChain());
        SamplerStage opaque = new SamplerStage() {
            @Override public void apply(Sampler.Candidates candidates) { }
            @Override public boolean constrains() { return true; }
        };
        assertFalse("a constraint that cannot answer per token",
                sampler(SamplerConfig.greedyDecoding(), Map.of(), List.of(opaque)).sampleFirstChain());
    }

    @Test
    public void everyWholeSetStageKeepsTheWholeRowMask() {
        Grammar grammar = RandomGrammars.next(new Random(2));
        Function<SamplerConfig, Boolean> qualifies = config -> sampler(config, Map.of(),
                List.of(new GrammarStage(grammar, VOCAB))).sampleFirstChain();
        assertTrue("XTC that never fires applies nothing",
                qualifies.apply(SamplerConfig.builder().greedy(true).xtcProbability(0f).build()));
        assertFalse("XTC removes the top candidates of the admitted set",
                qualifies.apply(SamplerConfig.builder().greedy(true).xtcProbability(0.5f).xtcThreshold(0.1f).build()));
        assertFalse("top-n-sigma reads the admitted set's spread",
                qualifies.apply(SamplerConfig.builder().greedy(true).topNSigma(1f).build()));
        assertFalse("infill merges and renormalizes the admitted set",
                qualifies.apply(SamplerConfig.builder().greedy(true)
                        .samplers(List.of(StageKind.INFILL, StageKind.TEMPERATURE)).build()));
        assertFalse("mirostat ends the chain in its own selector",
                qualifies.apply(SamplerConfig.builder().greedy(true).mirostat(2).build()));
        assertFalse("adaptive-p ends the chain in its own selector",
                qualifies.apply(SamplerConfig.builder().greedy(true).adaptiveTarget(0.5f)
                        .samplers(List.of(StageKind.TEMPERATURE, StageKind.ADAPTIVE_P)).build()));
        SamplerStage processor = new SamplerStage() {
            @Override public void apply(Sampler.Candidates candidates) { }
        };
        assertFalse("a logits processor may read the whole set",
                Sampler.create(SamplerConfig.greedyDecoding(), new int[0], Map.of(),
                        List.of(new GrammarStage(grammar, VOCAB)), List.of(processor), VOCAB).sampleFirstChain());
        ReasoningBudgetStage budget = ReasoningBudgetStage.of(ReasoningFilter.thinkTags(),
                GenerationRequest.builder().maxNewTokens(64).reasoningBudget(2).build(), BYTES);
        assertFalse("a reasoning budget forces its own tokens behind the mask",
                sampler(SamplerConfig.greedyDecoding(), Map.of(),
                        List.of(new GrammarStage(grammar, VOCAB), budget)).sampleFirstChain());
    }

    @Test
    public void aTokenEndingInAPartialCharacterGetsTheMaskAnswer() {
        Grammar ascii = GbnfParser.parse("root ::= \"a\" [a-z]+");
        Sampler first = partialSampler(new GrammarStage(ascii, PARTIAL));
        Sampler masked = partialSampler(new GrammarStage(ascii, PARTIAL));
        agree(first, masked, partialRow(A_LEAD, A), A, "no completion of the lead byte is in [a-z]");
        agree(first, masked, partialRow(A_LEAD, B), B, "nor after a letter");

        Grammar han = GbnfParser.parse("root ::= \"a\" \"\u53eb\" \"b\"");
        first = partialSampler(new GrammarStage(han, PARTIAL));
        masked = partialSampler(new GrammarStage(han, PARTIAL));
        agree(first, masked, partialRow(A_LEAD, A), A_LEAD, "the lead byte can complete to \u53eb");
        agree(first, masked, partialRow(A, TAIL), TAIL, "a letter cannot continue the pending character");
        agree(first, masked, partialRow(A_LEAD, B), B, "the character is complete");
    }

    @Test
    public void aTriggerWordEndingInAPartialCharacterGetsTheMaskAnswer() {
        Grammar ascii = GbnfParser.parse("root ::= \"a\" [a-z]+");
        GrammarTriggers triggers = GrammarTriggers.lazyWords(List.of("a"));
        Sampler first = partialSampler(new TriggeredGrammarStage(ascii, triggers, PARTIAL));
        Sampler masked = partialSampler(new TriggeredGrammarStage(ascii, triggers, PARTIAL));
        agree(first, masked, partialRow(A_LEAD, A), A, "the trigger's grammar refuses the partial character");
        agree(first, masked, partialRow(A_LEAD, B), B, "the active grammar refuses it too");
    }

    private static final int A = 1;
    /** "a" followed by the lead byte of \u53eb (E5 8F AB): a piece ending inside a character. */
    private static final int A_LEAD = 2;
    private static final int B = 3;
    /** The two bytes that complete \u53eb after its lead byte. */
    private static final int TAIL = 4;

    /** Pieces around one split character; id 0 ends generation. */
    private static final SamplerVocab PARTIAL = new SamplerVocab() {
        private final byte[][] pieces = {{}, {'a'}, {'a', (byte) 0xE5}, {'b'}, {(byte) 0x8F, (byte) 0xAB}};
        @Override public int vocabSize() { return pieces.length; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) { return pieces[id].clone(); }
        @Override public String pieceText(int id) { return new String(pieces[id], StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) { return new int[0]; }
    };

    private static Sampler partialSampler(SamplerStage stage) {
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0], Map.of(), List.of(stage), PARTIAL);
        assertTrue(sampler.sampleFirstChain());
        return sampler;
    }

    /** A row whose argmax is {@code top} and whose runner-up is {@code next}. */
    private static float[] partialRow(int top, int next) {
        float[] row = new float[PARTIAL.vocabSize()];
        row[top] = 5f;
        row[next] = 4f;
        return row;
    }

    /**
     * The sample-first chain and the whole-row mask both select {@code expected} over {@code row}, and both accept
     * it.
     */
    private static void agree(Sampler first, Sampler masked, float[] row, int expected, String why) {
        assertEquals(why + ": the whole-row mask", expected, masked.apply(row).selectedId());
        assertEquals(why + ": sample-first", expected, first.sample(row));
        masked.accept(expected);
        first.accept(expected);
    }

    /** Ids 0-255 render their byte, so the reasoning budget can tokenize its tags; 0 ends generation. */
    private static final SamplerVocab BYTES = new SamplerVocab() {
        @Override public int vocabSize() { return 256; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) { return id <= 0 ? new byte[0] : new byte[]{(byte) id}; }
        @Override public String pieceText(int id) { return new String(piece(id), StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) {
            byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
            int[] ids = new int[bytes.length];
            for (int i = 0; i < bytes.length; i++) {
                ids[i] = bytes[i] & 0xFF;
            }
            return ids;
        }
    };

    private static Sampler sampler(SamplerConfig config, Map<Integer, Float> bias, List<SamplerStage> stages) {
        return Sampler.create(config, new int[]{4}, bias, stages, VOCAB);
    }

    /**
     * Runs {@code grammars} random grammars through two chains built by {@code stages}, one selecting sample-first
     * and one over the whole-row mask, and checks they agree at every step.
     */
    private static void lockstep(SamplerConfig config, Map<Integer, Float> bias,
                                 Function<Grammar, List<SamplerStage>> stages, long seed) {
        Random rnd = new Random(seed);
        int decided = 0;
        int fellBack = 0;
        for (int g = 0; g < 300; g++) {
            Grammar grammar = RandomGrammars.next(rnd);
            Sampler first = sampler(config, bias, stages.apply(grammar));
            Sampler masked = sampler(config, bias, stages.apply(grammar));
            assertTrue(first.sampleFirstChain());
            for (int step = 0; step < 20; step++) {
                float[] row = row(rnd);
                String where = "grammar " + g + " step " + step;
                Object expected = outcome(() -> masked.apply(row).selectedId());
                int quick = first.sampleFirst(row);
                Object actual = quick >= 0 ? (Object) quick : outcome(() -> first.apply(row).selectedId());
                if (quick >= 0) {
                    decided++;
                } else {
                    fellBack++;
                }
                assertEquals(where, expected, actual);
                if (!(expected instanceof Integer) || (Integer) expected == 0) {
                    break;
                }
                int token = (Integer) expected;
                Object acceptedMasked = outcome(() -> {
                    masked.accept(token);
                    return token;
                });
                Object acceptedFirst = outcome(() -> {
                    first.accept(token);
                    return token;
                });
                assertEquals(where + " accept", acceptedMasked, acceptedFirst);
                if (!(acceptedMasked instanceof Integer)) {
                    break;
                }
            }
        }
        assertTrue("sample-first decided " + decided, decided > 500);
        assertTrue("the masked fallback ran " + fellBack, fellBack > 200);
    }

    /** A random row; half the time one token is lifted far above the rest, as a confident model's row is. */
    private static float[] row(Random rnd) {
        float[] row = new float[VOCAB.vocabSize()];
        for (int i = 0; i < row.length; i++) {
            row[i] = (float) rnd.nextGaussian();
        }
        if (rnd.nextBoolean()) {
            row[rnd.nextInt(row.length)] += 6f;
        }
        if (rnd.nextInt(8) == 0) {
            row[rnd.nextInt(row.length)] = row[rnd.nextInt(row.length)];
        }
        return row;
    }

    /** The selected token, or the exception's class name and message when the step throws. */
    private static Object outcome(java.util.function.Supplier<Integer> step) {
        try {
            return step.get();
        } catch (RuntimeException e) {
            return e.getClass().getSimpleName() + ": " + e.getMessage();
        }
    }
}

package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.ConstraintDeadEndException;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The constraint head's masks and the truncating stages compose (llama.cpp applies the grammar before the chain on
 * its grammar-first resample, vLLM applies {@code allowed_token_ids} and bad words to the logits before any
 * sampling stage): the truncating stages operate on the admitted set, so a candidate the grammar admits and a
 * request's allowed id or non-banned token survive any top-k, top-p and min-p cut at a temperature above 0, the
 * min-tokens ban yields behind the other masks, and a constraint whose admitted set is empty refuses sampling
 * with the chain's named error instead of leaving the draw nothing to pick.
 */
public class SamplerConstraintCompositionTest {

    private static final long SEED = 7L;

    /** ASCII byte tokens: id 0 is EOG, ids 1..127 render their code point's byte. */
    private static final class ByteVocab implements SamplerVocab {
        @Override public int vocabSize() { return 128; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) {
            return id <= 0 ? new byte[0] : new byte[]{(byte) id};
        }
        @Override public String pieceText(int id) { return new String(piece(id), StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) {
            byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
            int[] ids = new int[bytes.length];
            for (int i = 0; i < bytes.length; i++) {
                ids[i] = bytes[i] & 0xFF;
            }
            return ids;
        }
    }

    /** The default chain at {@code temperature}: top-k 40, top-p 0.95, min-p 0.05, a fixed seed. */
    private static SamplerConfig chain(float temperature) {
        return SamplerConfig.builder().temperature(temperature).seed(SEED).build();
    }

    /** A tight chain at {@code temperature}: top-k, top-p and min-p small enough to cut everything below the peak. */
    private static SamplerConfig tightChain(float temperature, int topK) {
        return SamplerConfig.builder().temperature(temperature).seed(SEED)
                .topK(topK).topP(0.1f).minP(0.9f).build();
    }

    private static Sampler sampler(SamplerConfig config, String grammarSource) {
        return sampler(config, grammarSource, new ByteVocab());
    }

    private static Sampler sampler(SamplerConfig config, String grammarSource, SamplerVocab vocab) {
        grammar = new GrammarStage(GbnfBuilder.compile(ConstraintSpec.gbnf(grammarSource)).grammar(), vocab);
        return Sampler.create(config, new int[0], List.of(grammar), vocab);
    }

    /** The grammar stage of the last {@link #sampler} build, which names the step's admitted set. */
    private static GrammarStage grammar;

    /** A row whose highest logits sit on letters, so the space byte (32) falls outside every truncation. */
    private static float[] letterRow() {
        float[] row = new float[128];
        Arrays.fill(row, -1.0f);
        for (int id = 97; id <= 122; id++) {
            row[id] = 8.0f - (id - 97) * 0.05f;
        }
        row[32] = -2.0f;
        return row;
    }

    @Test
    public void theTruncatingStagesOperateOnTheAdmittedSet() {
        Sampler sampler = sampler(chain(0.8f), "root ::= \" \"");
        int sampled = 0;
        for (int step = 0; step < 16; step++) {
            Sampler.Candidates candidates = sampler.apply(letterRow());
            assertTrue("the chain left " + candidates.size() + " candidates", candidates.size() >= 1);
            int selected = candidates.selectedId();
            assertTrue("the draw left the admitted set: " + selected, grammar.admits(selected));
            for (int i = 0; i < candidates.size(); i++) {
                if (Float.isFinite(candidates.logit(i))) {
                    assertTrue("a finite candidate the grammar does not admit: " + candidates.id(i),
                            grammar.admits(candidates.id(i)));
                }
            }
            sampler.accept(selected);
            sampled++;
        }
        assertEquals(16, sampled);
    }

    @Test
    public void everyTruncationKeepsAnAdmittedCandidate() {
        Sampler sampler = sampler(chain(1.0f), "root ::= \" \"");
        for (int step = 0; step < 16; step++) {
            int token = sampler.sample(letterRow());
            assertTrue("the draw left the admitted set: " + token, grammar.admits(token));
            sampler.accept(token);
        }
    }

    @Test
    public void aDeadEndConstraintRefusesSampling() {
        Sampler sampler = sampler(chain(0.8f), "root ::= \"\\xC3\"");
        try {
            sampler.sample(letterRow());
            throw new AssertionError("expected the dead end to refuse sampling");
        } catch (ConstraintDeadEndException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("no candidate"));
        }
    }

    @Test
    public void aChainWithoutConstraintKeepsTheSamplingRefusal() {
        Sampler sampler = Sampler.create(SamplerConfig.builder().topK(0).topP(1f).minP(0f).typicalP(1f)
                .temperature(0.8f).seed(SEED).build(), new int[0]);
        float[] row = new float[8];
        Arrays.fill(row, Float.NEGATIVE_INFINITY);
        try {
            sampler.sample(row);
            throw new AssertionError("expected the all-masked row to refuse sampling");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("no candidate has a finite logit"));
        }
    }

    // ================================================================== the request's masks

    /** The {@code letterRow} with the byte 'x' (120) raised above every letter, so a tight top-k keeps it alone. */
    private static float[] xTopRow() {
        float[] row = letterRow();
        row['x'] = 9.0f;
        return row;
    }

    @Test
    public void allowedTokenIdsSurviveTightTruncations() {
        // the space byte sits far below every letter: a top-k of 2 over the unmasked row keeps two letters, so
        // only the head-first mask lets the allowed id through (vLLM masks the logits before any sampling stage)
        for (float temperature : new float[] {0.8f, 1.0f}) {
            Sampler sampler = Sampler.create(tightChain(temperature, 2), new int[0],
                    List.of(new AllowedTokensStage(new int[] {' '})), new ByteVocab());
            for (int step = 0; step < 8; step++) {
                Sampler.Candidates candidates = sampler.apply(letterRow());
                int selected = candidates.selectedId();
                assertEquals("temperature " + temperature + " step " + step + " sampled " + selected,
                        ' ', selected);
                for (int i = 0; i < candidates.size(); i++) {
                    if (Float.isFinite(candidates.logit(i))) {
                        assertEquals("a finite candidate outside the allowed ids", ' ', candidates.id(i));
                    }
                }
                sampler.accept(selected);
            }
        }
    }

    @Test
    public void badWordsSurviveTightTruncations() {
        // 'x' tops the row and a top-k of 1 keeps it alone: only the head-first mask leaves the draw anything
        // else to pick (vLLM applies bad words to the logits before any sampling stage)
        for (float temperature : new float[] {0.8f, 1.0f}) {
            Sampler sampler = Sampler.create(tightChain(temperature, 1), new int[0],
                    List.of(new BadWordsStage(List.of(new int[] {'x'}), new ByteVocab())), new ByteVocab());
            for (int step = 0; step < 8; step++) {
                int token = sampler.sample(xTopRow());
                assertEquals("temperature " + temperature + " step " + step + " sampled the banned byte",
                        'a', token);
                sampler.accept(token);
            }
        }
    }

    @Test
    public void theMinTokensBanYieldsBehindTheRequestMasks() {
        // allowed_token_ids leaves the end-of-generation token alone; the min-tokens ban behind it must yield
        // (vLLM's MinTokensLogitsProcessor restores the stop-token logits) instead of emptying the set
        SamplerVocab vocab = new ByteVocab();
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                List.of(new AllowedTokensStage(new int[] {0}), new MinTokensStage(3, vocab)), vocab);
        assertEquals("the ban yields and the allowed end-of-generation token is sampled", 0,
                sampler.sample(letterRow()));
    }

    @Test
    public void aMaskDeadEndRefusesSampling() {
        // the allowed ids leave only 'A', and the bad word bans 'A': the mask leaves no candidate
        SamplerVocab vocab = new ByteVocab();
        Sampler sampler = Sampler.create(chain(0.8f), new int[0],
                List.of(new AllowedTokensStage(new int[] {'A'}),
                        new BadWordsStage(List.of(new int[] {'A'}), vocab)), vocab);
        try {
            sampler.sample(letterRow());
            throw new AssertionError("expected the mask dead end to refuse sampling");
        } catch (ConstraintDeadEndException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("no candidate"));
        }
    }
}

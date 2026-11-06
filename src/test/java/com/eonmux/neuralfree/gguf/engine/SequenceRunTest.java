package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter.Channel;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Traces of the {@link SequenceRun} token pipeline on an 8-token vocabulary {@code a .. h} with a greedy sampler
 * (every row is a one-hot argmax; the uniform row picks the first candidate {@code a}). The traces walk the stages
 * of {@code GenerationLoop.run}'s token loop (GenerationLoop.java:184-206) in its order: sample, accept, the stop
 * decision of the criteria session, the routed text delta, and the counters and timings of the
 * {@link GenerationResult}.
 */
public class SequenceRunTest {

    private static final int EOS_ID = 7;

    /** The routed think tags, given as token ids 3 and 4. */
    private static final String OPEN_TAG = "<" + "think" + ">";
    private static final String CLOSE_TAG = "<" + "/" + "think" + ">";

    /** Vocab {@code a .. h}; with {@code eos} the token {@code h} (id 7) is the end-of-generation token. */
    private static GGUFTokenizer tokenizer(boolean eos) {
        List<Object> tokens = new ArrayList<>(Arrays.<Object>asList("a", "b", "c", "d", "e", "f", "g", "h"));
        List<Object> types = new ArrayList<>(Arrays.<Object>asList(1, 1, 1, 1, 1, 1, 1, 1));
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, "gpt2");
        md.put(TokenizerVocab.KEY_PRE, "gpt-2");
        md.put(TokenizerVocab.KEY_TOKENS, tokens);
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, types);
        md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>());
        if (eos) {
            md.put("tokenizer.ggml.eos_token_id", EOS_ID);
        }
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(md));
    }

    /** argmax over the vocab: the single 1.0 entry wins; the uniform row picks {@code a}. */
    private static float[] row(int winner) {
        float[] row = new float[8];
        if (winner >= 0) {
            row[winner] = 1f;
        }
        return row;
    }

    private static GenerationRequest request(int maxNewTokens, boolean cancellable) {
        GenerationRequest.Builder b = GenerationRequest.builder().sampler(SamplerConfig.greedyDecoding());
        if (maxNewTokens != GenerationRequest.UNLIMITED) {
            b.maxNewTokens(maxNewTokens);
        }
        if (cancellable) {
            b.cancellation(() -> true);
        }
        return b.build();
    }

    private static Sampler greedy(GenerationRequest request) {
        return Sampler.create(request.sampler());
    }

    private static SequenceRun run(GGUFTokenizer tok, GenerationRequest request) {
        return SequenceRun.over(request, tok).sampler(greedy(request)).build();
    }

    /** One piece of output text the session routed to the run's sink. */
    private static final class Piece {
        final Channel channel;
        final int tokenId;
        final String text;

        Piece(Channel channel, int tokenId, String text) {
            this.channel = channel;
            this.tokenId = tokenId;
            this.text = text;
        }
    }

    /** A run whose sink records every routed piece; {@code reasoning} drives the default stop criteria. */
    private static SequenceRun overCollecting(GGUFTokenizer tok, GenerationRequest request, ReasoningFilter reasoning,
                                              List<Piece> out) {
        return SequenceRun.over(request, tok).sampler(greedy(request))
                .stopCriteria(StopCriteria.of(request, tok, reasoning))
                .sink((channel, id, text) -> {
                    if (!text.isEmpty()) {
                        out.add(new Piece(channel, id, text));
                    }
                })
                .build();
    }

    private static List<String> texts(List<Piece> pieces, Channel channel) {
        List<String> out = new ArrayList<>();
        for (Piece p : pieces) {
            if (p.channel == channel) {
                out.add(p.text);
            }
        }
        return out;
    }

    /** Tag token ids for the think tags. */
    private static ReasoningFilter thinkFilter() {
        return ReasoningFilter.thinkTags().withTagTokens(t -> OPEN_TAG.equals(t) ? 3 : CLOSE_TAG.equals(t) ? 4 : -1);
    }

    @Test
    public void sampleAcceptOfferMatchesTheLoopOrder() {
        GGUFTokenizer tok = tokenizer(false);
        GenerationRequest request = request(4, false);
        SequenceRun run = run(tok, request);
        run.prefilled(2, 123L);

        int t0 = run.next(row(3));               // sample and accept d(3)
        SequenceRun.Step step = run.offer(t0);   // the session checks the stop and routes the text
        assertFalse(step.stopped());
        assertEquals("d", step.text());
        assertTrue(run.offer(run.next(row(2))).text().equals("c"));
        assertTrue(run.offer(run.next(row(1))).text().equals("b"));
        assertFalse(run.done());

        // the fourth generated token reaches the budget and stops the run
        SequenceRun.Step last = run.offer(run.next(row(2)));
        assertTrue(last.stopped());
        assertEquals(StopReason.MAX_TOKENS, last.reason());
        GenerationResult r = run.result();
        assertArrayEquals(new int[] {3, 2, 1, 2}, r.tokens());
        assertEquals("dcbc", r.text());
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        assertEquals(2, r.timings().promptTokens());
        assertEquals(123L, r.timings().promptNanos());
        assertEquals(4, r.timings().generatedTokens());
        assertTrue(r.timings().split());
        assertTrue(r.timings().generationNanos() >= 0);
    }

    @Test
    public void endOfGenerationTokenStopsAndStaysOutOfTheOutput() {
        GGUFTokenizer tok = tokenizer(true);
        GenerationRequest request = request(GenerationRequest.UNLIMITED, false);
        SequenceRun run = run(tok, request);

        assertFalse(run.offer(run.next(row(0))).stopped());
        int t1 = run.next(row(EOS_ID));
        SequenceRun.Step step = run.offer(t1);

        assertTrue(step.stopped());
        assertEquals(StopReason.EOS, step.reason());
        assertTrue(run.done());
        GenerationResult r = run.result();
        assertArrayEquals(new int[] {0}, r.tokens());
        assertEquals("a", r.text());
        assertEquals(StopReason.EOS, r.stopReason());
    }

    /** A stop string is completed, reported as the reason; its own text and the tail after it are withheld. */
    @Test
    public void stopStringStopsAndCutsTheTextBeforeIt() {
        GGUFTokenizer tok = tokenizer(false);
        GenerationRequest request = GenerationRequest.builder().sampler(SamplerConfig.greedyDecoding())
                .stopString("ac").build();
        SequenceRun run = run(tok, request);

        SequenceRun.Step first = run.offer(run.next(row(0)));
        assertFalse(first.stopped());
        assertEquals("", first.text());          // "a" is a proper prefix of "ac" and is held back
        SequenceRun.Step step = run.offer(run.next(row(2)));

        assertTrue(step.stopped());
        assertEquals(StopReason.STOP_STRING, step.reason());
        assertEquals("", step.text());
        GenerationResult r = run.result();
        assertArrayEquals(new int[] {0, 2}, r.tokens());       // the token is kept, llama.cpp keeps it too
        assertEquals("", r.text());
        assertEquals("ac", r.stopString());
        assertEquals(StopReason.STOP_STRING, r.stopReason());
    }

    @Test
    public void tokenBudgetStopsAtTheLastAllowedToken() {
        GGUFTokenizer tok = tokenizer(false);
        GenerationRequest request = request(2, false);
        SequenceRun run = run(tok, request);

        assertFalse(run.offer(run.next(row(0))).stopped());
        SequenceRun.Step second = run.offer(run.next(row(1)));

        assertTrue(second.stopped());
        assertEquals(StopReason.MAX_TOKENS, second.reason());
        GenerationResult r = run.result();
        assertArrayEquals(new int[] {0, 1}, r.tokens());
        assertEquals("ab", r.text());
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
    }

    /** The cancellation poll moves into the run; the stop comes from {@link SequenceRun#stop}, not a token. */
    @Test
    public void cancellationStopsWithoutAToken() {
        GGUFTokenizer tok = tokenizer(false);
        AtomicBoolean cancelled = new AtomicBoolean();
        GenerationRequest request = GenerationRequest.builder().sampler(SamplerConfig.greedyDecoding())
                .cancellation(cancelled::get).build();
        SequenceRun run = run(tok, request);
        run.prefilled(1, 10L);

        run.offer(run.next(row(0)));
        assertFalse(run.done());
        cancelled.set(true);
        assertTrue(run.cancelled());
        SequenceRun.Step step = run.stop(StopReason.CANCELLED);

        assertTrue(step.stopped());
        assertEquals(StopReason.CANCELLED, step.reason());
        assertTrue(run.done());
        GenerationResult r = run.result();
        assertArrayEquals(new int[] {0}, r.tokens());
        assertEquals("a", r.text());
        assertEquals(StopReason.CANCELLED, r.stopReason());
    }

    /** A context the memory cannot make room for stops the run with the driver's reason, also without a token. */
    @Test
    public void theDriverStopsWithAContextFullRun() {
        GGUFTokenizer tok = tokenizer(false);
        GenerationRequest request = request(8, false);
        SequenceRun run = run(tok, request);

        run.offer(run.next(row(1)));
        SequenceRun.Step step = run.stop(StopReason.CONTEXT_FULL);

        assertTrue(step.stopped());
        assertEquals(StopReason.CONTEXT_FULL, step.reason());
        assertTrue(run.done());
        GenerationResult r = run.result();
        assertArrayEquals(new int[] {1}, r.tokens());
        assertEquals(StopReason.CONTEXT_FULL, r.stopReason());
    }

    /** A token the drive already has (the {@code decodeGreedyOnDevice} fast path) still enters the counters. */
    @Test
    public void deviceTokensAreAcceptedIntoTheHistoriesAndCounters() {
        GGUFTokenizer tok = tokenizer(false);
        GenerationRequest request = request(2, false);
        SequenceRun run = run(tok, request);

        run.accept(0);
        run.offer(0);
        run.accept(1);
        run.offer(1);

        GenerationResult r = run.result();
        assertArrayEquals(new int[] {0, 1}, r.tokens());
        assertEquals("ab", r.text());
        assertEquals(2, r.timings().generatedTokens());
    }

    /** The sink receives every channel; the step's delta carries exactly the text its token routed. */
    @Test
    public void theRoutingStageSplitsTagsTextIntoChannelsAndCarriesTheStepDelta() {
        GGUFTokenizer tok = tokenizer(false);
        GenerationRequest request = GenerationRequest.builder().sampler(SamplerConfig.greedyDecoding())
                .maxNewTokens(5).build();
        List<Piece> sink = new ArrayList<>();
        SequenceRun run = overCollecting(tok, request, thinkFilter(), sink);
        // the tags are given as token ids 3 and 4, so the session renders those ids as their tag text

        assertEquals("f", run.offer(run.next(row(5))).text()); // before the opening tag: prelude content
        run.accept(3);
        assertEquals("", run.offer(3).text());                 // the opening tag opens the block, routes nothing
        assertEquals("g", run.offer(run.next(row(6))).text()); // inside the block: reasoning
        run.accept(4);
        assertEquals("", run.offer(4).text());                 // the closing tag closes the block, routes nothing
        assertEquals("f", run.offer(run.next(row(5))).text()); // after the block: content

        // the fifth offered token reaches the budget and stops the run, tag tokens included
        assertTrue(run.done());
        GenerationResult r = run.result();
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        assertArrayEquals(new String[] {"f", "f"}, texts(sink, Channel.CONTENT).toArray(new String[0]));
        assertArrayEquals(new String[] {"g"}, texts(sink, Channel.REASONING).toArray(new String[0]));
        assertEquals(5, r.timings().generatedTokens());
        assertEquals("only the token inside the block counts as reasoning, not its tags", 1, r.reasoningTokens());
    }

    /**
     * Tags spelled over several tokens: the tokens of the closing tag do not count as reasoning, though their text
     * is held back while still in the block, and a held possible tag that turns out to be text counts.
     */
    @Test
    public void reasoningCountsExcludeAClosingTagSpreadOverTokens() {
        GGUFTokenizer tok = tokenizer(false);
        GenerationRequest request = GenerationRequest.builder().sampler(SamplerConfig.greedyDecoding())
                .maxNewTokens(9).build();
        ReasoningFilter filter = ReasoningFilter.thinkTags(Collections.singletonList("hh"),
                Collections.singletonList("cd"), Collections.<String>emptyList());
        List<Piece> sink = new ArrayList<>();
        SequenceRun run = overCollecting(tok, request, filter, sink);
        // h h opens, g c g is reasoning (c a false start of the tag), c d closes, f f is content
        for (int id : new int[] {7, 7, 6, 2, 6, 2, 3, 5, 5}) {
            run.offer(run.next(row(id)));
        }
        GenerationResult r = run.result();
        assertEquals("gcg", String.join("", texts(sink, Channel.REASONING)));
        assertEquals("ff", String.join("", texts(sink, Channel.CONTENT)));
        assertEquals("the three tokens inside the block count, not the tags' four", 3, r.reasoningTokens());
    }

    /** Constraints of the run's own surface: stop reasons are single, validated and precede the result. */
    @Test
    public void theStopIsSingleValidatedAndPrecedesTheResult() {
        GGUFTokenizer tok = tokenizer(false);
        GenerationRequest request = request(4, false);
        SequenceRun run = run(tok, request);

        assertThrows(IllegalStateException.class, run::result);   // result before a stop
        run.stop(StopReason.CANCELLED);
        assertThrows(IllegalStateException.class, () -> run.stop(StopReason.CONTEXT_FULL));   // a second stop
        assertThrows(IllegalArgumentException.class, () -> run.stop(StopReason.EOS));         // token-level
        assertThrows(IllegalArgumentException.class, () -> run.stop(StopReason.STOP_STRING));
        assertThrows(IllegalArgumentException.class, () -> run.stop(StopReason.MAX_TOKENS));
        assertThrows(IllegalArgumentException.class, () -> run.stop(null));
        run.result();                                            // the withheld tail releases
        assertThrows(IllegalStateException.class, run::result);   // result twice

        SequenceRun fresh = run(tok, request);
        fresh.offer(fresh.next(row(2)));
        assertThrows(IllegalArgumentException.class, () -> fresh.stop(StopReason.MAX_TOKENS));
        assertFalse(fresh.done());
    }
}

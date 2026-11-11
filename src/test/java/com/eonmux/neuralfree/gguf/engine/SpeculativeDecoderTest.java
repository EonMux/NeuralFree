package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Traces of {@link SpeculativeDecoder} on an 8-token vocabulary {@code a .. h} with a scripted causal target whose
 * argmax after token {@code t} is {@code (t + 1) mod 8}, and a scripted drafter. With an argmax sampler every
 * trace must emit exactly the plain greedy continuation {@code 1, 2, 3, ..} of the prompt {@code [0]}; the traces
 * check which forward calls, rollbacks, restores and drafter reports produce it.
 */
public class SpeculativeDecoderTest {

    private static final int VOCAB = 8;
    private static final int N_CTX = 32;
    private static final int UBATCH = 8;

    private static GGUFTokenizer tokenizer() {
        List<Object> tokens = new ArrayList<>(Arrays.<Object>asList("a", "b", "c", "d", "e", "f", "g", "h"));
        List<Object> types = new ArrayList<>(Arrays.<Object>asList(1, 1, 1, 1, 1, 1, 1, 1));
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, "gpt2");
        md.put(TokenizerVocab.KEY_PRE, "gpt-2");
        md.put(TokenizerVocab.KEY_TOKENS, tokens);
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, types);
        md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>());
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(md));
    }

    private static GenerationRequest greedy(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static SpeculativeDecoder decoder(FakeTarget target, FakeDrafter drafter, int draftMax, int draftMin) {
        return SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMax(draftMax).draftMin(draftMin).build();
    }

    /**
     * Round 1 at P = 1 drafts {@code [2, 3, 7, 7]}: the target's rows sample 2 (= d0), 3 (= d1), then 4 (!= 7), so
     * two drafted tokens are confirmed, 4 is the target's own token, and the last two of the five verified
     * positions are rolled back (memory {@code [0, 1, 2, 3]}). Round 2 at P = 4 asks the full budget (the draft
     * length does not shrink with the token budget, as llama.cpp's loop drafts {@code n_draft} regardless): the
     * drafter proposes {@code [5]}, it is confirmed and 6 ends the budget.
     */
    @Test
    public void verifyKeepsTheLongestMatchingPrefixAndRollsBackTheRest() {        FakeTarget target = new FakeTarget(false);
        FakeDrafter drafter = new FakeDrafter((last, n) -> last == 1 ? new int[]{2, 3, 7, 7} : new int[]{5});
        List<SpeculativeDecoder.Stats> stats = new ArrayList<>();
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMax(4)
                .statsSink(stats::add).build();

        GenerationResult r = dec.generate(new int[]{0}, greedy(6), (TokenCallback) null);

        assertArrayEquals(new int[]{1, 2, 3, 4, 5, 6}, r.tokens());
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        assertEquals(2, target.verifyCalls.size());
        assertArrayEquals(new int[]{1, 2, 3, 7, 7}, target.verifyCalls.get(0));
        assertArrayEquals(new int[]{4, 5}, target.verifyCalls.get(1));
        assertEquals(Arrays.asList(1, 4), target.verifyPositions);
        assertEquals(Arrays.asList(2), target.rollbacks);
        assertEquals(6, target.memory.sequence().nPast());
        assertEquals(Arrays.asList(4, 4), drafter.requested);
        assertEquals(Arrays.asList(2, 1), drafter.accepted);
        assertEquals(1, drafter.resets);
        assertEquals(1, stats.size());
        assertEquals(2, stats.get(0).rounds());
        assertEquals(5, stats.get(0).drafted());
        assertEquals(3, stats.get(0).accepted());
    }

    /**
     * Prefix reuse across the decoder's generations (llama.cpp's speculative loop over a cached prompt): the
     * second call whose prompt extends the stored sequence keeps the common prefix behind a rollback, prefills
     * only the tail (the prefill lands at the reused position), and emits the same continuation the full prefill
     * produces. With reuse off the whole prompt prefills from position 0.
     */
    @Test
    public void aSecondGenerationReusesTheStoredTargetPrefix() {
        FakeTarget target = new FakeTarget(false);
        FakeDrafter drafter = new FakeDrafter((last, n) -> last == 1 ? new int[]{2, 3, 7, 7} : new int[]{5});
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMax(4)
                .prefixReuse(true).build();

        GenerationResult first = dec.generate(new int[]{0}, greedy(6), (TokenCallback) null);
        GenerationResult second = dec.generate(new int[]{0, 1, 2, 3}, greedy(3), (TokenCallback) null);

        assertArrayEquals(new int[]{1, 2, 3, 4, 5, 6}, first.tokens());
        assertArrayEquals("the reused run emits the stored prompt's own continuation", new int[]{4, 5, 6},
                second.tokens());
        assertEquals(2, target.prefills.size());
        assertArrayEquals("the second prefill evaluated only the last prompt token", new int[]{3},
                target.prefills.get(1));
        assertEquals(Integer.valueOf(3), target.prefillPositions.get(1));
        assertEquals("generation 1 rolls back its rejected drafts, generation 2 drops the suffix"
                        + " past the reused prefix", Arrays.asList(2, 3), target.rollbacks);
        assertEquals(1, second.timings().promptTokens());
    }

    @Test
    public void theReuseSwitchPrefillsTheWholePrompt() {
        FakeTarget target = new FakeTarget(false);
        FakeDrafter drafter = new FakeDrafter((last, n) -> last == 1 ? new int[]{2, 3, 7, 7} : new int[]{5});
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMax(4)
                .prefixReuse(false).build();

        dec.generate(new int[]{0}, greedy(6), (TokenCallback) null);
        GenerationResult second = dec.generate(new int[]{0, 1, 2, 3}, greedy(3), (TokenCallback) null);

        assertArrayEquals(new int[]{4, 5, 6}, second.tokens());
        assertEquals(2, target.prefills.size());
        assertArrayEquals(new int[]{0, 1, 2, 3}, target.prefills.get(1));
        assertEquals(Integer.valueOf(0), target.prefillPositions.get(1));
    }

    /**
     * The logprobs capture over speculation (MATCH verify): every emitted token — the prompt's first sample, the
     * confirmed drafts and the target's own tokens — reports its post-chain distribution as log probabilities,
     * the sampled token first and the alternatives in descending probability, one entry per generated token.
     */
    @Test
    public void logprobsReportEveryEmittedTokensPostChainDistribution() {
        FakeTarget target = new FakeTarget(false);
        FakeDrafter drafter = new FakeDrafter((last, n) -> last == 1 ? new int[]{2, 3, 7, 7} : new int[]{5});
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMax(4).build();

        GenerationRequest request = GenerationRequest.builder().maxNewTokens(6)
                .sampler(SamplerConfig.greedyDecoding()).logprobs(3).build();
        GenerationResult r = dec.generate(new int[]{0}, request, (TokenCallback) null);

        assertArrayEquals(new int[]{1, 2, 3, 4, 5, 6}, r.tokens());
        assertEquals("one capture per emitted token, including the verified rounds", 6, r.logprobs().size());
        for (int step = 0; step < r.logprobs().size(); step++) {
            TokenScore[] scores = r.logprobs().get(step);
            assertEquals(3, scores.length);
            assertEquals("the sampled token leads", r.tokens()[step], scores[0].tokenId());
            assertTrue("the sampled token's logprob is at most 0", scores[0].logprob() <= 0f);
            for (int k = 1; k < scores.length; k++) {
                assertTrue("alternatives follow in descending probability",
                        scores[k].logprob() <= scores[k - 1].logprob());
            }
        }
    }

    /**
     * Device selection (llama.cpp backend sampling): a greedy MATCH generation verifies through
     * {@code forwardRowsGreedy} and decodes its draftless rounds through {@code decodeGreedyOnDevice}, reading back
     * ids only, and emits exactly the host path's tokens with the same verify batches and rollbacks.
     */
    @Test
    public void greedyMatchRoundsSelectOnTheDevice() {
        FakeDrafter script = new FakeDrafter((last, n) -> last == 1 ? new int[]{2, 3, 7, 7}
                : last == 4 ? new int[0] : new int[]{6});
        FakeTarget device = new FakeTarget(false);
        GenerationResult onDevice = SpeculativeDecoder.builder(device, script, tokenizer()).draftMax(4).build()
                .generate(new int[]{0}, greedy(7), (TokenCallback) null);
        FakeDrafter hostScript = new FakeDrafter((last, n) -> last == 1 ? new int[]{2, 3, 7, 7}
                : last == 4 ? new int[0] : new int[]{6});
        FakeTarget host = new FakeTarget(false);
        GenerationResult onHost = SpeculativeDecoder.builder(host, hostScript, tokenizer()).draftMax(4)
                .verifyOnDevice(false).build()
                .generate(new int[]{0}, greedy(7), (TokenCallback) null);

        assertArrayEquals(new int[]{1, 2, 3, 4, 5, 6, 7}, onDevice.tokens());
        assertArrayEquals(onHost.tokens(), onDevice.tokens());
        assertEquals("both verify rounds read greedy ids", 2, device.greedyVerifies);
        assertEquals("the draftless round decoded on the device", 1, device.greedyDecodes);
        assertEquals(0, host.greedyVerifies);
        assertEquals(0, host.greedyDecodes);
        assertEquals(host.verifyCalls.size(), device.verifyCalls.size());
        for (int i = 0; i < host.verifyCalls.size(); i++) {
            assertArrayEquals(host.verifyCalls.get(i), device.verifyCalls.get(i));
        }
        assertEquals(host.rollbacks, device.rollbacks);
        assertEquals(host.decodes, device.decodes);
        assertEquals(hostScript.accepted, script.accepted);
    }

    /** A logprobs capture needs the rows' distributions, so a greedy chain with logprobs verifies on the host. */
    @Test
    public void aLogprobsRequestVerifiesOnTheHost() {
        FakeTarget target = new FakeTarget(false);
        FakeDrafter drafter = new FakeDrafter((last, n) -> last == 1 ? new int[]{2, 3, 7, 7} : new int[]{5});
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(6)
                .sampler(SamplerConfig.greedyDecoding()).logprobs(2).build();

        GenerationResult r = SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMax(4).build()
                .generate(new int[]{0}, request, (TokenCallback) null);

        assertArrayEquals(new int[]{1, 2, 3, 4, 5, 6}, r.tokens());
        assertEquals(6, r.logprobs().size());
        assertEquals(0, target.greedyVerifies);
        assertEquals(0, target.greedyDecodes);
    }

    /** A stochastic chain samples every row on the host; only an argmax chain selects on the device. */
    @Test
    public void aStochasticChainVerifiesOnTheHost() {
        FakeTarget target = new FakeTarget(false);
        FakeDrafter drafter = new FakeDrafter((last, n) -> new int[]{(last + 1) % VOCAB});
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(6)
                .sampler(SamplerConfig.builder().temperature(0.8f).seed(7).build()).build();

        GenerationResult r = SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMax(2).build()
                .generate(new int[]{0}, request, (TokenCallback) null);

        assertEquals(6, r.tokens().length);
        assertTrue(target.verifyCalls.size() > 0);
        assertEquals(0, target.greedyVerifies);
        assertEquals(0, target.greedyDecodes);
    }

    /**
     * A chain whose top-k cone needs its frequency penalty applied first verifies over prefiltered device readouts
     * when the backend applies the edits on the device: verify row {@code i} is edited over the history plus the
     * draft's first {@code i} tokens, draftless rounds decode through the prefiltered single-row readout, and the
     * tokens equal the host path's under the same seed. The drafts are the reference run's own continuation, so
     * every verify row is accepted and the penalty counts of the draft tokens decide the later rows' draws.
     */
    @Test
    public void aPenalizedConeVerifiesOverPrefilteredDeviceReadouts() {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(16)
                .sampler(SamplerConfig.builder().temperature(0.4f).topK(3).frequencyPenalty(1.5f).penaltyLastN(8)
                        .seed(11).build()).build();
        FakeTarget reference = new FakeTarget(false);
        reference.twoPeaks = true;
        int[] ref = decoder(reference, new FakeDrafter((last, n) -> new int[0]), 3, 1)
                .generate(new int[]{0}, request, (TokenCallback) null).tokens();

        FakeTarget device = new FakeTarget(false);
        device.twoPeaks = true;
        device.prefilterDevice = true;
        GenerationResult onDevice = perfectDrafts(device, ref, request);
        FakeTarget host = new FakeTarget(false);
        host.twoPeaks = true;
        GenerationResult onHost = perfectDrafts(host, ref, request);

        assertArrayEquals(ref, onHost.tokens());
        assertArrayEquals(ref, onDevice.tokens());
        assertTrue("the verify rounds read prefiltered candidates", device.prefilteredVerifies > 0);
        assertTrue("the draftless rounds decoded through the prefiltered readout", device.prefilteredDecodes > 0);
        assertEquals(device.prefilteredVerifies, device.verifyCalls.size());
        assertEquals(host.verifyCalls.size(), device.verifyCalls.size());
        for (int i = 0; i < host.verifyCalls.size(); i++) {
            assertArrayEquals(host.verifyCalls.get(i), device.verifyCalls.get(i));
        }
        assertEquals(0, host.prefilteredVerifies);
    }

    /**
     * A generation over {@code target} whose drafts continue {@code ref} from the tokens emitted so far, three at a
     * time, with every third round left without a proposal.
     */
    private static GenerationResult perfectDrafts(FakeTarget target, int[] ref, GenerationRequest request) {
        int[] emitted = new int[1];
        int[] rounds = new int[1];
        FakeDrafter drafter = new FakeDrafter((last, n) -> {
            int from = emitted[0];
            if (rounds[0]++ % 3 == 2 || from >= ref.length) {
                return new int[0];
            }
            return Arrays.copyOfRange(ref, from, Math.min(ref.length, from + Math.min(n, 3)));
        });
        return decoder(target, drafter, 3, 1).generate(new int[]{0}, request, (id, text) -> emitted[0]++);
    }

    /**
     * The same round on recurrent memory: the state is snapshot before the verify call, restored after it, and the
     * kept tokens {@code [1, 2, 3]} are prefilled again from P = 1; nothing is rolled back.
     */
    @Test
    public void recurrentMemoryRestoresTheSnapshotAndReplaysTheKeptTokens() {
        FakeTarget target = new FakeTarget(true);
        FakeDrafter drafter = new FakeDrafter((last, n) -> {
            if (last != 1) {
                return new int[0];
            }
            int[] full = {2, 3, 7, 7};
            return n < full.length ? Arrays.copyOf(full, n) : full;
        });

        GenerationResult r = decoder(target, drafter, 4, 0).generate(new int[]{0}, greedy(5), (TokenCallback) null);

        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, r.tokens());
        assertEquals(1, target.snapshots);
        assertEquals(1, target.restores);
        assertTrue(target.rollbacks.isEmpty());
        assertEquals(2, target.prefills.size());
        assertArrayEquals(new int[]{0}, target.prefills.get(0));
        assertArrayEquals(new int[]{1, 2, 3}, target.prefills.get(1));
        assertEquals(1, target.prefillPositions.get(1).intValue());
        assertEquals(Arrays.asList(4), target.decodes);
        assertEquals(5, target.memory.sequence().nPast());
        assertEquals(Arrays.asList(2, 0), drafter.accepted);
    }

    /** All drafted tokens confirmed: nothing is dropped and the row after the last draft gives the next token.
     * The budget is wide enough that no round is cut mid-verify, so no position is ever rolled back. */
    @Test
    public void fullyAcceptedDraftNeedsNoRollback() {
        FakeTarget target = new FakeTarget(false);
        FakeDrafter drafter = new FakeDrafter((last, n) -> {
            int[] d = new int[n];
            for (int i = 0; i < n; i++) {
                d[i] = (last + 1 + i) % VOCAB;
            }
            return d;
        });

        GenerationResult r = decoder(target, drafter, 3, 0).generate(new int[]{0}, greedy(9), (TokenCallback) null);

        assertArrayEquals(new int[]{1, 2, 3, 4, 5, 6, 7, 0, 1}, r.tokens());
        assertTrue(target.rollbacks.isEmpty());
        assertArrayEquals(new int[]{1, 2, 3, 4}, target.verifyCalls.get(0));
        assertEquals(3, drafter.accepted.get(0).intValue());
    }

    /**
     * {@code repetition_detection} inside a verify round: the target's period-8 output repeats twice at token 16,
     * mid-way through the round verifying tokens 14 to 17, so generation ends there with
     * {@link StopReason#REPETITION} and the confirmed tokens after it are dropped.
     */
    @Test
    public void aRepetitionStopEndsGenerationInsideAVerifyRound() {
        FakeTarget target = new FakeTarget(false);
        FakeDrafter drafter = new FakeDrafter((last, n) -> {
            int[] d = new int[n];
            for (int i = 0; i < n; i++) {
                d[i] = (last + 1 + i) % VOCAB;
            }
            return d;
        });
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(28)
                .sampler(SamplerConfig.greedyDecoding()).repetitionDetection(RepetitionDetection.of(8, 2, 2)).build();

        GenerationResult r = decoder(target, drafter, 3, 0).generate(new int[]{0}, request, (TokenCallback) null);

        assertEquals(StopReason.REPETITION, r.stopReason());
        assertArrayEquals(new int[]{1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0}, r.tokens());
    }

    /** An empty proposal, or one shorter than draft-min, decodes the token alone and reports zero accepted. */
    @Test
    public void shortProposalsFallBackToSingleTokenDecoding() {
        FakeTarget target = new FakeTarget(false);
        FakeDrafter drafter = new FakeDrafter((last, n) -> last % 2 == 0 ? new int[0] : new int[]{(last + 1) % VOCAB});

        GenerationResult r = decoder(target, drafter, 4, 2).generate(new int[]{0}, greedy(4), (TokenCallback) null);

        assertArrayEquals(new int[]{1, 2, 3, 4}, r.tokens());
        assertTrue(target.verifyCalls.isEmpty());
        assertEquals(Arrays.asList(1, 2, 3), target.decodes);
        assertEquals(Arrays.asList(0, 0, 0), drafter.accepted);
    }

    /** A proposal longer than requested is a drafter fault; the target memory and the drafter are reset. */
    @Test
    public void oversizedProposalFailsAndResets() {
        FakeTarget target = new FakeTarget(false);
        FakeDrafter drafter = new FakeDrafter((last, n) -> new int[n + 1]);
        SpeculativeDecoder dec = decoder(target, drafter, 2, 0);

        assertThrows(IllegalStateException.class, () -> dec.generate(new int[]{0}, greedy(4), (TokenCallback) null));
        assertEquals(0, target.memory.sequence().nPast());
        assertEquals(2, drafter.resets);
    }

    /**
     * Rings that keep their windows only when at most two positions are dropped: each round asks for two tokens
     * although draft-max is four.
     */
    @Test
    public void draftLengthIsTheLongestTheRingsCanDrop() {
        FakeTarget target = new FakeTarget(new FakeMemory(false, false, 2));
        FakeDrafter drafter = new FakeDrafter((last, n) -> {
            int[] d = new int[n];
            for (int i = 0; i < n; i++) {
                d[i] = (last + 1 + i) % VOCAB;
            }
            return d;
        });

        GenerationResult r = decoder(target, drafter, 4, 0).generate(new int[]{0}, greedy(7), (TokenCallback) null);

        assertArrayEquals(new int[]{1, 2, 3, 4, 5, 6, 7}, r.tokens());
        assertEquals(Arrays.asList(2, 2), drafter.requested);
        assertArrayEquals(new int[]{1, 2, 3}, target.verifyCalls.get(0));
        assertArrayEquals(new int[]{4, 5, 6}, target.verifyCalls.get(1));
    }

    /** Recurrent state without snapshot shadows cannot drop drafted positions, so every round decodes one token. */
    @Test
    public void recurrentMemoryWithoutSnapshotsDoesNotDraft() {
        FakeTarget target = new FakeTarget(new FakeMemory(true, false, Integer.MAX_VALUE));
        FakeDrafter drafter = new FakeDrafter((last, n) -> new int[]{(last + 1) % VOCAB});

        GenerationResult r = decoder(target, drafter, 4, 0).generate(new int[]{0}, greedy(4), (TokenCallback) null);

        assertArrayEquals(new int[]{1, 2, 3, 4}, r.tokens());
        assertTrue(drafter.requested.isEmpty());
        assertTrue(target.verifyCalls.isEmpty());
        assertEquals(0, target.snapshots);
        assertEquals(Arrays.asList(1, 2, 3), target.decodes);
    }

    /**
     * DSV4 compressed memory drafts at most its compressor rollback snapshots ({@code n_rs_seq}) per round, as
     * llama.cpp bounds the draft by {@code llama_n_rs_seq} (tools/server/server-context.cpp:3805-3830), and none
     * without snapshots.
     */
    @Test
    public void compressedMemoryDraftsWithinItsRollbackSnapshots() {
        FakeTarget target = new FakeTarget(FakeMemory.dsv4(2));
        FakeDrafter drafter = new FakeDrafter((last, n) -> {
            int[] d = new int[n];
            for (int i = 0; i < n; i++) {
                d[i] = (last + 1 + i) % VOCAB;
            }
            return d;
        });

        GenerationResult r = decoder(target, drafter, 4, 0).generate(new int[]{0}, greedy(7), (TokenCallback) null);

        assertArrayEquals(new int[]{1, 2, 3, 4, 5, 6, 7}, r.tokens());
        assertEquals(Arrays.asList(2, 2), drafter.requested);

        FakeTarget plain = new FakeTarget(FakeMemory.dsv4(0));
        FakeDrafter idle = new FakeDrafter((last, n) -> new int[]{(last + 1) % VOCAB});
        GenerationResult single = decoder(plain, idle, 4, 0).generate(new int[]{0}, greedy(3), (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 3}, single.tokens());
        assertTrue(idle.requested.isEmpty());
        assertTrue(plain.verifyCalls.isEmpty());
    }

    @Test
    public void builderRejectsInvalidDraftLengths() {
        FakeTarget target = new FakeTarget(false);
        FakeDrafter drafter = new FakeDrafter((last, n) -> new int[0]);
        assertThrows(IllegalArgumentException.class,
                () -> SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMax(0));
        assertThrows(IllegalArgumentException.class,
                () -> SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMin(-1));
        assertThrows(IllegalArgumentException.class,
                () -> SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMax(2).draftMin(3).build());
    }

    /** Peaked logits for the target's next token after {@code token}. */
    private static float[] next(int token) {
        float[] l = new float[VOCAB];
        l[(token + 1) % VOCAB] = 10f;
        return l;
    }

    /**
     * Memory of one KV-less layer with a real sequence record and a host snapshot of its length. {@code snapshots}
     * says whether recurrent state has snapshot shadows; {@code maxWindowDrop} is the most positions a speculative
     * call may drop while every sliding window stays whole.
     */
    private static final class FakeMemory implements ModelMemory {
        private final MemoryPlan plan;
        private final SequenceState sequence = new SequenceState(N_CTX);
        private final boolean recurrent;
        private final boolean snapshots;
        private final int maxWindowDrop;
        private int saved = -1;

        FakeMemory(boolean recurrent, boolean snapshots, int maxWindowDrop) {
            this(recurrent, snapshots, maxWindowDrop, MemoryPlan.builder(MemoryKind.KV, N_CTX, UBATCH)
                    .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.NONE).build())
                    .build());
        }

        FakeMemory(boolean recurrent, boolean snapshots, int maxWindowDrop, MemoryPlan plan) {
            this.recurrent = recurrent;
            this.snapshots = snapshots;
            this.maxWindowDrop = maxWindowDrop;
            this.plan = plan;
        }

        /** One DSV4 HCA layer whose compressor keeps {@code rollbackSnapshots} snapshot planes. */
        static FakeMemory dsv4(int rollbackSnapshots) {
            KvWindow window = KvWindow.of(SwaType.STANDARD, 8, true);
            KvLayout raw = KvLayout.of(N_CTX, true, 8, 8, KvDType.F32, KvDType.F32, window);
            MemoryPlan plan = MemoryPlan.builder(MemoryKind.DSV4, N_CTX, UBATCH)
                    .group(MemoryPlan.KvGroup.of(0, N_CTX, true, window))
                    .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.DSV4).kv(raw, 0)
                            .compressed(LayerMemoryPlan.CompressedKvPlan.of(LayerMemoryPlan.CompressedKvPlan.Kind.HCA,
                                    N_CTX, 8, 0, KvDType.F32, rollbackSnapshots))
                            .build())
                    .build();
            return new FakeMemory(false, false, Integer.MAX_VALUE, plan);
        }

        @Override public MemoryPlan plan() { return plan; }
        @Override public SequenceState sequence() { return sequence; }
        @Override public void reset() { sequence.clear(); saved = -1; }
        @Override public boolean canRollback(int nTokens) { return !recurrent && nTokens <= sequence.nPast(); }

        @Override
        public void rollback(int nTokens) {
            if (!canRollback(nTokens)) {
                throw new IllegalStateException("cannot drop " + nTokens);
            }
            sequence.truncate(nTokens);
        }

        @Override public boolean hasRecurrentState() { return recurrent; }
        @Override public boolean canSnapshot() { return recurrent && snapshots; }
        @Override public boolean ringsKeepWindow(int nWrite, int nDrop) { return nDrop <= maxWindowDrop; }

        @Override
        public void snapshot() {
            if (!canSnapshot()) {
                throw new UnsupportedOperationException("no recurrent state");
            }
            saved = sequence.nPast();
        }

        @Override
        public void restore() {
            if (saved < 0) {
                throw new IllegalStateException("no snapshot");
            }
            sequence.truncate(sequence.nPast() - saved);
        }

        @Override public void close() { }
    }

    /** Causal target scripted by {@link #next}; records every call. */
    private static final class FakeTarget implements ForwardBackend {
        final FakeMemory memory;
        final List<int[]> prefills = new ArrayList<>();
        final List<Integer> prefillPositions = new ArrayList<>();
        final List<Integer> decodes = new ArrayList<>();
        final List<int[]> verifyCalls = new ArrayList<>();
        final List<Integer> verifyPositions = new ArrayList<>();
        final List<Integer> rollbacks = new ArrayList<>();
        int snapshots;
        int restores;
        /** Verify calls and plain decodes that selected on the "device" (read back ids, not logits). */
        int greedyVerifies;
        int greedyDecodes;
        /** Rows of two close peaks, so a penalty on either can hand the draw to the other. */
        boolean twoPeaks;
        /** Whether the "device" applies logit prefilters before its top-k readout. */
        boolean prefilterDevice;
        /** Verify calls and plain decodes that read a prefiltered top-k. */
        int prefilteredVerifies;
        int prefilteredDecodes;

        FakeTarget(boolean recurrent) {
            this(new FakeMemory(recurrent, true, Integer.MAX_VALUE));
        }

        FakeTarget(FakeMemory memory) {
            this.memory = memory;
        }

        private void store(int[] tokens, int pos0) {
            if (pos0 != memory.sequence.nPast()) {
                throw new IllegalStateException("call at " + pos0 + " after " + memory.sequence.nPast() + " positions");
            }
            memory.sequence.append(tokens, 0, tokens.length);
        }

        @Override
        public float[] prefill(int[] tokens, int pos0) {
            store(tokens, pos0);
            prefills.add(tokens.clone());
            prefillPositions.add(pos0);
            return row(tokens[tokens.length - 1]);
        }

        @Override
        public float[] decode(int token, int pos) {
            store(new int[]{token}, pos);
            decodes.add(token);
            return row(token);
        }

        @Override
        public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
            store(tokens, pos0);
            verifyCalls.add(tokens.clone());
            verifyPositions.add(pos0);
            int[] rows = outputRows;
            if (rows == null) {
                rows = new int[tokens.length];
                for (int i = 0; i < rows.length; i++) {
                    rows[i] = i;
                }
            }
            float[][] out = new float[rows.length][];
            for (int i = 0; i < rows.length; i++) {
                out[i] = row(tokens[rows[i]]);
            }
            return out;
        }

        private float[] row(int token) {
            if (!twoPeaks) {
                return next(token);
            }
            float[] l = new float[VOCAB];
            // the runner-up is the token before this one, which a verify row's pending draft tokens hold
            l[(token + 1) % VOCAB] = 6f;
            l[(token + VOCAB - 1) % VOCAB] = 5.5f;
            l[(token + 4) % VOCAB] = 2f;
            return l;
        }

        @Override
        public boolean topKOnDevice() {
            return prefilterDevice;
        }

        @Override
        public boolean topKPrefilterOnDevice() {
            return prefilterDevice;
        }

        @Override
        public RowTopK forwardRowsTopK(int[] tokens, int pos0, int k, int[] outputRows, LogitPrefilter prefilter) {
            if (prefilter == null || prefilter.isEmpty()) {
                return ForwardBackend.super.forwardRowsTopK(tokens, pos0, k, outputRows, prefilter);
            }
            prefilteredVerifies++;
            float[][] rows = forwardRows(tokens, pos0, outputRows);
            assertEquals("one prefilter row per verify row", rows.length, prefilter.rows());
            for (int i = 0; i < rows.length; i++) {
                rows[i] = SamplerPrefilterTest.applyPrefilter(rows[i], prefilter.row(i));
            }
            return RowTopK.of(rows, k);
        }

        @Override
        public RowTopK decodeTopKOnDevice(int token, int pos, int k, LogitPrefilter prefilter) {
            if (prefilter == null || prefilter.isEmpty()) {
                return ForwardBackend.super.decodeTopKOnDevice(token, pos, k, prefilter);
            }
            prefilteredDecodes++;
            float[] edited = SamplerPrefilterTest.applyPrefilter(decode(token, pos), prefilter.row(0));
            return RowTopK.of(new float[][]{edited}, k);
        }

        @Override
        public int[] forwardRowsGreedy(int[] tokens, int pos0, int[] outputRows) {
            greedyVerifies++;
            return ForwardBackend.super.forwardRowsGreedy(tokens, pos0, outputRows);
        }

        @Override
        public int decodeGreedyOnDevice(int token, int pos) {
            greedyDecodes++;
            return ForwardBackend.super.decodeGreedyOnDevice(token, pos);
        }

        @Override
        public float[][] forwardAll(int[] tokens, int[] outputRows) {
            throw new UnsupportedOperationException("not used by speculative decoding");
        }

        @Override
        public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
            throw new UnsupportedOperationException("not used by speculative decoding");
        }

        @Override
        public void reset() {
            memory.reset();
        }

        @Override
        public boolean canRollback(int nTokens) {
            return memory.canRollback(nTokens);
        }

        @Override
        public void rollback(int nTokens) {
            memory.rollback(nTokens);
            rollbacks.add(nTokens);
        }

        @Override
        public void shiftContext(int nKeep, int nDiscard) {
            throw new UnsupportedOperationException("not used by speculative decoding");
        }

        @Override
        public ModelMemory memory() {
            return new ModelMemory() {
                @Override public MemoryPlan plan() { return memory.plan(); }
                @Override public SequenceState sequence() { return memory.sequence(); }
                @Override public void reset() { memory.reset(); }
                @Override public boolean canRollback(int nTokens) { return memory.canRollback(nTokens); }
                @Override public void rollback(int nTokens) { memory.rollback(nTokens); }
                @Override public boolean hasRecurrentState() { return memory.hasRecurrentState(); }
                @Override public boolean canSnapshot() { return memory.canSnapshot(); }
                @Override public boolean ringsKeepWindow(int nWrite, int nDrop) {
                    return memory.ringsKeepWindow(nWrite, nDrop);
                }
                @Override public void snapshot() { snapshots++; memory.snapshot(); }
                @Override public void restore() { restores++; memory.restore(); }
                @Override public void close() { }
            };
        }

        @Override
        public void setHiddenTap(HiddenTap tap) {
        }

        @Override
        public void close() {
        }
    }

    /** Drafter answering from a script of {@code (lastToken, n)}; records requests, reports and resets. */
    private static final class FakeDrafter implements DraftModel {
        private final BiFunction<Integer, Integer, int[]> script;
        final List<Integer> requested = new ArrayList<>();
        final List<Integer> accepted = new ArrayList<>();
        int resets;

        FakeDrafter(BiFunction<Integer, Integer, int[]> script) {
            this.script = script;
        }

        @Override
        public void bindTarget(EngineModel target) {
        }

        @Override
        public int[] draft(int lastToken, int n) {
            requested.add(n);
            return script.apply(lastToken, n);
        }

        @Override
        public void accept(int nAccepted) {
            accepted.add(nAccepted);
        }

        @Override
        public void reset() {
            resets++;
        }
    }
}

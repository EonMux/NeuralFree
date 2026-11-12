package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.engine.sampling.TokenGate;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The speculative decoder selects each emitted row's token sample-first ({@link Sampler#sample}) as the plain loop
 * does: a greedy chain whose constraint head keeps the unmasked argmax never masks the whole row, the first token,
 * the draftless decode and every verified row alike; a logprobs capture reads the chain's candidates over the
 * masked row instead. A greedy JSON-schema generation emits the same tokens on either path, with every verification
 * mode.
 */
public class SpeculativeSelectionTest {

    private static final int VOCAB = 8;

    /** A per-token constraint refusing {@code banned}, counting its whole-row masks and its per-token answers. */
    private static final class CountingGate extends SamplerStage implements TokenGate {
        private final int banned;
        int masks;
        int answers;

        CountingGate(int banned) {
            this.banned = banned;
        }

        @Override
        public void apply(Sampler.Candidates candidates) {
            masks++;
            for (int i = 0; i < candidates.size(); i++) {
                if (candidates.id(i) == banned) {
                    candidates.setLogit(i, Float.NEGATIVE_INFINITY);
                }
            }
        }

        @Override
        public boolean keeps(int token) {
            answers++;
            return token != banned;
        }

        @Override
        public boolean constrains() {
            return true;
        }
    }

    private static final SamplerVocab BYTES = new SamplerVocab() {
        @Override public int vocabSize() { return VOCAB; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) { return new byte[]{(byte) ('a' + id)}; }
        @Override public String pieceText(int id) { return new String(piece(id), StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) { return new int[0]; }
    };

    private static Sampler sampler(CountingGate gate) {
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0], Map.of(), List.of(gate), BYTES);
        assertTrue(sampler.sampleFirstChain());
        return sampler;
    }

    private static float[] row(int top, int next) {
        float[] row = new float[VOCAB];
        row[top] = 5f;
        row[next] = 4f;
        return row;
    }

    private static SpeculativeDecoder.Scores scores(GenerationRequest request, List<TokenScore[]> out) {
        return new SpeculativeDecoder.Scores(LogprobCapture.of(request), out, (channel, tokenId, text) -> { });
    }

    @Test
    public void aRowWithoutALogprobsCaptureSelectsSampleFirst() {
        CountingGate gate = new CountingGate(6);
        List<TokenScore[]> out = new ArrayList<>();
        SpeculativeDecoder.Scores scores = scores(GenerationRequest.builder().maxNewTokens(4).build(), out);
        Sampler sampler = sampler(gate);
        assertEquals(3, scores.select(sampler, row(3, 5)));
        assertEquals("a kept argmax needs no whole-row mask", 0, gate.masks);
        assertEquals(1, gate.answers);
        sampler.accept(3);
        assertEquals("a refused argmax falls back to the mask", 5, scores.select(sampler, row(6, 5)));
        assertEquals(1, gate.masks);
        assertTrue("nothing is captured", out.isEmpty());
    }

    @Test
    public void aRowWithALogprobsCaptureReadsTheMaskedRow() {
        CountingGate gate = new CountingGate(6);
        List<TokenScore[]> out = new ArrayList<>();
        SpeculativeDecoder.Scores scores = scores(GenerationRequest.builder().maxNewTokens(4).logprobs(2).build(),
                out);
        assertEquals(3, scores.select(sampler(gate), row(3, 6)));
        assertEquals("the capture reads the chain's masked candidates", 1, gate.masks);
        assertEquals(1, out.size());
        assertEquals("the sampled token leads the capture", 3, out.get(0)[0].tokenId());
    }

    /**
     * A greedy JSON-schema generation over the synthetic llama emits the same tokens through the speculative
     * decoder with and without a logprobs capture (the whole-row mask on every row against sample-first), with a
     * drafter proposing text the grammar cuts, under every verification mode, and the plain loop's tokens.
     */
    @Test
    public void aGreedyJsonSchemaGenerationEmitsTheSameTokensOnBothSelectionPaths() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel one = new GGUFModel(model.toString(), config())) {
            ConstraintSpec schema = ConstraintSpec.jsonSchema(
                    "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},\"required\":[\"a\"]}",
                    null);
            GenerationRequest plainRequest = GenerationRequest.builder().maxNewTokens(16)
                    .sampler(SamplerConfig.greedyDecoding()).constraint(schema).build();
            GenerationRequest capturing = plainRequest.toBuilder().logprobs(1).build();
            int[] plain = GenerationLoop.forModel(one, config()).generate(prompt(), plainRequest, null).tokens();
            assertTrue("the plain loop generated tokens", plain.length > 3);
            int[] proposal = one.tokenizer().encode("{\"a\":\"no\"}", false, false);
            for (VerifyMode mode : new VerifyMode[]{VerifyMode.MATCH, VerifyMode.REJECTION, VerifyMode.BLOCK}) {
                int[] first = speculative(one, proposal, mode).generate(prompt(), plainRequest, null).tokens();
                int[] masked = speculative(one, proposal, mode).generate(prompt(), capturing, null).tokens();
                assertArrayEquals(mode + ": sample-first against the plain loop", plain, first);
                assertArrayEquals(mode + ": the whole-row mask against sample-first", first, masked);
            }
        } finally {
            Files.deleteIfExists(model);
        }
    }

    private static SpeculativeDecoder speculative(GGUFModel one, int[] proposal, VerifyMode mode) {
        DraftModel drafter = new DraftModel() {
            @Override public void bindTarget(EngineModel target) { }
            @Override public int[] draft(int lastToken, int n) {
                return Arrays.copyOf(proposal, Math.min(n, proposal.length));
            }
            @Override public void accept(int nAccepted) { }
        };
        return SpeculativeDecoder.builder(one.forwardBackend(), drafter, one.tokenizer()).draftMax(4)
                .verifyMode(mode).build();
    }

    private static EngineConfig config() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(96).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static int[] prompt() {
        return new int[]{11, 24, 37, 50, 63, 76};
    }
}

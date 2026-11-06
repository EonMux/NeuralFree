package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * A grammar-constrained sequence of the {@link BatchEngine} under the default chain at a temperature above 0 (the
 * plain decode path, {@code MATCH} verification included): the constraint's mask composes with the truncating
 * stages the way llama.cpp's grammar-first resample and vLLM's pre-sampling bitmask keep it, so the step always
 * has a candidate to sample and the text stays inside the grammar; the request's own masks ({@code
 * allowed_token_ids}, bad words) mask from the same head before the truncating stages.
 */
public class BatchEngineConstraintSamplingTest {

    private static final int N_CTX = 96;

    private static EngineConfig config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(N_CTX).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32);
        if (parallel > 1) {
            mc.parallel(parallel);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    /** The default chain (temperature 0.8, top-k 40, top-p 0.95, min-p 0.05) with a fixed seed. */
    private static GenerationRequest constrained(float temperature, ConstraintSpec constraint) {
        return GenerationRequest.builder().maxNewTokens(120)
                .sampler(SamplerConfig.builder().temperature(temperature).seed(7L).build())
                .constraint(constraint)
                .build();
    }

    private static int[] prompt() {
        return new int[]{11, 24, 37, 50, 63, 76};
    }

    @Test
    public void aJsonObjectConstraintDecodesAtTheDefaultTemperature() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(2);
        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg)) {
            BatchSequence sequence = engine.submit(prompt(), constrained(0.8f, ConstraintSpec.jsonObject(null)),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(String.valueOf(sequence.error()), BatchSequence.State.DONE, sequence.state());
            GenerationResult result = sequence.result();
            assertNotNull(result);
            assertTrue("generated " + result.tokenCount() + " tokens", result.tokenCount() >= 1);
            Json.parse(result.text().trim(), Json.Limits.UNTRUSTED);
        }
    }

    @Test
    public void aGrammarConstraintDecodesAtTemperatureOne() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(2);
        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg)) {
            BatchSequence sequence = engine.submit(prompt(),
                    constrained(1.0f, ConstraintSpec.gbnf("root ::= \"yes\" (\"es\")*")),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(String.valueOf(sequence.error()), BatchSequence.State.DONE, sequence.state());
            String text = sequence.result().text();
            assertTrue("constrained text was '" + text + "'", text.matches("yes(es)*"));
        }
    }

    /**
     * The request's masks ({@code allowed_token_ids}, bad words) decode through the plain decode path under a
     * tight chain at a temperature above 0: the constraint head masks before the truncating stages, so the tight
     * top-k/top-p/min-p operate on what the masks leave and only the kept token is sampled (vLLM applies the
     * masks to the logits before any sampling stage).
     */
    @Test
    public void theRequestMasksDecodeThroughTightTruncations() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(2);
        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg)) {
            int banned = roundTrippingTokens(two.tokenizer(), 1)[0];
            int kept = roundTrippingTokens(two.tokenizer(), 2)[1];
            for (float temperature : new float[] {0.8f, 1.0f}) {
                GenerationRequest request = GenerationRequest.builder().maxNewTokens(8)
                        .sampler(SamplerConfig.builder().temperature(temperature).seed(7L)
                                .topK(2).topP(0.1f).minP(0.9f).build())
                        .allowedTokenIds(banned, kept)
                        .badWord(two.tokenizer().decode(new int[]{banned}))
                        .build();
                BatchSequence sequence = engine.submit(prompt(), request,
                        (com.eonmux.neuralfree.gguf.TokenCallback) null);
                engine.drain();
                assertEquals(String.valueOf(sequence.error()), BatchSequence.State.DONE, sequence.state());
                int[] tokens = sequence.result().tokens();
                assertTrue("temperature " + temperature + " generated " + tokens.length + " tokens",
                        tokens.length > 0);
                for (int token : tokens) {
                    assertEquals("temperature " + temperature + " sampled the banned token " + token,
                            kept, token);
                }
            }
        }
    }

    /**
     * The first {@code n} distinct vocabulary ids whose decoded text re-encodes to the id alone: a bad word
     * spelled as that text tokenizes to exactly this token, so the request's masks and the assertions refer to
     * the same id whatever the tokenizer's piece shapes are.
     */
    private static int[] roundTrippingTokens(GGUFTokenizer tokenizer, int n) {
        int[] out = new int[n];
        int found = 0;
        for (int id = 0; id < tokenizer.vocabSize() && found < n; id++) {
            String text = tokenizer.decode(new int[]{id});
            if (text.isEmpty()) {
                continue;
            }
            int[] encoded = tokenizer.encode(text, false, false);
            if (encoded.length == 1 && encoded[0] == id) {
                out[found++] = id;
            }
        }
        org.junit.Assert.assertEquals("found only " + found + " round-tripping tokens", n, found);
        return out;
    }

    /** Proposes the start of {@code text} every round (the scratch drafter of the speculation tests). */
    private static final class TextDrafter implements BatchDrafter {
        private final int[] tokens;

        TextDrafter(int[] tokens) {
            this.tokens = tokens.clone();
        }

        @Override
        public int[] draft(int seq, int lastToken, int n) {
            return java.util.Arrays.copyOf(tokens, Math.min(n, tokens.length));
        }

        @Override
        public void accept(int seq, int nAccepted) {
        }
    }

    @Test
    public void aSpeculatingSequenceVerifiesUnderTheConstraintAtTemperature() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(2);
        try (GGUFModel two = new GGUFModel(model.toString(), cfg)) {
            int[] chars = charTokens(two.tokenizer(), "yeses");
            try (BatchEngine engine = BatchEngine.forModel(two, cfg, new TextDrafter(chars),
                    new BatchSpec(4, 0))) {
                BatchSequence sequence = engine.submit(prompt(),
                        constrained(0.8f, ConstraintSpec.gbnf("root ::= \"yes\" (\"es\")*")),
                        (com.eonmux.neuralfree.gguf.TokenCallback) null);
                engine.drain();
                assertEquals(String.valueOf(sequence.error()), BatchSequence.State.DONE, sequence.state());
                String text = sequence.result().text();
                assertTrue("constrained text was '" + text + "'", text.matches("yes(es)*"));
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
            org.junit.Assert.assertTrue("no single-character token for '" + ch + "'", out[c] >= 0);
        }
        return out;
    }
}

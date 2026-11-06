package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The {@code GenerationRequest} constraint of the CLI ({@code --grammar}, {@code --json-schema}) reaches the
 * generation loop's sampler: a constrained generation over the synthetic llama spells only what the grammar
 * allows, the constraint runs alongside the request's own stages ({@code min_tokens}), and an unusable grammar
 * refuses at the loop's sampler build with the compiler's error. At a temperature above 0 the constraint head's
 * masks (the grammar, {@code allowed_token_ids}, bad words) still leave the chain a candidate: the truncating
 * stages (top-k, top-p, min-p of the default and of a tight chain) operate on the admitted set, as llama.cpp's
 * grammar-first resample and vLLM's pre-sampling logits processors keep them, and a mask dead end refuses the
 * generation by name.
 */
public class GenerationLoopConstraintTest {

    private static final long SEED = 7L;

    /** The default chain (temperature 0.8, top-k 40, top-p 0.95, min-p 0.05) with a fixed seed. */
    private static SamplerConfig sampled(float temperature) {
        return SamplerConfig.builder().temperature(temperature).seed(SEED).build();
    }

    private static EngineConfig config() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static int[] promptTokens(int seed) {
        int[] t = new int[6];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    @Test
    public void aConstrainedGenerationSpellsOnlyWhatTheGrammarAllows() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            GenerationRequest constrained = GenerationRequest.builder()
                    .maxNewTokens(10)
                    .sampler(SamplerConfig.greedyDecoding())
                    .constraint(ConstraintSpec.choice(List.of("yes", "no")))
                    .build();
            GenerationResult result = GenerationLoop.forModel(one, cfg)
                    .generate(promptTokens(1), constrained, null);
            // the finished word leaves the completed grammar only its end-of-generation tokens, so the run ends
            // on EOS well inside the budget
            assertTrue(result.tokenCount() < 10);
            String text = result.text();
            assertTrue("constrained text was '" + text + "'", text.matches("(yes|no)+"));

            GenerationRequest free = GenerationRequest.builder()
                    .maxNewTokens(10)
                    .sampler(SamplerConfig.greedyDecoding())
                    .build();
            GenerationResult freeRun = GenerationLoop.forModel(one, cfg)
                    .generate(promptTokens(1), free, null);
            assertTrue(freeRun.tokenCount() > 0);
        }
    }

    @Test
    public void anUnusableConstraintIsRefusedAtTheSamplerBuild() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            GenerationRequest request = GenerationRequest.builder()
                    .maxNewTokens(4)
                    .sampler(SamplerConfig.greedyDecoding())
                    .constraint(ConstraintSpec.gbnf("root ::= (unclosed"))
                    .build();
            try {
                GenerationLoop.forModel(one, cfg).generate(promptTokens(1), request, null);
                throw new AssertionError("expected a refusal");
            } catch (IllegalArgumentException e) {
                assertTrue(e.getMessage(), !e.getMessage().isEmpty());
            }
        }
    }

    @Test
    public void theMinTokensBanRunsAlongsideTheConstraint() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            // the grammar's repetition leaves the run open past the minimum; the unspent min-tokens budget
            // bans end-of-generation behind the grammar's mask until six tokens were generated
            GenerationRequest request = GenerationRequest.builder()
                    .maxNewTokens(16)
                    .sampler(SamplerConfig.greedyDecoding())
                    .minTokens(6)
                    .constraint(ConstraintSpec.gbnf("root ::= \"yes\" (\"es\")*"))
                    .build();
            GenerationResult result = GenerationLoop.forModel(one, cfg)
                    .generate(promptTokens(1), request, null);
            assertTrue("generated " + result.tokenCount() + " tokens, the minimum is 6",
                    result.tokenCount() >= 6);
            String text = result.text();
            // the token budget can cut the repetition's second byte (the scratch run ended on "e")
            assertTrue("constrained text was '" + text + "'", text.matches("yes(es)*e?"));
        }
    }

    @Test
    public void aJsonObjectConstraintSamplesAtTemperaturesAboveZero() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            for (float temperature : new float[] {0.8f, 1.0f}) {
                GenerationRequest request = GenerationRequest.builder()
                        .maxNewTokens(48)
                        .sampler(sampled(temperature))
                        .constraint(ConstraintSpec.jsonObject(null))
                        .build();
                GenerationResult result = GenerationLoop.forModel(one, cfg)
                        .generate(promptTokens(1), request, null);
                assertTrue("temperature " + temperature + " generated " + result.tokenCount() + " tokens",
                        result.tokenCount() >= 1);
                Json.parse(result.text().trim(), Json.Limits.UNTRUSTED);
            }
        }
    }

    @Test
    public void theConstraintKindsSampleAtTemperaturesAboveZero() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            String tag = Json.toString(Json.parse("{\"type\": \"structural_tag\", \"format\": {\"type\": \"tag\","
                            + " \"begin\": \"<answer>\", \"content\": {\"type\": \"json_schema\","
                            + " \"json_schema\": {\"const\": \"yes\"}}, \"end\": \"</answer>\"}}",
                    Json.Limits.UNTRUSTED), Json.Style.CANONICAL);
            List<ConstraintSpec> kinds = List.of(
                    ConstraintSpec.gbnf("root ::= \"yes\" (\"es\")*"),
                    ConstraintSpec.regex("yes|no", false),
                    ConstraintSpec.choice(List.of("yes", "no")),
                    ConstraintSpec.structuralTag(tag));
            List<String> shapes = List.of("yes(es)*", "yes|no", "yes|no",
                    "<answer>\"yes\"</answer>");
            for (float temperature : new float[] {0.8f, 1.0f}) {
                for (int k = 0; k < kinds.size(); k++) {
                    GenerationRequest request = GenerationRequest.builder()
                            .maxNewTokens(48)
                            .sampler(sampled(temperature))
                            .constraint(kinds.get(k))
                            .build();
                    GenerationResult result = GenerationLoop.forModel(one, cfg)
                            .generate(promptTokens(1), request, null);
                    String text = result.text();
                    assertTrue("temperature " + temperature + " constrained text was '" + text + "'",
                            text.matches(shapes.get(k)));
                }
            }
        }
    }

    @Test
    public void aDeadEndConstraintRefusesByNameAtATemperatureAboveZero() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            // the grammar asks for the vocabulary token 999, which no candidate carries: its admitted set is empty
            GenerationRequest request = GenerationRequest.builder()
                    .maxNewTokens(8)
                    .sampler(sampled(0.8f))
                    .constraint(ConstraintSpec.gbnf("root ::= <[999]>"))
                    .build();
            try {
                GenerationLoop.forModel(one, cfg).generate(promptTokens(1), request, null);
                throw new AssertionError("expected the dead end to refuse the generation");
            } catch (ConstraintDeadEndException e) {
                assertTrue(e.getMessage(), e.getMessage().contains("no candidate"));
            }
        }
    }

    // ================================================================== the request's masks

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
        assertEquals("found only " + found + " round-tripping tokens", n, found);
        return out;
    }

    /** The tight chain over the default one: top-k 2, top-p 0.1, min-p 0.9 at {@code temperature}, a fixed seed. */
    private static SamplerConfig tight(float temperature) {
        return SamplerConfig.builder().temperature(temperature).seed(SEED)
                .topK(2).topP(0.1f).minP(0.9f).build();
    }

    @Test
    public void theRequestMasksConfineTightTruncations() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            int banned = roundTrippingTokens(one.tokenizer(), 1)[0];
            int kept = roundTrippingTokens(one.tokenizer(), 2)[1];
            // allowed_token_ids confines the step to the two tokens and bad_words bans one of them: the head
            // masks first, so the tight top-k/top-p/min-p operate on what is left and only the kept token is
            // sampled (vLLM applies allowed_token_ids and bad words to the logits before any sampling stage)
            for (float temperature : new float[] {0.8f, 1.0f}) {
                GenerationRequest request = GenerationRequest.builder()
                        .maxNewTokens(8)
                        .sampler(tight(temperature))
                        .allowedTokenIds(banned, kept)
                        .badWord(one.tokenizer().decode(new int[]{banned}))
                        .build();
                GenerationResult result = GenerationLoop.forModel(one, cfg)
                        .generate(promptTokens(1), request, null);
                int[] tokens = result.tokens();
                assertTrue("temperature " + temperature + " generated " + tokens.length + " tokens",
                        tokens.length > 0);
                for (int token : tokens) {
                    assertEquals("temperature " + temperature + " sampled the banned token " + token,
                            kept, token);
                }
            }
        }
    }

    @Test
    public void aMaskDeadEndRefusesTheGenerationByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            int only = roundTrippingTokens(one.tokenizer(), 1)[0];
            // the allowed ids leave only the one token and the bad word bans that token: the mask leaves no
            // candidate, the same named refusal as a grammar dead end
            GenerationRequest request = GenerationRequest.builder()
                    .maxNewTokens(8)
                    .sampler(sampled(0.8f))
                    .allowedTokenIds(only)
                    .badWord(one.tokenizer().decode(new int[]{only}))
                    .build();
            try {
                GenerationLoop.forModel(one, cfg).generate(promptTokens(1), request, null);
                throw new AssertionError("expected the mask dead end to refuse the generation");
            } catch (ConstraintDeadEndException ex) {
                assertTrue(ex.getMessage(), ex.getMessage().contains("no candidate"));
            }
        }
    }
}

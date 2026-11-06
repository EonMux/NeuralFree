package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The sampler fields of a request body: OpenAI's, vLLM's and llama.cpp's, their checks and their defaults. */
class SamplingParamsTest {

    private static SamplerConfig of(String json) {
        return SamplingParams.of(view(json));
    }

    @Test
    void llamaCppFieldsReachTheSampler() {
        SamplerConfig c = of("""
                {"temperature": 0.7, "dynatemp_range": 0.2, "dynatemp_exponent": 1.5, "top_k": 30, "top_p": 0.9,
                 "min_p": 0.05, "top_n_sigma": 1.5, "xtc_probability": 0.4, "xtc_threshold": 0.2, "typical_p": 0.95,
                 "repeat_last_n": 128, "repeat_penalty": 1.1, "presence_penalty": 0.3, "frequency_penalty": 0.2,
                 "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 3, "dry_penalty_last_n": 256,
                 "dry_sequence_breakers": ["\\n", "\\n", ";"], "mirostat": 2, "mirostat_tau": 4.0, "mirostat_eta": 0.2,
                 "adaptive_target": 0.6, "adaptive_decay": 0.8, "min_keep": 2, "seed": 9}""");
        assertEquals(0.7f, c.temperature());
        assertEquals(0.2f, c.dynatempRange());
        assertEquals(1.5f, c.dynatempExponent());
        assertEquals(30, c.topK());
        assertEquals(1.5f, c.topNSigma());
        assertEquals(0.4f, c.xtcProbability());
        assertEquals(0.2f, c.xtcThreshold());
        assertEquals(0.95f, c.typicalP());
        assertEquals(128, c.penaltyLastN());
        assertEquals(0.8f, c.dryMultiplier());
        assertEquals(1.75f, c.dryBase());
        assertEquals(3, c.dryAllowedLength());
        assertEquals(256, c.dryPenaltyLastN());
        assertEquals(List.of("\n", ";"), c.drySequenceBreakers(), "duplicate breakers are dropped");
        assertEquals(2, c.mirostat());
        assertEquals(4.0f, c.mirostatTau());
        assertEquals(0.2f, c.mirostatEta());
        assertEquals(0.6f, c.adaptiveTarget());
        assertEquals(0.8f, c.adaptiveDecay());
        assertEquals(2, c.minKeep());
        assertEquals(9L, c.seed());
    }

    @Test
    void theSamplerOrderTakesNamesAliasesOrCharacters() {
        assertEquals(List.of(StageKind.TOP_K, StageKind.TOP_P, StageKind.TYPICAL_P, StageKind.TEMPERATURE,
                StageKind.MIN_P), of("{\"samplers\": [\"top-k\", \"nucleus\", \"typ\", \"Temp\", \"minp\"]}").order());
        assertEquals(List.of(StageKind.PENALTIES, StageKind.DRY, StageKind.TOP_K, StageKind.MIN_P,
                StageKind.TEMPERATURE), of("{\"samplers\": \"edkmt\"}").order());
    }

    @Test
    void nullFieldsKeepTheDefaults() {
        SamplerConfig defaults = GenerationRequest.defaults().sampler();
        SamplerConfig c = of("{\"temperature\": null, \"top_k\": null, \"seed\": null, \"samplers\": null}");
        assertEquals(defaults.temperature(), c.temperature());
        assertEquals(defaults.topK(), c.topK());
        assertEquals(defaults.order(), c.order());
        assertEquals(SamplerConfig.RANDOM_SEED, c.seed());
    }

    @Test
    void malformedOrOutOfRangeValuesRefuse() {
        for (String body : new String[]{
                "{\"samplers\": [\"top_k\", \"bogus\"]}",
                "{\"samplers\": \"kz\"}",
                "{\"samplers\": [\"top_k\", \"topk\"]}",
                "{\"samplers\": 3}",
                "{\"dry_sequence_breakers\": []}",
                "{\"dry_sequence_breakers\": [1]}",
                "{\"mirostat\": 3}",
                "{\"top_p\": 1.5}",
                "{\"repeat_last_n\": -2}",
                "{\"adaptive_decay\": 2}",
                "{\"temperature\": \"hot\"}"}) {
            assertEquals(400, statusOf(() -> of(body)), body);
        }
        assertTrue(errorOf(() -> of("{\"samplers\": [\"bogus\"]}")).contains("bogus"));
    }

    @Test
    void atMostSixtyFourDistinctBreakers() {
        StringBuilder many = new StringBuilder("{\"dry_sequence_breakers\": [");
        for (int i = 0; i <= SamplingParams.MAX_DRY_SEQUENCE_BREAKERS; i++) {
            many.append(i == 0 ? "" : ", ").append("\"b").append(i).append('"');
        }
        String body = many.append("]}").toString();
        assertEquals(400, statusOf(() -> of(body)));
        assertEquals(Collections.nCopies(1, "x"), of("{\"dry_sequence_breakers\": " + "[\"x\", \"x\"]}")
                .drySequenceBreakers());
    }
}

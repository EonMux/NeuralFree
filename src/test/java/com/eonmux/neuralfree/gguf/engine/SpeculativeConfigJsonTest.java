package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link SpeculativeConfigJson}: vLLM's {@code --speculative-config} keys become the speculative flags with vLLM's
 * resolution rules (the method from the model, one prompt-lookup bound setting both, suffix decoding's draft max
 * from the tree depth), and every key or method NeuralFree does not run is refused by name.
 */
public class SpeculativeConfigJsonTest {

    /** The settings as {@code flag value} strings, in order. */
    private static List<String> flags(String json) {
        List<String> out = new ArrayList<>();
        for (SpeculativeConfigJson.Setting s : SpeculativeConfigJson.parse(json)) {
            out.add(s.flag() + " " + s.value());
        }
        return out;
    }

    private static String refusal(String json) {
        return assertThrows(IllegalArgumentException.class, () -> SpeculativeConfigJson.parse(json)).getMessage();
    }

    @Test
    public void theNgramMethodMapsToPromptLookup() {
        assertEquals(Arrays.asList("--spec-type ngram-simple", "--spec-draft-n-max 5",
                "--spec-ngram-simple-size-n 4", "--spec-ngram-simple-size-n-min 2"),
                flags("{\"method\": \"ngram\", \"num_speculative_tokens\": 5, \"prompt_lookup_max\": 4,"
                        + " \"prompt_lookup_min\": 2}"));
        assertEquals("one bound alone sets both, as vLLM resolves it",
                Arrays.asList("--spec-type ngram-simple", "--spec-ngram-simple-size-n 3",
                        "--spec-ngram-simple-size-n-min 3"),
                flags("{\"method\": \"ngram\", \"prompt_lookup_min\": 3}"));
        assertEquals(Arrays.asList("--spec-type ngram-simple", "--spec-ngram-simple-size-n 6",
                "--spec-ngram-simple-size-n-min 6"), flags("{\"model\": \"[ngram]\", \"prompt_lookup_max\": 6}"));
        assertEquals("absent keys keep the configuration", Arrays.asList("--spec-type ngram-simple"),
                flags("{\"method\": \"NGRAM\", \"prompt_lookup_max\": null}"));
    }

    @Test
    public void suffixDecodingKeysMapToTheSuffixLimits() {
        assertEquals(Arrays.asList("--spec-type suffix", "--spec-draft-n-max 24", "--spec-suffix-max-tree-depth 24",
                "--spec-suffix-max-cached-requests 100", "--spec-suffix-max-spec-factor 1.5",
                "--spec-suffix-min-token-prob 0.2"),
                flags("{\"method\": \"suffix\", \"suffix_decoding_max_tree_depth\": 24,"
                        + " \"suffix_decoding_max_cached_requests\": 100, \"suffix_decoding_max_spec_factor\": 1.5,"
                        + " \"suffix_decoding_min_token_prob\": 0.2}"));
        assertEquals("without a depth the draft max is the default depth",
                Arrays.asList("--spec-type suffix", "--spec-draft-n-max 64"), flags("{\"method\": \"suffix\"}"));
        assertEquals(Arrays.asList("--spec-type suffix", "--spec-draft-n-max 8"),
                flags("{\"method\": \"suffix\", \"num_speculative_tokens\": 8}"));
    }

    @Test
    public void useHeterogeneousVocabMapsToItsFlag() {
        assertEquals(Arrays.asList("--spec-type draft-simple", "--spec-draft-model small.gguf",
                "--spec-draft-heterogeneous-vocab true"),
                flags("{\"method\": \"draft_model\", \"model\": \"small.gguf\","
                        + " \"use_heterogeneous_vocab\": true}"));
        assertEquals(Arrays.asList("--spec-type draft-simple", "--spec-draft-model small.gguf",
                "--no-spec-draft-heterogeneous-vocab false"),
                flags("{\"method\": \"draft_model\", \"model\": \"small.gguf\","
                        + " \"use_heterogeneous_vocab\": false}"));
        assertEquals("a model without a method leaves the route to the file",
                Arrays.asList("--spec-type auto", "--spec-draft-model small.gguf",
                        "--spec-draft-heterogeneous-vocab true"),
                flags("{\"model\": \"small.gguf\", \"use_heterogeneous_vocab\": true}"));
        assertTrue(refusal("{\"method\": \"ngram\", \"use_heterogeneous_vocab\": true}")
                .contains("use_heterogeneous_vocab requires method='draft_model'"));
        assertTrue(refusal("{\"method\": \"draft_model\", \"model\": \"small.gguf\","
                + " \"use_heterogeneous_vocab\": \"yes\"}").contains("must be a boolean"));
    }

    @Test
    public void theDraftHeadQuantizationMapsToTheDerivedHeadPrec() {
        assertEquals(Arrays.asList("--spec-type auto", "--spec-draft-model dflash.gguf", "--draft-head-prec fp8"),
                flags("{\"method\": \"dflash\", \"model\": \"dflash.gguf\","
                        + " \"draft_lm_head_quantization\": \"FP8\"}"));
        assertEquals(Arrays.asList("--spec-type auto", "--spec-draft-model dflash.gguf", "--draft-head-prec q8_0"),
                flags("{\"model\": \"dflash.gguf\", \"draft_lm_head_quantization\": \"q8_0\"}"));
        assertEquals(Arrays.asList("--spec-type auto", "--spec-draft-model dflash.gguf", "--draft-head-prec nvfp4"),
                flags("{\"model\": \"dflash.gguf\", \"draft_lm_head_quantization\": \"nvfp4\"}"));
        String int4 = refusal("{\"model\": \"dflash.gguf\", \"draft_lm_head_quantization\": \"int4\"}");
        assertTrue(int4, int4.contains("int4"));
        assertTrue(int4, int4.contains("draft_lm_head_quantization"));
    }

    @Test
    public void draftFileMethodsNeedTheirModel() {
        assertEquals(Arrays.asList("--spec-type draft-simple", "--spec-draft-model small.gguf",
                "--spec-draft-n-max 4"),
                flags("{\"method\": \"draft_model\", \"model\": \"small.gguf\", \"num_speculative_tokens\": 4}"));
        assertEquals(Arrays.asList("--spec-type auto", "--spec-draft-model eagle.gguf"),
                flags("{\"method\": \"eagle3\", \"model\": \"eagle.gguf\"}"));
        assertEquals("a model without a method is detected from the file",
                Arrays.asList("--spec-type auto", "--spec-draft-model dflash.gguf"),
                flags("{\"model\": \"dflash.gguf\"}"));
        assertEquals(Arrays.asList("--spec-type draft-mtp"), flags("{\"method\": \"qwen3_next_mtp\"}"));
        assertEquals(Arrays.asList("--spec-type draft-mtp", "--spec-draft-model mtp.gguf"),
                flags("{\"method\": \"mtp\", \"model\": \"mtp.gguf\"}"));
        assertTrue(refusal("{\"method\": \"draft_model\"}").contains("method draft_model needs a model"));
        assertTrue(refusal("{\"method\": \"eagle\"}").contains("needs a model"));
        assertTrue(refusal("{\"method\": \"ngram\", \"model\": \"x.gguf\"}").contains("drafts without a model"));
        assertTrue(refusal("{\"num_speculative_tokens\": 3}").contains("needs a method or a model"));
    }

    @Test
    public void verificationKeysMapToTheVerifyFlags() {
        assertEquals(Arrays.asList("--spec-type ngram-simple", "--rejection-sample-method rejection",
                "--spec-fly-window 3", "--spec-fly-entropy-threshold 0.5", "--spec-fly-entropy-top-k 4"),
                flags("{\"method\": \"ngram\", \"rejection_sample_method\": \"standard\", \"fly_window_size\": 3,"
                        + " \"fly_entropy_threshold\": 0.5, \"fly_entropy_top_k\": 4}"));
        assertEquals(Arrays.asList("--spec-type ngram-simple", "--spec-draft-n-max 2",
                "--rejection-sample-method synthetic", "--spec-synth-rates 0.9,0.5"),
                flags("{\"method\": \"ngram\", \"num_speculative_tokens\": 2,"
                        + " \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_rates\": [0.9, 0.5]}"));
    }

    @Test
    public void unsupportedKeysAndMethodsAreRefusedByName() {
        String tp = refusal("{\"method\": \"ngram\", \"draft_tensor_parallel_size\": 1}");
        assertTrue(tp, tp.startsWith("--speculative-config key 'draft_tensor_parallel_size' is not supported"));
        String wrongTp = refusal("{\"method\": \"ngram\", \"tensor_parallel_size\": 2}");
        assertTrue(wrongTp, wrongTp.contains("'tensor_parallel_size' is not a speculative config key"));
        String unknown = refusal("{\"method\": \"ngram\", \"quantization\": \"fp8\"}");
        assertTrue(unknown, unknown.startsWith("--speculative-config key 'quantization' is not supported"));
        assertTrue(unknown, unknown.contains("prompt_lookup_min"));
        assertTrue(refusal("{\"method\": \"ngram_gpu\"}").contains("method ngram_gpu is not supported"));
        assertTrue(refusal("{\"method\": \"dspark\"}").contains("method dspark is not supported"));
        assertTrue(refusal("{\"method\": \"lookahead\"}").contains("method 'lookahead' is unknown"));
    }

    @Test
    public void malformedValuesAreRefusedByKey() {
        assertTrue(refusal("[1, 2]").contains("must be a JSON object"));
        assertTrue(refusal("{\"method\": ").startsWith("--speculative-config is not valid JSON"));
        assertTrue(refusal("{\"method\": 3}").contains("method must be a string"));
        assertTrue(refusal("{\"method\": \"ngram\", \"num_speculative_tokens\": 2.5}")
                .contains("num_speculative_tokens must be an integer"));
        assertTrue(refusal("{\"method\": \"suffix\", \"suffix_decoding_min_token_prob\": \"high\"}")
                .contains("suffix_decoding_min_token_prob must be a number"));
        assertTrue(refusal("{\"method\": \"ngram\", \"prompt_lookup_max\": 2, \"prompt_lookup_min\": 3}")
                .contains("prompt_lookup_min=3 must be <= prompt_lookup_max=2"));
        assertTrue(refusal("{\"method\": \"ngram\", \"synthetic_acceptance_rates\": []}")
                .contains("synthetic_acceptance_rates must be a non-empty array"));
        assertTrue(refusal("{\"method\": \"ngram\", \"synthetic_acceptance_rates\": [0.5, \"x\"]}")
                .contains("synthetic_acceptance_rates holds x"));
        SpeculativeConfigJson.Setting s = SpeculativeConfigJson.parse("{\"method\": \"ngram\"}").get(0);
        assertEquals("the refusal quotes the JSON value", "--speculative-config method=ngram: refused",
                s.refusal("refused").getMessage());
        SpeculativeConfigJson.Setting standard = SpeculativeConfigJson.parse(
                "{\"method\": \"ngram\", \"rejection_sample_method\": \"standard\"}").get(1);
        assertEquals("standard", standard.source());
        assertEquals("rejection", standard.value());
    }
}

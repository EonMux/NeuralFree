package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.Test;

import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.OptionalInt;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfGenerationDefaults}: the llama.cpp sampling keys of generation_config.json with their writer types, the
 * NeuralFree penalties and {@code max_new_tokens}, the extra end-of-generation ids and {@code suppress_tokens}.
 */
public class HfGenerationDefaultsTest {

    private static JsonView view(String singleQuoted, String source) {
        return JsonView.of(Json.parse(singleQuoted.replace('\'', '"')), source);
    }

    private static Map<String, Object> of(String generation, String config, int tokens) {
        return HfGenerationDefaults.of(generation == null ? null : view(generation, "generation_config.json"),
                view(config, "config.json"), OptionalInt.of(tokens));
    }

    @Test
    public void samplingKeysPenaltiesAndMaxNewTokensAreWrittenTyped() {
        Map<String, Object> md = of("{'top_k':40,'top_p':0.9,'min_p':0.05,'temperature':0.7,"
                + "'repetition_penalty':1.1,'presence_penalty':0.5,'frequency_penalty':0.25,'max_new_tokens':256,"
                + "'do_sample':true,'mirostat':2,'sequence':'abc'}", "{}", 10);
        assertEquals(40, md.get(GgufKeys.GENERAL_SAMPLING_TOP_K));
        assertEquals(0.9f, md.get(GgufKeys.GENERAL_SAMPLING_TOP_P));
        assertEquals(0.05f, md.get(GgufKeys.GENERAL_SAMPLING_MIN_P));
        assertEquals(0.7f, md.get(GgufKeys.GENERAL_SAMPLING_TEMP));
        assertEquals(1.1f, md.get(GgufKeys.GENERAL_SAMPLING_PENALTY_REPEAT));
        assertEquals(2, md.get(GgufKeys.GENERAL_SAMPLING_MIROSTAT));
        assertEquals("abc", md.get(GgufKeys.GENERAL_SAMPLING_SEQUENCE));
        assertEquals(0.5f, md.get(GgufKeys.NF_SAMPLING_PRESENCE_PENALTY));
        assertEquals(0.25f, md.get(GgufKeys.NF_SAMPLING_FREQUENCY_PENALTY));
        assertEquals(256, md.get(GgufKeys.NF_SAMPLING_MAX_NEW_TOKENS));
        assertEquals("do_sample is ignored", 10, md.size());
        assertThrows(UnsupportedOperationException.class, () -> md.put("x", 1));
    }

    @Test
    public void penaltyRepeatWinsOverRepetitionPenalty() {
        Map<String, Object> md = of("{'penalty_repeat':1.3,'repetition_penalty':1.1}", "{}", 10);
        assertEquals(1.3f, md.get(GgufKeys.GENERAL_SAMPLING_PENALTY_REPEAT));
    }

    @Test
    public void valuesOfTheWrongTypeAreSkipped() {
        Map<String, Object> md = of("{'top_k':'40','top_p':true,'temperature':null,'max_new_tokens':1e12,"
                + "'min_p':[0.1],'mirostat':2.0}", "{}", 10);
        assertEquals("an integral float is an integer", 2, md.get(GgufKeys.GENERAL_SAMPLING_MIROSTAT));
        assertEquals(1, md.size());
    }

    @Test
    public void eosIdsOfBothFilesBecomeExtraEogIdsWithinTheVocabulary() {
        Map<String, Object> md = of("{'eos_token_id':[1,7,12,'x']}", "{'eos_token_id':2}", 10);
        assertEquals(Arrays.asList(1, 7, 2), md.get(GgufKeys.NF_TOKENIZER_EXTRA_EOG_IDS));

        Map<String, Object> textConfig = of(null, "{'text_config':{'eos_token_id':[5,5]}}", 10);
        assertEquals(Collections.singletonList(5), textConfig.get(GgufKeys.NF_TOKENIZER_EXTRA_EOG_IDS));
    }

    @Test
    public void suppressTokensKeepInRangeIds() {
        Map<String, Object> md = of("{'suppress_tokens':[3,99,3]}", "{}", 10);
        assertEquals(Collections.singletonList(3), md.get(GgufKeys.TOKENIZER_SUPPRESS_TOKENS));
        assertEquals(Collections.emptyList(),
                of("{'suppress_tokens':[]}", "{}", 10).get(GgufKeys.TOKENIZER_SUPPRESS_TOKENS));
    }

    @Test
    public void withoutAVocabularyNoTokenIdsAreWritten() {
        Map<String, Object> md = HfGenerationDefaults.of(view("{'eos_token_id':1,'suppress_tokens':[1],'top_k':5}",
                "generation_config.json"), view("{'eos_token_id':2}", "config.json"), OptionalInt.empty());
        assertEquals(Collections.singletonMap(GgufKeys.GENERAL_SAMPLING_TOP_K, 5), md);
    }

    @Test
    public void aConfigWithoutGenerationConfigGivesOnlyItsEosIds() {
        Map<String, Object> md = HfGenerationDefaults.of(
                HfConfigLoader.fromJson(Paths.get("."), "{\"architectures\":[\"A\"],\"eos_token_id\":[2,3]}"),
                OptionalInt.of(10));
        assertEquals(Collections.singletonMap(GgufKeys.NF_TOKENIZER_EXTRA_EOG_IDS, Arrays.asList(2, 3)), md);
        assertTrue(HfGenerationDefaults.of(HfConfigLoader.fromJson(Paths.get("."), "{\"architectures\":[\"A\"]}"),
                OptionalInt.of(10)).isEmpty());
        assertFalse(md.containsKey(GgufKeys.TOKENIZER_SUPPRESS_TOKENS));
    }
}

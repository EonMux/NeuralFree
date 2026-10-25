package com.eonmux.neuralfree.utils.json;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** Typed access over parsed documents and its error messages. */
public class JsonViewTest {

    private static final String CONFIG = "{"
            + "\"model_type\": \"llama\", \"hidden_size\": 4096, \"rms_norm_eps\": 1e-05, \"rope_theta\": 10000,"
            + "\"tie_word_embeddings\": false, \"max_position_embeddings\": 8192.0, \"sliding_window\": null,"
            + "\"architectures\": [\"LlamaForCausalLM\"], \"eos_token_id\": [128001, 128009],"
            + "\"model_max_length\": 1000000000000000019884624838656,"
            + "\"text_config\": {\"num_hidden_layers\": true, \"rope_scaling\": {\"factor\": \"x\"}},"
            + "\"added_tokens\": [{\"content\": \"<s>\"}, {\"content\": 7}]"
            + "}";

    private static JsonView view(String text) {
        return JsonView.of(Json.parse(text), "config.json");
    }

    private static String message(Runnable r) {
        JsonException e = assertThrows(JsonException.class, r::run);
        assertEquals(JsonException.NO_POSITION, e.position());
        return e.getMessage();
    }

    @Test
    public void readsTypedMembers() {
        JsonView v = view(CONFIG);
        assertEquals("llama", v.str("model_type"));
        assertEquals(4096, v.i32("hidden_size"));
        assertEquals(4096L, v.i64("hidden_size"));
        assertEquals(1e-05, v.f64("rms_norm_eps"), 0.0);
        assertEquals(10000.0, v.f64("rope_theta"), 0.0);
        assertFalse(v.bool("tie_word_embeddings"));
        assertEquals(8192, v.i32("max_position_embeddings"));
        assertEquals(Arrays.asList("LlamaForCausalLM"), v.strList("architectures"));
        assertEquals(Arrays.asList(128001L, 128009L), v.i64List("eos_token_id"));
        assertEquals("<s>", ((Map<?, ?>) v.list("added_tokens").get(0)).get("content"));
        assertTrue(v.has("sliding_window"));
        assertFalse(v.has("rope_scaling"));
        assertEquals("model_type", new ArrayList<>(v.keys()).get(0));
        assertEquals("config.json", v.source());
        assertEquals("", v.path());
    }

    @Test
    public void optionalMembersDefaultWhenAbsentOrNull() {
        JsonView v = view(CONFIG);
        assertEquals(-1, v.optI32("sliding_window", -1));
        assertEquals(-1L, v.optI64("missing", -1L));
        assertEquals(0.5, v.optF64("missing", 0.5), 0.0);
        assertNull(v.optStr("sliding_window", null));
        assertTrue(v.optBool("missing", true));
        assertEquals(4096, v.optI32("hidden_size", -1));
        JsonView absent = v.optObj("rope_scaling");
        assertTrue(absent.isEmpty());
        assertEquals("rope_scaling", absent.path());
        assertEquals("linear", absent.optStr("type", "linear"));
        assertTrue(v.optObj("sliding_window").isEmpty());
    }

    @Test
    public void typeMismatchesNameTheSourceAndPath() {
        JsonView v = view(CONFIG);
        assertEquals("config.json:model_type: expected integer, got string", message(() -> v.i32("model_type")));
        assertEquals("config.json:hidden_size: expected string, got integer", message(() -> v.str("hidden_size")));
        assertEquals("config.json:architectures: expected object, got array", message(() -> v.obj("architectures")));
        assertEquals("config.json:model_type: expected array, got string", message(() -> v.list("model_type")));
        assertEquals("config.json:model_type: expected boolean, got string", message(() -> v.bool("model_type")));
        assertEquals("config.json:model_type: expected number, got string", message(() -> v.f64("model_type")));
        assertEquals("config.json:text_config.num_hidden_layers: expected integer, got boolean",
                message(() -> v.obj("text_config").i32("num_hidden_layers")));
        assertEquals("config.json:text_config.rope_scaling.factor: expected number, got string",
                message(() -> v.obj("text_config").optObj("rope_scaling").f64("factor")));
        assertEquals("config.json:added_tokens[0]: expected string, got object",
                message(() -> v.strList("added_tokens")));
        assertEquals("config.json:l[1]: expected string, got integer",
                message(() -> view("{\"l\": [\"a\", 1]}").strList("l")));
    }

    @Test
    public void missingAndNullRequiredMembersFail() {
        JsonView v = view(CONFIG);
        assertEquals("config.json:vocab_size: required member is missing", message(() -> v.i32("vocab_size")));
        assertEquals("config.json:sliding_window: expected integer, got null", message(() -> v.i64("sliding_window")));
        assertEquals("config.json:sliding_window: expected object, got null", message(() -> v.obj("sliding_window")));
    }

    @Test
    public void integerAccessorsRejectFractionsAndOutOfRangeValues() {
        JsonView v = view("{\"r\": 1.5, \"big\": 3000000000, \"huge\": 1e300, \"l\": [1, 2.5], \"neg\": -1e30}");
        assertEquals("config.json:r: expected integer, got number 1.5", message(() -> v.i64("r")));
        assertEquals("config.json:big: expected int32, got integer 3000000000 (out of range)", message(() -> v.i32("big")));
        assertEquals(3000000000L, v.i64("big"));
        assertEquals("config.json:huge: expected int64, got number 1e+300 (out of range)", message(() -> v.i64("huge")));
        assertEquals("config.json:l[1]: expected integer, got number 2.5", message(() -> v.i64List("l")));
        assertEquals(Long.MAX_VALUE, v.i64Saturating("huge", 0));
        assertEquals(Long.MIN_VALUE, v.i64Saturating("neg", 0));
    }

    @Test
    public void veryLargeIntegerIsRejectedByI64AndSaturatedByI64Saturating() {
        JsonView v = view(CONFIG);
        assertEquals("config.json:model_max_length: expected int64, got integer 1000000000000000019884624838656 (out of range)",
                message(() -> v.i64("model_max_length")));
        assertEquals("config.json:model_max_length: expected int32, got integer 1000000000000000019884624838656 (out of range)",
                message(() -> v.i32("model_max_length")));
        assertEquals(Long.MAX_VALUE, v.i64Saturating("model_max_length", 2048));
        assertEquals(2048L, v.i64Saturating("missing", 2048));
        assertEquals(4096L, v.i64Saturating("hidden_size", 2048));
        assertEquals("config.json:rms_norm_eps: expected integer, got number 1e-05",
                message(() -> v.i64Saturating("rms_norm_eps", 0)));
    }

    @Test
    public void rootMustBeAnObject() {
        assertEquals("config.json:<root>: expected object, got array",
                message(() -> JsonView.of(Json.parse("[1]"), "config.json")));
        assertEquals("tokenizer.json:model: expected object, got null",
                message(() -> JsonView.of(null, "tokenizer.json", "model")));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void containersAreReadOnlyAtEveryDepth() {
        JsonView v = view(CONFIG);
        List<Object> tokens = v.list("added_tokens");
        assertThrows(UnsupportedOperationException.class, () -> tokens.add("x"));
        Map<Object, Object> first = (Map<Object, Object>) tokens.get(0);
        assertThrows(UnsupportedOperationException.class, () -> first.put("content", "x"));
        Map<Object, Object> text = (Map<Object, Object>) v.raw("text_config");
        Map<Object, Object> rope = (Map<Object, Object>) text.get("rope_scaling");
        assertThrows(UnsupportedOperationException.class, () -> rope.remove("factor"));
        Map<Object, Object> viaValues = (Map<Object, Object>) new ArrayList<>(text.values()).get(1);
        assertThrows(UnsupportedOperationException.class, viaValues::clear);
        assertThrows(UnsupportedOperationException.class, () -> v.keys().remove("model_type"));
        assertThrows(UnsupportedOperationException.class, () -> v.strList("architectures").add("x"));
        assertEquals("<s>", first.get("content"));
    }
}

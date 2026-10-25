package com.eonmux.neuralfree.gguf.hf.meta;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Test;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfMetadataBuilder}: keys typed by the generated key table, value conversion into the GGUF Java value model,
 * the converter's writer semantics (a repeated key replaces its value in place, an empty array is skipped) and the
 * typed escape hatches.
 */
public class HfMetadataBuilderTest {

    private final HfMetadataBuilder kv = new HfMetadataBuilder("llama");

    @Test
    public void archScopedKeysAreTypedByTheTable() {
        kv.set(GgufKeys.CONTEXT_LENGTH, 4096L);
        kv.set(GgufKeys.ROPE_FREQ_BASE, 10000);
        kv.set(GgufKeys.ATTENTION_CAUSAL, false);
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "yarn");
        kv.set(GgufKeys.GENERAL_SAMPLING_TOP_K, -1);
        HfMetadata m = kv.build();
        assertEquals(Arrays.asList("llama.context_length", "llama.rope.freq_base", "llama.attention.causal",
                "llama.rope.scaling.type", "general.sampling.top_k"), new ArrayList<>(m.keys()));
        assertEquals(4096, m.get("llama.context_length"));
        assertEquals(GGUFValueType.UINT32, m.type("llama.context_length"));
        assertEquals(10000f, m.get("llama.rope.freq_base"));
        assertEquals(GGUFValueType.FLOAT32, m.type("llama.rope.freq_base"));
        assertEquals(false, m.get("llama.attention.causal"));
        assertEquals(GGUFValueType.BOOL, m.type("llama.attention.causal"));
        assertEquals("yarn", m.get("llama.rope.scaling.type"));
        assertEquals(-1, m.get("general.sampling.top_k"));
        assertEquals(GGUFValueType.INT32, m.type("general.sampling.top_k"));
        assertNull(m.elementType("llama.context_length"));
    }

    @Test
    public void integersAreRangeCheckedAndKeepTheirBitPattern() {
        kv.set(GgufKeys.CONTEXT_LENGTH, 4294967295L);
        assertEquals(-1, kv.get(GgufKeys.CONTEXT_LENGTH));
        kv.set(GgufKeys.CONTEXT_LENGTH, 8.0);
        assertEquals(8, kv.get(GgufKeys.CONTEXT_LENGTH));
        kv.set(GgufKeys.CONTEXT_LENGTH, BigInteger.valueOf(16));
        assertEquals(16, kv.get(GgufKeys.CONTEXT_LENGTH));
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.CONTEXT_LENGTH, -1));
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.CONTEXT_LENGTH, 4294967296L));
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.CONTEXT_LENGTH, 1.5));
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.CONTEXT_LENGTH, true));
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.CONTEXT_LENGTH, "4096"));
        assertThrows(IllegalArgumentException.class, () -> kv.set("split.count", 65536));
        kv.set("split.count", 65535);
        assertEquals(65535, kv.get("split.count"));
        assertEquals(GGUFValueType.UINT16, kv.type("split.count"));
    }

    @Test
    public void unsigned64ArraysTakeBigIntegers() {
        kv.set("%s.ple.head_offsets", Arrays.asList(BigInteger.ONE.shiftLeft(63), 5L));
        HfMetadata m = kv.build();
        assertEquals(Arrays.asList(Long.MIN_VALUE, 5L), m.get("llama.ple.head_offsets"));
        assertEquals(GGUFValueType.UINT64, m.elementType("llama.ple.head_offsets"));
        assertThrows(IllegalArgumentException.class,
                () -> kv.set("%s.ple.head_offsets", Collections.singletonList(BigInteger.ONE.shiftLeft(64))));
        assertThrows(IllegalArgumentException.class,
                () -> kv.set("%s.ple.head_offsets", Collections.singletonList(-1L)));
    }

    @Test
    public void scalarsAreStrict() {
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.ROPE_FREQ_BASE, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.ROPE_FREQ_BASE, "1e4"));
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.ATTENTION_CAUSAL, "false"));
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.ATTENTION_CAUSAL, 0));
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.ROPE_SCALING_TYPE, 1));
        assertThrows(NullPointerException.class, () -> kv.set(GgufKeys.ROPE_SCALING_TYPE, null));
        assertEquals(0, kv.build().size());
    }

    @Test
    public void anyKeysFollowPythonTyping() {
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, 11008);
        assertEquals(GGUFValueType.UINT32, kv.type(GgufKeys.FEED_FORWARD_LENGTH));
        assertEquals(11008, kv.get(GgufKeys.FEED_FORWARD_LENGTH));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, Arrays.asList(8L, 0L, 8L));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, new int[] {32, 16});
        HfMetadata m = kv.build();
        assertEquals(GGUFValueType.ARRAY, m.type("llama.attention.head_count_kv"));
        assertEquals(GGUFValueType.INT32, m.elementType("llama.attention.head_count_kv"));
        assertEquals(Arrays.asList(8, 0, 8), m.get("llama.attention.head_count_kv"));
        assertEquals(Arrays.asList(32, 16), m.get("llama.attention.head_count"));
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, Arrays.asList(1.5, 2.0));
        assertEquals(GGUFValueType.FLOAT32, kv.build().elementType("llama.feed_forward_length"));
        assertEquals(Arrays.asList(1.5f, 2f), kv.get(GgufKeys.FEED_FORWARD_LENGTH));
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.FEED_FORWARD_LENGTH, Arrays.asList(1, 1.5)));
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.FEED_FORWARD_LENGTH, "11008"));
    }

    @Test
    public void typedArraysConvertTheirElements() {
        kv.set(GgufKeys.ROPE_DIMENSION_SECTIONS, new long[] {16, 24, 24, 0});
        kv.set(GgufKeys.TOKENIZER_LIST, new String[] {"<s>", "a"});
        kv.set(GgufKeys.TOKENIZER_SCORES, Arrays.asList(0, -1.5));
        HfMetadata m = kv.build();
        assertEquals(Arrays.asList(16, 24, 24, 0), m.get("llama.rope.dimension_sections"));
        assertEquals(Arrays.asList("<s>", "a"), m.get("tokenizer.ggml.tokens"));
        assertEquals(Arrays.asList(0f, -1.5f), m.get("tokenizer.ggml.scores"));
        assertEquals(GGUFValueType.FLOAT32, m.elementType("tokenizer.ggml.scores"));
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.TOKENIZER_LIST, Arrays.asList("a", 1)));
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.TOKENIZER_LIST, "a"));
        assertThrows(UnsupportedOperationException.class,
                () -> ((List<?>) m.get("tokenizer.ggml.tokens")).clear());
    }

    @Test
    public void emptyArraysAreSkippedLikeAddArray() {
        kv.set(GgufKeys.TOKENIZER_LIST, Collections.emptyList());
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, new int[0]);
        assertFalse(kv.has(GgufKeys.TOKENIZER_LIST));
        assertFalse(kv.has(GgufKeys.ATTENTION_HEAD_COUNT));
        kv.setTyped("%s.custom_list", GGUFValueType.ARRAY, GGUFValueType.INT32, Collections.emptyList());
        assertEquals(Collections.emptyList(), kv.get("%s.custom_list"));
    }

    @Test
    public void repeatedKeysReplaceInPlace() {
        kv.set(GgufKeys.CONTEXT_LENGTH, 1);
        kv.set(GgufKeys.EMBEDDING_LENGTH, 2);
        kv.set(GgufKeys.CONTEXT_LENGTH, 3);
        HfMetadata m = kv.build();
        assertEquals(Arrays.asList("llama.context_length", "llama.embedding_length"), new ArrayList<>(m.keys()));
        assertEquals(3, m.get("llama.context_length"));
        assertEquals(2, kv.remove(GgufKeys.EMBEDDING_LENGTH));
        assertNull(kv.remove(GgufKeys.EMBEDDING_LENGTH));
        assertFalse(kv.has(GgufKeys.EMBEDDING_LENGTH));
        assertTrue(m.has("llama.embedding_length"));
    }

    @Test
    public void placeholderTemplatesNeedSetAt() {
        assertThrows(IllegalArgumentException.class, () -> kv.set(GgufKeys.GENERAL_BASE_MODEL_NAME, "base"));
        kv.setAt(GgufKeys.GENERAL_BASE_MODEL_NAME, "base", 0);
        kv.setAt(GgufKeys.TOKENIZER_CHAT_TEMPLATE_N, "{{ x }}", "rag");
        kv.setAt(GgufKeys.DECISION_TEMPERATURE, 0.5, "route");
        kv.setAt(GgufKeys.DECISION_SCORE_BIAS, 1.25, 2, 3);
        HfMetadata m = kv.build();
        assertEquals("base", m.get("general.base_model.0.name"));
        assertEquals("{{ x }}", m.get("tokenizer.chat_template.rag"));
        assertEquals(0.5f, m.get("llama.decision.temperature.route"));
        assertEquals(1.25f, m.get("llama.decision.score_bias.2.3"));
        assertThrows(IllegalArgumentException.class, () -> kv.setAt(GgufKeys.GENERAL_BASE_MODEL_NAME, "b"));
        assertThrows(IllegalArgumentException.class, () -> kv.setAt(GgufKeys.GENERAL_BASE_MODEL_NAME, "b", 0, 1));
        assertThrows(IllegalArgumentException.class, () -> kv.setAt(GgufKeys.GENERAL_BASE_MODEL_NAME, "b", "x"));
        assertThrows(IllegalArgumentException.class, () -> kv.setAt(GgufKeys.TOKENIZER_CHAT_TEMPLATE_N, "t", "%d"));
    }

    @Test
    public void denseFeatureKeysMatchTheirPattern() {
        kv.set(GgufKeys.DENSE_2_FEAT_IN, 768);
        kv.set("%s.dense_5_feat_out", 3072);
        assertEquals(768, kv.get(GgufKeys.DENSE_2_FEAT_IN));
        assertEquals(GGUFValueType.UINT32, kv.type("%s.dense_5_feat_out"));
        assertThrows(IllegalArgumentException.class, () -> kv.set("%s.dense_x_feat_out", 1));
        assertThrows(IllegalArgumentException.class, () -> kv.set("%s.{dense}_feat_out", 1));
    }

    @Test
    public void unknownKeysAndTypesAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> kv.set("%s.no_such_key", 1));
        assertThrows(IllegalArgumentException.class, () -> kv.setTyped("%s.x", GGUFValueType.ARRAY, 1));
        assertThrows(IllegalArgumentException.class,
                () -> kv.setTyped("%s.x", GGUFValueType.UINT32, GGUFValueType.INT32, 1));
        assertThrows(IllegalArgumentException.class,
                () -> kv.setTyped("%s.x", GGUFValueType.ARRAY, GGUFValueType.ARRAY, Collections.emptyList()));
        kv.setTyped("%s.x", GGUFValueType.UINT8, 255);
        assertEquals(255, kv.get("llama.x"));
        assertThrows(IllegalArgumentException.class, () -> kv.setTyped("%s.x", GGUFValueType.UINT8, 256));
        assertThrows(IllegalArgumentException.class, () -> new HfMetadataBuilder("a.b"));
        assertThrows(IllegalArgumentException.class, () -> new HfMetadataBuilder(""));
    }

    @Test
    public void putAllResolvesConcreteKeys() {
        Map<String, Object> in = new LinkedHashMap<>();
        in.put("llama.context_length", 4096L);
        in.put("tokenizer.ggml.tokens", Arrays.asList("a", "b"));
        in.put("tokenizer.chat_template.rag", "t");
        in.put("general.base_model.1.name", "n");
        in.put("llama.dense_3_feat_in", 4);
        kv.putAll(in);
        HfMetadata m = kv.build();
        assertEquals(new ArrayList<>(in.keySet()), new ArrayList<>(m.keys()));
        assertEquals(4096, m.get("llama.context_length"));
        assertEquals(GGUFValueType.STRING, m.type("general.base_model.1.name"));
        assertEquals(GGUFValueType.UINT32, m.type("llama.dense_3_feat_in"));
        assertThrows(IllegalArgumentException.class,
                () -> kv.putAll(Collections.singletonMap("llama.no_such_key", 1)));
        assertThrows(IllegalArgumentException.class,
                () -> kv.putAll(Collections.singletonMap("qwen2.context_length", 1)));
    }

    @Test
    public void snapshotsAreImmutableAndIndependent() {
        kv.set(GgufKeys.CONTEXT_LENGTH, 1);
        HfMetadata m = kv.build();
        kv.set(GgufKeys.EMBEDDING_LENGTH, 2);
        assertEquals(1, m.size());
        assertThrows(UnsupportedOperationException.class, () -> m.values().put("x", 1));
        assertThrows(UnsupportedOperationException.class, () -> m.types().clear());
    }
}

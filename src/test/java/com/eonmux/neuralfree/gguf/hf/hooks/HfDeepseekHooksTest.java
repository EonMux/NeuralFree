package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfDeepseekHooks}: the DEEPSEEK2 / DEEPSEEK2OCR resolver, the MLA metadata and its OCR differences
 * (LC/conversion/deepseek.py:319-388), the {@code kv_b_proj} split with the checkpoint's head count, the per-layer
 * expert stacking with the MTP and tied-head drops, the vision filter, and the Kimi-K2 tiktoken vocabulary.
 */
public class HfDeepseekHooksTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static final String V3 = "'architectures': ['DeepseekV3ForCausalLM'], 'hidden_size': 7168,"
            + " 'intermediate_size': 18432, 'moe_intermediate_size': 2048, 'num_attention_heads': 128,"
            + " 'num_key_value_heads': 128, 'num_hidden_layers': 61, 'max_position_embeddings': 163840,"
            + " 'vocab_size': 129280, 'rope_theta': 10000, 'q_lora_rank': 1536, 'kv_lora_rank': 512,"
            + " 'qk_nope_head_dim': 128, 'qk_rope_head_dim': 64, 'v_head_dim': 128, 'n_routed_experts': 256,"
            + " 'num_experts_per_tok': 8, 'n_shared_experts': 1, 'first_k_dense_replace': 3,"
            + " 'routed_scaling_factor': 2.5, 'norm_topk_prob': true, 'scoring_func': 'sigmoid',"
            + " 'rope_scaling': {'type': 'yarn', 'factor': 40, 'original_max_position_embeddings': 4096,"
            + " 'beta_fast': 32, 'beta_slow': 1, 'mscale': 1.0, 'mscale_all_dim': 1.0}";

    private static final String OCR = "'architectures': ['DeepseekOCRForCausalLM'], 'hidden_size': 1280,"
            + " 'intermediate_size': 6848, 'moe_intermediate_size': 896, 'num_attention_heads': 10,"
            + " 'num_key_value_heads': 10, 'num_hidden_layers': 12, 'max_position_embeddings': 8192,"
            + " 'vocab_size': 129280, 'q_lora_rank': null, 'kv_lora_rank': null, 'qk_nope_head_dim': 0,"
            + " 'qk_rope_head_dim': 0, 'v_head_dim': 0, 'n_routed_experts': 64, 'num_experts_per_tok': 6,"
            + " 'n_shared_experts': 2, 'first_k_dense_replace': 1, 'sliding_window_size': 4096";

    private static final String SMALL = "'architectures': ['DeepseekV3ForCausalLM'], 'hidden_size': 4,"
            + " 'intermediate_size': 8, 'moe_intermediate_size': 4, 'num_attention_heads': 2,"
            + " 'num_key_value_heads': 2, 'num_hidden_layers': 2, 'vocab_size': 16, 'kv_lora_rank': 4,"
            + " 'qk_nope_head_dim': 2, 'qk_rope_head_dim': 2, 'v_head_dim': 1, 'n_routed_experts': 2,"
            + " 'num_experts_per_tok': 1, 'tie_word_embeddings': true";

    private static HfConfig config(Path dir, String singleQuotedJson) {
        return HfConfigLoader.fromJson(dir, "{" + singleQuotedJson.replace('\'', '"') + "}");
    }

    private static HfArchEntry entry(String hfName) {
        return HfArchEntry.of(List.of(hfName), HfDeepseekHooks::arch, HfDeepseekHooks::meta,
                HfDeepseekHooks::tensors, HfDeepseekHooks::vocab)
                .withExtra(HfDeepseekHooks::extra)
                .withNameFilter(HfDeepseekHooks::filter);
    }

    private static HfLoadContext context(String json, Map<String, RowSource> index) {
        HfConfig cfg = config(Path.of("."), json);
        HfArchEntry e = entry(cfg.architecture());
        Arch arch = e.arch().resolve(cfg, HfTensorIndex.of(index));
        return HfLoadContext.builder(cfg, e, arch).index(HfTensorIndex.of(index)).build();
    }

    private static HfMetadata meta(HfLoadContext ctx) {
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        HfDeepseekHooks.meta(ctx, kv, null);
        return kv.build();
    }

    private static String key(HfLoadContext ctx, String template) {
        return new HfMetadataBuilder(ctx.arch().ggufName()).key(template);
    }

    @Test
    public void resolvesTheOcrArchitecturesToDeepseek2Ocr() {
        HfTensorIndex none = HfTensorIndex.of(Map.of());
        for (String name : List.of("DeepseekOCRForCausalLM", "DeepseekOCR2ForCausalLM", "UnlimitedOCRForCausalLM")) {
            HfConfig cfg = config(Path.of("."), "'architectures': ['" + name + "'], 'num_hidden_layers': 2");
            assertEquals(name, Arch.DEEPSEEK2OCR, HfDeepseekHooks.arch(cfg, none));
        }
        for (String name : List.of("DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM")) {
            HfConfig cfg = config(Path.of("."), "'architectures': ['" + name + "'], 'num_hidden_layers': 2");
            assertEquals(name, Arch.DEEPSEEK2, HfDeepseekHooks.arch(cfg, none));
        }
        assertEquals(HfDeepseekHooks.OCR_CHAT_TEMPLATE, HfDeepseekHooks.fallbackChatTemplate(Arch.DEEPSEEK2OCR));
        assertNull(HfDeepseekHooks.fallbackChatTemplate(Arch.DEEPSEEK2));
    }

    @Test
    public void writesTheMlaKeySetAsMqa() {
        HfLoadContext ctx = context(V3, Map.of());
        HfMetadata m = meta(ctx);
        assertEquals(1, m.get(key(ctx, GgufKeys.ATTENTION_HEAD_COUNT_KV)));
        assertEquals(128, m.get(key(ctx, GgufKeys.ATTENTION_HEAD_COUNT)));
        assertEquals(1e-6f, (Float) m.get(key(ctx, GgufKeys.ATTENTION_LAYERNORM_RMS_EPS)), 0f);
        assertEquals(3, m.get(key(ctx, GgufKeys.LEADING_DENSE_BLOCK_COUNT)));
        assertEquals(129280, m.get(key(ctx, GgufKeys.VOCAB_SIZE)));
        assertEquals(1536, m.get(key(ctx, GgufKeys.ATTENTION_Q_LORA_RANK)));
        assertEquals(512, m.get(key(ctx, GgufKeys.ATTENTION_KV_LORA_RANK)));
        assertEquals(576, m.get(key(ctx, GgufKeys.ATTENTION_KEY_LENGTH)));
        assertEquals(512, m.get(key(ctx, GgufKeys.ATTENTION_VALUE_LENGTH)));
        assertEquals(192, m.get(key(ctx, GgufKeys.ATTENTION_KEY_LENGTH_MLA)));
        assertEquals(128, m.get(key(ctx, GgufKeys.ATTENTION_VALUE_LENGTH_MLA)));
        assertEquals(2048, m.get(key(ctx, GgufKeys.EXPERT_FEED_FORWARD_LENGTH)));
        assertEquals(256, m.get(key(ctx, GgufKeys.EXPERT_COUNT)));
        assertEquals(1, m.get(key(ctx, GgufKeys.EXPERT_SHARED_COUNT)));
        assertEquals(2.5f, (Float) m.get(key(ctx, GgufKeys.EXPERT_WEIGHTS_SCALE)), 0f);
        assertEquals(Boolean.TRUE, m.get(key(ctx, GgufKeys.EXPERT_WEIGHTS_NORM)));
        assertEquals(64, m.get(key(ctx, GgufKeys.ROPE_DIMENSION_COUNT)));
        assertEquals(0.1f, (Float) m.get(key(ctx, GgufKeys.ROPE_SCALING_YARN_LOG_MUL)), 1e-7f);
        assertEquals(GGUFValueType.UINT32, m.type(key(ctx, GgufKeys.ATTENTION_KV_LORA_RANK)));
        assertFalse(m.has(key(ctx, GgufKeys.ATTENTION_SLIDING_WINDOW)));
    }

    @Test
    public void ocrKeepsItsHeadCountAndSkipsTheMlaLengths() {
        HfLoadContext ctx = context(OCR, Map.of());
        assertEquals(Arch.DEEPSEEK2OCR, ctx.arch());
        HfMetadata m = meta(ctx);
        assertEquals(10, m.get(key(ctx, GgufKeys.ATTENTION_HEAD_COUNT_KV)));
        for (String absent : List.of(GgufKeys.ATTENTION_KV_LORA_RANK, GgufKeys.ATTENTION_KEY_LENGTH_MLA,
                GgufKeys.ATTENTION_VALUE_LENGTH_MLA, GgufKeys.ATTENTION_Q_LORA_RANK,
                GgufKeys.ROPE_SCALING_YARN_LOG_MUL, GgufKeys.ROPE_FREQ_BASE, GgufKeys.EXPERT_WEIGHTS_SCALE,
                GgufKeys.EXPERT_WEIGHTS_NORM)) {
            assertFalse(absent, m.has(key(ctx, absent)));
        }
        assertEquals(4096, m.get(key(ctx, GgufKeys.ATTENTION_SLIDING_WINDOW)));
        assertEquals(1, m.get(key(ctx, GgufKeys.LEADING_DENSE_BLOCK_COUNT)));
        assertEquals(2, m.get(key(ctx, GgufKeys.EXPERT_SHARED_COUNT)));
        assertEquals(896, m.get(key(ctx, GgufKeys.EXPERT_FEED_FORWARD_LENGTH)));
        assertEquals(1e-6f, (Float) m.get(key(ctx, GgufKeys.ATTENTION_LAYERNORM_RMS_EPS)), 0f);
    }

    @Test
    public void denseModelsDefaultTheLeadingDenseCountToEveryBlock() {
        String dense = SMALL.replace(" 'n_routed_experts': 2,", "");
        HfLoadContext ctx = context(dense, Map.of());
        HfMetadata m = meta(ctx);
        assertEquals(2, m.get(key(ctx, GgufKeys.LEADING_DENSE_BLOCK_COUNT)));
        assertEquals(0, m.get(key(ctx, GgufKeys.EXPERT_SHARED_COUNT)));
        assertFalse(m.has(key(ctx, GgufKeys.EXPERT_COUNT)));
    }

    @Test
    public void kvBSplitsPerHeadWithTheCheckpointHeadCount() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        index.put("model.layers.0.self_attn.kv_b_proj.weight", floats(6, 4));
        HfLoadContext ctx = context(SMALL, index);
        Sink out = plan(ctx);
        RowSource kB = out.added.get("blk.0.attn_k_b.weight");
        RowSource vB = out.added.get("blk.0.attn_v_b.weight");
        assertArrayEquals(new long[]{2, 4, 2}, kB.shape());
        assertArrayEquals(new long[]{2, 1, 4}, vB.shape());
        float[] k = values(kB);
        assertArrayEquals(new float[]{0, 4, 1, 5, 2, 6, 3, 7}, Arrays.copyOf(k, 8), 0f);
        assertArrayEquals(new float[]{8, 9, 10, 11, 20, 21, 22, 23}, values(vB), 0f);

        Map<String, RowSource> bad = new LinkedHashMap<>();
        bad.put("model.layers.0.self_attn.kv_b_proj.weight", floats(5, 4));
        assertThrows(GgufFormatException.class, () -> plan(context(SMALL, bad)));
    }

    @Test
    public void stacksExpertsPerLayerAndDropsMtpLayersAndTheTiedHead() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        for (int x = 0; x < 2; x++) {
            for (String part : List.of("gate_proj", "up_proj", "down_proj")) {
                index.put("model.layers.1.mlp.experts." + x + "." + part + ".weight", floats(3, 4));
            }
        }
        index.put("model.layers.1.self_attn.q_a_proj.weight", floats(2, 4));
        index.put("model.layers.2.self_attn.q_a_proj.weight", floats(2, 4));
        index.put("lm_head.weight", floats(16, 4));
        HfLoadContext ctx = context(SMALL, index);
        Sink out = plan(ctx);
        HfDeepseekHooks.extra(ctx, out);
        assertEquals(List.of("blk.1.ffn_down_exps.weight", "blk.1.ffn_gate_exps.weight", "blk.1.ffn_up_exps.weight",
                "blk.1.attn_q_a.weight"), new ArrayList<>(out.added.keySet()));
        assertArrayEquals(new long[]{2, 3, 4}, out.added.get("blk.1.ffn_gate_exps.weight").shape());
        assertTrue(out.dropped.contains("model.layers.2.self_attn.q_a_proj.weight"));
        assertTrue(out.dropped.contains("lm_head.weight"));

        Map<String, RowSource> incomplete = new LinkedHashMap<>(index);
        incomplete.remove("model.layers.1.mlp.experts.1.up_proj.weight");
        assertThrows(GgufFormatException.class, () -> plan(context(SMALL, incomplete)));
        Map<String, RowSource> stray = new LinkedHashMap<>();
        stray.put("model.layers.1.mlp.experts.scales", floats(2));
        assertThrows(GgufFormatException.class, () -> plan(context(SMALL, stray)));
    }

    @Test
    public void subclassFormKeepsMtpLayersAndPreStackedExperts() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        index.put("model.layers.2.self_attn.q_a_proj.weight", floats(2, 4));
        index.put("model.layers.1.mlp.experts.gate_proj.weight", floats(2, 3, 4));
        HfConfig cfg = config(Path.of("."), SMALL);
        HfArchEntry e = entry(cfg.architecture()).withBlockCount(c -> c.requireInt("num_hidden_layers") + 1);
        HfLoadContext ctx = HfLoadContext.builder(cfg, e, Arch.DEEPSEEK2).index(HfTensorIndex.of(index)).build();
        Sink out = new Sink();
        for (String name : ctx.index().names()) {
            HfDeepseekHooks.tensors(ctx, name, ctx.index().get(name), out, false, false);
        }
        HfDeepseekHooks.extra(ctx, out);
        assertEquals(List.of("blk.2.attn_q_a.weight", "blk.1.ffn_gate_exps.weight"),
                new ArrayList<>(out.added.keySet()));
        assertTrue(out.dropped.isEmpty());
        assertTrue(out.consumed.isEmpty());
    }

    @Test
    public void filterDropsTheOcrVisionTowers() {
        assertNull(HfDeepseekHooks.filter("model.sam_model.blocks.0.attn.qkv.weight", n -> n));
        assertNull(HfDeepseekHooks.filter("model.qwen2_model.layers.0.mlp.up_proj.weight", n -> n));
        assertEquals("model.layers.0.mlp.up_proj.weight!",
                HfDeepseekHooks.filter("model.layers.0.mlp.up_proj.weight", n -> n + "!"));
    }

    @Test
    public void kimiTiktokenVocabularyKeepsTwoPartMergesAndReservedSpecials() throws IOException {
        Path dir = tmp.newFolder("kimi").toPath();
        StringBuilder ranks = new StringBuilder();
        int rank = 0;
        for (int b = 0; b < 256; b++) {
            ranks.append(Base64.getEncoder().encodeToString(new byte[]{(byte) b})).append(' ').append(rank++)
                    .append('\n');
        }
        for (String t : List.of("ab", "abc", "xyz")) {
            ranks.append(Base64.getEncoder().encodeToString(t.getBytes(StandardCharsets.US_ASCII))).append(' ')
                    .append(rank++).append('\n');
        }
        Files.write(dir.resolve("tiktoken.model"), ranks.toString().getBytes(StandardCharsets.US_ASCII));
        Files.write(dir.resolve("tokenizer_config.json"), ("{\"added_tokens_decoder\": {\"259\": {\"content\":"
                + " \"[BOS]\", \"special\": true}}}").getBytes(StandardCharsets.UTF_8));
        HfConfig cfg = config(dir, "'architectures': ['DeepseekV3ForCausalLM'], 'num_hidden_layers': 1,"
                + " 'vocab_size': 520, 'bos_token_id': 259");
        HfLoadContext ctx = HfLoadContext.builder(cfg, entry("DeepseekV3ForCausalLM"), Arch.DEEPSEEK2)
                .tokenizerFiles(HfTokenizerFiles.locate(dir)).build();

        Map<String, Object> md = HfDeepseekHooks.vocab(ctx, (tj, encode) -> {
            assertNull(tj);
            assertTrue(encode.apply("abc xyz").length > 0);
            return "kimi-k2";
        });
        assertEquals("gpt2", md.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals("kimi-k2", md.get(GgufKeys.TOKENIZER_PRE));
        assertEquals(List.of("a b", "ab c"), md.get(GgufKeys.TOKENIZER_MERGES));
        List<?> tokens = (List<?>) md.get(GgufKeys.TOKENIZER_LIST);
        List<?> types = (List<?>) md.get(GgufKeys.TOKENIZER_TOKEN_TYPE);
        assertEquals(520, tokens.size());
        assertEquals("a", tokens.get('a'));
        assertEquals("Ġ", tokens.get(' '));
        assertEquals("abc", tokens.get(257));
        assertEquals("xyz", tokens.get(258));
        assertEquals("[BOS]", tokens.get(259));
        assertEquals(3, types.get(259));
        assertEquals("<|reserved_token_260|>", tokens.get(260));
        assertEquals(3, types.get(514));
        assertEquals("[PAD515]", tokens.get(515));
        assertEquals(5, types.get(515));
        assertEquals(1, types.get(258));
        assertEquals(259, md.get(GgufKeys.TOKENIZER_BOS_ID));

        assertThrows(GgufFormatException.class, () -> HfDeepseekHooks.vocab(ctx, (tj, encode) -> "deepseek-v3"));
    }

    // ------------------------------------------------------------------ helpers

    /** Runs the tensor hook over the index in order, skipping consumed names, as the planner does. */
    private static Sink plan(HfLoadContext ctx) throws IOException {
        Sink out = new Sink();
        HfTensorHook hook = ctx.entry().tensors();
        for (String name : ctx.index().names()) {
            if (!out.isConsumed(name)) {
                hook.apply(ctx, name, ctx.index().get(name), out);
            }
        }
        return out;
    }

    private static RowSource floats(long... shape) {
        int n = 1;
        for (long d : shape) {
            n *= (int) d;
        }
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = i;
        }
        return RopeTensors.fromFloats(shape, v);
    }

    private static float[] values(RowSource s) throws IOException {
        ByteBuffer b = ByteBuffer.allocate((int) (s.rows() * s.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
        s.readRows(0, (int) s.rows(), b);
        b.flip();
        float[] out = new float[b.remaining() / 4];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getFloat();
        }
        return out;
    }

    /** A sink recording what the hooks plan, drop and consume. */
    static final class Sink implements TensorSink {
        final Map<String, RowSource> added = new LinkedHashMap<>();
        final Set<String> dropped = new HashSet<>();
        final Set<String> consumed = new HashSet<>();

        @Override
        public void add(String ggufName, RowSource src) {
            if (added.put(ggufName, src) != null) {
                throw new AssertionError("planned twice: " + ggufName);
            }
        }

        @Override
        public void addKv(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
            throw new AssertionError("unexpected tensor-derived key " + key);
        }

        @Override
        public void drop(String hfName, String reason) {
            dropped.add(hfName);
        }

        @Override
        public boolean isConsumed(String hfName) {
            return consumed.contains(hfName);
        }

        @Override
        public void consume(String hfName) {
            consumed.add(hfName);
        }
    }
}

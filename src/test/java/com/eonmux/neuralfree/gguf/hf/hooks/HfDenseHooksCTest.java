package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.RecordingSink;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.planAll;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link HfDenseHooksC} against the converter classes it ports: the Olmo clamp and Q/K permutation
 * (LC/conversion/olmo.py:18-39), the Olmo 2 sliding window and its olmo-2 period (olmo.py:51-69), the Apertus xIELU
 * per-layer parameter arrays (llama.py:434-469, with the entry's {@code withUndoPermute(false)}), the HunYuan NTK-alpha
 * dynamic RoPE base (hunyuan.py:248-283) and the Hunyuan-VL XD-RoPE fixup and metadata (hunyuan.py:332-360), plus the
 * table rows of {@link HfArchTableDense}.
 */
public class HfDenseHooksCTest {

    private static final String OLMO_BASE = "'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2,"
            + " 'num_key_value_heads': 1, 'num_hidden_layers': 2, 'max_position_embeddings': 64,"
            + " 'rms_norm_eps': 1e-05, 'vocab_size': 8, 'rope_theta': 10000.0";

    // ------------------------------------------------------------------ OlmoModel

    @Test
    public void olmoTensorStorePermutesQAndK() throws IOException {
        Map<String, RowSource> tensors = new LinkedHashMap<>();
        tensors.put("model.layers.0.self_attn.q_proj.weight", iota(8, 2));
        tensors.put("model.layers.0.self_attn.k_proj.weight", iota(4, 2));
        tensors.put("model.layers.0.self_attn.v_proj.weight", iota(4, 2));
        RecordingSink out = planAll(context(Arch.OLMO, "OLMoForCausalLM", OLMO_BASE, tensors),
                HfDenseHooksC::olmoTensors);
        assertArrayEquals(new float[] {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15},
                values(out.tensors.get("blk.0.attn_q.weight")), 0f);
        assertArrayEquals(new float[] {0, 1, 4, 5, 2, 3, 6, 7},
                values(out.tensors.get("blk.0.attn_k.weight")), 0f);
        assertArrayEquals(new float[] {0, 1, 2, 3, 4, 5, 6, 7},
                values(out.tensors.get("blk.0.attn_v.weight")), 0f);
    }

    @Test
    public void olmoMetaWritesTheFixedEpsilonAndTheClamp() throws IOException {
        HfMetadata m = meta(HfDenseHooksC::olmoMeta, OLMO_BASE + ", 'clip_qkv': 4.0", Arch.OLMO, "OLMoForCausalLM");
        assertEquals(1e-05f, num(m.get(key(Arch.OLMO, "attention.layer_norm_epsilon"))), 1e-9f);
        assertEquals(4.0f, num(m.get(key(Arch.OLMO, "attention.clamp_kqv"))), 0f);
        HfMetadata withoutClamp = meta(HfDenseHooksC::olmoMeta, OLMO_BASE, Arch.OLMO, "OLMoForCausalLM");
        assertNull(withoutClamp.get(key(Arch.OLMO, "attention.clamp_kqv")));
    }

    // ------------------------------------------------------------------ Olmo2Model

    @Test
    public void olmo2MetaWritesTheWindowAndTheOlmo2Period() throws IOException {
        HfMetadata m = meta(HfDenseHooksC::olmo2Meta,
                OLMO_BASE + ", 'num_hidden_layers': 4, 'sliding_window': 512", Arch.OLMO2, "Olmo2ForCausalLM");
        assertEquals(512, num(m.get(key(Arch.OLMO2, "attention.sliding_window"))), 0f);
        assertArrayEquals(new Object[] {true, true, true, false},
                list(m.get(key(Arch.OLMO2, "attention.sliding_window_pattern"))));
    }

    @Test
    public void olmo2MetaFollowsTheLayerTypesPattern() throws IOException {
        HfMetadata m = meta(HfDenseHooksC::olmo2Meta, OLMO_BASE + ", 'sliding_window': 4096,"
                        + " 'layer_types': ['full_attention', 'sliding_attention', 'full_attention',"
                        + " 'sliding_attention']",
                Arch.OLMO2, "Olmo3ForCausalLM");
        assertArrayEquals(new Object[] {false, true, false, true},
                list(m.get(key(Arch.OLMO2, "attention.sliding_window_pattern"))));
    }

    @Test
    public void olmo2WithoutAWindowWritesNeitherKey() throws IOException {
        HfMetadata m = meta(HfDenseHooksC::olmo2Meta, OLMO_BASE, Arch.OLMO2, "Olmo2ForCausalLM");
        assertNull(m.get(key(Arch.OLMO2, "attention.sliding_window")));
        assertNull(m.get(key(Arch.OLMO2, "attention.sliding_window_pattern")));
    }

    // ------------------------------------------------------------------ ApertusModel

    /** The act_fn scalar of layer {@code layer} for the xIELU parameter {@code param}. */
    private static String xieluName(int layer, String param) {
        return "model.layers." + layer + ".act_fn." + param;
    }

    /** A one-element F32 tensor with the value {@code v}. */
    private static RowSource scalar(float v) {
        return RopeTensors.fromFloats(new long[] {1}, new float[] {v});
    }

    @Test
    public void apertusCollectsTheXieluScalarsPerLayer() throws IOException {
        HfLoadContext ctx = context(Arch.APERTUS, "ApertusForCausalLM", OLMO_BASE, Map.of());
        RecordingSink out = new RecordingSink();
        float[][] scalars = {{0.1f, 0.7f, 3.0f, 0.01f}, {0.2f, 0.8f, 3.5f, 0.02f}};
        String[] params = {"alpha_n", "alpha_p", "beta", "eps"};
        for (int l = 0; l < 2; l++) {
            for (int k = 0; k < params.length; k++) {
                HfDenseHooksC.apertusTensors(ctx, xieluName(l, params[k]), scalar(scalars[l][k]), out);
            }
        }
        // layer 0's scalars are dropped; layer 1 completes the per-parameter array and is consumed
        for (String param : params) {
            assertTrue(out.dropped.containsKey(xieluName(0, param)));
        }
        assertArrayEquals(new float[] {0.1f, 0.2f}, xielu(out.kv.get(GgufKeys.XIELU_ALPHA_N)), 0f);
        assertArrayEquals(new float[] {0.7f, 0.8f}, xielu(out.kv.get(GgufKeys.XIELU_ALPHA_P)), 0f);
        assertArrayEquals(new float[] {3.0f, 3.5f}, xielu(out.kv.get(GgufKeys.XIELU_BETA)), 0f);
        assertArrayEquals(new float[] {0.01f, 0.02f}, xielu(out.kv.get(GgufKeys.XIELU_EPS)), 0f);
        assertEquals(GGUFValueType.ARRAY, out.kvTypes.get(GgufKeys.XIELU_ALPHA_N));
        assertNull(out.tensors.get(xieluName(0, "alpha_n")));
    }

    private static float[] xielu(Object value) {
        java.util.List<?> l = (java.util.List<?>) value;
        float[] out = new float[l.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) l.get(i)).floatValue();
        }
        return out;
    }

    @Test
    public void apertusRepeatedAndUnlayeredScalarsAreRefused() throws IOException {
        HfLoadContext ctx = context(Arch.APERTUS, "ApertusForCausalLM", OLMO_BASE, Map.of());
        RecordingSink out = new RecordingSink();
        HfDenseHooksC.apertusTensors(ctx, xieluName(0, "alpha_n"), scalar(0.1f), out);
        assertThrows(GgufFormatException.class,
                () -> HfDenseHooksC.apertusTensors(ctx, xieluName(0, "alpha_n"), scalar(0.1f), out));
        assertThrows(GgufFormatException.class,
                () -> HfDenseHooksC.apertusTensors(ctx, "model.act_fn.alpha_n", scalar(0.1f), out));
        assertEquals(1, out.dropped.size());
        HfDenseHooksC.apertusTensors(ctx, xieluName(0, "alpha_p"), scalar(0.7f), out);
        assertEquals(2, out.dropped.size());
    }

    @Test
    public void apertusOtherTensorsFollowTheLlamaPathWithoutThePermutation() throws IOException {
        Map<String, RowSource> tensors = new LinkedHashMap<>();
        tensors.put("model.layers.0.self_attn.q_proj.weight", iota(8, 2));
        tensors.put("model.layers.0.mlp.up_proj.weight", iota(16, 8));
        RecordingSink out = planAll(context(Arch.APERTUS, "ApertusForCausalLM", OLMO_BASE, tensors),
                HfDenseHooksC::apertusTensors);
        // the entry stores undo_permute false: q and k convert as stored (llama.py:436)
        assertArrayEquals(new float[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                values(out.tensors.get("blk.0.attn_q.weight")), 0f);
        // the up projection keeps its [16, 8] shape and values
        assertArrayEquals(new float[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
                92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127},
                values(out.tensors.get("blk.0.ffn_up.weight")), 0f);
    }

    // ------------------------------------------------------------------ HunYuanModel

    /** The HunYuan dense configuration with the dynamic-RoPE parameters of the dense checkpoints. */
    private static final String HUNYUAN = OLMO_BASE + ", 'head_dim': 4, 'max_position_embeddings': 32768,"
            + " 'rope_scaling': {'rope_type': 'dynamic', 'alpha': 50.0}";

    @Test
    public void hunyuanMetaWritesTheNtkAlphaScaledBase() throws IOException {
        HfMetadata m = meta(HfDenseHooksC::hunyuanMeta, HUNYUAN, Arch.HUNYUAN_DENSE, "HunYuanDenseV1ForCausalLM");
        String g = Arch.HUNYUAN_DENSE.ggufName();
        // theta * alpha^(dim / (dim - 2)) = 10000 * 50^2
        assertEquals(25_000_000f, num(m.get(String.format(GgufKeys.ROPE_FREQ_BASE, g))), 0f);
        assertEquals("none", m.get(String.format(GgufKeys.ROPE_SCALING_TYPE, g)));
        assertEquals(1f, num(m.get(String.format(GgufKeys.ROPE_SCALING_FACTOR, g))), 0f);
        assertEquals(262144, num(m.get(String.format(GgufKeys.ROPE_SCALING_ORIG_CTX_LEN, g))), 0f);
        assertEquals(262144, num(m.get(String.format(GgufKeys.CONTEXT_LENGTH, g))), 0f);
        assertNull(m.get(String.format(GgufKeys.EXPERT_COUNT, g)));
    }

    @Test
    public void hunyuanMetaRefusesChangedDynamicScalingAssumptions() {
        HfLoadContext ctx = context(Arch.HUNYUAN_DENSE, "HunYuanDenseV1ForCausalLM", OLMO_BASE + ", 'head_dim': 4,"
                + " 'max_position_embeddings': 32768, 'rope_scaling': {'rope_type': 'dynamic',"
                + " 'rope_theta': 20000.0, 'alpha': 50.0}", Map.of());
        try {
            HfDenseHooksC.hunyuanMeta(ctx, new HfMetadataBuilder(Arch.HUNYUAN_DENSE.ggufName()), null);
            fail("a changed dynamic scaling base must be refused");
        } catch (GgufFormatException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("assumptions changed"));
        }
    }

    @Test
    public void hunyuanTensorStoreDropsTheTiedOutputLayer() throws IOException {
        Map<String, RowSource> tensors = new LinkedHashMap<>();
        tensors.put("lm_head.weight", iota(8, 8));
        tensors.put("model.embed_tokens.weight", iota(8, 8));
        RecordingSink out = planAll(context(Arch.HUNYUAN_DENSE, "HunYuanDenseV1ForCausalLM",
                HUNYUAN + ", 'tie_word_embeddings': true", tensors), HfDenseHooksC::hunyuanTensors);
        assertEquals(1, out.dropped.size());
        assertTrue(out.dropped.containsKey("lm_head.weight"));
        assertEquals(1, out.tensors.size());
        assertTrue(out.tensors.containsKey("token_embd.weight"));
    }

    // ------------------------------------------------------------------ HunyuanVLTextModel

    @Test
    public void hunyuanVlFixupTurnsDynamicMRopeIntoXdrope() throws IOException {
        HfLoadContext ctx = context(Arch.HUNYUAN_VL, "HunYuanVLForConditionalGeneration", OLMO_BASE
                + ", 'head_dim': 4, 'rope_scaling': {'rope_type': 'dynamic', 'mrope_section': [2, 2, 2],"
                + " 'alpha': 50.0, 'factor': 2.0}", Map.of());
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.HUNYUAN_VL.ggufName());
        HfDenseHooksC.hunyuanVlMeta(ctx, kv, null);
        HfMetadata m = kv.build();
        String g = Arch.HUNYUAN_VL.ggufName();
        assertEquals("xdrope", ctx.config().rope().raw().get("rope_type"));
        assertEquals(10_000f, num(m.get(String.format(GgufKeys.ROPE_FREQ_BASE, g))), 0f);
        assertEquals(50f, num(m.get(String.format(GgufKeys.ROPE_SCALING_ALPHA, g))), 0f);
        assertEquals("none", m.get(String.format(GgufKeys.ROPE_SCALING_TYPE, g)));
        assertEquals(2f, num(m.get(String.format(GgufKeys.ROPE_SCALING_FACTOR, g))), 0f);
        assertEquals(64, num(m.get(String.format(GgufKeys.ROPE_SCALING_ORIG_CTX_LEN, g))), 0f);
        assertEquals(64, num(m.get(String.format(GgufKeys.CONTEXT_LENGTH, g))), 0f);
        assertArrayEquals(new Object[] {2, 2, 2},
                list(m.get(String.format(GgufKeys.ROPE_DIMENSION_SECTIONS, g))));
    }

    @Test
    public void hunyuanVlFixupKeepsOtherRopeTypes() throws IOException {
        HfLoadContext ctx = context(Arch.HUNYUAN_VL, "HunYuanVLForConditionalGeneration", OLMO_BASE
                + ", 'head_dim': 4", Map.of());
        assertNull(ctx.config().rope().raw().get("rope_type"));
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.HUNYUAN_VL.ggufName());
        HfDenseHooksC.hunyuanVlMeta(ctx, kv, null);
        assertNull(kv.build().get(String.format(GgufKeys.ROPE_DIMENSION_SECTIONS, Arch.HUNYUAN_VL.ggufName())));
    }

    // ------------------------------------------------------------------ the registered table rows

    /** Every {@link HfArchTableDense} name resolves to its row's architecture. */
    @Test
    public void everyTableHkD2NameResolvesToItsArch() {
        HfTensorIndex empty = HfTensorIndex.of(new HashMap<>());
        for (Object[] row : tableNames()) {
            String hfName = (String) row[0];
            Arch arch = (Arch) row[1];
            String members = "{'architectures': ['" + hfName + "'], " + OLMO_BASE + "}";
            if (hfName.equals("Ministral3Model") || hfName.equals("Mistral3Model")) {
                members = "{'architectures': ['" + hfName + "'], " + OLMO_BASE + ", 'is_causal': false}";
            }
            HfConfig cfg = config(members);
            assertEquals(hfName, arch, HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT).arch().resolve(cfg, empty));
        }
    }

    /** The table's rows: HF name to the row's architecture, in table order. */
    private static Object[][] tableNames() {
        return new Object[][] {
                {"NEOChatModel", Arch.SENSENOVA_U1},
                {"ApertusForCausalLM", Arch.APERTUS},
                {"ArceeForCausalLM", Arch.ARCEE},
                {"AudioFlamingo3ForConditionalGeneration", Arch.QWEN2},
                {"DotsOCRForCausalLM", Arch.QWEN2},
                {"KORMoForCausalLM", Arch.QWEN2},
                {"Qwen2AudioForConditionalGeneration", Arch.QWEN2},
                {"Qwen2ForCausalLM", Arch.QWEN2},
                {"Qwen2Model", Arch.QWEN2},
                {"Ernie4_5ForCausalLM", Arch.ERNIE4_5},
                {"Ernie4_5_ForCausalLM", Arch.ERNIE4_5},
                {"Gemma2ForCausalLM", Arch.GEMMA2},
                {"Gemma3ForCausalLM", Arch.GEMMA3},
                {"Gemma3ForConditionalGeneration", Arch.GEMMA3},
                {"Gemma3nForCausalLM", Arch.GEMMA3N},
                {"Gemma3nForConditionalGeneration", Arch.GEMMA3N},
                {"GemmaForCausalLM", Arch.GEMMA},
                {"HunYuanDenseV1ForCausalLM", Arch.HUNYUAN_DENSE},
                {"HunYuanVLForConditionalGeneration", Arch.HUNYUAN_VL},
                {"Ministral3ForCausalLM", Arch.MISTRAL3},
                {"Ministral3Model", Arch.MISTRAL3},
                {"Mistral3ForConditionalGeneration", Arch.MISTRAL3},
                {"Mistral3Model", Arch.MISTRAL3},
                {"VoxtralRealtimeForConditionalGeneration", Arch.MISTRAL3},
                {"OLMoForCausalLM", Arch.OLMO},
                {"OlmoForCausalLM", Arch.OLMO},
                {"Olmo2ForCausalLM", Arch.OLMO2},
                {"Olmo3ForCausalLM", Arch.OLMO2},
                {"PaddleOCRVLForConditionalGeneration", Arch.PADDLEOCR},
                {"Phi3ForCausalLM", Arch.PHI3},
                {"Phi4ForCausalLMV", Arch.PHI3},
                {"QWenLMHeadModel", Arch.QWEN},
                {"Qwen2VLForConditionalGeneration", Arch.QWEN2VL},
                {"Qwen2VLModel", Arch.QWEN2VL},
                {"Qwen2_5OmniModel", Arch.QWEN2VL},
                {"Qwen2_5_VLForConditionalGeneration", Arch.QWEN2VL},
                {"Qwen3ASRForConditionalGeneration", Arch.QWEN3VL},
                {"Qwen3ForCausalLM", Arch.QWEN3},
                {"Qwen3Model", Arch.QWEN3},
                {"Qwen3MoeForCausalLM", Arch.QWEN3MOE},
                {"Qwen3VLForConditionalGeneration", Arch.QWEN3VL},
                {"SeedOssForCausalLM", Arch.SEED_OSS},
                {"SmolLM3ForCausalLM", Arch.SMOLLM3},
                {"StepVLForConditionalGeneration", Arch.QWEN3}};
    }

    // ------------------------------------------------------------------ helpers

    private static HfConfig config(String singleQuotedJson) {
        return HfConfigLoader.fromJson(Path.of("."), singleQuotedJson.replace('\'', '"'));
    }

    private static HfArchEntry entry(String hfName) {
        return HfArchRegistry.resolve(config("{'architectures': ['" + hfName + "'], 'num_hidden_layers': 1}"),
                HfArchEntry.Kind.TEXT);
    }

    /** The context the assembler derives: resolve on the raw configuration, then the entry's fixup. */
    private static HfLoadContext context(Arch arch, String hfArch, String members, Map<String, RowSource> tensors) {
        HfTensorIndex index = HfTensorIndex.of(tensors == null ? new HashMap<>() : new HashMap<>(tensors));
        HfConfig cfg = config("{'architectures': ['" + hfArch + "'], " + members + "}");
        HfArchEntry entry = entry(hfArch);
        Arch resolved = entry.arch().resolve(cfg, index);
        return HfLoadContext.builder(entry.configFixup().apply(cfg), entry, resolved).index(index).build();
    }

    private interface MetaHook {
        void apply(HfLoadContext ctx, HfMetadataBuilder kv, com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView plan)
                throws IOException;
    }

    private static HfMetadata meta(MetaHook hook, String members, Arch arch, String hfArch) throws IOException {
        HfLoadContext ctx = context(arch, hfArch, members, null);
        HfMetadataBuilder kv = new HfMetadataBuilder(arch.ggufName());
        hook.apply(ctx, kv, null);
        return kv.build();
    }

    private static float num(Object v) {
        return ((Number) v).floatValue();
    }

    private static Object[] list(Object l) {
        return ((java.util.List<?>) l).toArray();
    }

    private static String key(Arch arch, String tail) {
        return arch.ggufName() + "." + tail;
    }

    /** An F32 tensor of {@code shape} with the values 0, 1, 2, ... */
    private static RowSource iota(long... shape) {
        long n = 1;
        for (long d : shape) {
            n *= d;
        }
        float[] v = new float[(int) n];
        for (int i = 0; i < n; i++) {
            v[i] = i;
        }
        return RopeTensors.fromFloats(shape, v);
    }

    /** The values of an F32 source. */
    private static float[] values(RowSource src) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(src.rows() * src.rowBytes()))
                .order(ByteOrder.LITTLE_ENDIAN);
        src.readRows(0, Math.toIntExact(src.rows()), b);
        b.flip();
        float[] out = new float[b.capacity() / Float.BYTES];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getFloat(i * Float.BYTES);
        }
        return out;
    }
}

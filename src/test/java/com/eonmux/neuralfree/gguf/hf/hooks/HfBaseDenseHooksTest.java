package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfMetadataHook;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
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
 * {@link HfBaseDenseHooks} against the converter classes it ports: Exaone4 (LC/conversion/exaone.py:65-129), Ernie 4.5
 * (LC/conversion/ernie.py:19-75) and Phi-3 (LC/conversion/phi.py:41-211), on synthetic configurations and in-memory
 * tensors. Also holds the recording sink and context helpers the Gemma hook tests share.
 */
public class HfBaseDenseHooksTest {

    // ------------------------------------------------------------------ shared fixtures

    /** A {@link TensorSink} that records what the hooks plan. */
    static final class RecordingSink implements TensorSink {
        final Map<String, RowSource> tensors = new LinkedHashMap<>();
        final Map<String, Object> kv = new LinkedHashMap<>();
        final Map<String, GGUFValueType> kvTypes = new LinkedHashMap<>();
        final Map<String, String> dropped = new LinkedHashMap<>();
        final Set<String> consumed = new HashSet<>();

        @Override
        public void add(String ggufName, RowSource src) {
            if (tensors.containsKey(ggufName)) {
                throw new IllegalArgumentException("tensor '" + ggufName + "' is already planned");
            }
            tensors.put(ggufName, src);
        }

        @Override
        public void addKv(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
            if (kv.containsKey(key)) {
                throw new IllegalArgumentException("metadata key '" + key + "' is already recorded");
            }
            kv.put(key, value);
            kvTypes.put(key, type);
        }

        @Override
        public void drop(String hfName, String reason) {
            dropped.put(hfName, reason);
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

    /** A source with an output scale, as the quantization rewrite produces for scaled formats. */
    static final class Scaled implements RowSource {
        private final RowSource base;
        private final float[] scale;

        Scaled(RowSource base, float... scale) {
            this.base = base;
            this.scale = scale.clone();
        }

        @Override
        public String describe() {
            return base.describe() + " with output scale " + Arrays.toString(scale);
        }

        @Override
        public ElementFormat format() {
            return base.format();
        }

        @Override
        public long[] shape() {
            return base.shape();
        }

        @Override
        public long rows() {
            return base.rows();
        }

        @Override
        public long rowBytes() {
            return base.rowBytes();
        }

        @Override
        public boolean rowLocal() {
            return base.rowLocal();
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            base.readRows(first, count, dst);
        }

        @Override
        public float[] outputScale() {
            return scale.clone();
        }
    }

    static String json(String singleQuoted) {
        return singleQuoted.replace('\'', '"');
    }

    /** A context for {@code arch} over the config.json members {@code members} (single-quoted) and {@code tensors}. */
    static HfLoadContext context(Arch arch, String hfArch, String members, Map<String, RowSource> tensors) {
        HfArchEntry entry = HfArchEntry.of(List.of(hfArch), HfArchResolver.of(arch), HfMetadataHook.GENERIC,
                HfTensorHook.DEFAULT, (ctx, pre) -> {
                    throw new UnsupportedOperationException("vocabulary is not part of this test");
                });
        String config = "{'architectures': ['" + hfArch + "'], " + members + "}";
        return HfLoadContext.builder(HfConfigLoader.fromJson(Path.of("."), json(config)), entry, arch)
                .index(HfTensorIndex.of(tensors)).build();
    }

    /** Runs {@code hook} over the index in order, skipping consumed tensors, as the planner does. */
    static RecordingSink planAll(HfLoadContext ctx, HfTensorHook hook) throws IOException {
        RecordingSink out = new RecordingSink();
        for (String name : ctx.index().names()) {
            if (!out.isConsumed(name)) {
                hook.apply(ctx, name, ctx.index().get(name), out);
            }
        }
        return out;
    }

    /** An F32 tensor of {@code shape} holding 0, 1, 2, ... */
    static RowSource iota(long... shape) {
        long n = 1;
        for (long d : shape) {
            n *= d;
        }
        float[] v = new float[(int) n];
        for (int i = 0; i < v.length; i++) {
            v[i] = i;
        }
        return RopeTensors.fromFloats(shape, v);
    }

    static float[] values(RowSource src) throws IOException {
        return HfGemma4Hooks.readFloats(src, src.describe(), Integer.MAX_VALUE);
    }

    static float[] range(int from, int count) {
        float[] v = new float[count];
        for (int i = 0; i < count; i++) {
            v[i] = from + i;
        }
        return v;
    }

    private static HfMetadata meta(HfLoadContext ctx, HfMetadataHook hook) throws IOException {
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        hook.apply(ctx, kv, null);
        return kv.build();
    }

    // ------------------------------------------------------------------ Exaone4

    private static final String EXAONE4 = "'hidden_size': 64, 'intermediate_size': 128, 'num_attention_heads': 4,"
            + " 'num_key_value_heads': 2, 'max_position_embeddings': 4096, 'rms_norm_eps': 1e-05,"
            + " 'vocab_size': 1000, 'rope_theta': 1000000.0";

    @Test
    public void exaone4WritesTheVocabSizeAndTheLayerTypesPattern() throws IOException {
        HfMetadata m = meta(context(Arch.EXAONE4, "Exaone4ForCausalLM", EXAONE4 + ", 'num_hidden_layers': 4,"
                + " 'sliding_window': 4096, 'layer_types': ['sliding_attention', 'sliding_attention',"
                + " 'sliding_attention', 'full_attention']", Map.of()), HfBaseDenseHooks::exaone4Meta);
        assertEquals(1000, m.get("exaone4.vocab_size"));
        assertEquals(4096, m.get("exaone4.attention.sliding_window"));
        assertEquals(Arrays.asList(true, true, true, false), m.get("exaone4.attention.sliding_window_pattern"));
        assertEquals(GGUFValueType.BOOL, m.elementType("exaone4.attention.sliding_window_pattern"));
        assertEquals(1e6f, m.get("exaone4.rope.freq_base"));
    }

    @Test
    public void exaone4ExpandsAStringOrPeriodPattern() throws IOException {
        HfMetadata s = meta(context(Arch.EXAONE4, "Exaone4ForCausalLM", EXAONE4 + ", 'num_hidden_layers': 5,"
                + " 'sliding_window': 128, 'sliding_window_pattern': 'LLLG'", Map.of()), HfBaseDenseHooks::exaone4Meta);
        assertEquals(Arrays.asList(true, true, true, false, true), s.get("exaone4.attention.sliding_window_pattern"));
        HfMetadata p = meta(context(Arch.EXAONE4, "Exaone4ForCausalLM", EXAONE4 + ", 'num_hidden_layers': 5,"
                + " 'sliding_window': 128, 'sliding_window_pattern': 4", Map.of()), HfBaseDenseHooks::exaone4Meta);
        assertEquals(Arrays.asList(true, true, true, false, true), p.get("exaone4.attention.sliding_window_pattern"));
    }

    @Test
    public void exaone4WithoutASlidingWindowWritesNoWindowKeys() throws IOException {
        HfMetadata m = meta(context(Arch.EXAONE4, "Exaone4ForCausalLM", EXAONE4 + ", 'num_hidden_layers': 2,"
                + " 'sliding_window': null, 'sliding_window_pattern': 'LG'", Map.of()), HfBaseDenseHooks::exaone4Meta);
        assertFalse(m.keys().contains("exaone4.attention.sliding_window"));
        assertFalse(m.keys().contains("exaone4.attention.sliding_window_pattern"));
    }

    @Test
    public void exaone4Llama3RopeFreqsUseFactorSixteenByDefault() throws IOException {
        HfLoadContext ctx = context(Arch.EXAONE4, "Exaone4ForCausalLM", EXAONE4 + ", 'num_hidden_layers': 2,"
                + " 'head_dim': 64, 'rope_scaling': {'rope_type': 'llama3', 'original_max_position_embeddings': 128,"
                + " 'low_freq_factor': 1.0, 'high_freq_factor': 4.0}", Map.of());
        RecordingSink out = new RecordingSink();
        HfBaseDenseHooks.exaone4Extra(ctx, out);
        assertEquals(List.of("rope_freqs.weight"), new ArrayList<>(out.tensors.keySet()));
        float[] expected = values(RopeTensors.llama3Freqs(1e6, 64, 0, 16.0, 1.0, 4.0, 128, false));
        float[] got = values(out.tensors.get("rope_freqs.weight"));
        assertArrayEquals(expected, got, 0f);
        assertEquals("the lowest frequency gets the full factor", 16f, got[got.length - 1], 0f);
        assertEquals("the highest frequency is unscaled", 1f, got[0], 0f);
    }

    @Test
    public void exaone4WithoutLlama3RopeHasNoExtraTensors() {
        RecordingSink out = new RecordingSink();
        HfBaseDenseHooks.exaone4Extra(context(Arch.EXAONE4, "Exaone4ForCausalLM", EXAONE4
                + ", 'num_hidden_layers': 2", Map.of()), out);
        assertTrue(out.tensors.isEmpty());
    }

    // ------------------------------------------------------------------ Ernie4.5

    private static final String ERNIE = "'hidden_size': 8, 'intermediate_size': 6, 'num_attention_heads': 4,"
            + " 'num_key_value_heads': 2, 'head_dim': 2, 'num_hidden_layers': 1, 'max_position_embeddings': 128,"
            + " 'rms_norm_eps': 1e-05";

    @Test
    public void ernieFilterRenamesTheErniePrefixBeforeTheBaseFilter() {
        List<String> seen = new ArrayList<>();
        String out = HfBaseDenseHooks.ernie45Filter("ernie.layers.0.self_attn.q_proj.weight", n -> {
            seen.add(n);
            return n;
        });
        assertEquals(List.of("model.layers.0.self_attn.q_proj.weight"), seen);
        assertEquals("model.layers.0.self_attn.q_proj.weight", out);
        assertNull(HfBaseDenseHooks.ernie45Filter("ernie.vision.x", n -> null));
    }

    @Test
    public void ernieSplitsTheFusedQkvByHeadCounts() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put("model.layers.0.self_attn.qkv_proj.weight", iota(16, 8));
        RecordingSink out = planAll(context(Arch.ERNIE4_5, "Ernie4_5_ForCausalLM", ERNIE, t),
                HfBaseDenseHooks::ernie45Tensors);
        assertEquals(List.of("blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight"),
                new ArrayList<>(out.tensors.keySet()));
        assertArrayEquals(new long[]{8, 8}, out.tensors.get("blk.0.attn_q.weight").shape());
        assertArrayEquals(new long[]{4, 8}, out.tensors.get("blk.0.attn_k.weight").shape());
        assertArrayEquals(new long[]{4, 8}, out.tensors.get("blk.0.attn_v.weight").shape());
        assertArrayEquals(range(0, 64), values(out.tensors.get("blk.0.attn_q.weight")), 0f);
        assertArrayEquals(range(64, 32), values(out.tensors.get("blk.0.attn_k.weight")), 0f);
        assertArrayEquals(range(96, 32), values(out.tensors.get("blk.0.attn_v.weight")), 0f);
    }

    @Test
    public void ernieSplitsUpGateIntoGateThenUp() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put("model.layers.0.mlp.up_gate_proj.weight", iota(12, 8));
        RecordingSink out = planAll(context(Arch.ERNIE4_5, "Ernie4_5_ForCausalLM", ERNIE, t),
                HfBaseDenseHooks::ernie45Tensors);
        assertEquals(List.of("blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"), new ArrayList<>(out.tensors.keySet()));
        assertArrayEquals("the first half is the gate", range(0, 48), values(out.tensors.get("blk.0.ffn_gate.weight")),
                0f);
        assertArrayEquals(range(48, 48), values(out.tensors.get("blk.0.ffn_up.weight")), 0f);
    }

    @Test
    public void ernieRejectsAnOddUpGateTensor() {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put("model.layers.0.mlp.up_gate_proj.weight", iota(5, 8));
        HfLoadContext ctx = context(Arch.ERNIE4_5, "Ernie4_5_ForCausalLM", ERNIE, t);
        assertThrows(GgufFormatException.class, () -> planAll(ctx, HfBaseDenseHooks::ernie45Tensors));
    }

    // ------------------------------------------------------------------ Phi-3

    private static final String PHI3 = "'hidden_size': 64, 'intermediate_size': 256, 'num_attention_heads': 4,"
            + " 'num_key_value_heads': 4, 'num_hidden_layers': 2, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
            + " 'original_max_position_embeddings': 4096, 'vocab_size': 32064, 'head_dim': 16";

    private static String longrope(String type, int maxPos) {
        return PHI3 + ", 'max_position_embeddings': " + maxPos + ", 'rope_scaling': {"
                + (type == null ? "" : "'type': '" + type + "', ")
                + "'long_factor': [1, 2, 3, 4, 5, 6, 7, 8], 'short_factor': [1, 1, 1, 1, 1, 1, 1, 1.5]}";
    }

    @Test
    public void phi3WritesOnlyItsOwnKeys() throws IOException {
        HfMetadata m = meta(context(Arch.PHI3, "Phi3ForCausalLM", longrope("longrope", 131072)
                + ", 'partial_rotary_factor': 0.75", Map.of()), HfBaseDenseHooks::phi3Meta);
        assertEquals("no generic keys (no key_length, no rope.scaling.type)", Arrays.asList("phi3.context_length",
                "phi3.rope.scaling.original_context_length", "phi3.embedding_length", "phi3.feed_forward_length",
                "phi3.block_count", "phi3.attention.head_count", "phi3.attention.head_count_kv",
                "phi3.attention.layer_norm_rms_epsilon", "phi3.rope.dimension_count", "phi3.rope.freq_base",
                "phi3.attention.sliding_window"), new ArrayList<>(m.keys()));
        assertEquals(131072, m.get("phi3.context_length"));
        assertEquals(4096, m.get("phi3.rope.scaling.original_context_length"));
        assertEquals("int(0.75 * 64) // 4", 12, m.get("phi3.rope.dimension_count"));
        assertEquals(10000f, m.get("phi3.rope.freq_base"));
        assertEquals("no sliding window is written as 0", 0, m.get("phi3.attention.sliding_window"));
        assertEquals(2, m.get("phi3.block_count"));
    }

    @Test
    public void phi3RequiresTheOriginalContext() {
        HfLoadContext ctx = context(Arch.PHI3, "Phi3ForCausalLM", "'hidden_size': 64, 'intermediate_size': 256,"
                + " 'num_attention_heads': 4, 'num_key_value_heads': 4, 'num_hidden_layers': 2,"
                + " 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0, 'max_position_embeddings': 4096", Map.of());
        assertThrows(GgufFormatException.class, () -> meta(ctx, HfBaseDenseHooks::phi3Meta));
    }

    @Test
    public void phi3LongRopeWritesBothFactorListsAndTheAttentionFactor() throws IOException {
        RecordingSink out = new RecordingSink();
        HfBaseDenseHooks.phi3Extra(context(Arch.PHI3, "Phi3ForCausalLM", longrope("longrope", 131072), Map.of()),
                out);
        assertEquals(List.of("rope_factors_long.weight", "rope_factors_short.weight"),
                new ArrayList<>(out.tensors.keySet()));
        assertArrayEquals(range(1, 8), values(out.tensors.get("rope_factors_long.weight")), 0f);
        assertArrayEquals(new float[]{1, 1, 1, 1, 1, 1, 1, 1.5f}, values(out.tensors.get("rope_factors_short.weight")),
                0f);
        float s = 131072f / 4096f;
        assertEquals((float) Math.sqrt(1 + Math.log(s) / Math.log(4096)), (Float) out.kv.get(
                "phi3.rope.scaling.attn_factor"), 1e-7f);
        assertEquals(GGUFValueType.FLOAT32, out.kvTypes.get("phi3.rope.scaling.attn_factor"));
    }

    @Test
    public void phi3AttentionFactorIsOneWithoutContextExtension() {
        RecordingSink out = new RecordingSink();
        HfBaseDenseHooks.phi3Extra(context(Arch.PHI3, "Phi3ForCausalLM", longrope("su", 4096), Map.of()), out);
        assertEquals(1f, (Float) out.kv.get("phi3.rope.scaling.attn_factor"), 0f);
        RecordingSink shorter = new RecordingSink();
        HfBaseDenseHooks.phi3Extra(context(Arch.PHI3, "Phi3ForCausalLM", longrope("yarn", 2048), Map.of()), shorter);
        assertEquals(1f, (Float) shorter.kv.get("phi3.rope.scaling.attn_factor"), 0f);
    }

    @Test
    public void phi3YarnAttentionFactor() {
        RecordingSink out = new RecordingSink();
        HfBaseDenseHooks.phi3Extra(context(Arch.PHI3, "Phi3ForCausalLM", longrope("yarn", 8192), Map.of()), out);
        assertEquals((float) (0.1 * Math.log(2.0) + 1.0), (Float) out.kv.get("phi3.rope.scaling.attn_factor"), 1e-7f);
        assertEquals(2, out.tensors.size());
    }

    @Test
    public void phi3RejectsAMissingOrUnsupportedScalingType() {
        assertThrows(GgufFormatException.class, () -> HfBaseDenseHooks.phi3Extra(context(Arch.PHI3,
                "Phi3ForCausalLM", longrope(null, 131072), Map.of()), new RecordingSink()));
        assertThrows(GgufFormatException.class, () -> HfBaseDenseHooks.phi3Extra(context(Arch.PHI3,
                "Phi3ForCausalLM", longrope("linear", 131072), Map.of()), new RecordingSink()));
    }

    @Test
    public void phi3WithoutLongFactorsHasNoExtras() {
        RecordingSink out = new RecordingSink();
        HfBaseDenseHooks.phi3Extra(context(Arch.PHI3, "Phi3ForCausalLM", PHI3 + ", 'max_position_embeddings': 4096",
                Map.of()), out);
        assertTrue(out.tensors.isEmpty());
        assertTrue(out.kv.isEmpty());
    }

    @Test
    public void phi3KeepsTheFusedProjections() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put("model.layers.0.self_attn.qkv_proj.weight", iota(12, 4));
        t.put("model.layers.0.mlp.gate_up_proj.weight", iota(8, 4));
        RecordingSink out = planAll(context(Arch.PHI3, "Phi3ForCausalLM", PHI3 + ", 'max_position_embeddings': 4096",
                t), HfTensorHook.DEFAULT);
        assertEquals(List.of("blk.0.attn_qkv.weight", "blk.0.ffn_up.weight"), new ArrayList<>(out.tensors.keySet()));
        assertArrayEquals(new long[]{12, 4}, out.tensors.get("blk.0.attn_qkv.weight").shape());
    }
}

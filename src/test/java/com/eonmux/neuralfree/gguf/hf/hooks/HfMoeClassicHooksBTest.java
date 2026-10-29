package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfMetadataHook;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.intValue;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.key;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfMoeClassicHooksB} against the Bailing MoE converters (LC/conversion/bailingmoe.py:15-219): the v1 fused
 * QKV split with the pair permutation and the L2-normalized norm head, the V2/Sarvam router-bias handling and
 * NextN counts, and the Dots1 fixup and the Ernie4.5-MoE metadata (ernie.py:86-97) and MTP filter (ernie.py:100-121).
 */
public class HfMoeClassicHooksBTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    private static final String BAILING_CONFIG = "'moe_intermediate_size': 2, 'num_shared_experts': 3,"
            + " 'first_k_dense_replace': 1, 'norm_topk_prob': true, 'num_local_experts': 2";

    private static final UnaryOperator<String> IDENTITY = UnaryOperator.identity();

    private Path dir() throws IOException {
        return tmp.newFolder().toPath();
    }

    private static HfMetadata meta(HfLoadContext ctx, HfMetadataHook hook) throws IOException {
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        hook.apply(ctx, kv, null);
        return kv.build();
    }

    // ------------------------------------------------------------------ BailingMoeModel

    @Test
    public void bailingFusedQkvIsSplitAndPairPermuted() throws IOException {
        String name = "model.layers.0.self_attn.query_key_value.weight";
        Map<String, RowSource> index = Collections.singletonMap(name, seq(16, 4));
        HfLoadContext ctx = context(dir(), Arch.BAILINGMOE, "BailingMoeForCausalLM", BAILING_CONFIG, index, false);
        RecordingSink sink = new RecordingSink();
        HfMoeClassicHooksB.bailingMoeTensors(ctx, name, ctx.index().get(name), sink);
        assertEquals(Arrays.asList("blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight"),
                List.copyOf(sink.tensors.keySet()));
        // the pair swap walks rows [0, 2, 1, 3] (q head 0) then [4, 6, 5, 7] (q head 1)
        assertArrayEquals(floats(0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15,
                16, 17, 18, 19, 24, 25, 26, 27, 20, 21, 22, 23, 28, 29, 30, 31),
                values(sink.tensors.get("blk.0.attn_q.weight")), 0f);
        // k runs through the single-group swap: rows [8, 10, 9, 11]
        assertArrayEquals(floats(32, 33, 34, 35, 40, 41, 42, 43, 36, 37, 38, 39, 44, 45, 46, 47),
                values(sink.tensors.get("blk.0.attn_k.weight")), 0f);
        assertArrayEquals(floats(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63),
                values(sink.tensors.get("blk.0.attn_v.weight")), 0f);
    }

    @Test
    public void bailingAttentionDenseMapsToAttnOut() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.BAILINGMOE, "BailingMoeForCausalLM", "'num_local_experts': 2");
        RecordingSink sink = new RecordingSink();
        HfMoeClassicHooksB.bailingMoeTensors(ctx, "model.layers.1.self_attn.attention.dense.weight", seq(8, 8), sink);
        assertEquals(Collections.singleton("blk.1.attn_output.weight"), sink.tensors.keySet());
    }

    @Test
    public void bailingOutputWeightIsL2NormalizedWhenNormHead() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.BAILINGMOE, "BailingMoeForCausalLM", "'norm_head': true");
        RecordingSink sink = new RecordingSink();
        HfMoeClassicHooksB.bailingMoeTensors(ctx, "output.weight", seq(2, 3), sink);
        // each column is divided by its L2 norm over dim 0 (plus 1e-7): norms 3, sqrt(17), sqrt(29)
        float[] expected = floats(0f / 3f, 1f / (float) Math.sqrt(17f), 2f / (float) Math.sqrt(29f),
                3f / 3f, 4f / (float) Math.sqrt(17f), 5f / (float) Math.sqrt(29f));
        assertArrayEquals(expected, values(sink.tensors.get("output.weight")), 1e-4f);
    }

    @Test
    public void bailingPerExpertWeightsStackWhenTheLayerCompletes() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        for (int x = 0; x < 2; x++) {
            index.put("model.layers.0.mlp.experts." + x + ".down_proj.weight", filled(10 * x, 8, 2));
            index.put("model.layers.0.mlp.experts." + x + ".gate_proj.weight", filled(10 * x + 1, 2, 8));
            index.put("model.layers.0.mlp.experts." + x + ".up_proj.weight", filled(10 * x + 2, 2, 8));
        }
        index.put("model.layers.0.mlp.gate.weight", seq(2, 8));
        HfLoadContext ctx = context(dir(), Arch.BAILINGMOE, "BailingMoeForCausalLM", BAILING_CONFIG, index, false);
        RecordingSink sink = new RecordingSink();
        for (String name : ctx.index().names()) {
            if (!sink.isConsumed(name)) {
                HfMoeClassicHooksB.bailingMoeTensors(ctx, name, ctx.index().get(name), sink);
            }
        }
        HfMoeClassicHooksB.bailingMoeExtra(ctx, sink);
        assertArrayEquals(new long[]{2, 8, 2}, sink.tensors.get("blk.0.ffn_down_exps.weight").shape());
        assertArrayEquals(new long[]{2, 2, 8}, sink.tensors.get("blk.0.ffn_gate_exps.weight").shape());
        float[] gate = new float[32];
        for (int i = 0; i < 16; i++) {
            gate[i] = 1;
            gate[16 + i] = 11;
        }
        assertArrayEquals(gate, values(sink.tensors.get("blk.0.ffn_gate_exps.weight")), 0f);
    }

    @Test
    public void bailingLeftoverExpertsFailTheCompletenessCheck() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        String w1 = "model.layers.0.mlp.experts.0.down_proj.weight";
        for (int x = 0; x < 2; x++) {
            for (String part : new String[]{"down_proj", "gate_proj", "up_proj"}) {
                index.put("model.layers.0.mlp.experts." + x + "." + part + ".weight", seq(16));
            }
        }
        HfLoadContext ctx = context(dir(), Arch.BAILINGMOE, "BailingMoeForCausalLM", BAILING_CONFIG, index, false);
        RecordingSink sink = new RecordingSink();
        // one member offered of the six planned: the leftovers fail the extra check
        HfMoeClassicHooksB.bailingMoeTensors(ctx, w1, ctx.index().get(w1), sink);
        assertThrows(GgufFormatException.class, () -> HfMoeClassicHooksB.bailingMoeExtra(ctx, sink));
    }

    // ------------------------------------------------------------------ BailingMoeV2Model

    @Test
    public void v2RouterBiasIsRenamedButNotNormalized() {
        assertEquals("model.layers.0.mlp.expert_bias.bias",
                HfMoeClassicHooksB.BAILING_V2_FILTER.filter("model.layers.0.mlp.expert_bias", IDENTITY));
        assertEquals("model.layers.0.mlp.experts.0.gate_proj.weight",
                HfMoeClassicHooksB.BAILING_V2_FILTER.filter("model.layers.0.mlp.experts.0.gate_proj.weight",
                        IDENTITY));
    }

    @Test
    public void v2TensorHookPassesTheRenamedBiasThroughUnchanged() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.BAILINGMOE2, "BailingMoeV2ForCausalLM", "");
        RecordingSink sink = new RecordingSink();
        HfMoeClassicHooksB.bailingMoeV2Tensors(ctx, "model.layers.0.mlp.expert_bias.bias", seq(2), sink);
        assertArrayEquals(floats(0, 1), values(sink.tensors.get("blk.0.exp_probs_b.bias")), 0f);
    }

    @Test
    public void sarvamRouterBiasIsZeroMeanNormalized() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.BAILINGMOE2, "SarvamMoEForCausalLM", BAILING_CONFIG);
        RecordingSink sink = new RecordingSink();
        HfMoeClassicHooksB.sarvamMoeTensors(ctx, "model.layers.0.mlp.expert_bias.bias",
                com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors.fromFloats(new long[]{2, 1}, new float[]{4, 2}), sink);
        assertArrayEquals(floats(1, -1), values(sink.tensors.get("blk.0.exp_probs_b.bias")), 0f);
    }

    @Test
    public void v2MetadataUsesThePartialRotaryFactorAndTheSharedWidth() throws IOException {
        HfLoadContext shared = context(dir(), Arch.BAILINGMOE2, "BailingMoeV2ForCausalLM",
                BAILING_CONFIG + ", 'moe_shared_expert_intermediate_size': 10, 'num_nextn_predict_layers': 1");
        HfMetadata m = meta(shared, HfMoeClassicHooksB::bailingMoeV2Meta);
        // rope width 4 with the 0.5 default partial factor
        assertEquals(2, intValue(m, key(GgufKeys.ROPE_DIMENSION_COUNT, Arch.BAILINGMOE2)));
        assertEquals(1, intValue(m, key(GgufKeys.LEADING_DENSE_BLOCK_COUNT, Arch.BAILINGMOE2)));
        assertEquals(2, intValue(m, key(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, Arch.BAILINGMOE2)));
        assertEquals(10, intValue(m, key(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, Arch.BAILINGMOE2)));
        assertEquals(3, intValue(m, key(GgufKeys.EXPERT_SHARED_COUNT, Arch.BAILINGMOE2)));
        assertEquals(1, intValue(m, key(GgufKeys.NEXTN_PREDICT_LAYERS, Arch.BAILINGMOE2)));
        assertEquals(1, HfMoeClassicHooksB.nextnPredictLayers(shared.config()));
        // without the explicit shared width the converter multiplies MoE width by shared experts
        HfLoadContext derived = context(dir(), Arch.BAILINGMOE2, "BailingMoeV2ForCausalLM", BAILING_CONFIG);
        HfMetadata m2 = meta(derived, HfMoeClassicHooksB::bailingMoeV2Meta);
        assertEquals(6, intValue(m2, key(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, Arch.BAILINGMOE2)));
        assertEquals(0, HfMoeClassicHooksB.nextnPredictLayers(derived.config()));
    }

    @Test
    public void sarvamMetadataOverridesThePartialRotaryValue() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.BAILINGMOE2, "SarvamMoEForCausalLM", BAILING_CONFIG);
        HfMetadata m = meta(ctx, HfMoeClassicHooksB::sarvamMoeMeta);
        assertEquals(4, intValue(m, key(GgufKeys.ROPE_DIMENSION_COUNT, Arch.BAILINGMOE2)));
    }

    // ------------------------------------------------------------------ Dots1Model, Ernie4_5MoeModel

    @Test
    public void dots1AliasesNumExpertsFromTheRoutedCount() throws IOException {
        HfLoadContext without = context(dir(), Arch.DOTS1, "Dots1ForCausalLM", "");
        GgufFormatException missing = assertThrows(GgufFormatException.class,
                () -> HfMoeClassicHooksB.dots1Fixup(without.config()));
        assertTrue(missing.getMessage(), missing.getMessage().contains("n_routed_experts"));
        HfLoadContext ctx = context(dir(), Arch.DOTS1, "Dots1ForCausalLM", "'n_routed_experts': 6");
        assertEquals(6, HfMoeClassicHooksB.dots1Fixup(ctx.config()).findInt("num_experts").getAsInt());
    }

    @Test
    public void ernie45MoeSharedFeedForwardIsTheIntermediateSizePerKvHeadAsWritten() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.ERNIE4_5_MOE, "Ernie4_5_MoeForCausalLM",
                "'moe_num_experts': 4, 'moe_k': 2, 'moe_layer_interval': 2, 'moe_layer_start_index': 1,"
                        + " 'moe_intermediate_size': 8, 'moe_num_shared_experts': 2, 'intermediate_size': 20,"
                        + " 'num_key_value_heads': 4");
        HfMetadata m = meta(ctx, HfMoeClassicHooksB::ernie45MoeMeta);
        assertEquals(4, intValue(m, key(GgufKeys.EXPERT_COUNT, Arch.ERNIE4_5_MOE)));
        assertEquals(2, intValue(m, key(GgufKeys.EXPERT_USED_COUNT, Arch.ERNIE4_5_MOE)));
        assertEquals(2, intValue(m, key(GgufKeys.INTERLEAVE_MOE_LAYER_STEP, Arch.ERNIE4_5_MOE)));
        assertEquals(1, intValue(m, key(GgufKeys.LEADING_DENSE_BLOCK_COUNT, Arch.ERNIE4_5_MOE)));
        assertEquals(8, intValue(m, key(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, Arch.ERNIE4_5_MOE)));
        // reproduced as written at ernie.py:96-97: the intermediate size per kv head, 20 / 4 = 5
        assertEquals(5, intValue(m, key(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, Arch.ERNIE4_5_MOE)));
    }

    @Test
    public void ernie45MoeFilterDropsTheMtpTensors() {
        assertNull(HfMoeClassicHooksB.ernie45MoeFilter("model.mtp_block.0.weight", IDENTITY));
        assertNull(HfMoeClassicHooksB.ernie45MoeFilter("model.mtp_emb_norm.0.weight", IDENTITY));
        assertNull(HfMoeClassicHooksB.ernie45MoeFilter("model.mtp_hidden_norm.1.weight", IDENTITY));
        assertNull(HfMoeClassicHooksB.ernie45MoeFilter("model.mtp_linear_proj.2.weight", IDENTITY));
        assertEquals("model.layers.0.mlp.down_proj.weight",
                HfMoeClassicHooksB.ernie45MoeFilter("model.layers.0.mlp.down_proj.weight", IDENTITY));
    }

    /** A {@code [rows, cols]} F32 tensor filled with {@code value}. */
    private static RowSource filled(float value, long rows, long cols) {
        float[] v = new float[(int) (rows * cols)];
        Arrays.fill(v, value);
        return RopeTensors.fromFloats(new long[]{rows, cols}, v);
    }
}

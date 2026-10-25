package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.intValue;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.key;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.values;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.writeJson;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfQwenLinearHooks} against LC/conversion/qwen.py: the {@code Qwen3NextModel} metadata and tensor
 * transforms (QKVZ regrouping, the unsplit {@code in_proj_ba}, {@code A_log}, {@code dt_bias}, {@code conv1d}, the
 * +1 norms), the {@code _LinearAttentionVReorderBase} grouped-to-tiled value-head reorder of rows and column groups
 * with its Hadamard-folded exception, and the Qwen3.5 rope sections and pooling.
 */
public class HfQwenLinearHooksTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    /** Two key heads of one entry, four value heads of {@code hv} entries, two layers: one linear, one full. */
    private static String linearConfig(int nK, int nV, int hv) {
        return "'linear_num_key_heads': " + nK + ", 'linear_num_value_heads': " + nV + ", 'linear_key_head_dim': 1,"
                + " 'linear_value_head_dim': " + hv + ", 'linear_conv_kernel_dim': 4, 'num_experts': 2,"
                + " 'num_experts_per_tok': 1, 'moe_intermediate_size': 2, 'shared_expert_intermediate_size': 6,"
                + " 'layer_types': ['linear_attention', 'full_attention'], 'full_attention_interval': 2,"
                + " 'head_dim': 8";
    }

    private HfLoadContext context(Arch arch, String hfArch, String config, boolean includeMtp) throws IOException {
        return context(tmp.newFolder().toPath(), arch, hfArch, config, includeMtp);
    }

    private static HfLoadContext context(Path dir, Arch arch, String hfArch, String config, boolean includeMtp)
            throws IOException {
        return HfQwenHooksTest.context(dir, arch, hfArch, config, Collections.emptyMap(), includeMtp);
    }

    private HfLoadContext next() throws IOException {
        return context(Arch.QWEN3NEXT, "Qwen3NextForCausalLM", linearConfig(2, 4, 1), false);
    }

    /** A Qwen3.5 context with two key heads, four value heads of two entries each. */
    private HfLoadContext qwen35(Path dir) throws IOException {
        return context(dir, Arch.QWEN35, "Qwen3_5ForConditionalGeneration", linearConfig(2, 4, 2), false);
    }

    private static RecordingSink next(HfLoadContext ctx, String name, RowSource src) throws IOException {
        RecordingSink sink = new RecordingSink();
        HfQwenLinearHooks.qwen3NextTensors(ctx, name, src, sink);
        return sink;
    }

    private static RecordingSink reorder(HfLoadContext ctx, String name, RowSource src) throws IOException {
        RecordingSink sink = new RecordingSink();
        HfQwenLinearHooks.vReorderTensors(ctx, name, src, sink);
        return sink;
    }

    private static float[] only(RecordingSink sink, String ggufName) throws IOException {
        assertEquals(Collections.singleton(ggufName), sink.tensors.keySet());
        return values(sink.tensors.get(ggufName));
    }

    // ------------------------------------------------------------------ Qwen3NextModel tensors

    @Test
    public void fusedQkvzOfTwoKeyHeadsIsRegroupedIntoQkvAndGate() throws IOException {
        RecordingSink sink = next(next(), "model.layers.0.linear_attn.in_proj_qkvz.weight", seq(12, 1));
        assertEquals(Arrays.asList("blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"),
                List.copyOf(sink.tensors.keySet()));
        assertArrayEquals(floats(0, 6, 1, 7, 2, 3, 8, 9), values(sink.tensors.get("blk.0.attn_qkv.weight")), 0f);
        assertArrayEquals(floats(4, 5, 10, 11), values(sink.tensors.get("blk.0.attn_gate.weight")), 0f);
    }

    @Test
    public void fusedQkvzWhoseValueHeadsDoNotGroupIsRejected() throws IOException {
        HfLoadContext ctx = context(Arch.QWEN3NEXT, "Qwen3NextForCausalLM", linearConfig(3, 4, 1), false);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> next(ctx, "model.layers.0.linear_attn.in_proj_qkvz.weight", seq(14, 1)));
        assertTrue(e.getMessage(), e.getMessage().endsWith("4 linear-attention value heads do not group over 3 key"
                + " heads"));
    }

    @Test
    public void fusedBetaAlphaStaysWhole() throws IOException {
        RecordingSink sink = next(next(), "model.layers.0.linear_attn.in_proj_ba.weight", seq(4, 8));
        assertArrayEquals(values(seq(4, 8)), only(sink, "blk.0.ssm_ba.weight"), 0f);
        assertArrayEquals(new long[]{4, 8}, sink.tensors.get("blk.0.ssm_ba.weight").shape());
    }

    @Test
    public void aLogBecomesNegativeExp() throws IOException {
        float[] v = only(next(next(), "model.layers.0.linear_attn.A_log", seq(2)), "blk.0.ssm_a");
        assertArrayEquals(floats(-1f, (float) -Math.E), v, 1e-6f);
    }

    @Test
    public void dtBiasIsRenamedToTheDtProjectionBias() throws IOException {
        assertArrayEquals(floats(0, 1, 2, 3),
                only(next(next(), "model.layers.0.linear_attn.dt_bias", seq(4)), "blk.0.ssm_dt.bias"), 0f);
    }

    @Test
    public void conv1dIsSqueezed() throws IOException {
        RecordingSink sink = next(next(), "model.layers.0.linear_attn.conv1d.weight", seq(6, 1, 4));
        assertArrayEquals(new long[]{6, 4}, sink.tensors.get("blk.0.ssm_conv1d.weight").shape());
    }

    @Test
    public void normsGainOneExceptTheLinearAttentionNorm() throws IOException {
        HfLoadContext ctx = next();
        assertArrayEquals(floats(1, 2),
                only(next(ctx, "model.layers.0.input_layernorm.weight", seq(2)), "blk.0.attn_norm.weight"), 0f);
        assertArrayEquals(floats(0, 1),
                only(next(ctx, "model.layers.0.linear_attn.norm.weight", seq(2)), "blk.0.ssm_norm.weight"), 0f);
    }

    @Test
    public void expertTensorsFallThroughToTheMoeHook() throws IOException {
        RecordingSink sink = next(next(), "model.layers.1.mlp.experts.gate_up_proj", seq(2, 4, 8));
        assertEquals(Arrays.asList("blk.1.ffn_gate_exps.weight", "blk.1.ffn_up_exps.weight"),
                List.copyOf(sink.tensors.keySet()));
    }

    // ------------------------------------------------------------------ value-head reorder

    @Test
    public void qkvReordersOnlyTheValueRows() throws IOException {
        float[] v = only(reorder(qwen35(tmp.newFolder().toPath()), "model.layers.0.linear_attn.in_proj_qkv.weight",
                seq(12, 1)), "blk.0.attn_qkv.weight");
        assertArrayEquals(floats(0, 1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11), v, 0f);
    }

    @Test
    public void gateRowsAreReorderedPerValueHead() throws IOException {
        float[] v = only(reorder(qwen35(tmp.newFolder().toPath()), "model.layers.0.linear_attn.in_proj_z.weight",
                seq(8, 1)), "blk.0.attn_gate.weight");
        assertArrayEquals(floats(0, 1, 4, 5, 2, 3, 6, 7), v, 0f);
    }

    @Test
    public void betaAndAlphaRowsAreReorderedOnePerHead() throws IOException {
        HfLoadContext ctx = qwen35(tmp.newFolder().toPath());
        float[] tiled = floats(0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11);
        assertArrayEquals(tiled, only(reorder(ctx, "model.layers.0.linear_attn.in_proj_b.weight", seq(4, 3)),
                "blk.0.ssm_beta.weight"), 0f);
        assertArrayEquals(tiled, only(reorder(ctx, "model.layers.0.linear_attn.in_proj_a.weight", seq(4, 3)),
                "blk.0.ssm_alpha.weight"), 0f);
    }

    @Test
    public void perHeadParametersAreReorderedBeforeTheirTransforms() throws IOException {
        HfLoadContext ctx = qwen35(tmp.newFolder().toPath());
        assertArrayEquals(floats(-1f, (float) -Math.exp(2), (float) -Math.E, (float) -Math.exp(3)),
                only(reorder(ctx, "model.layers.0.linear_attn.A_log", seq(4)), "blk.0.ssm_a"), 1e-5f);
        assertArrayEquals(floats(0, 2, 1, 3),
                only(reorder(ctx, "model.layers.0.linear_attn.dt_bias", seq(4)), "blk.0.ssm_dt.bias"), 0f);
        assertArrayEquals(floats(0, 2, 1, 3, 4, 6, 5, 7),
                only(reorder(ctx, "model.layers.0.linear_attn.dt_proj.weight", seq(2, 4)), "blk.0.ssm_dt.weight"),
                0f);
    }

    @Test
    public void conv1dReordersOnlyTheValueChannels() throws IOException {
        RecordingSink sink = reorder(qwen35(tmp.newFolder().toPath()), "model.layers.0.linear_attn.conv1d.weight",
                seq(12, 1, 2));
        RowSource conv = sink.tensors.get("blk.0.ssm_conv1d.weight");
        assertArrayEquals(new long[]{12, 2}, conv.shape());
        assertArrayEquals(floats(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15, 20, 21, 22,
                23), values(conv), 0f);
    }

    @Test
    public void outProjectionColumnGroupsAreReordered() throws IOException {
        float[] v = only(reorder(qwen35(tmp.newFolder().toPath()), "model.layers.0.linear_attn.out_proj.weight",
                seq(2, 8)), "blk.0.ssm_out.weight");
        assertArrayEquals(floats(0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15), v, 0f);
    }

    @Test
    public void hadamardFoldedOutProjectionKeepsTheGroupedOrderAndIsRecordedOnce() throws IOException {
        Path dir = tmp.newFolder().toPath();
        writeJson(dir, "hadamard_packing.json", "{'tensors': [{'name': 'model.layers.0.linear_attn.out_proj.weight'},"
                + " {'name': 'layers.1.linear_attn.out_proj.weight'}]}");
        HfLoadContext ctx = qwen35(dir);
        RecordingSink sink = new RecordingSink();
        HfQwenLinearHooks.vReorderTensors(ctx, "model.layers.0.linear_attn.out_proj.weight", seq(2, 8), sink);
        HfQwenLinearHooks.vReorderTensors(ctx, "model.layers.1.linear_attn.out_proj.weight", seq(2, 8), sink);
        assertArrayEquals(values(seq(2, 8)), values(sink.tensors.get("blk.0.ssm_out.weight")), 0f);
        assertArrayEquals(values(seq(2, 8)), values(sink.tensors.get("blk.1.ssm_out.weight")), 0f);
        assertEquals(Collections.singleton(GgufKeys.PRISM_HADAMARD_GDN_V_GROUPED), sink.kv.keySet());
        assertEquals(Boolean.TRUE, sink.kv.get(GgufKeys.PRISM_HADAMARD_GDN_V_GROUPED));
        assertEquals(GGUFValueType.BOOL, sink.kvTypes.get(GgufKeys.PRISM_HADAMARD_GDN_V_GROUPED));

        RecordingSink other = new RecordingSink();
        HfQwenLinearHooks.vReorderTensors(ctx, "model.layers.0.linear_attn.in_proj_z.weight", seq(8, 1), other);
        assertArrayEquals(floats(0, 1, 4, 5, 2, 3, 6, 7), values(other.tensors.get("blk.0.attn_gate.weight")), 0f);
    }

    @Test
    public void equalKeyAndValueHeadsAreNotReordered() throws IOException {
        HfLoadContext ctx = context(Arch.QWEN35, "Qwen3_5ForConditionalGeneration", linearConfig(4, 4, 2), false);
        assertArrayEquals(values(seq(8, 1)), only(reorder(ctx, "model.layers.0.linear_attn.in_proj_z.weight",
                seq(8, 1)), "blk.0.attn_gate.weight"), 0f);
    }

    @Test
    public void valueHeadsThatAreNotAMultipleOfTheKeyHeadsAreRejected() throws IOException {
        HfLoadContext ctx = context(Arch.QWEN35, "Qwen3_5ForConditionalGeneration", linearConfig(3, 4, 2), false);
        assertThrows(GgufFormatException.class, () -> reorder(ctx, "model.layers.0.linear_attn.in_proj_z.weight",
                seq(8, 1)));
    }

    @Test
    public void tensorsThatDoNotHoldTheValueHeadsAreRejected() throws IOException {
        HfLoadContext ctx = qwen35(tmp.newFolder().toPath());
        GgufFormatException rows = assertThrows(GgufFormatException.class,
                () -> reorder(ctx, "model.layers.0.linear_attn.in_proj_z.weight", seq(6, 1)));
        assertTrue(rows.getMessage(), rows.getMessage().contains("does not hold 4 value heads of 2"));
        assertThrows(GgufFormatException.class,
                () -> reorder(ctx, "model.layers.0.linear_attn.in_proj_qkv.weight", seq(10, 1)));
        assertThrows(GgufFormatException.class,
                () -> reorder(ctx, "model.layers.0.linear_attn.out_proj.weight", seq(2, 6)));
    }

    // ------------------------------------------------------------------ metadata

    @Test
    public void qwen3NextMetadataWritesTheSsmLayoutAndRope() throws IOException {
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN3NEXT.ggufName());
        HfQwenLinearHooks.qwen3NextMeta(next(), kv, null);
        HfMetadata m = kv.build();
        Arch a = Arch.QWEN3NEXT;
        assertEquals(2, intValue(m, key(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, a)));
        assertEquals(4, intValue(m, key(GgufKeys.SSM_CONV_KERNEL, a)));
        assertEquals(1, intValue(m, key(GgufKeys.SSM_STATE_SIZE, a)));
        assertEquals(2, intValue(m, key(GgufKeys.SSM_GROUP_COUNT, a)));
        assertEquals(4, intValue(m, key(GgufKeys.SSM_TIME_STEP_RANK, a)));
        assertEquals(4, intValue(m, key(GgufKeys.SSM_INNER_SIZE, a)));
        assertEquals(Arrays.asList(true, false), m.get(key(GgufKeys.ATTENTION_RECURRENT_LAYERS, a)));
        assertEquals(2, intValue(m, key(GgufKeys.FULL_ATTENTION_INTERVAL, a)));
        assertEquals(2, intValue(m, key(GgufKeys.ROPE_DIMENSION_COUNT, a)));
        assertFalse(m.has(key(GgufKeys.NEXTN_PREDICT_LAYERS, a)));
    }

    @Test
    public void mtpLayersAreAttentionLayersAndSetTheNextnCount() throws IOException {
        HfLoadContext ctx = context(Arch.QWEN3NEXT, "Qwen3NextForCausalLM", linearConfig(2, 4, 1), true);
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN3NEXT.ggufName());
        HfQwenLinearHooks.qwen3NextMeta(ctx.withBlockCount(3), kv, null);
        HfMetadata m = kv.build();
        assertEquals(Arrays.asList(true, false, false),
                m.get(key(GgufKeys.ATTENTION_RECURRENT_LAYERS, Arch.QWEN3NEXT)));
        assertEquals(1, intValue(m, key(GgufKeys.NEXTN_PREDICT_LAYERS, Arch.QWEN3NEXT)));
    }

    @Test
    public void partialRotaryFactorScalesTheRopeDimension() throws IOException {
        HfLoadContext ctx = context(Arch.QWEN3NEXT, "Qwen3NextForCausalLM",
                linearConfig(2, 4, 1) + ", 'partial_rotary_factor': 0.5", false);
        assertEquals(4L, HfQwenLinearHooks.ropeDimensionCount(ctx.config()));
    }

    @Test
    public void layerTypesOfAnotherLengthAreRejected() throws IOException {
        HfLoadContext ctx = context(Arch.QWEN3NEXT, "Qwen3NextForCausalLM", linearConfig(2, 4, 1)
                .replace("'layer_types': ['linear_attention', 'full_attention']",
                        "'layer_types': ['linear_attention', 'linear_attention', 'full_attention']"), false);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfQwenLinearHooks.qwen3NextMeta(ctx, new HfMetadataBuilder(Arch.QWEN3NEXT.ggufName()), null));
        assertTrue(e.getMessage(),
                e.getMessage().endsWith("layer_types has 3 entries, expected num_hidden_layers (2)"));
    }

    @Test
    public void qwen35WritesTheDefaultRopeSectionsAndPooling() throws IOException {
        Path dir = tmp.newFolder().toPath();
        writeJson(dir, "embedding_config.json", "{'pooling': 'last_token'}");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN35.ggufName());
        HfQwenLinearHooks.qwen35Meta(qwen35(dir), kv, null);
        HfMetadata m = kv.build();
        assertEquals(HfQwenLinearHooks.DEFAULT_MROPE_SECTIONS,
                m.get(key(GgufKeys.ROPE_DIMENSION_SECTIONS, Arch.QWEN35)));
        assertEquals(Arrays.asList(11, 11, 10, 0), HfQwenLinearHooks.DEFAULT_MROPE_SECTIONS);
        assertTrue(m.has(key(GgufKeys.POOLING_TYPE, Arch.QWEN35)));
        assertEquals(8, intValue(m, key(GgufKeys.SSM_INNER_SIZE, Arch.QWEN35)));
    }

    @Test
    public void qwen35KeepsTheCheckpointRopeSections() throws IOException {
        HfLoadContext ctx = context(Arch.QWEN35, "Qwen3_5ForConditionalGeneration", linearConfig(2, 4, 2)
                + ", 'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000.0,"
                + " 'mrope_section': [2, 1, 1]}", false);
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN35.ggufName());
        HfQwenLinearHooks.mropeSections(ctx, kv);
        assertFalse(kv.build().has(key(GgufKeys.ROPE_DIMENSION_SECTIONS, Arch.QWEN35)));

        HfMetadataBuilder full = new HfMetadataBuilder(Arch.QWEN35.ggufName());
        HfQwenLinearHooks.qwen35Meta(ctx, full, null);
        assertEquals(Arrays.asList(2, 1, 1, 0), full.build().get(key(GgufKeys.ROPE_DIMENSION_SECTIONS, Arch.QWEN35)));
    }

    @Test
    public void qwen35MoeWritesTheRopeSectionsWithoutPooling() throws IOException {
        Path dir = tmp.newFolder().toPath();
        writeJson(dir, "embedding_config.json", "{'pooling': 'mean'}");
        HfLoadContext ctx = context(dir, Arch.QWEN35MOE, "Qwen3_5MoeForConditionalGeneration",
                linearConfig(2, 4, 2), false);
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN35MOE.ggufName());
        HfQwenLinearHooks.qwen35MoeMeta(ctx, kv, null);
        HfMetadata m = kv.build();
        assertEquals(HfQwenLinearHooks.DEFAULT_MROPE_SECTIONS,
                m.get(key(GgufKeys.ROPE_DIMENSION_SECTIONS, Arch.QWEN35MOE)));
        assertEquals(2, intValue(m, key(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, Arch.QWEN35MOE)));
        assertFalse(m.has(key(GgufKeys.POOLING_TYPE, Arch.QWEN35MOE)));
    }
}

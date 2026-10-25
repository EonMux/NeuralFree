package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.names.HfTensorNameMap;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.function.ToIntFunction;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfDeepseekV4Hooks} and {@link HfDeepseekV4Names}: the native name table against llama.cpp's GGUF names, the
 * RoPE fixup, the MTP renaming filter and block count, the tensor routing, the hash tables, stacked experts and MTP
 * {@code eh_proj} concatenation of the extra tensors, the forced types (LC/conversion/deepseek.py:894-906) and the
 * metadata.
 */
public class HfDeepseekV4HooksTest {

    private static final String V4 = "'architectures': ['DeepseekV4ForCausalLM'], 'hidden_size': 64,"
            + " 'intermediate_size': 128, 'moe_intermediate_size': 32, 'num_attention_heads': 4,"
            + " 'num_key_value_heads': 1, 'head_dim': 32, 'num_hidden_layers': 2, 'num_nextn_predict_layers': 1,"
            + " 'max_position_embeddings': 1024, 'rms_norm_eps': 1e-06, 'vocab_size': 100, 'rope_theta': 10000.0,"
            + " 'qk_rope_head_dim': 16, 'q_lora_rank': 24, 'sliding_window': 128, 'n_routed_experts': 2,"
            + " 'num_experts_per_tok': 1, 'n_shared_experts': 1, 'routed_scaling_factor': 1.5,"
            + " 'norm_topk_prob': true, 'swiglu_limit': 10.0, 'index_n_heads': 2, 'index_head_dim': 16,"
            + " 'index_topk': 8, 'o_groups': 2, 'o_lora_rank': 16, 'compress_ratios': [0, 4, 128],"
            + " 'compress_rope_theta': 160000.0, 'hc_mult': 4, 'hc_sinkhorn_iters': 20, 'hc_eps': 1e-06,"
            + " 'num_hash_layers': 1, 'scoring_func': 'sqrtsoftplus'";

    private static HfConfig config(String singleQuotedJson) {
        return HfConfigLoader.fromJson(Path.of("."), "{" + singleQuotedJson.replace('\'', '"') + "}");
    }

    /** The entry as the architecture table registers it, with the block count of a view that keeps the MTP layer. */
    private static HfArchEntry entry() {
        return HfArchEntry.of(List.of("DeepseekV4ForCausalLM"), HfArchResolver.of(Arch.DEEPSEEK4),
                HfDeepseekV4Hooks::meta, HfDeepseekV4Hooks::tensors, HfDeepseekV4Hooks::vocab)
                .withExtra(HfDeepseekV4Hooks::extra)
                .withConfigFixup(HfDeepseekV4Hooks::configFixup)
                .withForceType(HfDeepseekV4Hooks::forceType)
                .withNameFilter(HfDeepseekV4Hooks::filterWithoutMtp)
                .withBlockCount(c -> c.requireInt("num_hidden_layers") + c.findInt(0, "num_nextn_predict_layers"));
    }

    private static HfLoadContext context(Map<String, RowSource> index, ModelSourceOptions options) {
        return HfLoadContext.builder(config(V4), entry(), Arch.DEEPSEEK4).options(options)
                .index(HfTensorIndex.of(index)).build();
    }

    private static HfLoadContext context(Map<String, RowSource> index) {
        return context(index, ModelSourceOptions.defaults());
    }

    @Test
    public void nameTableRoundTripsThroughTheGgufNames() {
        HfTensorNameMap names = HfTensorNameMap.forArch(Arch.DEEPSEEK4, 4);
        Map<String, String> sample = new LinkedHashMap<>();
        sample.put("embed.weight", "token_embd.weight");
        sample.put("norm.weight", "output_norm.weight");
        sample.put("head.weight", "output.weight");
        sample.put("hc_head_fn", "output_hc_fn.weight");
        sample.put("layers.0.attn.attn_sink", "blk.0.attn_sinks.weight");
        sample.put("layers.2.attn.wkv.weight", "blk.2.attn_kv.weight");
        sample.put("layers.2.attn.kv_norm.weight", "blk.2.attn_kv_a_norm.weight");
        sample.put("layers.1.attn.wo_a.weight", "blk.1.attn_output_a.weight");
        sample.put("layers.1.attn.compressor.ape", "blk.1.attn_compressor_ape.weight");
        sample.put("layers.0.attn.indexer.weights_proj.weight", "blk.0.indexer.proj.weight");
        sample.put("layers.1.hc_attn_fn", "blk.1.hc_attn_fn.weight");
        sample.put("layers.1.ffn.gate.bias", "blk.1.exp_probs_b.bias");
        sample.put("layers.1.ffn.gate.bias_vl", "blk.1.exp_probs_b_vl.bias");
        sample.put("layers.0.ffn.gate.tid2eid", "blk.0.ffn_gate_tid2eid.weight");
        sample.put("layers.3.nextn.eh_proj.weight", "blk.3.nextn.eh_proj.weight");
        sample.put("layers.3.nextn.shared_head_norm.weight", "blk.3.nextn.shared_head_norm.weight");
        sample.put("layers.2.ffn.experts.7.w3.scale", "blk.2.ffn_gate_exps.weight");
        for (Map.Entry<String, String> e : sample.entrySet()) {
            int bid = HfLoadContext.bidOf(e.getKey());
            HfDeepseekV4Names.Mapped m = HfDeepseekV4Names.map(e.getKey(), bid);
            String gguf = HfDeepseekV4Names.ggufName(names, e.getKey(), bid);
            assertEquals(e.getKey(), e.getValue(), gguf);
            assertTrue(e.getKey(), names.matches(gguf, m.tensor(), m.root() ? -1 : bid, m.suffix()));
        }
        for (HfDeepseekV4Names.Mapped m : HfDeepseekV4Names.rootTable().values()) {
            assertTrue(m.tensor(), m.root() && names.hasTensor(m.tensor()));
        }
        for (HfDeepseekV4Names.Mapped m : HfDeepseekV4Names.layerTable().values()) {
            assertTrue(m.tensor(), !m.root() && names.hasTensor(m.tensor()));
        }
        assertThrows(GgufFormatException.class, () -> HfDeepseekV4Names.map("layers.1.attn.wq_a.scale", 1));
        assertThrows(GgufFormatException.class, () -> HfDeepseekV4Names.map("layers.1.attn.wq_a.weight", 2));
        assertThrows(GgufFormatException.class, () -> HfDeepseekV4Names.map("model.embed_tokens.weight", -1));
    }

    @Test
    public void ropeScalingFillsMissingRopeParameters() {
        HfConfig cfg = config("'architectures': ['DeepseekV4ForCausalLM'], 'num_hidden_layers': 2,"
                + " 'rope_parameters': {'rope_theta': 10000.0}, 'rope_scaling': {'type': 'yarn', 'factor': 16.0,"
                + " 'original_max_position_embeddings': 65536}");
        HfConfig fixed = HfDeepseekV4Hooks.configFixup(cfg);
        assertEquals("yarn", fixed.rope().global().type().get());
        assertEquals(16.0, fixed.rope().global().factor().getAsDouble(), 0);
        assertEquals(65536, fixed.rope().global().originalMaxPositionEmbeddings().getAsInt());
        assertEquals(10000.0, fixed.rope().theta().getAsDouble(), 0);

        HfConfig typed = config("'architectures': ['DeepseekV4ForCausalLM'], 'num_hidden_layers': 2,"
                + " 'rope_scaling': {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 4096}");
        assertSame(typed, HfDeepseekV4Hooks.configFixup(typed));
    }

    @Test
    public void filterRenamesMtpIntoExtraBlocks() {
        HfNameFilter keep = HfDeepseekV4Hooks.filter(context(Map.of()));
        assertEquals("layers.2.nextn.e_proj.weight", keep.filter("mtp.0.e_proj.weight", n -> n));
        assertEquals("layers.2.nextn.h_proj.scale", keep.filter("mtp.0.h_proj.scale", n -> n));
        assertEquals("layers.2.nextn.enorm.weight", keep.filter("mtp.0.enorm.weight", n -> n));
        assertEquals("layers.2.nextn.shared_head_norm.weight", keep.filter("mtp.0.norm.weight", n -> n));
        assertEquals("layers.2.nextn.hc_head_fn", keep.filter("mtp.0.hc_head_fn", n -> n));
        assertEquals("layers.2.attn.wq_a.weight", keep.filter("mtp.0.attn.wq_a.weight", n -> n));
        assertEquals("embed.weight!", keep.filter("embed.weight", n -> n + "!"));
        assertNull(keep.filter("aligner.proj.weight", n -> n));
        assertNull(keep.filter("image_newline", n -> n));
        assertThrows(GgufFormatException.class, () -> keep.filter("mtp.1.enorm.weight", n -> n));
        assertThrows(GgufFormatException.class, () -> keep.filter("mtp.x.enorm.weight", n -> n));

        ModelSourceOptions noMtp = ModelSourceOptions.builder().includeMtp(false).build();
        HfNameFilter drop = HfDeepseekV4Hooks.filter(context(Map.of(), noMtp));
        assertNull(drop.filter("mtp.0.e_proj.weight", n -> n));
        assertNull(HfDeepseekV4Hooks.filterWithoutMtp("mtp.0.e_proj.weight", n -> n));

        Function<HfLoadContext, HfNameFilter> loadFilter = HfDeepseekV4Hooks::filter;
        ToIntFunction<HfLoadContext> blocks = HfDeepseekV4Hooks::blockCount;
        Map<String, RowSource> withMtp = Map.of("mtp.0.enorm.weight", zeros(4), "embed.weight", zeros(2, 4));
        assertEquals(3, blocks.applyAsInt(context(withMtp)));
        assertEquals(2, blocks.applyAsInt(context(withMtp, noMtp)));
        assertEquals(2, blocks.applyAsInt(context(Map.of("embed.weight", zeros(2, 4)))));
        assertTrue(loadFilter.apply(context(Map.of())) != null);
    }

    @Test
    public void tensorsRouteNativeNamesAndLeaveGeneratedOnesToTheExtraTensors() {
        HfLoadContext ctx = context(Map.of());
        Sink out = new Sink();
        HfDeepseekV4Hooks.tensors(ctx, "layers.0.ffn.experts.1.w2.weight", zeros(2, 4), out);
        HfDeepseekV4Hooks.tensors(ctx, "layers.2.nextn.e_proj.weight", zeros(2, 4), out);
        HfDeepseekV4Hooks.tensors(ctx, "layers.0.ffn.gate.tid2eid", zeros(4, 2), out);
        HfDeepseekV4Hooks.tensors(ctx, "layers.1.ffn.gate.tid2eid", zeros(4, 2), out);
        HfDeepseekV4Hooks.tensors(ctx, "layers.0.ffn.gate.bias", zeros(2), out);
        HfDeepseekV4Hooks.tensors(ctx, "layers.1.ffn.gate.bias", zeros(2), out);
        HfDeepseekV4Hooks.tensors(ctx, "layers.2.nextn.hc_head_fn", zeros(4, 4), out);
        HfDeepseekV4Hooks.tensors(ctx, "hc_head_base", zeros(4), out);
        HfDeepseekV4Hooks.tensors(ctx, "layers.1.attn.wq_b.weight", zeros(4, 4), out);
        assertEquals(Set.of("layers.0.ffn.experts.1.w2.weight", "layers.2.nextn.e_proj.weight",
                "layers.0.ffn.gate.tid2eid"), out.consumed);
        assertEquals(Set.of("layers.1.ffn.gate.tid2eid", "layers.0.ffn.gate.bias", "layers.2.nextn.hc_head_fn"),
                out.dropped);
        assertEquals(List.of("blk.1.exp_probs_b.bias", "output_hc_base.weight", "blk.1.attn_q_b.weight"),
                new ArrayList<>(out.added.keySet()));
        assertThrows(GgufFormatException.class,
                () -> HfDeepseekV4Hooks.tensors(ctx, "layers.1.attn.wq_b.scale", zeros(1, 1), out));
        assertThrows(GgufFormatException.class,
                () -> HfDeepseekV4Hooks.tensors(ctx, "layers.0.nextn.e_proj.weight", zeros(2, 4), out));
    }

    @Test
    public void tensorsConsultASubclassTableBeforeTheDefault() {
        HfLoadContext ctx = context(Map.of());
        HfDeepseekV4Names.Table extended = (name, bid) -> name.equals("main_norm.weight")
                ? new HfDeepseekV4Names.Mapped("OUTPUT_NORM", ".weight", true)
                : HfDeepseekV4Names.Table.DEFAULT.map(name, bid);
        Sink out = new Sink();
        HfDeepseekV4Hooks.tensors(ctx, "main_norm.weight", zeros(4), out, extended);
        HfDeepseekV4Hooks.tensors(ctx, "layers.1.attn.wq_a.weight", zeros(4, 4), out, extended);
        assertEquals(List.of("output_norm.weight", "blk.1.attn_q_a.weight"), new ArrayList<>(out.added.keySet()));
        assertThrows(GgufFormatException.class,
                () -> HfDeepseekV4Hooks.tensors(ctx, "main_norm.weight", zeros(4), new Sink()));
        assertThrows(IllegalArgumentException.class, () -> new HfDeepseekV4Names.Mapped("OUTPUT_NORM", ".scale", true));
    }

    @Test
    public void extraTensorsWriteHashTablesStackedExpertsAndTheMtpConcat() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        index.put("layers.0.ffn.gate.tid2eid", RopeTensors.fromLongs(new long[]{3, 2}, new long[]{0, 1, 1, 0, 1, 1}));
        for (int bid = 0; bid < 3; bid++) {
            for (int x = 0; x < 2; x++) {
                for (String w : List.of("w1", "w2", "w3")) {
                    index.put("layers." + bid + ".ffn.experts." + x + "." + w + ".weight",
                            RopeTensors.constant(new long[]{2, 4}, 10 * x + bid));
                }
            }
        }
        index.put("layers.2.nextn.e_proj.weight", RopeTensors.constant(new long[]{2, 3}, 1f));
        index.put("layers.2.nextn.h_proj.weight", RopeTensors.constant(new long[]{2, 5}, 2f));
        HfLoadContext ctx = context(index);
        assertEquals(3, ctx.blockCount());
        Sink out = new Sink();
        HfDeepseekV4Hooks.extra(ctx, out);

        assertEquals(List.of("blk.0.ffn_gate_tid2eid.weight",
                "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight", "blk.0.ffn_up_exps.weight",
                "blk.1.ffn_gate_exps.weight", "blk.1.ffn_down_exps.weight", "blk.1.ffn_up_exps.weight",
                "blk.2.ffn_gate_exps.weight", "blk.2.ffn_down_exps.weight", "blk.2.ffn_up_exps.weight",
                "blk.2.nextn.eh_proj.weight"), new ArrayList<>(out.added.keySet()));

        RowSource tid2eid = out.added.get("blk.0.ffn_gate_tid2eid.weight");
        assertEquals(ElementFormat.of(GGMLType.I32), tid2eid.format());
        assertArrayEquals(new int[]{0, 1, 1, 0, 1, 1}, ints(tid2eid));

        RowSource experts = out.added.get("blk.1.ffn_up_exps.weight");
        assertArrayEquals(new long[]{2, 2, 4}, experts.shape());
        float[] ev = floatValues(experts);
        assertEquals(1f, ev[0], 0f);
        assertEquals(11f, ev[8], 0f);

        RowSource eh = out.added.get("blk.2.nextn.eh_proj.weight");
        assertArrayEquals(new long[]{2, 8}, eh.shape());
        assertArrayEquals(new float[]{1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2}, floatValues(eh), 0f);
    }

    @Test
    public void extraTensorsRejectIncompleteSources() {
        Map<String, RowSource> index = new LinkedHashMap<>();
        index.put("layers.0.ffn.gate.tid2eid", RopeTensors.fromInts(new long[]{3, 2}, new int[]{0, 1, 1, 0, 1, 1}));
        for (int bid = 0; bid < 3; bid++) {
            for (int x = 0; x < 2; x++) {
                for (String w : List.of("w1", "w2", "w3")) {
                    index.put("layers." + bid + ".ffn.experts." + x + "." + w + ".weight", zeros(2, 4));
                }
            }
        }
        index.put("layers.2.nextn.e_proj.weight", zeros(2, 3));
        assertThrows(GgufFormatException.class, () -> HfDeepseekV4Hooks.extra(context(index), new Sink()));

        index.remove("layers.2.nextn.e_proj.weight");
        index.put("layers.1.ffn.experts.0.w2.scale", zeros(2, 1));
        assertThrows(GgufFormatException.class, () -> HfDeepseekV4Hooks.extra(context(index), new Sink()));

        index.remove("layers.1.ffn.experts.0.w2.scale");
        index.remove("layers.1.ffn.experts.1.w3.weight");
        assertThrows(GgufFormatException.class, () -> HfDeepseekV4Hooks.extra(context(index), new Sink()));

        Map<String, RowSource> noHash = new LinkedHashMap<>();
        assertThrows(GgufFormatException.class, () -> HfDeepseekV4Hooks.extra(context(noHash), new Sink()));
    }

    @Test
    public void forceTypeFollowsTheConverterRules() {
        ElementFormat f32 = ElementFormat.of(GGMLType.F32);
        ElementFormat bf16 = ElementFormat.of(GGMLType.BF16);
        ElementFormat fp8 = ElementFormat.of(GGMLType.F8_E4M3_B128);
        ElementFormat mxfp4 = ElementFormat.of(GGMLType.MXFP4);
        String eh = "blk.2.nextn.eh_proj.weight";
        assertEquals(GGMLType.Q8_0, HfDeepseekV4Hooks.forceType(eh, 2, fp8, OutType.F16));
        assertEquals(GGMLType.Q8_0, HfDeepseekV4Hooks.forceType(eh, 2, bf16, OutType.AUTO));
        assertNull(HfDeepseekV4Hooks.forceType(eh, 2, fp8, OutType.SOURCE));
        assertEquals(GGMLType.BF16, HfDeepseekV4Hooks.forceType(eh, 2, bf16, OutType.SOURCE));
        assertEquals(GGMLType.F32, HfDeepseekV4Hooks.forceType("blk.0.hc_attn_fn.weight", 2, f32, OutType.F16));
        assertEquals(GGMLType.F32, HfDeepseekV4Hooks.forceType("blk.0.attn_sinks.weight", 1, f32, OutType.BF16));
        assertEquals(GGMLType.BF16, HfDeepseekV4Hooks.forceType("blk.0.attn_q_a.weight", 2, bf16, OutType.F16));
        assertNull(HfDeepseekV4Hooks.forceType("blk.0.attn_norm.weight", 1, bf16, OutType.F16));
        assertNull(HfDeepseekV4Hooks.forceType("blk.0.attn_q_a.weight", 2, fp8, OutType.F16));
        assertNull(HfDeepseekV4Hooks.forceType("blk.0.ffn_up_exps.weight", 3, mxfp4, OutType.Q8_0));
    }

    @Test
    public void metadataCarriesTheV4KeySet() {
        HfLoadContext ctx = context(Map.of());
        HfMetadataBuilder kv = new HfMetadataBuilder("deepseek4");
        HfDeepseekV4Hooks.meta(ctx, kv, null);
        HfMetadata m = kv.build();
        assertEquals(3, m.get("deepseek4.block_count"));
        assertEquals(16, m.get("deepseek4.rope.dimension_count"));
        assertEquals(24, m.get("deepseek4.attention.q_lora_rank"));
        assertEquals(128, m.get("deepseek4.attention.sliding_window"));
        assertEquals(32, m.get("deepseek4.expert_feed_forward_length"));
        assertEquals(1, m.get("deepseek4.expert_shared_count"));
        assertEquals(1.5f, (Float) m.get("deepseek4.expert_weights_scale"), 0f);
        assertEquals(Boolean.TRUE, m.get("deepseek4.expert_weights_norm"));
        List<?> clamp = (List<?>) m.get("deepseek4.swiglu_clamp_exp");
        assertEquals(3, clamp.size());
        assertEquals(10f, ((Number) clamp.get(2)).floatValue(), 0f);
        assertEquals(3, ((List<?>) m.get("deepseek4.swiglu_clamp_shexp")).size());
        assertEquals(2, m.get("deepseek4.attention.indexer.head_count"));
        assertEquals(16, m.get("deepseek4.attention.indexer.key_length"));
        assertEquals(8, m.get("deepseek4.attention.indexer.top_k"));
        assertEquals(2, m.get("deepseek4.attention.output_group_count"));
        assertEquals(16, m.get("deepseek4.attention.output_lora_rank"));
        assertEquals(GGUFValueType.INT32, m.elementType("deepseek4.attention.compress_ratios"));
        assertEquals(3, ((List<?>) m.get("deepseek4.attention.compress_ratios")).size());
        assertEquals(160000f, (Float) m.get("deepseek4.attention.compress_rope_freq_base"), 0f);
        assertEquals(4, m.get("deepseek4.hyper_connection.count"));
        assertEquals(20, m.get("deepseek4.hyper_connection.sinkhorn_iterations"));
        assertEquals(1e-6f, (Float) m.get("deepseek4.hyper_connection.epsilon"), 0f);
        assertEquals(1, m.get("deepseek4.hash_layer_count"));
        assertEquals(256, m.get("deepseek4.embedding_length_out"));
        assertEquals(1, m.get("deepseek4.nextn_predict_layers"));
        assertTrue(m.has("deepseek4.expert_gating_func"));

        HfConfig missing = config(V4.replace(" 'hc_eps': 1e-06,", ""));
        HfLoadContext bad = HfLoadContext.builder(missing, entry(), Arch.DEEPSEEK4).build();
        assertThrows(GgufFormatException.class,
                () -> HfDeepseekV4Hooks.meta(bad, new HfMetadataBuilder("deepseek4"), null));
    }

    @Test
    public void defaultTemplateFollowsTheModelIdHint() {
        assertEquals(HfDeepseekV4Hooks.TEMPLATE, HfDeepseekV4Hooks.defaultTemplate(Path.of("/m/DeepSeek-V4-Flash")));
        assertEquals(HfDeepseekV4Hooks.TEMPLATE_0731,
                HfDeepseekV4Hooks.defaultTemplate(Path.of("/m/DeepSeek-V4-Flash-0731")));
        assertEquals(HfDeepseekV4Hooks.TEMPLATE_0731, HfDeepseekV4Hooks.defaultTemplate(
                Path.of("/c/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/0123abcd")));
        assertEquals(HfDeepseekV4Hooks.TEMPLATE, HfDeepseekV4Hooks.defaultTemplate(
                Path.of("/c/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/0123abcd")));
    }

    // ------------------------------------------------------------------ helpers

    private static RowSource zeros(long... shape) {
        return RopeTensors.constant(shape, 0f);
    }

    private static ByteBuffer bytes(RowSource s) throws IOException {
        ByteBuffer b = ByteBuffer.allocate((int) (s.rows() * s.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
        s.readRows(0, (int) s.rows(), b);
        b.flip();
        return b;
    }

    private static float[] floatValues(RowSource s) throws IOException {
        ByteBuffer b = bytes(s);
        float[] out = new float[b.remaining() / 4];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getFloat();
        }
        return out;
    }

    private static int[] ints(RowSource s) throws IOException {
        ByteBuffer b = bytes(s);
        int[] out = new int[b.remaining() / 4];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getInt();
        }
        return out;
    }

    /** A sink recording what the hooks plan, drop and consume. */
    private static final class Sink implements TensorSink {
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

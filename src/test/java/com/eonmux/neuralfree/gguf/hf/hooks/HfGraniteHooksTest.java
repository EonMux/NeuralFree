package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Checkpoint;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Sink;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.UnaryOperator;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.LLAMA_BASE;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.config;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.plan;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.seq;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfGraniteHooks} on synthetic Granite and GraniteMoe configurations and checkpoints: the multiplier scales,
 * the popped {@code head_dim}, the Granite 4 Vision deepstack mapping, the multimodal name filter, and the split of
 * the fused JetMoe expert and shared-MLP projections with and without routed experts.
 */
public class HfGraniteHooksTest {

    private static final String SCALES = "'attention_multiplier': 0.015625, 'embedding_multiplier': 12.0,"
            + " 'residual_multiplier': 0, 'logits_scaling': 8.0";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Checkpoint ckpt;

    @After
    public void close() throws IOException {
        if (ckpt != null) {
            ckpt.close();
        }
    }

    private static HfArchEntry granite() {
        return HfArchEntry.of(Collections.singletonList("GraniteForCausalLM"), HfArchResolver.of(Arch.GRANITE),
                HfGraniteHooks::meta, HfLlamaHooks::tensors, HfLlamaHooks::vocab)
                .withNameFilter(HfGraniteHooks.FILTER).withExtra(HfLlamaHooks::extra);
    }

    private static HfArchEntry graniteMoe(Arch arch) {
        return HfArchEntry.of(Collections.singletonList("GraniteMoeForCausalLM"), HfArchResolver.of(arch),
                HfGraniteHooks::moeMeta, HfGraniteHooks::moeTensors, HfLlamaHooks::vocab)
                .withNameFilter(HfGraniteHooks.FILTER).withExtra(HfLlamaHooks::extra);
    }

    private static HfMetadata meta(HfArchEntry entry, Arch arch, String json) throws IOException {
        HfConfig cfg = config(json);
        HfLoadContext ctx = HfLoadContext.builder(cfg, entry, arch).build();
        HfMetadataBuilder kv = new HfMetadataBuilder(arch.ggufName());
        entry.meta().apply(ctx, kv, null);
        return kv.build();
    }

    @Test
    public void graniteWritesTruthyScalesAndIgnoresHeadDim() throws IOException {
        HfMetadata m = meta(granite(), Arch.GRANITE, "{'architectures': ['GraniteForCausalLM'], 'head_dim': 6, "
                + SCALES + ", " + LLAMA_BASE + "}");
        assertEquals(0.015625f, m.get("granite.attention.scale"));
        assertEquals(12f, m.get("granite.embedding_scale"));
        assertEquals(8f, m.get("granite.logit_scale"));
        assertFalse(m.has("granite.residual_scale"));
        assertFalse(m.has("granite.attention.key_length"));
        assertFalse(m.has("granite.attention.value_length"));
        assertEquals(4, m.get("granite.rope.dimension_count"));
        assertEquals(10, m.get("granite.vocab_size"));
        assertFalse(m.has("granite.deepstack_mapping"));
    }

    @Test
    public void graniteVisionDeepstackMappingFollowsTheNormalizedProjectorMap() throws IOException {
        String base = "'architectures': ['GraniteForCausalLM'], 'hidden_size': 8, 'num_attention_heads': 2,"
                + " 'vocab_size': 10, 'text_config': {'num_hidden_layers': 4},"
                + " 'vision_config': {'num_hidden_layers': 24}";
        HfMetadata both = meta(granite(), Arch.GRANITE, "{" + base + ", 'deepstack_layer_map': [[20, 3], [10, 0]],"
                + " 'spatial_target_layers': [1, 2]}");
        assertEquals(Arrays.asList(-1, 1, 2, 3), both.get("granite.deepstack_mapping"));
        HfMetadata negative = meta(granite(), Arch.GRANITE, "{" + base
                + ", 'deepstack_layer_map': [[-4, -1], [0, 0]]}");
        assertEquals(Arrays.asList(-1, -1, -1, 1), negative.get("granite.deepstack_mapping"));
        assertThrows(GgufFormatException.class, () -> meta(granite(), Arch.GRANITE, "{" + base
                + ", 'deepstack_layer_map': [[0, 0], [1, 9]]}"));
    }

    @Test
    public void graniteFilterDropsMultimodalTensors() {
        List<String> seen = new ArrayList<>();
        UnaryOperator<String> base = n -> {
            seen.add(n);
            return n;
        };
        assertNull(HfGraniteHooks.filter("encoder.layers.0.weight", base));
        assertNull(HfGraniteHooks.filter("model.image_newline", base));
        assertNull(HfGraniteHooks.filter("model.layerwise_projectors.0.weight", base));
        assertNull(HfGraniteHooks.filter("model.spatial_projectors.1.weight", base));
        assertTrue(seen.isEmpty());
        assertEquals("model.layers.0.mlp.up_proj.weight",
                HfGraniteHooks.FILTER.filter("model.text_model.layers.0.mlp.up_proj.weight", base));
    }

    @Test
    public void graniteMoeSplitsFusedExpertAndSharedProjections() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt.f32("model.layers.0.block_sparse_moe.input_linear.weight", new long[]{2, 4, 8},
                seq(64))
                .f32("model.layers.0.block_sparse_moe.output_linear.weight", new long[]{2, 8, 2}, seq(32))
                .f32("model.layers.0.shared_mlp.input_linear.weight", new long[]{6, 8}, seq(48))
                .f32("model.layers.0.shared_mlp.output_linear.weight", new long[]{8, 3}, seq(24)).index();
        HfConfig cfg = config("{'architectures': ['GraniteMoeForCausalLM'], 'num_local_experts': 2,"
                + " 'shared_intermediate_size': 3, " + SCALES + ", "
                + LLAMA_BASE.replace("'intermediate_size': 16", "'intermediate_size': 2") + "}");
        HfArchEntry entry = graniteMoe(Arch.GRANITE_MOE);
        Sink out = plan(HfLoadContext.builder(cfg, entry, Arch.GRANITE_MOE).index(index).build());
        assertArrayEquals(concat(range(0, 16), range(32, 48)), floats(out.get("blk.0.ffn_gate_exps.weight")), 0f);
        assertArrayEquals(concat(range(16, 32), range(48, 64)), floats(out.get("blk.0.ffn_up_exps.weight")), 0f);
        assertArrayEquals(new long[]{2, 2, 8}, out.get("blk.0.ffn_gate_exps.weight").shape());
        assertArrayEquals(seq(32), floats(out.get("blk.0.ffn_down_exps.weight")), 0f);
        assertArrayEquals(range(0, 24), floats(out.get("blk.0.ffn_gate_shexp.weight")), 0f);
        assertArrayEquals(range(24, 48), floats(out.get("blk.0.ffn_up_shexp.weight")), 0f);
        assertArrayEquals(seq(24), floats(out.get("blk.0.ffn_down_shexp.weight")), 0f);

        HfMetadataBuilder kv = new HfMetadataBuilder("granitemoe");
        entry.meta().apply(HfLoadContext.builder(cfg, entry, Arch.GRANITE_MOE).build(), kv, null);
        assertEquals(3, kv.build().get("granitemoe.expert_shared_feed_forward_length"));
    }

    @Test
    public void graniteMoeWithoutExpertsMakesTheSharedMlpADenseFfn() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt.f32("model.layers.0.shared_mlp.input_linear.weight", new long[]{6, 8}, seq(48))
                .f32("model.layers.0.shared_mlp.output_linear.weight", new long[]{8, 3}, seq(24)).index();
        HfConfig cfg = config("{'architectures': ['GraniteMoeForCausalLM'], 'num_local_experts': 0,"
                + " 'shared_intermediate_size': 3, " + LLAMA_BASE + "}");
        HfArchEntry entry = graniteMoe(Arch.GRANITE_HYBRID);
        Sink out = plan(HfLoadContext.builder(cfg, entry, Arch.GRANITE_HYBRID).index(index).build());
        assertArrayEquals(range(0, 24), floats(out.get("blk.0.ffn_gate.weight")), 0f);
        assertArrayEquals(range(24, 48), floats(out.get("blk.0.ffn_up.weight")), 0f);
        assertArrayEquals(seq(24), floats(out.get("blk.0.ffn_down.weight")), 0f);
    }

    @Test
    public void graniteMoeRejectsAFusedTensorOfTheWrongSize() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt.f32("model.layers.0.block_sparse_moe.input_linear.weight", new long[]{2, 6, 8},
                seq(96)).index();
        HfConfig cfg = config("{'architectures': ['GraniteMoeForCausalLM'], 'num_local_experts': 2, "
                + LLAMA_BASE.replace("'intermediate_size': 16", "'intermediate_size': 2") + "}");
        HfArchEntry entry = graniteMoe(Arch.GRANITE_MOE);
        HfLoadContext ctx = HfLoadContext.builder(cfg, entry, Arch.GRANITE_MOE).index(index).build();
        assertThrows(GgufFormatException.class, () -> plan(ctx));
    }

    private static float[] range(int from, int to) {
        float[] v = new float[to - from];
        for (int i = 0; i < v.length; i++) {
            v[i] = from + i;
        }
        return v;
    }

    private static float[] concat(float[] a, float[] b) {
        return HfLlamaHooksTest.concat(a, b);
    }
}

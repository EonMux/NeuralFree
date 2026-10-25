package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The part A hook rows: the Flint dense lead / sliding-and-rope patterns / router bias (flint.py:12-37), the
 * Afmoe per-expert stacking (afmoe.py:24-79), the Kolibri-1 fixup (kolibri1.py:14-22) and the Granite SWA
 * metadata (granite.py:91-143).
 */
public class HfMoeSwaHooksATest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path dir() throws IOException {
        return tmp.newFolder().toPath();
    }

    // ------------------------------------------------------------------ FlintModel

    @Test
    public void flintDenseLayersMustPrecedeTheMoeLayers() throws IOException {
        HfLoadContext bad = context(dir(), Arch.FLINT, "FlintForCausalLM", "'mlp_only_layers': [1, 2],"
                + " 'num_experts': 2, 'moe_intermediate_size': 4,"
                + " 'shared_expert_intermediate_size': 8, 'sliding_window': 4,"
                + " 'layer_types': ['full_attention', 'full_attention']");
        assertThrows(GgufFormatException.class, () -> HfMoeSwaHooksA.flintMeta(bad, new HfMetadataBuilder(
                Arch.FLINT.ggufName()), null));
        HfLoadContext ok = context(dir(), Arch.FLINT, "FlintForCausalLM", "'mlp_only_layers': [0, 1],"
                + " 'num_experts': 2, 'moe_intermediate_size': 4,"
                + " 'shared_expert_intermediate_size': 8, 'sliding_window': 4,"
                + " 'layer_types': ['sliding_attention', 'full_attention'], 'no_rope_layer_indices': [0]");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.FLINT.ggufName());
        HfMoeSwaHooksA.flintMeta(ok, kv, null);
        assertEquals(2, kv.get(GgufKeys.LEADING_DENSE_BLOCK_COUNT));
        assertEquals(1, kv.get(GgufKeys.EXPERT_SHARED_COUNT));
        assertEquals(4, kv.get(GgufKeys.ATTENTION_SLIDING_WINDOW));
    }

    @Test
    public void flintTensorsDropTheRouterBiasAndRenameTheSoftmaxOffset() throws IOException {
        Map<String, com.eonmux.neuralfree.gguf.hf.tensor.RowSource> index = new LinkedHashMap<>();
        index.put("model.layers.1.mlp.softmax_offset", seq(2));
        index.put("model.layers.0.mlp.gate.expert_bias", seq(2));
        HfLoadContext ctx = context(dir(), Arch.FLINT, "FlintForCausalLM", "'num_experts': 2,"
                + " 'router_expert_bias': false", index, false);
        RecordingSink sink = new RecordingSink();
        for (String name : index.keySet()) {
            HfMoeSwaHooksA.flintTensors(ctx, name, ctx.index().get(name), sink);
        }
        assertTrue(sink.dropped.containsKey("model.layers.0.mlp.gate.expert_bias"));
        assertTrue(sink.tensors.size() == 1);
    }

    // ------------------------------------------------------------------ AfmoeModel

    @Test
    public void afmoeExpertsStackAndLeftoversAreInvalid() throws IOException {
        Map<String, com.eonmux.neuralfree.gguf.hf.tensor.RowSource> index = new LinkedHashMap<>();
        for (int x = 0; x < 2; x++) {
            index.put("model.layers.0.mlp.experts." + x + ".gate_proj.weight", seq(2, 8));
            index.put("model.layers.0.mlp.experts." + x + ".up_proj.weight", seq(2, 8));
            index.put("model.layers.0.mlp.experts." + x + ".down_proj.weight", seq(2, 2));
        }
        HfLoadContext ctx = context(dir(), Arch.AFMOE, "AfmoeForCausalLM", "'num_local_experts': 2", index, false);
        RecordingSink sink = new RecordingSink();
        for (String name : index.keySet()) {
            HfMoeSwaHooksA.afmoeTensors(ctx, name, ctx.index().get(name), sink);
        }
        HfMoeSwaHooksA.afmoeExtra(ctx, sink);
        assertEquals(3, sink.tensors.size());
        HfLoadContext partial = context(dir(), Arch.AFMOE, "AfmoeForCausalLM", "'num_local_experts': 2",
                java.util.Collections.singletonMap("model.layers.0.mlp.experts.0.gate_proj.weight", seq(2, 8)),
                false);
        assertThrows(GgufFormatException.class, () -> HfMoeSwaHooksA.afmoeExtra(partial, new RecordingSink()));
    }

    // ------------------------------------------------------------------ Kolibri1Model

    @Test
    public void kolibri1FixupRenamesTheAfmoeAliases() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.KOLIBRI1, "Kolibri1ForCausalLM", "'norm_topk_prob': true,"
                + " 'moe_intermediate_size': 4, 'shared_expert_intermediate_size': 8,"
                + " 'layer_types': ['full_attention', 'sliding_attention']");
        com.eonmux.neuralfree.gguf.hf.config.HfConfig fixed =
                HfMoeSwaHooksA.kolibri1Fixup(ctx.config());
        assertEquals(1, fixed.findInt(0, "num_shared_experts"));
        assertTrue(fixed.find("route_norm").orElse(Boolean.FALSE).equals(Boolean.TRUE));
        assertTrue(fixed.find("intermediate_size").orElse(0).equals(4));
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.KOLIBRI1.ggufName());
        HfMoeSwaHooksA.kolibri1Meta(ctx, kv, null);
        assertEquals(8, kv.get(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH));
        assertEquals(2, kv.get(GgufKeys.EXPERT_GATING_FUNC));
        assertEquals(java.util.Arrays.asList(false, true), kv.get(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN));
    }

    // ------------------------------------------------------------------ Granite SWA rows

    @Test
    public void graniteSwaDefaultsToThePeriodFourWindowAndRopePattern() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.GRANITE_SWA, "GraniteSWAForCausalLM",
                "'no_rope_layers': [0]", java.util.Collections.emptyMap(), false);
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.GRANITE_SWA.ggufName());
        HfMoeSwaHooksA.graniteSwaMeta(ctx, kv, null);
        assertEquals(128, kv.get(GgufKeys.ATTENTION_SLIDING_WINDOW));
        assertEquals(java.util.Arrays.asList(true, false),
                kv.get(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN));
        assertEquals(java.util.Arrays.asList(false, true), kv.get(GgufKeys.ATTENTION_ROPE_PATTERN));
        HfLoadContext moe = context(dir(), Arch.GRANITE_SWA, "GraniteMoeSWAForCausalLM",
                "'layer_types': ['sliding_attention', 'full_attention']",
                java.util.Collections.emptyMap(), false);
        HfMetadataBuilder kv2 = new HfMetadataBuilder(Arch.GRANITE_SWA.ggufName());
        HfMoeSwaHooksA.graniteMoeSwaMeta(moe, kv2, null);
        assertEquals(java.util.Arrays.asList(true, false), kv2.get(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN));
    }
}

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
 * The part C hooks: the Laguna attention-gate validation (laguna.py:124-176), the Maple reference SwiGLU clamp and
 * silu-only reject (maple.py:20-41), the Mellum Qwen-style MTP (mellum.py:23-88), the Exaone-MoE MTP replication
 * into every NextN block (exaone.py:142-229) and the SmallThinker sliding-window layout (smallthinker.py:17-49).
 */
public class HfMoeSwaHooksCTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path dir() throws IOException {
        return tmp.newFolder().toPath();
    }

    // ------------------------------------------------------------------ LagunaModel

    @Test
    public void lagunaGateTypesFailLoudlyAndTheGateWidthMustMatch() throws IOException {
        HfLoadContext unknown = context(dir(), Arch.LAGUNA, "LagunaForCausalLM", "'gating': 'per-element-unknown'");
        assertThrows(GgufFormatException.class, () -> HfMoeSwaHooksC.lagunaMeta(unknown, new HfMetadataBuilder(
                Arch.LAGUNA.ggufName()), null));
        HfLoadContext perHead = context(dir(), Arch.LAGUNA, "LagunaForCausalLM", "'gating': 'per-head',"
                + " 'num_attention_heads_per_layer': [2, 2], 'moe_intermediate_size': 4,"
                + " 'shared_expert_intermediate_size': 8, 'moe_routed_scaling_factor': 1.0,"
                + " 'num_experts': 2, 'head_dim': 4");
        Map<String, com.eonmux.neuralfree.gguf.hf.tensor.RowSource> index = new LinkedHashMap<>();
        index.put("model.layers.0.self_attn.g_proj.weight", seq(2, 5));
        index.put("model.layers.1.mlp.experts.0.gate_proj.weight", seq(2, 8));
        index.put("model.layers.1.mlp.experts.0.up_proj.weight", seq(2, 8));
        index.put("model.layers.1.mlp.experts.0.down_proj.weight", seq(2, 2));
        index.put("model.layers.1.mlp.experts.1.gate_proj.weight", seq(2, 8));
        index.put("model.layers.1.mlp.experts.1.up_proj.weight", seq(2, 8));
        index.put("model.layers.1.mlp.experts.1.down_proj.weight", seq(2, 2));
        HfLoadContext gproj = context(dir(), Arch.LAGUNA, "LagunaForCausalLM", "'gating': 'per-head',"
                + " 'moe_intermediate_size': 4, 'shared_expert_intermediate_size': 8,"
                + " 'moe_routed_scaling_factor': 1.0, 'num_experts': 2, 'head_dim': 4", index, false);
        RecordingSink sink = new RecordingSink();
        for (String name : index.keySet()) {
            HfMoeSwaHooksC.lagunaTensors(gproj, name, gproj.index().get(name), sink);
        }
        HfMoeSwaHooksC.lagunaExtra(gproj, sink);
        assertEquals(3, sink.tensors.size());
        Map<String, com.eonmux.neuralfree.gguf.hf.tensor.RowSource> wide = new LinkedHashMap<>();
        wide.put("model.layers.0.self_attn.g_proj.weight", seq(16, 5));
        HfLoadContext contradicting = context(dir(), Arch.LAGUNA, "LagunaForCausalLM", "'gating': 'per-head',"
                + " 'head_dim': 4", wide, false);
        assertThrows(GgufFormatException.class, () -> HfMoeSwaHooksC.lagunaTensors(contradicting,
                "model.layers.0.self_attn.g_proj.weight",
                contradicting.index().get("model.layers.0.self_attn.g_proj.weight"), new RecordingSink()));
    }

    // ------------------------------------------------------------------ MapleModel

    @Test
    public void mapleIsSiluOnlyAndWritesTheSevenClamp() throws IOException {
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.MAPLE.ggufName());
        HfMoeSwaHooksC.mapleMeta(context(dir(), Arch.MAPLE, "MapleForCausalLM", "'head_dim': 4,"
                        + " 'partial_rotary_factor': 0.5, 'sliding_window': 4,"
                        + " 'layer_types': ['full_attention', 'sliding_attention'], 'moe_intermediate_size': 4"),
                kv, null);
        assertEquals(java.util.Arrays.asList(7.0, 7.0), kv.get(GgufKeys.SWIGLU_CLAMP_EXP));
        assertThrows(GgufFormatException.class, () -> HfMoeSwaHooksC.mapleMeta(
                context(dir(), Arch.MAPLE, "MapleForCausalLM", "'hidden_act': 'gelu', 'head_dim': 4,"
                        + " 'sliding_window': 4, 'layer_types': ['full_attention', 'full_attention']"
                        + ", 'moe_intermediate_size': 4"),
                new HfMetadataBuilder(Arch.MAPLE.ggufName()), null));
    }

    // ------------------------------------------------------------------ MellumModel

    @Test
    public void mellumWritesTheMtpLayerCount() throws IOException {
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.MELLUM.ggufName());
        HfMoeSwaHooksC.mellumMeta(context(dir(), Arch.MELLUM, "MellumForCausalLM", "'num_mtp_layers': 1,"
                + " 'use_sliding_window': true, 'sliding_window': 4,"
                + " 'layer_types': ['sliding_attention', 'full_attention'], 'mtp_layer_types': ['full_attention']"),
                kv, null);
        assertEquals(1, kv.get(GgufKeys.NEXTN_PREDICT_LAYERS));
        assertEquals(4, kv.get(GgufKeys.ATTENTION_SLIDING_WINDOW));
        assertEquals(java.util.Arrays.asList(true, false, false),
                kv.get(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN));
    }

    // ------------------------------------------------------------------ ExaoneMoEModel

    @Test
    public void exaoneMoeReplicatesTheSharedMtpHeadIntoEveryNextNBlock() throws IOException {
        Map<String, com.eonmux.neuralfree.gguf.hf.tensor.RowSource> index = new LinkedHashMap<>();
        index.put("mtp.fc.weight", seq(8, 8));
        index.put("mtp.norm.weight", seq(8));
        index.put("mtp.layers.0.mlp.experts.0.gate_proj.weight", seq(2, 8));
        index.put("mtp.layers.0.mlp.experts.1.gate_proj.weight", seq(2, 8));
        HfLoadContext ctx = context(dir(), Arch.EXAONE_MOE, "ExaoneMoEForCausalLM",
                "'num_nextn_predict_layers': 2, 'moe_intermediate_size': 4, 'num_shared_experts': 1,"
                        + " 'routed_scaling_factor': 1.0, 'num_local_experts': 2", index, true);
        RecordingSink sink = new RecordingSink();
        for (String name : index.keySet()) {
            HfMoeSwaHooksC.exaoneMoeTensors(ctx, name, ctx.index().get(name), sink);
        }
        HfMoeSwaHooksC.exaoneMoeExtra(ctx, sink);
        // the shared mtp.fc / mtp.norm heads replicate into both NextN blocks (blocks 2 and 3); the
        // mtp.layers.0 per-expert names rename into the appended slots and map on the normal path
        long replicated = sink.tensors.keySet().stream()
                .filter(n -> n.endsWith("eh_proj.weight") || n.endsWith("shared_head.norm.weight")).count();
        assertEquals(4, replicated);
        assertTrue(sink.tensors.keySet().stream().anyMatch(n -> n.contains(".3.")));
        assertTrue(sink.tensors.keySet().stream().anyMatch(n -> n.contains(".2.")));
    }

    // ------------------------------------------------------------------ SmallThinkerModel

    @Test
    public void smallthinkerEnablesTheWindowOnlyOnANonzeroLayoutAndStacksItsExperts() throws IOException {
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.SMALLTHINKER.ggufName());
        HfMoeSwaHooksC.smallthinkerMeta(context(dir(), Arch.SMALLTHINKER, "SmallThinkerForCausalLM",
                "'moe_num_primary_experts': 2, 'moe_num_active_primary_experts': 1, 'moe_ffn_hidden_size': 4,"
                        + " 'sliding_window_layout': ['0', '4'], 'sliding_window_size': 3"), kv, null);
        assertEquals(2, kv.get(GgufKeys.EXPERT_COUNT));
        assertEquals(4, kv.get(GgufKeys.FEED_FORWARD_LENGTH));
        assertEquals(3, kv.get(GgufKeys.ATTENTION_SLIDING_WINDOW));
        HfMetadataBuilder off = new HfMetadataBuilder(Arch.SMALLTHINKER.ggufName());
        HfMoeSwaHooksC.smallthinkerMeta(context(dir(), Arch.SMALLTHINKER, "SmallThinkerForCausalLM",
                "'sliding_window_layout': ['0', '0'], 'sliding_window_size': 3"), off, null);
        assertEquals(null, off.get(GgufKeys.ATTENTION_SLIDING_WINDOW));
        Map<String, com.eonmux.neuralfree.gguf.hf.tensor.RowSource> index = new LinkedHashMap<>();
        for (int x = 0; x < 2; x++) {
            index.put("model.layers.0.block_sparse_moe.experts." + x + ".down.weight", seq(2, 2));
            index.put("model.layers.0.block_sparse_moe.experts." + x + ".gate.weight", seq(2, 8));
            index.put("model.layers.0.block_sparse_moe.experts." + x + ".up.weight", seq(2, 8));
        }
        HfLoadContext experts = context(dir(), Arch.SMALLTHINKER, "SmallThinkerForCausalLM",
                "'moe_num_primary_experts': 2", index, false);
        RecordingSink sink = new RecordingSink();
        for (String name : index.keySet()) {
            HfMoeSwaHooksC.smallthinkerTensors(experts, name, experts.index().get(name), sink);
        }
        HfMoeSwaHooksC.smallthinkerExtra(experts, sink);
        assertEquals(3, sink.tensors.size());
    }
}

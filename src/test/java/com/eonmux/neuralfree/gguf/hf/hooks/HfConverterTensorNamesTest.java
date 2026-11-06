package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Hooks that build tensor names with {@code format_tensor_name} pass the converter's {@code MODEL_TENSOR} names: the
 * Bailing V3 {@code f_proj} / {@code g_proj} renames (LC/conversion/bailingmoe3.py:133-188) and the RWKV6 fused
 * token-shift lerp (LC/conversion/rwkv.py:53-116).
 */
public class HfConverterTensorNamesTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void bailingV3GateProjectionsTakeTheirLayerKindsName() throws IOException {
        // four layers in groups of two: layers 1 and 3 are full attention
        HfLoadContext ctx = HfQwenHooksTest.context(tmp.newFolder().toPath(), Arch.BAILINGMOE3,
                "BailingMoeV3ForCausalLM", "'num_hidden_layers': 4, 'layer_group_size': 2",
                Collections.emptyMap(), false);
        RecordingSink sink = new RecordingSink();
        HfBailingV3Hooks.tensors(ctx, "model.layers.0.attention.f_proj.weight", seq(4, 8), sink);
        HfBailingV3Hooks.tensors(ctx, "model.layers.0.attention.g_proj.weight", seq(8, 8), sink);
        HfBailingV3Hooks.tensors(ctx, "model.layers.1.attention.g_proj.weight", seq(8, 8), sink);
        assertEquals(Arrays.asList("blk.0.ssm_f_a.weight", "blk.0.ssm_g_a.weight",
                "blk.1.attn_gate.weight"), new ArrayList<>(sink.tensors.keySet()));
    }

    @Test
    public void rwkv6LerpWeightsFuseInTheWkvrgOrder() throws IOException {
        HfLoadContext ctx = HfQwenHooksTest.context(tmp.newFolder().toPath(), Arch.RWKV6, "Rwkv6ForCausalLM", "",
                Collections.emptyMap(), false);
        RecordingSink sink = new RecordingSink();
        for (String part : new String[]{"w", "k", "v", "r", "g"}) {
            HfRwkvHooks.rwkv6Tensors(ctx, "rwkv.blocks.0.attention.time_mix_lerp_" + part + ".weight",
                    seq(1, 1, 8), sink);
        }
        assertEquals(Collections.singleton("blk.0.time_mix_lerp_fused.weight"), sink.tensors.keySet());
        assertArrayEquals("torch.stack(...).unsqueeze(1) (rwkv.py:79)", new long[]{5, 1, 1, 8},
                sink.tensors.get("blk.0.time_mix_lerp_fused.weight").shape());
    }

    @Test
    public void rwkv7LerpWeightsStackInTheRwkvagOrder() throws IOException {
        HfLoadContext ctx = HfQwenHooksTest.context(tmp.newFolder().toPath(), Arch.RWKV7, "Rwkv7ForCausalLM", "",
                Collections.emptyMap(), false);
        RecordingSink sink = new RecordingSink();
        for (String part : new String[]{"r", "w", "k", "v", "a", "g"}) {
            HfRwkvHooks.rwkv7Tensors(ctx, "model.layers.0.attention.x_" + part, seq(1, 1, 8), sink);
        }
        assertEquals(Collections.singleton("blk.0.time_mix_lerp_fused.weight"), sink.tensors.keySet());
        assertArrayEquals("torch.stack(x_*, dim=0) (rwkv.py:235)", new long[]{6, 1, 1, 8},
                sink.tensors.get("blk.0.time_mix_lerp_fused.weight").shape());
    }
}

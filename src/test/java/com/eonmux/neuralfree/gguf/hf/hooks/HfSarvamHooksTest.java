package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link HfSarvamHooks}: the zero-mean router bias and the configuration defaults and refusals. */
public class HfSarvamHooksTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path dir() throws IOException {
        return tmp.newFolder().toPath();
    }

    @Test
    public void centersTheRouterBias() throws IOException {
        String name = "model.layers.0.mlp.gate.e_score_correction.bias";
        Map<String, RowSource> index = Collections.singletonMap(name, seq(4));
        HfLoadContext ctx = context(dir(), Arch.DEEPSEEK2, "SarvamMLAForCausalLM", "", index, false);
        RecordingSink sink = new RecordingSink();
        HfSarvamHooks.tensors(ctx, name, ctx.index().get(name), sink);
        assertEquals(Collections.singleton("blk.0.exp_probs_b.bias"), sink.tensors.keySet());
        // 0, 1, 2, 3 has mean 1.5
        assertArrayEquals(floats(-1.5f, -0.5f, 0.5f, 1.5f), values(sink.tensors.get("blk.0.exp_probs_b.bias")), 1e-6f);
    }

    @Test
    public void appliesTheClassDefaultsToTheConfiguration() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.DEEPSEEK2, "SarvamMLAForCausalLM", "'n_routed_experts': 2,"
                + " 'num_attention_heads': 2, 'num_hidden_layers': 1", Collections.emptyMap(), false);
        com.eonmux.neuralfree.gguf.hf.config.HfConfig fixed = HfSarvamHooks.fixup(ctx.config());
        assertEquals(1, fixed.findInt("n_group").getAsInt());
        assertEquals(1, fixed.findInt("topk_group").getAsInt());
        assertTrue(fixed.findBool(false, "norm_topk_prob"));
        assertEquals(1, fixed.findInt("n_shared_experts").getAsInt());
        // the shared test configuration already names the KV heads; an existing value is preserved.
        assertEquals(1, fixed.findInt("num_key_value_heads").getAsInt());
        assertEquals(2, fixed.findInt("n_routed_experts").getAsInt());
        assertEquals("sigmoid", fixed.findString("scoring_func").get());
    }

    @Test
    public void refusesANonUnitMoeLayerFrequencyAndNoSharedExpert() throws IOException {
        for (Map.Entry<String, String> c : Map.of("'moe_layer_freq': 2", "moe_layer_freq",
                "'n_shared_experts': 0", "num_shared_experts").entrySet()) {
            HfLoadContext ctx = context(dir(), Arch.DEEPSEEK2, "SarvamMLAForCausalLM", c.getKey(),
                    Collections.emptyMap(), false);
            GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfSarvamHooks.fixup(ctx.config()));
            assertTrue(e.getMessage(), e.getMessage().contains(c.getValue()));
        }
    }

    private static RowSource seq(long... shape) {
        long n = 1;
        for (long d : shape) {
            n *= d;
        }
        float[] v = new float[(int) n];
        for (int i = 0; i < v.length; i++) {
            v[i] = i;
        }
        return com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors.fromFloats(shape, v);
    }
}

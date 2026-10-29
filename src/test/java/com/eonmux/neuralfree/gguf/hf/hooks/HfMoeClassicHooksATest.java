package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
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
import java.util.stream.Collectors;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfMoeClassicHooksA} against the Llama4, Dbrx and Grok converters (LC/conversion/llama.py:380-417,
 * dbrx.py:13-76, grok.py:17-117) and the Arctic expert stacking (arctic.py:115-154, 156-163): the Llama4 width fixup
 * and the transposed [E,D,2F] split, the DBRX expert view with the force type, the Grok TP-piece concat, and the
 * Arctic Mixtral stacking with its completeness check.
 */
public class HfMoeClassicHooksATest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    private static final String LLAMA4_CONFIG = "'intermediate_size': 4, 'intermediate_size_mlp': 8,"
            + " 'interleave_moe_layer_step': 1, 'tie_word_embeddings': false,"
            + " 'layer_types': ['full_attention', 'full_attention']";

    private static final String DBRX_CONFIG = "'d_model': 8, 'n_heads': 2, 'max_seq_len': 64,"
            + " 'num_hidden_layers': 2, 'attn_config': {'kv_n_heads': 1, 'clip_qkv': 4.0, 'rope_theta': 10000.0},"
            + " 'ffn_config': {'ffn_hidden_size': 8, 'moe_num_experts': 2, 'moe_top_k': 1}";

    private static final String GROK_CONFIG = "'num_local_experts': 2";

    private static final String ARCTIC_CONFIG = "'num_local_experts': 2, 'vocab_size': 4";

    private Path dir() throws IOException {
        return tmp.newFolder().toPath();
    }

    // ------------------------------------------------------------------ Llama4Model

    @Test
    public void llama4FixupSwapsTheMoeAndMlpWidths() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.LLAMA4, "Llama4ForCausalLM", LLAMA4_CONFIG);
        assertEquals(4, ctx.config().findInt("intermediate_size").getAsInt());
        HfConfig fixed = parser(ctx, HfMoeClassicHooksA.llama4Fixup(ctx.config()));
        fixed.assertInt("intermediate_size", 8);
        fixed.assertInt("intermediate_size_moe", 4);
        HfLoadContext broken = context(dir(), Arch.LLAMA4, "Llama4ForCausalLM",
                "'intermediate_size': 4, 'intermediate_size_moe': 3");
        assertThrows(GgufFormatException.class, () -> HfMoeClassicHooksA.llama4Fixup(broken.config()));
        HfLoadContext brokenToo = context(dir(), Arch.LLAMA4, "Llama4ForCausalLM", "'intermediate_size_mlp': 8");
        // the context helper's base config carries an intermediate_size, so the broken case is a bare config
        com.eonmux.neuralfree.gguf.hf.config.HfConfig bare =
                com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader.fromJson(dir(),
                        "{\"architectures\":[\"Llama4ForCausalLM\"],\"intermediate_size_mlp\":8}");
        assertThrows(GgufFormatException.class, () -> HfMoeClassicHooksA.llama4Fixup(bare));
    }

    private static HfConfig parser(HfLoadContext ctx, com.eonmux.neuralfree.gguf.hf.config.HfConfig raw) {
        return new HfConfig(raw, ctx.config().source());
    }

    @Test
    public void llama4FusedGateUpExpertsAreTransposedAndSplit() throws IOException {
        String name = "model.layers.1.feed_forward.experts.gate_up_proj";
        Map<String, RowSource> index = Collections.singletonMap(name, seq(2, 4, 4));
        HfLoadContext ctx = context(dir(), Arch.LLAMA4, "Llama4ForCausalLM", LLAMA4_CONFIG, index, false);
        RecordingSink sink = new RecordingSink();
        HfMoeClassicHooksA.llama4Tensors(ctx, name, ctx.index().get(name), sink);
        assertEquals(2, sink.tensors.size());
        RowSource gate = tensors(sink, "ffn_gate_exps").get(0);
        RowSource up = tensors(sink, "ffn_up_exps").get(0);
        assertArrayEquals(new long[]{2, 2, 4}, gate.shape());
        assertArrayEquals(floats(0, 4, 8, 12, 1, 5, 9, 13, 16, 20, 24, 28, 17, 21, 25, 29), values(gate), 0f);
        assertArrayEquals(floats(2, 6, 10, 14, 3, 7, 11, 15, 18, 22, 26, 30, 19, 23, 27, 31), values(up), 0f);
    }

    @Test
    public void llama4DownExpertsAreTransposedWithAWeightSuffix() throws IOException {
        String name = "model.layers.0.feed_forward.experts.down_proj";
        Map<String, RowSource> index = Collections.singletonMap(name, seq(2, 4, 8));
        HfLoadContext ctx = context(dir(), Arch.LLAMA4, "Llama4ForCausalLM", LLAMA4_CONFIG, index, false);
        RecordingSink sink = new RecordingSink();
        HfMoeClassicHooksA.llama4Tensors(ctx, name, ctx.index().get(name), sink);
        RowSource down = tensors(sink, "ffn_down_exps").get(0);
        assertArrayEquals(new long[]{2, 8, 4}, down.shape());
        // (e, j, i) of the transposed stack is (e, i, j) of the source
        float[] src = values(ctx.index().get(name));
        float[] expected = new float[src.length];
        for (int e = 0; e < 2; e++) {
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < 4; i++) {
                    expected[e * 32 + j * 4 + i] = src[e * 32 + i * 8 + j];
                }
            }
        }
        assertArrayEquals(expected, values(down), 0f);
    }

    // ------------------------------------------------------------------ DbrxModel

    @Test
    public void dbrxExpertsAreViewedWithAWeightSuffixAndW2IsTransposed() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        for (String p : Arrays.asList("w1", "w2", "v1")) {
            index.put("transformer.blocks.0.ffn.experts.mlp." + p, seq(2, 8, 8));
        }
        HfLoadContext ctx = context(dir(), Arch.DBRX, "DbrxForCausalLM", DBRX_CONFIG, index, false);
        RecordingSink sink = new RecordingSink();
        for (String p : Arrays.asList("w1", "w2", "v1")) {
            String name = "transformer.blocks.0.ffn.experts.mlp." + p;
            HfMoeClassicHooksA.dbrxTensors(ctx, name, ctx.index().get(name), sink);
        }
        assertEquals(3, sink.tensors.size());
        for (Map.Entry<String, RowSource> e : sink.tensors.entrySet()) {
            assertTrue(e.getKey() + " gains the weight suffix", e.getKey().endsWith(".weight"));
        }
        RowSource w1 = tensors(sink, "ffn_gate_exps").get(0);
        RowSource w2 = tensors(sink, "ffn_down_exps").get(0);
        RowSource v1 = tensors(sink, "ffn_up_exps").get(0);
        assertArrayEquals(new long[]{2, 8, 8}, w1.shape());
        assertArrayEquals(values(ctx.index().get("transformer.blocks.0.ffn.experts.mlp.w1")), values(w1), 0f);
        assertArrayEquals(values(ctx.index().get("transformer.blocks.0.ffn.experts.mlp.v1")), values(v1), 0f);
        // (e, j, k) of w2 is (e, k, j) of the source
        float[] w2Expected = new float[128];
        int w = 0;
        for (int e = 0; e < 2; e++) {
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 8; k++) {
                    w2Expected[w++] = e * 64 + k * 8 + j;
                }
            }
        }
        assertArrayEquals(w2Expected, values(w2), 0f);
    }

    @Test
    public void dbrxAnExpertOfAnotherElementCountIsRejected() throws IOException {
        String name = "transformer.blocks.0.ffn.experts.mlp.w1";
        Map<String, RowSource> index = Collections.singletonMap(name, seq(2, 4));
        HfLoadContext ctx = context(dir(), Arch.DBRX, "DbrxForCausalLM", DBRX_CONFIG, index, false);
        assertThrows(GgufFormatException.class,
                () -> HfMoeClassicHooksA.dbrxTensors(ctx, name, ctx.index().get(name), new RecordingSink()));
    }

    @Test
    public void dbrxForcesF32OnMultiDimensionalTensors() {
        assertEquals(GGMLType.F32, HfMoeClassicHooksA.dbrxForceType("a.weight", 2,
                ElementFormat.of(GGMLType.F32), OutType.F32));
        assertNull(HfMoeClassicHooksA.dbrxForceType("a.weight", 1, ElementFormat.of(GGMLType.F32), OutType.F32));
        assertThrows(IllegalArgumentException.class,
                () -> HfMoeClassicHooksA.dbrxForceType("a.weight", 0, ElementFormat.of(GGMLType.F32), OutType.F32));
    }

    // ------------------------------------------------------------------ GrokModel

    @Test
    public void grokCompleteExpertsUnderTheDecoderNamesAreStacked() throws IOException {
        String prefix = "transformer.decoder_layer.0.moe.";
        Map<String, RowSource> index = new LinkedHashMap<>();
        for (int x = 0; x < 2; x++) {
            index.put(prefix + x + ".linear.weight", filled(10 * x, 2, 4));
            index.put(prefix + x + ".linear_1.weight", filled(10 * x + 1, 4, 2));
            index.put(prefix + x + ".linear_v.weight", filled(10 * x + 2, 2, 4));
        }
        HfLoadContext ctx = context(dir(), Arch.GROK, "GrokForCausalLM", GROK_CONFIG, index, false);
        RecordingSink sink = new RecordingSink();
        int placed = 0;
        for (Map.Entry<String, RowSource> e : index.entrySet()) {
            HfMoeClassicHooksA.grokTensors(ctx, e.getKey(), e.getValue(), sink);
            placed++;
            assertEquals(placed < 6 ? 0 : 3, sink.tensors.size());
        }
        HfMoeClassicHooksA.grokTensors(ctx, "transformer.decoder_layer.0.rms_norm.weight", seq(8), sink);
        assertEquals(4, sink.tensors.size());
        assertArrayEquals(new long[]{2, 2, 4}, tensors(sink, "ffn_gate_exps").get(0).shape());
        assertArrayEquals(new long[]{2, 4, 2}, tensors(sink, "ffn_down_exps").get(0).shape());
        // expert x fills every element with 10x plus the part offset one for linear_1 and two for linear_v
        assertArrayEquals(floats(0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10),
                values(tensors(sink, "ffn_gate_exps").get(0)), 0f);
    }

    @Test
    public void grokLeftoverExpertsFailTheCompletenessCheck() throws IOException {
        String prefix = "transformer.decoder_layer.0.moe.";
        Map<String, RowSource> index = new LinkedHashMap<>();
        for (int x = 0; x < 2; x++) {
            index.put(prefix + x + ".linear.weight", filled(10 * x, 2, 4));
            index.put(prefix + x + ".linear_1.weight", filled(10 * x + 1, 4, 2));
        }
        index.put(prefix + "1.linear_v.weight", filled(12, 2, 4));
        HfLoadContext ctx = context(dir(), Arch.GROK, "GrokForCausalLM", GROK_CONFIG, index, false);
        RecordingSink sink = new RecordingSink();
        for (Map.Entry<String, RowSource> e : index.entrySet()) {
            HfMoeClassicHooksA.grokTensors(ctx, e.getKey(), e.getValue(), sink);
        }
        assertThrows(GgufFormatException.class, () -> HfMoeClassicHooksA.grokExtra(ctx, sink));
    }

    // ------------------------------------------------------------------ ArcticModel

    @Test
    public void arcticMixtralExpertsStackUnderTheSpelledFeedForwardNames() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        for (int x = 0; x < 2; x++) {
            index.put("model.layers.0.block_sparse_moe.experts." + x + ".w1.weight", filled(10 * x, 2, 4));
            index.put("model.layers.0.block_sparse_moe.experts." + x + ".w2.weight", filled(10 * x + 1, 4, 2));
            index.put("model.layers.0.block_sparse_moe.experts." + x + ".w3.weight", filled(10 * x + 2, 2, 4));
        }
        index.put("model.layers.0.self_attn.q_proj.weight", seq(8, 8));
        HfLoadContext ctx = context(dir(), Arch.ARCTIC, "ArcticForCausalLM", ARCTIC_CONFIG, index, false);
        RecordingSink sink = new RecordingSink();
        for (int x = 0; x < 2; x++) {
            for (int w = 1; w <= 3; w++) {
                String name = "model.layers.0.block_sparse_moe.experts." + x + ".w" + w + ".weight";
                HfMoeClassicHooksA.arcticTensors(ctx, name, ctx.index().get(name), sink);
            }
        }
        assertEquals(3, sink.tensors.size());
        RowSource w1 = tensors(sink, "ffn_gate_exps").get(0);
        assertArrayEquals(new long[]{2, 2, 4}, w1.shape());
        assertArrayEquals(floats(0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10), values(w1), 0f);
        assertArrayEquals(new long[]{2, 4, 2}, tensors(sink, "ffn_down_exps").get(0).shape());
        HfMoeClassicHooksA.arcticTensors(ctx, "model.layers.0.self_attn.q_proj.weight",
                ctx.index().get("model.layers.0.self_attn.q_proj.weight"), sink);
        assertEquals(4, sink.tensors.size());
        HfMoeClassicHooksA.arcticExtra(ctx, sink);
    }

    @Test
    public void arcticLeftoversFailTheCompletenessCheck() throws IOException {
        String w1 = "model.layers.0.block_sparse_moe.experts.0.w1.weight";
        Map<String, RowSource> index = new LinkedHashMap<>();
        for (int x = 0; x < 2; x++) {
            for (String part : Arrays.asList("w1", "w2", "w3")) {
                index.put("model.layers.0.block_sparse_moe.experts." + x + "." + part + ".weight", seq(2, 4));
            }
        }
        index.put("model.layers.0.self_attn.q_proj.weight", seq(8, 8));
        HfLoadContext ctx = context(dir(), Arch.ARCTIC, "ArcticForCausalLM", ARCTIC_CONFIG, index, false);
        RecordingSink sink = new RecordingSink();
        // one expert part offered of the six planned: the leftover members fail the extra check
        HfMoeClassicHooksA.arcticTensors(ctx, w1, ctx.index().get(w1), sink);
        HfMoeClassicHooksA.arcticTensors(ctx, "model.layers.0.self_attn.q_proj.weight",
                ctx.index().get("model.layers.0.self_attn.q_proj.weight"), sink);
        assertThrows(GgufFormatException.class, () -> HfMoeClassicHooksA.arcticExtra(ctx, sink));
    }

    @Test
    public void arcticExpertWeightsOnlyAcceptTheThreeParts() throws IOException {
        String name = "model.layers.0.block_sparse_moe.experts.0.w4.weight";
        Map<String, RowSource> index = Collections.singletonMap(name, seq(2, 4));
        HfLoadContext ctx = context(dir(), Arch.ARCTIC, "ArcticForCausalLM", ARCTIC_CONFIG, index, false);
        assertThrows(GgufFormatException.class,
                () -> HfMoeClassicHooksA.arcticTensors(ctx, name, ctx.index().get(name), new RecordingSink()));
    }

    private static RowSource filled(float value, long rows, long cols) {
        float[] v = new float[(int) (rows * cols)];
        Arrays.fill(v, value);
        return RopeTensors.fromFloats(new long[]{rows, cols}, v);
    }

    /** The planned tensors whose gguf name holds {@code part}, in insertion order. */
    private static List<RowSource> tensors(RecordingSink sink, String part) {
        return sink.tensors.entrySet().stream()
                .filter(e -> e.getKey().contains(part))
                .map(Map.Entry::getValue)
                .collect(Collectors.toList());
    }

    /** A tiny required-integer read over {@code HfConfig}, kept local to this test. */
    private static final class HfConfig {
        private final com.eonmux.neuralfree.gguf.hf.config.HfConfig cfg;
        private final String source;

        HfConfig(com.eonmux.neuralfree.gguf.hf.config.HfConfig cfg, String source) {
            this.cfg = cfg;
            this.source = source;
        }

        void assertInt(String key, int expected) {
            com.eonmux.neuralfree.gguf.hf.config.HfConfig c = cfg.findInt(key).isPresent() ? cfg : cfg;
            int v = c.findInt(key).orElseThrow();
            if (v != expected) {
                throw new AssertionError(source + ": " + key + " is " + v + ", not " + expected);
            }
        }
    }
}

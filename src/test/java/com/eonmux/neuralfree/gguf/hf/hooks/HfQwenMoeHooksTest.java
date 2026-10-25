package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.intValue;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.key;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfQwenMoeHooks} against {@code Qwen2MoeModel} (LC/conversion/qwen.py:81-161): the expert lengths, the
 * Transformers 5 fused {@code gate_up_proj}/{@code down_proj} experts, the per-expert tensors stacked per layer, and
 * the {@code Unprocessed experts} checks.
 */
public class HfQwenMoeHooksTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    private static final String MOE_CONFIG = "'num_experts': 2, 'num_experts_per_tok': 1,"
            + " 'moe_intermediate_size': 2, 'shared_expert_intermediate_size': 6";

    private HfLoadContext context(Map<String, RowSource> index) throws IOException {
        return HfQwenHooksTest.context(tmp.newFolder().toPath(), Arch.QWEN2MOE, "Qwen2MoeForCausalLM", MOE_CONFIG,
                index, false);
    }

    /** Runs the tensor hook over the index in order, skipping consumed names, then the completeness check. */
    private static RecordingSink plan(HfLoadContext ctx) throws IOException {
        RecordingSink sink = new RecordingSink();
        for (String name : ctx.index().names()) {
            if (!sink.isConsumed(name)) {
                HfQwenMoeHooks.qwen2MoeTensors(ctx, name, ctx.index().get(name), sink);
            }
        }
        HfQwenMoeHooks.qwen2MoeExtra(ctx, sink);
        return sink;
    }

    /** A {@code [rows, cols]} F32 tensor filled with {@code value}. */
    private static RowSource filled(float value, long rows, long cols) {
        float[] v = new float[(int) (rows * cols)];
        Arrays.fill(v, value);
        return RopeTensors.fromFloats(new long[]{rows, cols}, v);
    }

    // ------------------------------------------------------------------ metadata

    @Test
    public void metadataWritesTheExpertLengths() throws IOException {
        HfLoadContext ctx = context(Collections.emptyMap());
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN2MOE.ggufName());
        HfQwenMoeHooks.qwen2MoeMeta(ctx, kv, null);
        HfMetadata m = kv.build();
        assertEquals(2, intValue(m, key(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, Arch.QWEN2MOE)));
        assertEquals(6, intValue(m, key(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, Arch.QWEN2MOE)));
        assertTrue(m.has(key(GgufKeys.BLOCK_COUNT, Arch.QWEN2MOE)));
    }

    // ------------------------------------------------------------------ fused experts

    @Test
    public void fusedGateUpSplitsInHalvesGateFirst() throws IOException {
        HfLoadContext ctx = context(Collections.emptyMap());
        RecordingSink sink = new RecordingSink();
        HfQwenMoeHooks.qwen2MoeTensors(ctx, "model.layers.0.mlp.experts.gate_up_proj", seq(2, 4, 3), sink);
        assertEquals(Arrays.asList("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"),
                List.copyOf(sink.tensors.keySet()));
        RowSource gate = sink.tensors.get("blk.0.ffn_gate_exps.weight");
        RowSource up = sink.tensors.get("blk.0.ffn_up_exps.weight");
        assertArrayEquals(new long[]{2, 2, 3}, gate.shape());
        assertArrayEquals(new long[]{2, 2, 3}, up.shape());
        assertArrayEquals(floats(0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17), values(gate), 0f);
        assertArrayEquals(floats(6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23), values(up), 0f);
    }

    @Test
    public void fusedGateUpWithAWeightSuffixIsSplitAlike() throws IOException {
        HfLoadContext ctx = context(Collections.emptyMap());
        RecordingSink sink = new RecordingSink();
        HfQwenMoeHooks.qwen2MoeTensors(ctx, "model.layers.1.mlp.experts.gate_up_proj.weight", seq(2, 4, 3), sink);
        assertEquals(Arrays.asList("blk.1.ffn_gate_exps.weight", "blk.1.ffn_up_exps.weight"),
                List.copyOf(sink.tensors.keySet()));
    }

    @Test
    public void fusedDownProjGetsTheWeightSuffix() throws IOException {
        HfLoadContext ctx = context(Collections.emptyMap());
        RecordingSink sink = new RecordingSink();
        HfQwenMoeHooks.qwen2MoeTensors(ctx, "model.layers.1.mlp.experts.down_proj", seq(2, 8, 2), sink);
        HfQwenMoeHooks.qwen2MoeTensors(ctx, "model.layers.0.mlp.experts.down_proj.weight", seq(2, 8, 2), sink);
        assertEquals(Arrays.asList("blk.1.ffn_down_exps.weight", "blk.0.ffn_down_exps.weight"),
                List.copyOf(sink.tensors.keySet()));
    }

    @Test
    public void fusedGateUpOfAnOddOrFlatShapeIsRejected() throws IOException {
        HfLoadContext ctx = context(Collections.emptyMap());
        String name = "model.layers.0.mlp.experts.gate_up_proj";
        GgufFormatException odd = assertThrows(GgufFormatException.class,
                () -> HfQwenMoeHooks.qwen2MoeTensors(ctx, name, seq(2, 3, 3), new RecordingSink()));
        assertEquals("Unexpected gate_up_proj shape for " + name + ": (2, 3, 3)", odd.getMessage());
        GgufFormatException flat = assertThrows(GgufFormatException.class,
                () -> HfQwenMoeHooks.qwen2MoeTensors(ctx, name, seq(4, 3), new RecordingSink()));
        assertEquals("Unexpected gate_up_proj shape for " + name + ": (4, 3)", flat.getMessage());
    }

    // ------------------------------------------------------------------ per-expert tensors

    /** One layer of per-expert tensors in checkpoint order; expert {@code x} part {@code p} holds {@code 10x + p}. */
    private static Map<String, RowSource> perExpertIndex() {
        Map<String, RowSource> index = new LinkedHashMap<>();
        index.put("model.layers.0.mlp.gate.weight", seq(2, 8));
        for (int x = 0; x < 2; x++) {
            String prefix = "model.layers.0.mlp.experts." + x + ".";
            index.put(prefix + "down_proj.weight", filled(10 * x, 8, 2));
            index.put(prefix + "gate_proj.weight", filled(10 * x + 1, 2, 8));
            index.put(prefix + "up_proj.weight", filled(10 * x + 2, 2, 8));
        }
        index.put("model.layers.0.mlp.shared_expert.gate_proj.weight", seq(6, 8));
        index.put("model.layers.0.mlp.shared_expert_gate.weight", seq(1, 8));
        return index;
    }

    @Test
    public void perExpertTensorsAreStackedPerLayer() throws IOException {
        RecordingSink sink = plan(context(perExpertIndex()));
        assertEquals(Arrays.asList("blk.0.ffn_gate_inp.weight", "blk.0.ffn_down_exps.weight",
                "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_shexp.weight",
                "blk.0.ffn_gate_inp_shexp.weight"), List.copyOf(sink.tensors.keySet()));
        RowSource gate = sink.tensors.get("blk.0.ffn_gate_exps.weight");
        assertArrayEquals(new long[]{2, 2, 8}, gate.shape());
        float[] v = values(gate);
        for (int i = 0; i < v.length; i++) {
            assertEquals(i < 16 ? 1f : 11f, v[i], 0f);
        }
        assertArrayEquals(new long[]{2, 8, 2}, sink.tensors.get("blk.0.ffn_down_exps.weight").shape());
        assertEquals(5, sink.consumed.size());
    }

    @Test
    public void layerWhoseExpertsWereNotAllPlannedFailsTheCompletenessCheck() throws IOException {
        HfLoadContext ctx = context(perExpertIndex());
        RecordingSink sink = new RecordingSink();
        String first = "model.layers.0.mlp.experts.0.down_proj.weight";
        HfQwenMoeHooks.qwen2MoeTensors(ctx, first, ctx.index().get(first), sink);
        assertFalse(sink.tensors.containsKey("blk.0.ffn_down_exps.weight"));
        assertThrows(GgufFormatException.class, () -> HfQwenMoeHooks.qwen2MoeExtra(ctx, sink));
    }

    @Test
    public void completenessCheckPassesWithoutExperts() throws IOException {
        HfLoadContext ctx = context(Collections.emptyMap());
        HfQwenMoeHooks.qwen2MoeExtra(ctx, new RecordingSink());
    }

    @Test
    public void expertTensorOfAnotherPartIsUnprocessed() throws IOException {
        String name = "model.layers.0.mlp.experts.0.w1.weight";
        HfLoadContext ctx = context(Collections.singletonMap(name, seq(2, 8)));
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfQwenMoeHooks.qwen2MoeTensors(ctx, name, seq(2, 8), new RecordingSink()));
        assertEquals("Unprocessed experts: ['" + name + "']", e.getMessage());
    }

    @Test
    public void expertOutsideTheConfiguredCountIsRejected() throws IOException {
        String name = "model.layers.0.mlp.experts.2.up_proj.weight";
        HfLoadContext ctx = context(Collections.singletonMap(name, seq(2, 8)));
        assertThrows(GgufFormatException.class,
                () -> HfQwenMoeHooks.qwen2MoeTensors(ctx, name, seq(2, 8), new RecordingSink()));
    }
}

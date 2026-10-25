package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link HfMiniMaxHooks}: the MiniMax-M2 expert feed-forward length and RoPE dimension count
 * (LC/conversion/minimax.py:128-132) and the per-layer stacking of the {@code w1}/{@code w2}/{@code w3} experts
 * (minimax.py:134-161).
 */
public class HfMiniMaxHooksTest {

    private static final String M2 = "'architectures': ['MiniMaxM2ForCausalLM'], 'hidden_size': 8,"
            + " 'intermediate_size': 6, 'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 4,"
            + " 'num_hidden_layers': 2, 'max_position_embeddings': 4096, 'rms_norm_eps': 1e-06,"
            + " 'vocab_size': 32, 'rope_theta': 5000000, 'rotary_dim': 2, 'num_local_experts': 2,"
            + " 'num_experts_per_tok': 1, 'scoring_func': 'sigmoid'";

    private static HfArchEntry entry() {
        return HfArchEntry.of(List.of("MiniMaxM2ForCausalLM"), HfArchResolver.of(Arch.MINIMAX_M2),
                HfMiniMaxHooks::meta, HfMiniMaxHooks::tensors, HfMiniMaxHooks::vocab);
    }

    private static HfLoadContext context(String json, Map<String, RowSource> index) {
        return HfLoadContext.builder(HfConfigLoader.fromJson(Path.of("."), "{" + json.replace('\'', '"') + "}"),
                entry(), Arch.MINIMAX_M2).index(HfTensorIndex.of(index)).build();
    }

    @Test
    public void writesTheExpertLengthAndPartialRopeDimension() {
        HfLoadContext ctx = context(M2, Map.of());
        HfMetadataBuilder kv = new HfMetadataBuilder("minimax-m2");
        HfMiniMaxHooks.meta(ctx, kv, null);
        HfMetadata m = kv.build();
        assertEquals(6, m.get("minimax-m2.expert_feed_forward_length"));
        assertEquals(2, m.get("minimax-m2.rope.dimension_count"));
        assertEquals(GGUFValueType.UINT32, m.type("minimax-m2.rope.dimension_count"));

        HfLoadContext noRotary = context(M2.replace(" 'rotary_dim': 2,", ""), Map.of());
        assertThrows(GgufFormatException.class,
                () -> HfMiniMaxHooks.meta(noRotary, new HfMetadataBuilder("minimax-m2"), null));
    }

    @Test
    public void stacksTheW1W2W3ExpertsOfEachLayer() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        for (int x = 0; x < 2; x++) {
            for (String w : List.of("w3", "w1", "w2")) {
                index.put("model.layers.0.block_sparse_moe.experts." + x + "." + w + ".weight",
                        RopeTensors.constant(new long[]{3, 4}, x + 1));
            }
        }
        // the router bias as the inherited filter renames it (LC/conversion/base.py:588-589)
        index.put("model.layers.0.block_sparse_moe.e_score_correction.bias", RopeTensors.constant(new long[]{2}, 0f));
        index.put("model.layers.0.self_attn.q_proj.weight", RopeTensors.constant(new long[]{8, 8}, 0f));
        HfLoadContext ctx = context(M2, index);
        Sink out = new Sink();
        for (String name : ctx.index().names()) {
            if (!out.isConsumed(name)) {
                HfMiniMaxHooks.tensors(ctx, name, ctx.index().get(name), out);
            }
        }
        assertEquals(List.of("blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight", "blk.0.ffn_up_exps.weight",
                "blk.0.exp_probs_b.bias", "blk.0.attn_q.weight"), new ArrayList<>(out.added.keySet()));
        RowSource gate = out.added.get("blk.0.ffn_gate_exps.weight");
        assertArrayEquals(new long[]{2, 3, 4}, gate.shape());
        float[] v = values(gate);
        assertEquals(1f, v[0], 0f);
        assertEquals(2f, v[12], 0f);

        Map<String, RowSource> incomplete = new LinkedHashMap<>(index);
        incomplete.remove("model.layers.0.block_sparse_moe.experts.1.w2.weight");
        HfLoadContext bad = context(M2, incomplete);
        assertThrows(GgufFormatException.class, () -> HfMiniMaxHooks.tensors(bad,
                "model.layers.0.block_sparse_moe.experts.0.w3.weight",
                incomplete.get("model.layers.0.block_sparse_moe.experts.0.w3.weight"), new Sink()));
        HfLoadContext stray = context(M2, Map.of());
        assertThrows(GgufFormatException.class, () -> HfMiniMaxHooks.tensors(stray,
                "model.layers.0.block_sparse_moe.experts.scale", RopeTensors.constant(new long[]{1}, 0f),
                new Sink()));
    }

    private static float[] values(RowSource s) throws IOException {
        ByteBuffer b = ByteBuffer.allocate((int) (s.rows() * s.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
        s.readRows(0, (int) s.rows(), b);
        b.flip();
        float[] out = new float[b.remaining() / 4];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getFloat();
        }
        return out;
    }

    /** A sink recording what the hook plans and consumes. */
    private static final class Sink implements TensorSink {
        final Map<String, RowSource> added = new LinkedHashMap<>();
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
            throw new AssertionError("unexpected drop of " + hfName);
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

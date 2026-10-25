package com.eonmux.neuralfree.gguf.hf.names;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.util.Optional;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfTensorNameMap} against gguf-py's {@code TensorNameMap}: identity keys, suffix retry, block bounds,
 * last-wins pattern collisions, per-architecture block overrides that leave other architectures untouched, and the
 * name builders. Expected names were derived from the generated tables with the converter's algorithm.
 */
public class HfTensorNameMapTest {

    private static String map(Arch arch, int blocks, String hfName) {
        return HfTensorNameMap.forArch(arch, blocks).mapOrThrow(hfName);
    }

    @Test
    public void llamaNamesMapWithSuffixRetry() {
        assertEquals("blk.1.attn_q.weight", map(Arch.LLAMA, 2, "model.layers.1.self_attn.q_proj.weight"));
        assertEquals("token_embd.weight", map(Arch.LLAMA, 2, "model.embed_tokens.weight"));
        assertEquals("token_embd.weight", map(Arch.LLAMA, 2, "tok_embeddings.weight"));
        assertEquals("output.weight", map(Arch.LLAMA, 2, "lm_head.weight"));
        assertEquals("output_norm.weight", map(Arch.LLAMA, 2, "model.norm.weight"));
        assertEquals("output_norm.bias", map(Arch.LLAMA, 2, "model.norm.bias"));
        assertEquals("blk.0.ffn_norm.weight", map(Arch.LLAMA, 2, "model.layers.0.post_attention_layernorm.weight"));
        assertEquals("rope_freqs.weight", map(Arch.LLAMA, 2, "rope_freqs.weight"));
    }

    @Test
    public void ggufNamesAreIdentityKeys() {
        HfTensorNameMap m = HfTensorNameMap.forArch(Arch.LLAMA, 2);
        Optional<HfTensorNameMap.Mapped> exact = m.map("blk.0.attn_q");
        assertTrue(exact.isPresent());
        assertEquals("ATTN_Q", exact.get().tensor());
        assertEquals("blk.0.attn_q", exact.get().ggufName());
        assertEquals("blk.0.attn_q.weight", m.mapOrThrow("blk.0.attn_q.weight"));
        assertEquals("output_norm.weight", m.mapOrThrow("output_norm.weight"));
    }

    @Test
    public void suffixesAreTriedOnlyWhenGiven() {
        HfTensorNameMap m = HfTensorNameMap.forArch(Arch.LLAMA, 1);
        assertFalse(m.map("model.layers.0.self_attn.q_proj.weight").isPresent());
        assertEquals("blk.0.attn_q.weight",
                m.map("model.layers.0.self_attn.q_proj.weight", ".weight").get().ggufName());
        assertEquals("blk.0.attn_q.bias",
                m.map("model.layers.0.self_attn.q_proj.bias", ".weight", ".bias").get().ggufName());
        assertFalse(m.map("model.layers.0.self_attn.q_proj.scales", ".weight", ".bias").isPresent());
    }

    @Test
    public void blocksBeyondTheCountAndUnlistedTensorsDoNotMap() {
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> map(Arch.LLAMA, 2, "model.layers.2.self_attn.q_proj.weight"));
        assertEquals("Can not map tensor 'model.layers.2.self_attn.q_proj.weight'", e.getMessage());
        assertFalse(HfTensorNameMap.forArch(Arch.LLAMA, 2)
                .map("model.layers.0.self_attn.q_norm.weight", ".weight").isPresent());
        assertEquals("blk.0.attn_q_norm.weight", map(Arch.QWEN3, 2, "model.layers.0.self_attn.q_norm.weight"));
        assertEquals("blk.1.attn_k_norm.bias", map(Arch.QWEN3, 2, "model.layers.1.self_attn.k_norm.bias"));
    }

    @Test
    public void laterTensorsWinSharedPatterns() {
        // FFN_POST_NORM lists model.layers.{bid}.feed_forward.up_proj before FFN_UP does; both are gemma3 tensors.
        assertEquals("blk.0.ffn_up.weight", map(Arch.GEMMA3, 2, "model.layers.0.feed_forward.up_proj.weight"));
        // FFN_UP_SHEXP lists model.layers.{bid}.feed_forward.down_proj before FFN_DOWN does.
        assertEquals("blk.1.ffn_down.weight", map(Arch.DEEPSEEK2, 2, "model.layers.1.feed_forward.down_proj.weight"));
        // TOKEN_EMBD_NORM lists transformer.norm before OUTPUT_NORM does (model-level patterns).
        assertEquals("output_norm.weight", map(Arch.BERT, 1, "transformer.norm.weight"));
    }

    @Test
    public void emptyPatternListsMapOnlyTheGgufName() {
        HfTensorNameMap m = HfTensorNameMap.forArch(Arch.PHI3, 1);
        assertFalse(m.map("-").isPresent());
        assertEquals("rope_factors_long.weight", m.mapOrThrow("rope_factors_long.weight"));
    }

    @Test
    public void archBlockOverridesReplaceAndExtendPatterns() {
        assertEquals("blk.0.ffn_norm.weight", map(Arch.ARCTIC, 2, "model.layers.0.residual_layernorm.weight"));
        assertEquals("blk.1.ffn_norm_exps.weight", map(Arch.ARCTIC, 2,
                "model.layers.1.post_attention_layernorm.weight"));
        assertEquals("blk.2.hc_attn_norm.weight", map(Arch.QWEN4EXP, 3,
                "model.layers.2.attn_hyper_connection.hc_norm.weight"));
        assertEquals("output_hc_norm.weight", map(Arch.QWEN4EXP, 3, "model.hyper_connection_mixer.hc_norm.weight"));
        assertEquals("blk.0.indexer.q_norm.weight", map(Arch.QWEN4EXP, 1,
                "model.layers.0.self_attn.indexer.q_layernorm.weight"));
        assertFalse(HfTensorNameMap.forArch(Arch.QWEN4EXP, 1)
                .map("model.layers.0.self_attn.index_q_norm.weight", ".weight").isPresent());
        // The override of one architecture does not leak into another.
        assertEquals("blk.0.indexer.q_norm.weight", map(Arch.MINIMAX_M3, 1,
                "model.layers.0.self_attn.index_q_norm.weight"));
        assertEquals("blk.0.ffn_norm.weight", map(Arch.LLAMA, 1, "model.layers.0.post_attention_layernorm.weight"));
    }

    @Test
    public void mapsAreCachedPerArchAndBlockCount() {
        assertSame(HfTensorNameMap.forArch(Arch.LLAMA, 4), HfTensorNameMap.forArch(Arch.LLAMA, 4));
        assertEquals(4, HfTensorNameMap.forArch(Arch.LLAMA, 4).blockCount());
        assertEquals(Arch.LLAMA, HfTensorNameMap.forArch(Arch.LLAMA, 4).arch());
    }

    @Test
    public void invalidArgumentsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> HfTensorNameMap.forArch(Arch.UNKNOWN, 1));
        assertThrows(IllegalArgumentException.class, () -> HfTensorNameMap.forArch(Arch.LLAMA, -1));
        assertThrows(IllegalArgumentException.class, () -> HfTensorNameMap.forArch(Arch.LLAMA, 4097));
        assertThrows(NullPointerException.class, () -> HfTensorNameMap.forArch(null, 1));
    }

    @Test
    public void namesAreBuiltFromTensors() {
        HfTensorNameMap m = HfTensorNameMap.forArch(Arch.LLAMA, 4);
        assertEquals("blk.3.attn_q", m.ggufName("ATTN_Q", 3));
        assertEquals("blk.3.attn_q.weight", m.ggufName("ATTN_Q", 3, ".weight"));
        assertEquals("output", m.ggufName("OUTPUT", -1));
        assertThrows(IllegalArgumentException.class, () -> m.ggufName("ATTN_Q", -1));
        assertThrows(IllegalArgumentException.class, () -> m.ggufName("OUTPUT", 0));
        assertThrows(IllegalArgumentException.class, () -> m.ggufName("ATTN_Q_NORM", 0));
        assertTrue(m.hasTensor("ATTN_Q"));
        assertFalse(m.hasTensor("ATTN_Q_NORM"));
        assertTrue(m.matches("blk.2.ffn_up.weight", "FFN_UP", 2, ".weight"));
        assertFalse(m.matches("blk.2.ffn_up.weight", "FFN_UP", 1, ".weight"));
        assertFalse(m.matches("output.weight", "OUTPUT", 0, ".weight"));
        assertTrue(m.matches("output.weight", "OUTPUT", -1, ".weight"));
        assertFalse(m.matches("blk.0.attn_q_norm.weight", "ATTN_Q_NORM", 0, ".weight"));
    }
}

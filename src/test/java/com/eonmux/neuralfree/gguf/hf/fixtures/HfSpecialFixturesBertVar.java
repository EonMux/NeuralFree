package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.hf.fixtures.HfSpecialFixturesBert;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

/**
 * Synthetic checkpoint of the DistilBert row: {@code DistilBertModel} on the
 * {@code distil} hooks of {@link com.eonmux.neuralfree.gguf.hf.hooks.HfBertVariantHooks} over the shared
 * phantom-space WordPiece tokenizer. Two layers, hidden size 8, the exact tensor names and config keys of
 * LC/conversion/bert.py:243-264.
 */
public final class HfSpecialFixturesBertVar {

    private HfSpecialFixturesBertVar() {
    }

    /**
     * {@code DistilBertModel}: the {@code distilbert.} strip, the {@code vocab_} drops and the
     * {@code transformer.layer.N} names with the 1e-12 layer-norm epsilon.
     */
    public static SyntheticHf.Builder distilbert() {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json(
                "{'architectures': ['DistilBertModel'], 'model_type': 'distilbert', 'dim': 8,"
                        + " 'hidden_size': 8, 'n_layers': 2, 'num_hidden_layers': 2, 'n_heads': 2,"
                        + " 'num_attention_heads': 2, 'intermediate_size': 16, 'max_position_embeddings': 64,"
                        + " 'vocab_size': 8, 'pad_token_id': 0}"))
                .tokenizerJson(HfSpecialFixturesBert.WORD_PIECE_TOKENIZER)
                .tokenizerConfigJson(HfSpecialFixturesBert.WORD_PIECE_CONFIG);
        b.bf16("distilbert.embeddings.word_embeddings.weight", 8, 8);
        b.bf16("distilbert.embeddings.position_embeddings.weight", 64, 8);
        b.bf16("distilbert.embeddings.LayerNorm.weight", 8);
        b.bf16("distilbert.embeddings.LayerNorm.bias", 8);
        for (int l = 0; l < 2; l++) {
            b.bf16("distilbert.transformer.layer." + l + ".attention.q_lin.weight", 8, 8);
            b.bf16("distilbert.transformer.layer." + l + ".attention.q_lin.bias", 8);
            b.bf16("distilbert.transformer.layer." + l + ".attention.k_lin.weight", 8, 8);
            b.bf16("distilbert.transformer.layer." + l + ".attention.k_lin.bias", 8);
            b.bf16("distilbert.transformer.layer." + l + ".attention.v_lin.weight", 8, 8);
            b.bf16("distilbert.transformer.layer." + l + ".attention.v_lin.bias", 8);
            b.bf16("distilbert.transformer.layer." + l + ".attention.out_lin.weight", 8, 8);
            b.bf16("distilbert.transformer.layer." + l + ".attention.out_lin.bias", 8);
            b.bf16("distilbert.transformer.layer." + l + ".sa_layer_norm.weight", 8);
            b.bf16("distilbert.transformer.layer." + l + ".sa_layer_norm.bias", 8);
            b.bf16("distilbert.transformer.layer." + l + ".ffn.lin1.weight", 16, 8);
            b.bf16("distilbert.transformer.layer." + l + ".ffn.lin1.bias", 16);
            b.bf16("distilbert.transformer.layer." + l + ".ffn.lin2.weight", 8, 16);
            b.bf16("distilbert.transformer.layer." + l + ".ffn.lin2.bias", 8);
            b.bf16("distilbert.transformer.layer." + l + ".output_layer_norm.weight", 8);
            b.bf16("distilbert.transformer.layer." + l + ".output_layer_norm.bias", 8);
        }
        b.bf16("vocab_.LayerNorm.weight", 8);
        return b;
    }

    /** The single fixture of this file. */
    public static java.util.List<HfSpecialFixturesBert.Named> all() {
        return java.util.Collections.singletonList(HfSpecialFixturesBert.of("DistilBertModel",
                HfSpecialFixturesBertVar::distilbert));
    }
}

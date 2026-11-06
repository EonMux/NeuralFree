package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic WordPiece/BPE checkpoints of the plain BERT rows on
 * {@link com.eonmux.neuralfree.gguf.hf.hooks.HfBertHooks}: {@code BertModel}, {@code
 * BertForSequenceClassification} and {@code RobertaModel}. Two layers, hidden size 8, two heads, the
 * sentences-transformers phantom-space WordPiece vocabulary and the exact tensor names and config keys of
 * LC/conversion/bert.py.
 */
public final class HfSpecialFixturesBert {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int FF = 16;
    private static final int LABELS = 3;

    private HfSpecialFixturesBert() {
    }

    /**
     * The phantom-space WordPiece tokenizer.json of a BERT checkpoint: {@code [PAD] [UNK] [CLS] [SEP]}
     * , then two continuing pieces and one base piece. Also loaded with phantom space markers.
     */
    public static final String WORD_PIECE_TOKENIZER = SyntheticHf.json("{'added_tokens':["
            + "{'id':0,'content':'[PAD]','special':true,'normalized':false},"
            + "{'id':1,'content':'[UNK]','special':true,'normalized':false},"
            + "{'id':2,'content':'[CLS]','special':true,'normalized':false},"
            + "{'id':3,'content':'[SEP]','special':true,'normalized':false},"
            + "{'id':4,'content':'[UNK]','special':true,'normalized':false}],"
            + "'normalizer':null,"
            + "'pre_tokenizer':{'type':'BertPreTokenizer'},"
            + "'post_processor':null,"
            + "'decoder':{'type':'WordPiece','prefix':'##'},"
            + "'model':{'type':'WordPiece','unk_token':'[UNK]','continuing_subword_prefix':'##',"
            + "'vocab':{'[PAD]':0,'[UNK]':1,'[CLS]':2,'[SEP]':3,'a':4,'b':5,'##x':6,'##y':7}}}");

    /** The tokenizer_config.json of {@link #WORD_PIECE_TOKENIZER}. */
    public static final String WORD_PIECE_CONFIG = SyntheticHf.json(
            "{'cls_token':'[CLS]','sep_token':'[SEP]','mask_token':'[UNK]','pad_token':'[PAD]',"
                    + "'unk_token':'[UNK]','model_max_length':64}");

    /** The registers of a WordPiece checkpoint on the model directory. */
    static SyntheticHf.Builder bertBase(String singleQuotedConfig) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json(singleQuotedConfig))
                .tokenizerJson(WORD_PIECE_TOKENIZER)
                .tokenizerConfigJson(WORD_PIECE_CONFIG);
        b.bf16("bert.embeddings.word_embeddings.weight", 8, DIM);
        b.bf16("bert.embeddings.position_embeddings.weight", 64, DIM);
        b.bf16("bert.embeddings.token_type_embeddings.weight", 2, DIM);
        b.bf16("bert.embeddings.LayerNorm.weight", DIM);
        b.bf16("bert.embeddings.LayerNorm.bias", DIM);
        for (int l = 0; l < LAYERS; l++) {
            b.bf16("bert.encoder.layer." + l + ".attention.self.query.weight", DIM, DIM);
            b.bf16("bert.encoder.layer." + l + ".attention.self.query.bias", DIM);
            b.bf16("bert.encoder.layer." + l + ".attention.self.key.weight", DIM, DIM);
            b.bf16("bert.encoder.layer." + l + ".attention.self.key.bias", DIM);
            b.bf16("bert.encoder.layer." + l + ".attention.self.value.weight", DIM, DIM);
            b.bf16("bert.encoder.layer." + l + ".attention.self.value.bias", DIM);
            b.bf16("bert.encoder.layer." + l + ".attention.output.dense.weight", DIM, DIM);
            b.bf16("bert.encoder.layer." + l + ".attention.output.dense.bias", DIM);
            b.bf16("bert.encoder.layer." + l + ".attention.output.LayerNorm.weight", DIM);
            b.bf16("bert.encoder.layer." + l + ".attention.output.LayerNorm.bias", DIM);
            b.bf16("bert.encoder.layer." + l + ".intermediate.dense.weight", FF, DIM);
            b.bf16("bert.encoder.layer." + l + ".intermediate.dense.bias", FF);
            b.bf16("bert.encoder.layer." + l + ".output.dense.weight", DIM, FF);
            b.bf16("bert.encoder.layer." + l + ".output.dense.bias", DIM);
            b.bf16("bert.encoder.layer." + l + ".output.LayerNorm.weight", DIM);
            b.bf16("bert.encoder.layer." + l + ".output.LayerNorm.bias", DIM);
        }
        return b;
    }

    /** {@code BertModel}: the phantom-space WordPiece vocabulary and the classic samuel encoder names. */
    public static SyntheticHf.Builder bert() {
        return bertBase("{'architectures': ['BertModel'], 'model_type': 'bert', 'hidden_size': 8,"
                + " 'intermediate_size': 16, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 2, 'max_position_embeddings': 64, 'vocab_size': 8,"
                + " 'type_vocab_size': 2, 'pad_token_id': 0}");
    }

    /**
     * {@code BertForSequenceClassification}: real {@code id2label} labels, so {@code classifier.weight}
     * becomes {@code classifier.out_proj.weight}; the pooler of the checkpoint is dropped.
     */
    public static SyntheticHf.Builder bertClassification() {
        SyntheticHf.Builder b = bertBase("{'architectures': ['BertForSequenceClassification'],"
                + " 'model_type': 'bert', 'hidden_size': 8, 'intermediate_size': 16, 'num_hidden_layers': 2,"
                + " 'num_attention_heads': 2, 'num_key_value_heads': 2, 'max_position_embeddings': 64,"
                + " 'vocab_size': 8, 'type_vocab_size': 2, 'pad_token_id': 0,"
                + " 'id2label': {'0': 'entailment', '1': 'neutral', '2': 'contradiction'}}");
        b.bf16("bert.pooler.dense.weight", DIM, DIM);
        b.bf16("bert.pooler.dense.bias", DIM);
        b.bf16("classifier.weight", LABELS, DIM);
        b.bf16("classifier.bias", LABELS);
        return b;
    }

    /**
     * {@code RobertaModel}: a GPT-2 BPE tokenizer.json (the converter's RobertaModel branch), the
     * {@code pad_token_id} fixup that cuts {@code max_position_embeddings} by {@code pad + 1} and the
     * position-embedding row chop of bert.py:308-314.
     */
    public static SyntheticHf.Builder roberta() {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json(
                "{'architectures': ['RobertaModel'], 'model_type': 'roberta', 'hidden_size': 8,"
                        + " 'intermediate_size': 16, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                        + " 'num_key_value_heads': 2, 'max_position_embeddings': 10, 'vocab_size': 8,"
                        + " 'type_vocab_size': 2, 'pad_token_id': 2, 'position_embedding_type': 'absolute'}"));
        b.bf16("roberta.embeddings.word_embeddings.weight", 8, DIM);
        b.bf16("roberta.embeddings.position_embeddings.weight", 10, DIM);
        b.bf16("roberta.embeddings.token_type_embeddings.weight", 2, DIM);
        b.bf16("roberta.embeddings.LayerNorm.weight", DIM);
        b.bf16("roberta.embeddings.LayerNorm.bias", DIM);
        for (int l = 0; l < LAYERS; l++) {
            b.bf16("roberta.encoder.layer." + l + ".attention.self.query.weight", DIM, DIM);
            b.bf16("roberta.encoder.layer." + l + ".attention.self.query.bias", DIM);
            b.bf16("roberta.encoder.layer." + l + ".attention.self.key.weight", DIM, DIM);
            b.bf16("roberta.encoder.layer." + l + ".attention.self.key.bias", DIM);
            b.bf16("roberta.encoder.layer." + l + ".attention.self.value.weight", DIM, DIM);
            b.bf16("roberta.encoder.layer." + l + ".attention.self.value.bias", DIM);
            b.bf16("roberta.encoder.layer." + l + ".attention.output.dense.weight", DIM, DIM);
            b.bf16("roberta.encoder.layer." + l + ".attention.output.dense.bias", DIM);
            b.bf16("roberta.encoder.layer." + l + ".attention.output.LayerNorm.weight", DIM);
            b.bf16("roberta.encoder.layer." + l + ".attention.output.LayerNorm.bias", DIM);
            b.bf16("roberta.encoder.layer." + l + ".intermediate.dense.weight", FF, DIM);
            b.bf16("roberta.encoder.layer." + l + ".intermediate.dense.bias", FF);
            b.bf16("roberta.encoder.layer." + l + ".output.dense.weight", DIM, FF);
            b.bf16("roberta.encoder.layer." + l + ".output.dense.bias", DIM);
            b.bf16("roberta.encoder.layer." + l + ".output.LayerNorm.weight", DIM);
            b.bf16("roberta.encoder.layer." + l + ".output.LayerNorm.bias", DIM);
        }
        return b;
    }

    /** One fixture: its leading HF architecture name and the checkpoint builder. */
    public static final class Named {
        /** The HF architecture of the fixture. */
        public final String hfArch;
        /** The builder of the checkpoint. */
        public final Supplier<SyntheticHf.Builder> fixture;

        Named(String hfArch, Supplier<SyntheticHf.Builder> fixture) {
            this.hfArch = hfArch;
            this.fixture = fixture;
        }

        @Override
        public String toString() {
            return hfArch;
        }
    }

    private static Named named(String hfArch, Supplier<SyntheticHf.Builder> fixture) {
        return new Named(hfArch, fixture);
    }

    /** A fixture of an HF architecture name and its checkpoint builder (the other fixture files reuse it). */
    public static Named of(String hfArch, Supplier<SyntheticHf.Builder> fixture) {
        return named(hfArch, fixture);
    }

    /** Every fixture of this file, in table order. */
    public static List<Named> all() {
        return Arrays.asList(
                named("BertForSequenceClassification", HfSpecialFixturesBert::bertClassification),
                named("BertModel", HfSpecialFixturesBert::bert),
                named("RobertaModel", HfSpecialFixturesBert::roberta));
    }
}

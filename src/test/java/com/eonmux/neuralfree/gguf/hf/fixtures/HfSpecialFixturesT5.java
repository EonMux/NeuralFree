package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the T5 rows of Appendix A table HK-SP: {@code T5ForConditionalGeneration} and
 * {@code T5EncoderModel} on {@link com.eonmux.neuralfree.gguf.hf.hooks.HfT5Hooks}. Two encoder layers, two decoder
 * layers, hidden size 8, two heads with a key/value width of 4, FFN 16, the Unigram tokenizer.model of t5.py:30-119
 * and the exact tensor names and config keys of the t5.py converters.
 */
public final class HfSpecialFixturesT5 {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int FF = 16;
    private static final int KV = 4;
    private static final int BUCKETS = 32;

    private HfSpecialFixturesT5() {
    }

    /** The Unigram SentencePiece pieces of the test vocabulary: {text, score, type} triples. */
    public static final Object[] SPIECE_PIECES = {"<pad>", 0f, 3, "</s>", 0f, 3, "<unk>", 0f, 2,
            "▁a", -1.5f, 1, "▁b", -1.5f, 1, "▁c", -2f, 1, "▁ab", -2.5f, 1, "z", -3f, 1};

    /**
     * A minimal protobuf encoder for SentencePiece model files (one piece message per {text, score, type}
     * triple, a trainer spec whose model type is Unigram).
     */
    private static final class Proto {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();

        Proto varint(int field, long value) {
            tag(field, 0);
            raw(value);
            return this;
        }

        Proto fixed32(int field, float value) {
            tag(field, 5);
            int bits = Float.floatToIntBits(value);
            for (int i = 0; i < 4; i++) {
                out.write(bits >>> (8 * i));
            }
            return this;
        }

        Proto string(int field, String value) {
            byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
            tag(field, 2);
            raw(bytes.length);
            out.write(bytes, 0, bytes.length);
            return this;
        }

        Proto message(int field, Proto value) {
            byte[] bytes = value.out.toByteArray();
            tag(field, 2);
            raw(bytes.length);
            out.write(bytes, 0, bytes.length);
            return this;
        }

        byte[] toBytes() {
            return out.toByteArray();
        }

        private void tag(int field, int wire) {
            raw(((long) field << 3) | wire);
        }

        private void raw(long v) {
            while ((v & ~0x7FL) != 0) {
                out.write((int) ((v & 0x7F) | 0x80));
                v >>>= 7;
            }
            out.write((int) v);
        }
    }

    /** The {@code tokenizer.model} bytes of {@link #SPIECE_PIECES} with the Unigram model type. */
    public static byte[] unigramModel() {
        Proto model = new Proto();
        for (int i = 0; i < SPIECE_PIECES.length; i += 3) {
            model.message(1, new Proto().string(1, (String) SPIECE_PIECES[i])
                    .fixed32(2, (Float) SPIECE_PIECES[i + 1]).varint(3, (Integer) SPIECE_PIECES[i + 2]));
        }
        return model.message(2, new Proto().varint(3, 1)).toBytes();
    }

    /** {@code T5Config}: the d_model/d_kv/d_ff keys, the relative buckets and the decoder start id. */
    private static SyntheticHf.Builder t5Base(String hfArch, boolean withDecoder, boolean withShared) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json(
                "{'architectures': ['" + hfArch + "'], 'model_type': 't5', 'd_model': 8, 'd_kv': 4,"
                        + " 'd_ff': 16, 'num_heads': 2, 'num_layers': 2, 'num_decoder_layers':"
                        + (withDecoder ? " 2" : " 0")
                        + ", 'layer_norm_epsilon': 1e-06, 'relative_attention_num_buckets': 32,"
                        + " 'relative_attention_max_distance': 128, 'vocab_size': 8,"
                        + " 'decoder_start_token_id': 1, 'eos_token_id': 1, 'pad_token_id': 0}"));
        b.file("tokenizer.model", null);
        // the tokenizer.model is a binary file, written by unigramModel() below
        if (withShared) {
            b.bf16("shared.weight", 8, DIM);
        }
        b.bf16("encoder.embed_tokens.weight", 8, DIM);
        b.bf16("encoder.final_layer_norm.weight", DIM);
        for (int l = 0; l < LAYERS; l++) {
            b.bf16("encoder.block." + l + ".layer.0.layer_norm.weight", DIM);
            b.bf16("encoder.block." + l + ".layer.0.SelfAttention.q.weight", KV * 2, DIM);
            b.bf16("encoder.block." + l + ".layer.0.SelfAttention.k.weight", KV * 2, DIM);
            b.bf16("encoder.block." + l + ".layer.0.SelfAttention.v.weight", KV * 2, DIM);
            b.bf16("encoder.block." + l + ".layer.0.SelfAttention.o.weight", DIM, KV * 2);
            b.bf16("encoder.block." + l + ".layer.1.layer_norm.weight", DIM);
            b.bf16("encoder.block." + l + ".layer.1.DenseReluDense.wi_0.weight", FF, DIM);
            b.bf16("encoder.block." + l + ".layer.1.DenseReluDense.wi_1.weight", FF, DIM);
            b.bf16("encoder.block." + l + ".layer.1.DenseReluDense.wo.weight", DIM, FF);
        }
        if (withDecoder) {
            b.bf16("decoder.embed_tokens.weight", 8, DIM);
            b.bf16("decoder.final_layer_norm.weight", DIM);
            for (int l = 0; l < LAYERS; l++) {
                b.bf16("decoder.block." + l + ".layer.0.layer_norm.weight", DIM);
                b.bf16("decoder.block." + l + ".layer.0.SelfAttention.q.weight", KV * 2, DIM);
                b.bf16("decoder.block." + l + ".layer.0.SelfAttention.k.weight", KV * 2, DIM);
                b.bf16("decoder.block." + l + ".layer.0.SelfAttention.v.weight", KV * 2, DIM);
                b.bf16("decoder.block." + l + ".layer.0.SelfAttention.o.weight", DIM, KV * 2);
                b.bf16("decoder.block." + l + ".layer.1.layer_norm.weight", DIM);
                b.bf16("decoder.block." + l + ".layer.1.EncDecAttention.q.weight", KV * 2, DIM);
                b.bf16("decoder.block." + l + ".layer.1.EncDecAttention.k.weight", KV * 2, DIM);
                b.bf16("decoder.block." + l + ".layer.1.EncDecAttention.v.weight", KV * 2, DIM);
                b.bf16("decoder.block." + l + ".layer.1.EncDecAttention.o.weight", DIM, KV * 2);
                b.bf16("decoder.block." + l + ".layer.2.layer_norm.weight", DIM);
                b.bf16("decoder.block." + l + ".layer.2.DenseReluDense.wi_0.weight", FF, DIM);
                b.bf16("decoder.block." + l + ".layer.2.DenseReluDense.wi_1.weight", FF, DIM);
                b.bf16("decoder.block." + l + ".layer.2.DenseReluDense.wo.weight", DIM, FF);
            }
            b.bf16("decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", BUCKETS, 2);
            b.bf16("output.weight", 8, DIM);
        }
        return b;
    }

    /** {@code T5ForConditionalGeneration}: every tensor of the full model, the shared selection included. */
    public static SyntheticHf.Builder t5() {
        return t5Base("T5ForConditionalGeneration", true, true);
    }

    /** {@code T5EncoderModel}: encoder only, encoder-side embedding before the shared one. */
    public static SyntheticHf.Builder t5encoder() {
        return t5Base("T5EncoderModel", false, true);
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

    /** Every fixture of this file, in table order. */
    public static List<Named> all() {
        return Arrays.asList(
                named("T5EncoderModel", HfSpecialFixturesT5::t5encoder),
                named("T5ForConditionalGeneration", HfSpecialFixturesT5::t5));
    }
}

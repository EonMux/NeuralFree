package com.eonmux.neuralfree.gguf.hf.parity;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * PT-1 fixtures of the other architectures: Gemma 3 (stored-minus-one norms, the padded token embedding cut to the
 * vocabulary, the sliding window), Phi-3 Mini (LongRoPE factors and the attention factor), Mamba-2 (the SSM sizes,
 * the conv squeeze, the unsuffixed F32 A/D names, the grouped SSM norm) and BERT (the bert. renames and pooler
 * drop, the forced-F32 position and token-type embeddings, the phantom-space WordPiece vocabulary). Every expected
 * GGUF value and tensor is hand-derived from the converter rules the ports cite — HfGemmaHooks (gemma.py:128-183),
 * HfBaseDenseHooks (phi.py:148-173, 175-211), HfSsmHooks (mamba.py:126-200), HfBertHooks (bert.py:26-107) — never
 * read back from NeuralFree. The GPT-2 tokenizer derivations and the storage rules come from
 * {@link ParityFixturesDense}.
 */
public final class ParityFixturesOther {

    private ParityFixturesOther() {
    }

    /** The checkpoint element values of seed {@code seed}, shifted by {@code shift}. */
    private static float[] shifted(int seed, int n, float shift) {
        float[] v = ParityFixturesDense.valuesAt(seed, n);
        for (int i = 0; i < n; i++) {
            v[i] += shift;
        }
        return v;
    }

    /** A tensor's values as the checkpoint stores them (F32 bytes, or BF16 little-endian pairs). */
    private static byte[] bytesOf(StDType dtype, float[] v) {
        if (dtype == StDType.BF16) {
            return ParityFixturesDense.bf16bytes(v);
        }
        ByteBuffer b = ByteBuffer.allocate(4 * v.length).order(ByteOrder.LITTLE_ENDIAN);
        for (float x : v) {
            b.putFloat(x);
        }
        return b.array();
    }

    /** The Mamba A_log values: a dyadic table seeded by the rule of {@link ParityFixturesDense#valuesAt}. */
    private static float[] aLogValues() {
        return new float[]{-0.5f, 0.625f, -0.75f, 0.125f};
    }

    /** The Mamba D values: the same rule, seeded so each element differs. */
    private static float[] dValues() {
        float[] v = new float[16];
        for (int i = 0; i < v.length; i++) {
            v[i] = (((i * 7 + 5 * 13) % 13) - 6) / 8f;
        }
        return v;
    }

    // ------------------------------------------------------------------ gemma3

    /** The gemma3 checkpoint: the embedding carries two rows past the vocabulary the converter cuts off. */
    public static Path gemma3Checkpoint(Path root, StDType dtype) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder()
                .config(SyntheticHf.json("{'architectures':['Gemma3ForCausalLM'],'model_type':'gemma3_text',"
                        + "'hidden_size':16,'intermediate_size':32,'num_hidden_layers':2,"
                        + "'num_attention_heads':2,'num_key_value_heads':1,'head_dim':8,"
                        + "'max_position_embeddings':128,'rms_norm_eps':1e-05,'rope_theta':10000.0,"
                        + "'vocab_size':12,'sliding_window':4,'tie_word_embeddings':true,"
                        + "'torch_dtype':'" + ParityFixturesDense.torchDtype(dtype) + "'}"));
        b.tensor("model.embed_tokens.weight", dtype, new long[]{14, 16},
                bytesOf(dtype, ParityFixturesDense.valuesAt(0, 14 * 16)));
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            b.floats(dtype, p + "input_layernorm.weight", 16);
            b.floats(dtype, p + "self_attn.q_proj.weight", 16, 16);
            b.floats(dtype, p + "self_attn.q_norm.weight", 8);
            b.floats(dtype, p + "self_attn.k_proj.weight", 8, 16);
            b.floats(dtype, p + "self_attn.k_norm.weight", 8);
            b.floats(dtype, p + "self_attn.v_proj.weight", 8, 16);
            b.floats(dtype, p + "self_attn.o_proj.weight", 16, 16);
            b.floats(dtype, p + "post_attention_layernorm.weight", 16);
            b.floats(dtype, p + "mlp.gate_proj.weight", 32, 16);
            b.floats(dtype, p + "mlp.up_proj.weight", 32, 16);
            b.floats(dtype, p + "mlp.down_proj.weight", 16, 32);
        }
        b.floats(dtype, "model.norm.weight", 16);
        return b.write(root);
    }

    /** The expected GGUF view of {@link #gemma3Checkpoint}: norms shifted + 1, embedding cut to 12 rows. */
    public static GgufTestWriter gemma3Expected(StDType dtype) {
        GGMLType w = dtype == StDType.BF16 ? GGMLType.BF16 : GGMLType.F16; // --outtype auto defaults to F16 (base.py:181-196)
        GgufTestWriter g = new GgufTestWriter().architecture("gemma3");
        g.str("general.architecture", "gemma3");
        g.u32("general.file_type", dtype == StDType.BF16 ? 32 /* MOSTLY_BF16 */ : 1 /* MOSTLY_F16 */);
        g.str("general.type", "model");
        g.u32("general.quantization_version", 2);
        g.u32("gemma3.block_count", 2);
        g.u32("gemma3.context_length", 128);
        g.u32("gemma3.embedding_length", 16);
        g.u32("gemma3.feed_forward_length", 32);
        g.u32("gemma3.attention.head_count", 2);
        g.u32("gemma3.attention.head_count_kv", 1);
        g.u32("gemma3.attention.key_length", 8);
        g.u32("gemma3.attention.value_length", 8);
        g.u32("gemma3.attention.sliding_window", 4);
        // llama.cpp writes no {arch}.vocab_size for a gemma3 with a gpt2 vocab (only the mistral vocab
        // path calls add_vocab_size); the tokenizer keys carry the vocabulary size instead
        g.f32("gemma3.attention.layer_norm_rms_epsilon", 1e-5f);
        g.f32("gemma3.rope.freq_base", 10000.0f);
        ParityFixturesDense.tokenizerKv(g, ParityFixturesDense.gpt2Tokens(12),
                ParityFixturesDense.gpt2Types(12), "mpt");

        // The embedding keeps the first 12 rows of the 14 checkpoint rows (gemma.py:160-163).
        g.tensorFromFloats("token_embd.weight", w, new long[]{16, 12},
                ParityFixturesDense.valuesAt(0, 12 * 16));
        int seed = 1;
        for (int l = 0; l < 2; l++) {
            String b = "blk." + l + ".";
            int norm = seed++;
            g.tensorFromFloats(b + "attn_norm.weight", GGMLType.F32, new long[]{16}, shifted(norm, 16, 1f));
            ParityFixturesDense.planMatrix(g, b + "attn_q.weight", w, new long[]{16, 16}, seed++);
            int qNorm = seed++;
            g.tensorFromFloats(b + "attn_q_norm.weight", GGMLType.F32, new long[]{8}, shifted(qNorm, 8, 1f));
            ParityFixturesDense.planMatrix(g, b + "attn_k.weight", w, new long[]{16, 8}, seed++);
            int kNorm = seed++;
            g.tensorFromFloats(b + "attn_k_norm.weight", GGMLType.F32, new long[]{8}, shifted(kNorm, 8, 1f));
            ParityFixturesDense.planMatrix(g, b + "attn_v.weight", w, new long[]{16, 8}, seed++);
            ParityFixturesDense.planMatrix(g, b + "attn_output.weight", w, new long[]{16, 16}, seed++);
            int ffnNorm = seed++;
            g.tensorFromFloats(b + "post_attention_norm.weight", GGMLType.F32, new long[]{16}, shifted(ffnNorm, 16, 1f));
            ParityFixturesDense.planMatrix(g, b + "ffn_gate.weight", w, new long[]{16, 32}, seed++);
            ParityFixturesDense.planMatrix(g, b + "ffn_up.weight", w, new long[]{16, 32}, seed++);
            ParityFixturesDense.planMatrix(g, b + "ffn_down.weight", w, new long[]{32, 16}, seed++);
        }
        int finalNorm = seed;
        g.tensorFromFloats("output_norm.weight", GGMLType.F32, new long[]{16}, shifted(finalNorm, 16, 1f));
        return g;
    }

    // ------------------------------------------------------------------ phi3

    /** The Phi-3 checkpoint: llama-style tensors, no biases, with the su LongRoPE configuration. */
    public static Path phi3Checkpoint(Path root, StDType dtype) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder()
                .config(SyntheticHf.json("{'architectures':['Phi3ForCausalLM'],'model_type':'phi3',"
                        + "'hidden_size':16,'intermediate_size':32,'num_hidden_layers':2,"
                        + "'num_attention_heads':2,'num_key_value_heads':2,'max_position_embeddings':64,"
                        + "'vocab_size':16,'bos_token_id':0,'eos_token_id':1,'rms_norm_eps':1e-05,"
                        + "'rope_scaling':{'type':'su','original_max_position_embeddings':8,"
                        + "'long_factor':[1.0105,1.0115,1.0125,1.0135],"
                        + "'short_factor':[1.0005,1.0015,1.0025,1.0035]},'rope_theta':10000.0,"
                        + "'torch_dtype':'" + ParityFixturesDense.torchDtype(dtype) + "'}"));
        // The tokenizer_config names GPT2Tokenizer, so the converter takes the GPT-2 branch (phi.py:46-49).
        b.file("tokenizer_config.json", SyntheticHf.json("{'tokenizer_class':'GPT2Tokenizer',"
                + "'bos_token':'<s>','eos_token':'</s>','add_bos_token':false,"
                + "'clean_up_tokenization_spaces':false}"));
        b.floats(dtype, "model.embed_tokens.weight", 16, 16);
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            b.floats(dtype, p + "input_layernorm.weight", 16);
            b.floats(dtype, p + "self_attn.q_proj.weight", 16, 16);
            b.floats(dtype, p + "self_attn.k_proj.weight", 16, 16);
            b.floats(dtype, p + "self_attn.v_proj.weight", 16, 16);
            b.floats(dtype, p + "self_attn.o_proj.weight", 16, 16);
            b.floats(dtype, p + "post_attention_layernorm.weight", 16);
            // phi3 carries a fused gate_up projection (tensor_mapping.py FFN_UP row: model.layers.{bid}.mlp.gate_up_proj)
            b.floats(dtype, p + "mlp.gate_up_proj.weight", 64, 16);
            b.floats(dtype, p + "mlp.down_proj.weight", 16, 32);
        }
        b.floats(dtype, "model.norm.weight", 16);
        b.floats(dtype, "lm_head.weight", 16, 16);
        return b.write(root);
    }

    /** The expected GGUF view of {@link #phi3Checkpoint} with the LongRoPE scalings (phi.py:148-211). */
    public static GgufTestWriter phi3Expected(StDType dtype) {
        GGMLType w = dtype == StDType.BF16 ? GGMLType.BF16 : GGMLType.F16; // --outtype auto defaults to F16 (base.py:181-196)
        GgufTestWriter g = new GgufTestWriter().architecture("phi3");
        g.str("general.architecture", "phi3");
        g.u32("general.file_type", dtype == StDType.BF16 ? 32 : 1);
        g.str("general.type", "model");
        g.u32("general.quantization_version", 2);
        g.u32("phi3.block_count", 2);
        g.u32("phi3.context_length", 64);
        g.u32("phi3.rope.scaling.original_context_length", 8);
        g.u32("phi3.embedding_length", 16);
        g.u32("phi3.feed_forward_length", 32);
        g.u32("phi3.attention.head_count", 2);
        g.u32("phi3.attention.head_count_kv", 2);
        g.f32("phi3.attention.layer_norm_rms_epsilon", 1e-5f);
        g.u32("phi3.rope.dimension_count", 8); // int(partial_rotary_factor * hidden) / heads
        g.f32("phi3.rope.freq_base", 10000.0f);
        g.u32("phi3.attention.sliding_window", 0);
        g.f32("phi3.rope.scaling.attn_factor", (float) Math.sqrt(2.0)); // scale 64/8, type su
        ParityFixturesDense.tokenizerKv(g, ParityFixturesDense.gpt2Tokens(16),
                ParityFixturesDense.gpt2Types(16), "mpt");

        int seed = 0;
        ParityFixturesDense.planMatrix(g, "token_embd.weight", w, new long[]{16, 16}, seed++);
        for (int l = 0; l < 2; l++) {
            String b = "blk." + l + ".";
            int norm = seed++;
            g.tensorFromFloats(b + "attn_norm.weight", GGMLType.F32, new long[]{16},
                    ParityFixturesDense.valuesAt(norm, 16));
            ParityFixturesDense.planMatrix(g, b + "attn_q.weight", w, new long[]{16, 16}, seed++);
            ParityFixturesDense.planMatrix(g, b + "attn_k.weight", w, new long[]{16, 16}, seed++);
            ParityFixturesDense.planMatrix(g, b + "attn_v.weight", w, new long[]{16, 16}, seed++);
            ParityFixturesDense.planMatrix(g, b + "attn_output.weight", w, new long[]{16, 16}, seed++);
            int ffnNorm = seed++;
            g.tensorFromFloats(b + "ffn_norm.weight", GGMLType.F32, new long[]{16},
                    ParityFixturesDense.valuesAt(ffnNorm, 16));
            // the fused gate_up projection maps to FFN_UP alone; PHI3 has no FFN_GATE tensor
            ParityFixturesDense.planMatrix(g, b + "ffn_up.weight", w, new long[]{16, 64}, seed++);
            ParityFixturesDense.planMatrix(g, b + "ffn_down.weight", w, new long[]{32, 16}, seed++);
        }
        int finalNorm = seed++;
        g.tensorFromFloats("output_norm.weight", GGMLType.F32, new long[]{16},
                ParityFixturesDense.valuesAt(finalNorm, 16));
        ParityFixturesDense.planMatrix(g, "output.weight", w, new long[]{16, 16}, seed);
        // phi.py:180-202: one F32 factor per rope dimension pair, the configuration numbers as stored floats.
        g.tensorFromFloats("rope_factors_long.weight", GGMLType.F32, new long[]{4},
                ropeFactors("1.0105", "1.0115", "1.0125", "1.0135"));
        g.tensorFromFloats("rope_factors_short.weight", GGMLType.F32, new long[]{4},
                ropeFactors("1.0005", "1.0015", "1.0025", "1.0035"));
        return g;
    }

    /** The JSON numbers of a LongRoPE factor list as the float32 values the converter stores. */
    private static float[] ropeFactors(String... numbers) {
        float[] out = new float[numbers.length];
        for (int i = 0; i < numbers.length; i++) {
            out[i] = Float.parseFloat(numbers[i]);
        }
        return out;
    }

    // ------------------------------------------------------------------ mamba2

    /** The Mamba-2 checkpoint: backbone layers of mixer tensors, the conv kernel and the A_log/D vectors. */
    public static Path mamba2Checkpoint(Path root, StDType dtype) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder()
                .config(SyntheticHf.json("{'architectures':['Mamba2ForCausalLM'],'model_type':'mamba2',"
                        + "'hidden_size':8,'num_hidden_layers':2,'n_groups':1,'state_size':4,'conv_kernel':2,"
                        + "'mamba_d_head':4,'mamba_expand':2,'layer_norm_epsilon':1e-05,'vocab_size':12,"
                        + "'torch_dtype':'" + ParityFixturesDense.torchDtype(dtype) + "'}"));
        b.floats(dtype, "backbone.embeddings.weight", 16, 8);
        for (int l = 0; l < 2; l++) {
            String p = "backbone.layers." + l + ".mixer.";
            b.floats(dtype, p + "in_proj.weight", 44, 8);
            b.floats(dtype, p + "conv1d.weight", 16, 2, 1);
            b.tensor(p + "A_log", dtype, new long[]{1, 4}, bytesOf(dtype, aLogValues()));
            b.tensor(p + "D", dtype, new long[]{16}, bytesOf(dtype, dValues()));
            b.floats(dtype, p + "norm.weight", 16);
            b.floats(dtype, p + "out_proj.weight", 8, 16);
        }
        b.floats(dtype, "lm_head.weight", 16, 8);
        return b.write(root);
    }

    /** The expected GGUF view of {@link #mamba2Checkpoint} (mamba.py:126-200). */
    public static GgufTestWriter mamba2Expected(StDType dtype) {
        GGMLType w = dtype == StDType.BF16 ? GGMLType.BF16 : GGMLType.F16; // --outtype auto defaults to F16 (base.py:181-196)
        GgufTestWriter g = new GgufTestWriter().architecture("mamba2");
        g.str("general.architecture", "mamba2");
        g.u32("general.file_type", dtype == StDType.BF16 ? 32 : 1);
        g.str("general.type", "model");
        g.u32("general.quantization_version", 2);
        g.u32("mamba2.context_length", 1048576); // arbitrary (mamba.py:155)
        g.u32("mamba2.embedding_length", 8);
        g.u32("mamba2.feed_forward_length", 0);
        g.u32("mamba2.attention.head_count", 0);
        g.u32("mamba2.block_count", 2);
        g.u32("mamba2.ssm.conv_kernel", 2);
        g.u32("mamba2.ssm.inner_size", 16);
        g.u32("mamba2.ssm.state_size", 4);
        g.u32("mamba2.ssm.time_step_rank", 4); // d_inner / head_dim
        g.u32("mamba2.ssm.group_count", 1);
        g.f32("mamba2.attention.layer_norm_rms_epsilon", 1e-5f);
        ParityFixturesDense.tokenizerKv(g, ParityFixturesDense.gpt2Tokens(16),
                ParityFixturesDense.gpt2Types(16), "mpt");

        int seed = 0;
        ParityFixturesDense.planMatrix(g, "token_embd.weight", w, new long[]{8, 16}, seed++);
        for (int l = 0; l < 2; l++) {
            String b = "blk." + l + ".";
            ParityFixturesDense.planMatrix(g, b + "ssm_in.weight", w, new long[]{8, 44}, seed++);
            int conv = seed++;
            g.tensorFromFloats(b + "ssm_conv1d.weight", GGMLType.F32, new long[]{2, 16},
                    ParityFixturesDense.valuesAt(conv, 32)); // the squeeze keeps the flat order
            seed++; // A_log: stored under the unsuffixed F32 name blk.<l>.ssm_a with the +1 dim appended
            g.tensorFromFloats(b + "ssm_a", GGMLType.F32, new long[]{1, 4, 1},
                    negExp(aLogValues()));
            seed++; // D: stored under the unsuffixed F32 name blk.<l>.ssm_d with the +1 dim appended
            g.tensorFromFloats(b + "ssm_d", GGMLType.F32, new long[]{16, 1}, dValues());
            int norm = seed++;
            g.tensorFromFloats(b + "ssm_norm.weight", GGMLType.F32, new long[]{16, 1},
                    ParityFixturesDense.valuesAt(norm, 16));
            ParityFixturesDense.planMatrix(g, b + "ssm_out.weight", w, new long[]{16, 8}, seed++);
        }
        ParityFixturesDense.planMatrix(g, "output.weight", w, new long[]{8, 16}, seed);
        return g;
    }

    /** {@code -torch.exp(A_log)} of the checkpoint values, in float32 as the converter computes it. */
    private static float[] negExp(float[] v) {
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = (float) -Math.exp(v[i]);
        }
        return out;
    }

    // ------------------------------------------------------------------ bert

    /** A WordPiece tokenizer.json without added tokens, for the bertWpm vocabulary (bert.py:40-67). */
    private static final String WORDPIECE_TOKENIZER = SyntheticHf.json("{'version':'1.0','added_tokens':[],"
            + "'normalizer':null,'pre_tokenizer':null,'post_processor':null,'decoder':null,"
            + "'model':{'type':'WordPiece','unk_token':'[UNK]','continuing_subword_prefix':'##',"
            + "'max_input_chars_per_word':100,'vocab':{'[PAD]':0,'[UNK]':1,'[CLS]':2,'[SEP]':3,"
            + "'a':4,'b':5,'ab':6,'##x':7}}}");

    /** The BERT checkpoint: encoder layers, the dropped pooler, position and token-type embeddings. */
    public static Path bertCheckpoint(Path root, StDType dtype) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder()
                .config(SyntheticHf.json("{'architectures':['BertForMaskedLM'],'model_type':'bert',"
                        + "'hidden_size':16,'intermediate_size':32,'num_hidden_layers':2,"
                        + "'num_attention_heads':2,'max_position_embeddings':8,'type_vocab_size':2,"
                        + "'vocab_size':12,'torch_dtype':'" + ParityFixturesDense.torchDtype(dtype) + "'}"))
                .tokenizerJson(WORDPIECE_TOKENIZER)
                .tokenizerConfigJson(null);
        b.floats(dtype, "bert.embeddings.word_embeddings.weight", 12, 16);
        b.floats(dtype, "bert.embeddings.position_embeddings.weight", 8, 16);
        b.floats(dtype, "bert.embeddings.token_type_embeddings.weight", 2, 16);
        b.floats(dtype, "bert.embeddings.LayerNorm.weight", 16);
        for (int l = 0; l < 2; l++) {
            String p = "bert.encoder.layer." + l + ".";
            b.floats(dtype, p + "attention.self.query.weight", 16, 16);
            b.floats(dtype, p + "attention.self.key.weight", 16, 16);
            b.floats(dtype, p + "attention.self.value.weight", 16, 16);
            b.floats(dtype, p + "attention.output.dense.weight", 16, 16);
            b.floats(dtype, p + "attention.output.LayerNorm.weight", 16);
            b.floats(dtype, p + "intermediate.dense.weight", 32, 16);
            b.floats(dtype, p + "output.dense.weight", 16, 32);
            b.floats(dtype, p + "output.LayerNorm.weight", 16);
        }
        return b.write(root);
    }

    /** The expected GGUF view of {@link #bertCheckpoint} (bert.py:19-112). */
    public static GgufTestWriter bertExpected(StDType dtype) {
        GGMLType w = dtype == StDType.BF16 ? GGMLType.BF16 : GGMLType.F16; // --outtype auto defaults to F16 (base.py:181-196)
        GgufTestWriter g = new GgufTestWriter().architecture("bert");
        g.str("general.architecture", "bert");
        g.u32("general.file_type", dtype == StDType.BF16 ? 32 : 1);
        g.str("general.type", "model");
        g.u32("general.quantization_version", 2);
        g.u32("bert.block_count", 2);
        g.u32("bert.context_length", 8);
        g.u32("bert.embedding_length", 16);
        g.u32("bert.feed_forward_length", 32);
        g.u32("bert.attention.head_count", 2);
        g.bool("bert.attention.causal", false);
        g.u32("bert.vocab_size", 12);
        g.f32("bert.attention.layer_norm_epsilon", 1e-5f);
        bertVocabKv(g);

        int seed = 0;
        ParityFixturesDense.planMatrix(g, "token_embd.weight", w, new long[]{16, 12}, seed++);
        // forced_f32 POS_EMBD and TOKEN_TYPES (base.py:1260-1261)
        g.tensorFromFloats("position_embd.weight", GGMLType.F32, new long[]{16, 8},
                ParityFixturesDense.valuesAt(1, 8 * 16));
        g.tensorFromFloats("token_types.weight", GGMLType.F32, new long[]{16, 2},
                ParityFixturesDense.valuesAt(2, 2 * 16));
        g.tensorFromFloats("token_embd_norm.weight", GGMLType.F32, new long[]{16},
                ParityFixturesDense.valuesAt(3, 16));
        seed += 3;
        for (int l = 0; l < 2; l++) {
            String b = "blk." + l + ".";
            ParityFixturesDense.planMatrix(g, b + "attn_q.weight", w, new long[]{16, 16}, seed++);
            ParityFixturesDense.planMatrix(g, b + "attn_k.weight", w, new long[]{16, 16}, seed++);
            ParityFixturesDense.planMatrix(g, b + "attn_v.weight", w, new long[]{16, 16}, seed++);
            ParityFixturesDense.planMatrix(g, b + "attn_output.weight", w, new long[]{16, 16}, seed++);
            int attnNorm = seed++;
            g.tensorFromFloats(b + "attn_output_norm.weight", GGMLType.F32, new long[]{16},
                    ParityFixturesDense.valuesAt(attnNorm, 16));
            ParityFixturesDense.planMatrix(g, b + "ffn_up.weight", w, new long[]{16, 32}, seed++);
            ParityFixturesDense.planMatrix(g, b + "ffn_down.weight", w, new long[]{32, 16}, seed++);
            int outNorm = seed;
            g.tensorFromFloats(b + "layer_output_norm.weight", GGMLType.F32, new long[]{16},
                    ParityFixturesDense.valuesAt(outNorm, 16));
            seed++;
        }
        return g;
    }

    /**
     * The phantom-space WordPiece tokens (bert.py:40-67): base tokens prefixed with U+2581, {@code ##x} stripped to
     * {@code x}; the unused ids become {@code [PAD<i>]} UNUSED. No tokenizer.ggml.scores for the WordPiece class,
     * plus the {@code tokenizer.ggml.token_type_count} of the WordPiece tokenizer and the fingerprint default
     * {@code tokenizer.ggml.pre}.
     */
    private static void bertVocabKv(GgufTestWriter g) {
        String metaspace = new String(Character.toChars(0x2581));
        String[] base = {metaspace + "[PAD]", metaspace + "[UNK]", metaspace + "[CLS]", metaspace + "[SEP]",
            metaspace + "a", metaspace + "b", metaspace + "ab", "x"};
        List<String> tokens = new ArrayList<>();
        List<Integer> types = new ArrayList<>();
        for (int i = 0; i < base.length; i++) {
            tokens.add(base[i]);
            types.add(1); // NORMAL: a base token, not an added one
        }
        for (int i = base.length; i < 12; i++) {
            tokens.add("[PAD" + i + "]");
            types.add(5); // UNUSED
        }
        g.str("tokenizer.ggml.model", "bert");
        g.u32("tokenizer.ggml.token_type_count", 2);
        g.strArray("tokenizer.ggml.tokens", tokens.toArray(new String[0]));
        int[] typeArray = new int[12];
        for (int i = 0; i < 12; i++) {
            typeArray[i] = types.get(i);
        }
        g.i32Array("tokenizer.ggml.token_type", typeArray);
        g.str("tokenizer.ggml.pre", "default");
    }
}

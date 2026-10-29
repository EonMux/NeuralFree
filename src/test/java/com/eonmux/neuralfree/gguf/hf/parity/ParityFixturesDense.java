package com.eonmux.neuralfree.gguf.hf.parity;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * PT-1 fixtures of the dense families: Llama and Qwen2, each in its F32 and BF16 weight form. Every expected GGUF
 * value and tensor here is hand-derived from the converter rules the ports cite — never read back from NeuralFree:
 *
 * <ul>
 *   <li>{@code LlamaModel} metadata: llama.py:165-174 (vocab size, {@code rope.dimension_count}) after the generic
 *       rows of {@code TextModel.set_gguf_parameters} (base.py:1606-1720, read off hparam_aliases.tsv); the Q/K
 *       permutation of llama.py:176-182; the tensor names of tensor_mapping.py / tensor_patterns.tsv.</li>
 *   <li>{@code Qwen2Model} metadata: qwen.py:71-73 (generic rows + the pooling step, which is absent here); the
 *       default tensor mapping of base.py:845-905 (no Q/K permutation).</li>
 *   <li>The shared GPT-2 BPE vocabulary: base.py:1755-1797 ({@code get_vocab_base}) and 1758-1762 (the added-token
 *       types), and the special-token ids of gguf-py's {@code SpecialVocab}; hparam_aliases.tsv rows in table order.</li>
 *   <li>Tensor storage types: base.py:1240-1330 (rank ≤ 1 and {@code *_norm.weight} tensors are F32; forced-F32
 *       models from filters.tsv; every other {@code .weight} keeps the checkpoint's dtype through {@code --outtype
 *       auto}) and base.py:180-197 (the auto guess: BF16 stored weights win over F16).</li>
 * </ul>
 *
 * <p>The checkpoint element values are {@link SyntheticHf#values}: {@code (-6 + ((i * 7 + seed * 13) % 13)) / 8},
 * every earlier tensor in the builder raising the seed by one (SyntheticHf.Builder: the tensor's own position). Each
 * element value is an exact multiple of 1/8 in [-0.75, 0.75], exact in BF16 and F32, and widens from BF16 to F32
 * without loss.</p>
 */
public final class ParityFixturesDense {

    private ParityFixturesDense() {
    }

    // ------------------------------------------------------------------ shared derivations

    /** The checkpoint element values of seed {@code seed}, the {@link SyntheticHf#values} rule. */
    public static float[] valuesAt(int seed, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (((i * 7 + seed * 13) % 13) - 6) / 8f;
        }
        return v;
    }

    /** The value at flat index {@code i}. */
    public static float valueAt(int seed, int i) {
        return ((i * 7 + seed * 13) % 13 - 6) / 8f;
    }

    /** The BF16 bits (little-endian pairs) of the given values, as the checkpoint stores them. */
    public static byte[] bf16bytes(float[] v) {
        ByteBuffer b = ByteBuffer.allocate(2 * v.length).order(ByteOrder.LITTLE_ENDIAN);
        for (float x : v) {
            b.putShort((short) (Float.floatToRawIntBits(x) >>> 16));
        }
        return b.array();
    }

    /**
     * Adds {@code name} with {@code ne} to {@code w}: values of the seed, type-bound storage bytes exactly as a
     * GGUF writer of llama.cpp would lay them out.
     */
    public static void planMatrix(GgufTestWriter w, String name, GGMLType type, long[] ne, int seed) {
        long count = 1;
        for (long d : ne) {
            count *= d;
        }
        float[] v = valuesAt(seed, (int) count);
        w.tensorFromFloats(name, type, ne, v);
    }

    /** Writes the shared tokenizer keys of the {@link SyntheticHf#GPT2_TOKENIZER} fixture. The {@code pre} name is
     * the converter's chkhsh lookup result for that vocab (a miss raises in llama.cpp, base.py:2132-2140). */
    public static void tokenizerKv(GgufTestWriter w, List<String> tokens, int[] types, String pre) {
        w.str("tokenizer.ggml.model", "gpt2");
        w.str("tokenizer.ggml.pre", pre);
        w.strArray("tokenizer.ggml.tokens", tokens);
        w.i32Array("tokenizer.ggml.token_type", types);
        w.strArray("tokenizer.ggml.merges", "a b");
        w.u32("tokenizer.ggml.bos_token_id", 0);
        w.u32("tokenizer.ggml.eos_token_id", 1);
        w.bool("tokenizer.ggml.add_bos_token", false);
    }

    /**
     * The shared GPT-2 BPE token list of every non-SentencePiece fixture: the seven ids of
     * {@link SyntheticHf#GPT2_TOKENIZER} (vocab {'<s>' 0, '</s>' 1, a 2, b 3, ab 4} plus the added ids 5 and 6)
     * padded to {@code vocabSize} with {@code [PAD<i>]} UNUSED slots. Types: 1 NORMAL (a, b, ab), 3 CONTROL (the
     * special added tokens), 4 USER_DEFINED (Ġhi, an added byte-level token that keeps its text), 5 UNUSED.
     */
    public static List<String> gpt2Tokens(int vocabSize) {
        // the 7th token is stored byte-decoded, as llama.cpp's _set_vocab_gpt2 writes Ġ-decoded pieces
        List<String> tokens = new ArrayList<>(Arrays.asList("<s>", "</s>", "a", "b", "ab", "<|begin|>", " hi"));
        while (tokens.size() < vocabSize) {
            tokens.add("[PAD" + tokens.size() + "]");
        }
        return tokens;
    }

    /** Matching token types, index-aligned with {@link #gpt2Tokens}. */
    public static int[] gpt2Types(int vocabSize) {
        int[] types = new int[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            types[i] = i < 7 ? (i == 2 || i == 3 || i == 4 ? 1 : (i == 6 ? 4 : 3)) : 5;
        }
        return types;
    }

    // ------------------------------------------------------------------ llama

    public static Path llamaCheckpoint(Path root, StDType dtype) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder()
                .config(SyntheticHf.json("{'architectures':['LlamaForCausalLM'],'model_type':'llama',"
                        + "'hidden_size':64,'intermediate_size':128,'num_hidden_layers':2,"
                        + "'num_attention_heads':4,'num_key_value_heads':2,'max_position_embeddings':128,"
                        + "'rms_norm_eps':1e-05,'rope_theta':10000.0,'vocab_size':16,'bos_token_id':0,"
                        + "'eos_token_id':1,'torch_dtype':'" + torchDtype(dtype) + "'}"));
        b.floats(dtype, "model.embed_tokens.weight", 16, 64);
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            b.floats(dtype, p + "input_layernorm.weight", 64);
            b.floats(dtype, p + "self_attn.q_proj.weight", 64, 64);
            b.floats(dtype, p + "self_attn.k_proj.weight", 32, 64);
            b.floats(dtype, p + "self_attn.v_proj.weight", 32, 64);
            b.floats(dtype, p + "self_attn.o_proj.weight", 64, 64);
            b.floats(dtype, p + "post_attention_layernorm.weight", 64);
            b.floats(dtype, p + "mlp.gate_proj.weight", 128, 64);
            b.floats(dtype, p + "mlp.up_proj.weight", 128, 64);
            b.floats(dtype, p + "mlp.down_proj.weight", 64, 128);
        }
        b.floats(dtype, "model.norm.weight", 64);
        b.floats(dtype, "lm_head.weight", 16, 64);
        return b.write(root);
    }

    /** The expected GGUF view of {@link #llamaCheckpoint}. */
    public static GgufTestWriter llamaExpected(StDType dtype) {
        GGMLType w = dtype == StDType.BF16 ? GGMLType.BF16 : GGMLType.F16; // --outtype auto defaults to F16 (base.py:181-196)
        GgufTestWriter g = new GgufTestWriter().architecture("llama");
        g.str("general.architecture", "llama");
        g.u32("general.file_type", dtype == StDType.BF16 ? 32 /* MOSTLY_BF16 */ : 1 /* MOSTLY_F16 */);
        g.str("general.type", "model");
        g.u32("general.quantization_version", 2);
        g.u32("llama.block_count", 2);
        g.u32("llama.context_length", 128);
        g.u32("llama.embedding_length", 64);
        g.u32("llama.feed_forward_length", 128);
        g.u32("llama.attention.head_count", 4);
        g.u32("llama.attention.head_count_kv", 2);
        g.f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
        g.u32("llama.rope.dimension_count", 16);
        g.u32("llama.vocab_size", 16);
        g.f32("llama.rope.freq_base", 10000.0f);
        tokenizerKv(g, gpt2Tokens(16), gpt2Types(16), "mpt");

        int seed = 0;
        planMatrix(g, "token_embd.weight", w, new long[]{64, 16}, seed++);      // embed
        for (int l = 0; l < 2; l++) {
            String b = "blk." + l + ".";
            int norm = seed++;
            g.tensorFromFloats(b + "attn_norm.weight", GGMLType.F32, new long[]{64}, widened(valuesAt(norm, 64)));
            qPermuted(g, w, b + "attn_q.weight", seed++, 64, 64, 4);
            qPermuted(g, w, b + "attn_k.weight", seed++, 64, 32, 2);
            planMatrix(g, b + "attn_v.weight", w, new long[]{64, 32}, seed++);
            planMatrix(g, b + "attn_output.weight", w, new long[]{64, 64}, seed++);
            int ffnNorm = seed++;
            g.tensorFromFloats(b + "ffn_norm.weight", GGMLType.F32, new long[]{64}, widened(valuesAt(ffnNorm, 64)));
            planMatrix(g, b + "ffn_gate.weight", w, new long[]{64, 128}, seed++);
            planMatrix(g, b + "ffn_up.weight", w, new long[]{64, 128}, seed++);
            planMatrix(g, b + "ffn_down.weight", w, new long[]{128, 64}, seed++);
        }
        int finalNorm = seed;
        g.tensorFromFloats("output_norm.weight", GGMLType.F32, new long[]{64}, widened(valuesAt(finalNorm, 64)));
        planMatrix(g, "output.weight", w, new long[]{64, 16}, seed + 1);
        return g;
    }

    // ------------------------------------------------------------------ qwen2

    public static Path qwen2Checkpoint(Path root, StDType dtype) throws IOException {
        return qwenBuilder(dtype).write(root);
    }

    private static SyntheticHf.Builder qwenBuilder(StDType dtype) {
        SyntheticHf.Builder b = SyntheticHf.builder()
                .config(SyntheticHf.json("{'architectures':['Qwen2ForCausalLM'],'model_type':'qwen2',"
                        + "'hidden_size':64,'intermediate_size':128,'num_hidden_layers':2,"
                        + "'num_attention_heads':4,'num_key_value_heads':2,'max_position_embeddings':128,"
                        + "'rms_norm_eps':1e-05,'rope_theta':10000.0,'vocab_size':16,'bos_token_id':0,"
                        + "'eos_token_id':1}"));
        b.floats(dtype, "model.embed_tokens.weight", 16, 64);
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            b.floats(dtype, p + "input_layernorm.weight", 64);
            b.floats(dtype, p + "self_attn.q_proj.weight", 64, 64);
            b.floats(dtype, p + "self_attn.q_proj.bias", 64);
            b.floats(dtype, p + "self_attn.k_proj.weight", 32, 64);
            b.floats(dtype, p + "self_attn.k_proj.bias", 32);
            b.floats(dtype, p + "self_attn.v_proj.weight", 32, 64);
            b.floats(dtype, p + "self_attn.v_proj.bias", 32);
            b.floats(dtype, p + "self_attn.o_proj.weight", 64, 64);
            b.floats(dtype, p + "self_attn.o_proj.bias", 64);
            b.floats(dtype, p + "post_attention_layernorm.weight", 64);
            b.floats(dtype, p + "mlp.gate_proj.weight", 128, 64);
            b.floats(dtype, p + "mlp.up_proj.weight", 128, 64);
            b.floats(dtype, p + "mlp.down_proj.weight", 64, 128);
        }
        b.floats(dtype, "model.norm.weight", 64);
        b.floats(dtype, "lm_head.weight", 16, 64);
        return b;
    }

    /** The expected GGUF view of {@link #qwen2Checkpoint}: no permutation, biases preserved. */
    public static GgufTestWriter qwen2Expected(StDType dtype) {
        GGMLType w = dtype == StDType.BF16 ? GGMLType.BF16 : GGMLType.F16; // --outtype auto defaults to F16 (base.py:181-196)
        GgufTestWriter g = new GgufTestWriter().architecture("qwen2");
        g.str("general.architecture", "qwen2");
        g.u32("general.file_type", dtype == StDType.BF16 ? 32 : 1);
        g.str("general.type", "model");
        g.u32("general.quantization_version", 2);
        g.u32("qwen2.block_count", 2);
        g.u32("qwen2.context_length", 128);
        g.u32("qwen2.embedding_length", 64);
        g.u32("qwen2.feed_forward_length", 128);
        g.u32("qwen2.attention.head_count", 4);
        g.u32("qwen2.attention.head_count_kv", 2);
        g.f32("qwen2.attention.layer_norm_rms_epsilon", 1e-5f);
        g.f32("qwen2.rope.freq_base", 10000.0f);
        tokenizerKv(g, gpt2Tokens(16), gpt2Types(16), "mpt");

        int seed = 0;
        planMatrix(g, "token_embd.weight", w, new long[]{64, 16}, seed++);
        for (int l = 0; l < 2; l++) {
            String b = "blk." + l + ".";
            int norm = seed++;
            g.tensorFromFloats(b + "attn_norm.weight", GGMLType.F32, new long[]{64}, widened(valuesAt(norm, 64)));
            planMatrix(g, b + "attn_q.weight", w, new long[]{64, 64}, seed++);
            f32Rank1Matrix(g, b + "attn_q.bias", 64, seed++);
            planMatrix(g, b + "attn_k.weight", w, new long[]{64, 32}, seed++);
            f32Rank1Matrix(g, b + "attn_k.bias", 32, seed++);
            planMatrix(g, b + "attn_v.weight", w, new long[]{64, 32}, seed++);
            f32Rank1Matrix(g, b + "attn_v.bias", 32, seed++);
            planMatrix(g, b + "attn_output.weight", w, new long[]{64, 64}, seed++);
            f32Rank1Matrix(g, b + "attn_output.bias", 64, seed++);
            int ffnNorm = seed++;
            g.tensorFromFloats(b + "ffn_norm.weight", GGMLType.F32, new long[]{64}, widened(valuesAt(ffnNorm, 64)));
            planMatrix(g, b + "ffn_gate.weight", w, new long[]{64, 128}, seed++);
            planMatrix(g, b + "ffn_up.weight", w, new long[]{64, 128}, seed++);
            planMatrix(g, b + "ffn_down.weight", w, new long[]{128, 64}, seed++);
        }
        int finalNorm = seed;
        g.tensorFromFloats("output_norm.weight", GGMLType.F32, new long[]{64}, widened(valuesAt(finalNorm, 64)));
        planMatrix(g, "output.weight", w, new long[]{64, 16}, seed + 1);
        return g;
    }

    // ------------------------------------------------------------------ helpers

    /** A flat F32 tensor whose element order is the checkpoint's flat byte order. */
    private static void f32Rank1Matrix(GgufTestWriter g, String name, int n, int seed) {
        g.tensorFromFloats(name, GGMLType.F32, new long[]{n}, valuesAt(seed, n));
    }

    /** BF16-widened (exact) copies whose storage bytes are F32. */
    private static float[] widened(float[] v) {
        return v;
    }

    /**
     * The checker's data comes from planned row bytes; a matrix with {@code ne} whose flat element list is the
     * checkpoint's row-major order shares the HF bytes. Rank-1 norms enlarge via dtype only.
     */
    private static void qPermuted(GgufTestWriter g, GGMLType type, String name, int seed, int nEmbd, int rows,
                                  int headGroups) {
        // LlamaModel.permute (llama.py:176-179): the head count falls back to n_head_kv when they differ, and the
        // two halves of each head swap — new row (h, i, j) takes the checkpoint row (h, j, i).
        int half = rows / headGroups / 2;
        long count = (long) nEmbd * rows;
        float[] src = valuesAt(seed, (int) count);
        float[] dst = new float[src.length];
        for (int h = 0; h < headGroups; h++) {
            for (int i = 0; i < half; i++) {
                for (int j = 0; j < 2; j++) {
                    int dstRow = h * half * 2 + i * 2 + j;
                    int srcRow = h * half * 2 + j * half + i;
                    System.arraycopy(src, srcRow * nEmbd, dst, dstRow * nEmbd, nEmbd);
                }
            }
        }
        g.tensorFromFloats(name, type, new long[]{nEmbd, rows}, dst);
    }

    /** The torch dtype string of {@code dtype} in config.json. */
    static String torchDtype(StDType dtype) {
        return dtype == StDType.BF16 ? "bfloat16" : "float32";
    }
}

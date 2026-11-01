package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the HK-MC part C converter classes: {@code OlmoeModel} (olmo.py:74-124), the Qwen3-VL and
 * Omni MoE text models (qwen3vl.py:303-372) and {@code InternLM3Model} (internlm.py:175-234, an LLAMA row without
 * experts). Two layers, a hidden size of 8, two query heads, one key/value head, an MoE of two experts and the
 * checkpoint's real names and config keys.
 */
public final class HfMoeClassicFixturesC {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int KV = 4;
    private static final int EXPERTS = 2;
    private static final int FF = 4;

    private HfMoeClassicFixturesC() {
    }

    /** {@code OlmoeForCausalLM}: the Qwen-style per-expert down/gate/up names without the fused tensors. */
    public static SyntheticHf.Builder olmoe() {
        SyntheticHf.Builder b = builder("{'architectures': ['OlmoeForCausalLM'], 'hidden_size': 8,"
                + " 'intermediate_size': 4, 'num_local_experts': 2, 'num_experts_per_tok': 1,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1, 'vocab_size': 8,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            layer(b, p);
            b.bf16(p + "self_attn.q_norm.weight", KV);
            b.bf16(p + "self_attn.k_norm.weight", KV);
            // the OLMOE arch maps no dense FFN (arch_tensors.tsv) — the mlp is the routed experts
            b.remove(p + "mlp.gate_proj.weight");
            b.remove(p + "mlp.up_proj.weight");
            b.remove(p + "mlp.down_proj.weight");
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "mlp.experts." + x + ".gate_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".up_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".down_proj.weight", DIM, FF);
            }
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code Qwen3VLMoeForConditionalGeneration}: the thinker-prefixed fused experts that arrive transposed.
     * The qwen3moe-based arch maps no shared-expert tensors (constants.py:2961 QWEN3MOE). */
    public static SyntheticHf.Builder qwen3VlMoe() {
        SyntheticHf.Builder b = builder("{'architectures': ['Qwen3VLMoeForConditionalGeneration'], 'hidden_size': 8,"
                + " 'moe_intermediate_size': 4, 'num_experts': 2, 'num_experts_per_tok': 1,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 1, 'vocab_size': 8, 'max_position_embeddings': 64,"
                + " 'rms_norm_eps': 1e-06, 'rope_theta': 10000.0, 'vision_config': {'deepstack_visual_indexes': [5, 9]},"
                + " 'rope_scaling': {'rope_type': 'mrope', 'mrope_section': [1, 1, 0]}}");
        b.bf16("thinker.model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "thinker.model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.q_norm.weight", KV);
            b.bf16(p + "self_attn.k_norm.weight", KV);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            b.bf16(p + "mlp.experts.gate_up_proj.weight", EXPERTS, DIM, 2 * FF);
            b.bf16(p + "mlp.experts.down_proj.weight", EXPERTS, FF, DIM);
        }
        b.bf16("thinker.model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code Qwen3OmniMoeForConditionalGeneration}: the same body, with no deepstack layers. */
    public static SyntheticHf.Builder qwen3OmniMoe() {
        return qwen3VlMoe().config(SyntheticHf.json("{'architectures':"
                + " ['Qwen3OmniMoeForConditionalGeneration'], 'hidden_size': 8, 'moe_intermediate_size': 4,"
                + " 'num_experts': 2, 'num_experts_per_tok': 1,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1, 'vocab_size': 8,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-06, 'rope_theta': 10000.0,"
                + " 'rope_scaling': {'rope_type': 'mrope', 'mrope_section': [1, 1, 0]}}"))
                .file("tokenizer_config.json", SyntheticHf.json("{'bos_token': '<|im_end|>', 'eos_token':"
                        + " '<|im_end|>', 'add_bos_token': false, 'clean_up_tokenization_spaces': false,"
                        + " 'added_tokens_decoder': {'3': {'content': '<|im_end|>', 'special': true,"
                        + " 'normalized': false}}}"))
                .tokenizerJson(SyntheticHf.json("{'added_tokens':["
                        + "{'id':0,'content':'<|start|>','special':true,'normalized':false},"
                        + "{'id':1,'content':'<|end|>','special':true,'normalized':false},"
                        + "{'id':3,'content':'<|im_end|>','special':true,'normalized':false}],"
                        + "'normalizer':null,"
                        + "'pre_tokenizer':{'type':'ByteLevel','add_prefix_space':false,'trim_offsets':true,"
                        + "'use_regex':true},'post_processor':null,"
                        + "'decoder':{'type':'ByteLevel','add_prefix_space':true,'trim_offsets':true,"
                        + "'use_regex':true},'model':{'type':'BPE','unk_token':null,'byte_fallback':false,"
                        + "'ignore_merges':false,'vocab':{'<|start|>':0,'<|end|>':1,'a':2,'<|im_end|>':3},"
                        + "'merges':['a a']}}"));
    }

    /**
     * {@code InternLM3ForCausalLM}: a dense LLAMA-arch row with stripped vision prefixes (kept out here) and an SPM
     * tokenizer.
     */
    public static SyntheticHf.Builder internLm3() {
        SyntheticHf.Builder b = builder("{'architectures': ['InternLM3ForCausalLM'], 'hidden_size': 8,"
                + " 'intermediate_size': 8, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 1, 'vocab_size': 8, 'max_position_embeddings': 64,"
                + " 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0, 'tie_word_embeddings': false}");
        // the internlm3 vocabulary strictly requires a sentencepiece tokenizer.model
        b.binaryFile("tokenizer.model", HfSpecialFixturesT5.unigramModel());
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            layer(b, p);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** The Llama-style attention of a decoder layer under {@code prefix}. */
    private static void layer(SyntheticHf.Builder b, String p) {
        b.bf16(p + "input_layernorm.weight", DIM);
        b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
        b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
        b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
        b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
        b.bf16(p + "post_attention_layernorm.weight", DIM);
        b.bf16(p + "mlp.gate_proj.weight", FF, DIM);
        b.bf16(p + "mlp.up_proj.weight", FF, DIM);
        b.bf16(p + "mlp.down_proj.weight", DIM, FF);
    }

    /** One end-to-end fixture: its leading HF architecture name and the checkpoint builder. */
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
                named("InternLM3ForCausalLM", HfMoeClassicFixturesC::internLm3),
                named("OlmoeForCausalLM", HfMoeClassicFixturesC::olmoe),
                named("Qwen3OmniMoeForConditionalGeneration", HfMoeClassicFixturesC::qwen3OmniMoe),
                named("Qwen3VLMoeForConditionalGeneration", HfMoeClassicFixturesC::qwen3VlMoe));
    }

    private static SyntheticHf.Builder builder(String singleQuotedConfig) {
        return SyntheticHf.builder().config(SyntheticHf.json(singleQuotedConfig));
    }
}

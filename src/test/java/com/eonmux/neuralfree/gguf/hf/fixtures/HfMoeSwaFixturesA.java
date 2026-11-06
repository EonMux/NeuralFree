package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the converter classes: {@code AfmoeModel} (afmoe.py:17-80),
 * {@code FlintModel} (flint.py:9-37), {@code Kolibri1Model} (kolibri1.py:9-22), {@code Cohere2MoeModel}
 * (command_r.py:67-183), the Granite model (granite.py:78-200), {@code Kolibri1Model} and the
 * {@code Gemma4UnifiedModel} suppress-tokens row. Two layers, a hidden size of 8, two query heads, one
 * key/value head, an MoE of two experts and the checkpoint's real names and config keys.
 */
public final class HfMoeSwaFixturesA {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int KV = 4;
    private static final int EXPERTS = 2;
    private static final int FF = 4;

    private HfMoeSwaFixturesA() {
    }

    /** {@code AfmoeForCausalLM}: the Llama-style per-expert projections. */
    public static SyntheticHf.Builder afmoe() {
        SyntheticHf.Builder b = builder("{'architectures': ['AfmoeForCausalLM'], 'hidden_size': 8,"
                + " 'intermediate_size': 8, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 1, 'head_dim': 4, 'moe_intermediate_size': 4,"
                + " 'num_local_experts': 2, 'num_experts_per_tok': 1, 'num_shared_experts': 1, 'sliding_window': 4,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            // the afmoe dual norms and attention gating (afmoe.cpp:59-75): the q/k norms of
            // head_dim, the attention output gate, the ffn pre/post norms, the router bias and
            // the shared expert
            b.bf16(p + "self_attn.q_norm.weight", KV);
            b.bf16(p + "self_attn.k_norm.weight", KV);
            b.bf16(p + "self_attn.attn_gate_proj.weight", DIM, DIM);
            b.bf16(p + "pre_feedforward_layernorm.weight", DIM);
            b.bf16(p + "post_feedforward_layernorm.weight", DIM);
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            b.bf16(p + "mlp.gate.expert_bias", EXPERTS);
            b.bf16(p + "mlp.shared_expert.gate_proj.weight", FF, DIM);
            b.bf16(p + "mlp.shared_expert.up_proj.weight", FF, DIM);
            b.bf16(p + "mlp.shared_expert.down_proj.weight", DIM, FF);
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

    /** {@code FlintForCausalLM}: the Qwen2-MoE fused experts, softmax offset and router expert bias. */
    public static SyntheticHf.Builder flint() {
        SyntheticHf.Builder b = builder("{'architectures': ['FlintForCausalLM'], 'hidden_size': 8,"
                + " 'moe_intermediate_size': 4, 'num_local_experts': 2,"
                + " 'num_experts_per_tok': 1, 'shared_expert_intermediate_size': 8, 'norm_topk_prob': true,"
                + " 'score_func': 'sigmoid',"
                + " 'mlp_only_layers': [0], 'sliding_window': 4,"
                + " 'layer_types': ['full_attention', 'sliding_attention'], 'no_rope_layer_indices': [1],"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'head_dim': 4, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-06,"
                + " 'rope_theta': 10000.0, 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            b.bf16(p + "self_attn.q_norm.weight", KV);
            b.bf16(p + "self_attn.k_norm.weight", KV);
            // the per-head softmax offset is the attention sinks tensor (tensor_patterns.tsv:285)
            b.bf16(p + "self_attn.softmax_offset", KV);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            if (l < 1) {
                // mlp_only_layers [0]: layer 0 is dense, the rest are MoE
                b.bf16(p + "mlp.gate_proj.weight", FF, DIM);
                b.bf16(p + "mlp.up_proj.weight", FF, DIM);
                b.bf16(p + "mlp.down_proj.weight", DIM, FF);
                continue;
            }
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            b.bf16(p + "mlp.experts.gate_up_proj.weight", EXPERTS, DIM, 2 * FF);
            b.bf16(p + "mlp.experts.down_proj.weight", EXPERTS, DIM, FF);
            b.bf16(p + "mlp.shared_expert.gate_proj.weight", 2, DIM);
            b.bf16(p + "mlp.shared_expert.up_proj.weight", 2, DIM);
            b.bf16(p + "mlp.shared_expert.down_proj.weight", DIM, 2);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code Kolibri1ForCausalLM}: the Afmoe names under the kolibri alias config keys. */
    public static SyntheticHf.Builder kolibri1() {
        SyntheticHf.Builder b = afmoe().config(SyntheticHf.json("{'architectures': ['Kolibri1ForCausalLM'],"
                + " 'hidden_size': 8,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,'head_dim': 4,"
                + " 'moe_intermediate_size': 4, 'num_local_experts': 2, 'num_experts_per_tok': 1,"
                + " 'num_shared_experts': 1, 'score_func': 'sigmoid', 'norm_topk_prob': true,"
                + " 'sliding_window': 4,"
                + " 'shared_expert_intermediate_size': 8, 'layer_types': ['sliding_attention',"
                + " 'full_attention'], 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05,"
                + " 'rope_theta': 10000.0, 'vocab_size': 8}"));
        // the kolibri dual norms: FFN_NORM owns post_attention_layernorm (-> ffn_norm); the
        // attention post-norm comes from the post_attn_norm source (tensor_patterns.tsv:276);
        // the kolibri arch carries no attention gate
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.remove(p + "pre_feedforward_layernorm.weight");
            b.remove(p + "self_attn.attn_gate_proj.weight");
            b.bf16(p + "post_attn_norm.weight", 8);
        }
        return b;
    }

    /** {@code Cohere2MoeForCausalLM}: the iterate-per-expert names with zero biases. */
    public static SyntheticHf.Builder cohere2() {
        SyntheticHf.Builder b = builder("{'architectures': ['Cohere2MoeForCausalLM'], 'hidden_size': 8,"
                + " 'intermediate_size': 4, 'num_experts': 2, 'num_experts_per_tok': 1,"
                + " 'shared_expert_combination_strategy': 'average', 'head_dim': 4,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'layer_types': ['sliding_attention', 'full_attention'], 'sliding_window': 4,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            // the COHERE2MOE set carries no post-attention norm and no q/k norms
            b.remove(p + "post_attention_layernorm.weight");
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "mlp.experts." + x + ".gate_proj.bias", FF);
                b.bf16(p + "mlp.experts." + x + ".gate_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".up_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".down_proj.weight", DIM, FF);
            }
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code GraniteSWAForCausalLM}: a dense Granite body with the SWA metadata. */
    public static SyntheticHf.Builder graniteSwa() {
        SyntheticHf.Builder b = builder("{'architectures': ['GraniteSWAForCausalLM'], 'hidden_size': 8,"
                + " 'intermediate_size': 8, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 1, 'head_dim': 4, 'layer_types': ['sliding_attention',"
                + " 'full_attention'], 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05,"
                + " 'rope_theta': 10000.0, 'vocab_size': 8, 'logits_scaling': 8.0}");
        body(b, false);
        return b;
    }

    /** {@code GraniteMoeSWAForCausalLM}: the MoE Granite body with the shared expert width. */
    public static SyntheticHf.Builder graniteMoeSwa() {
        SyntheticHf.Builder b = builder("{'architectures': ['GraniteMoeSWAForCausalLM'], 'hidden_size': 8,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'head_dim': 4, 'moe_intermediate_size': 4, 'num_experts_per_tok': 1,"
                + " 'num_local_experts': 2, 'shared_intermediate_size': 8,"
                + " 'layer_types': ['sliding_attention', 'full_attention'],"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8, 'logits_scaling': 8.0}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            b.bf16(p + "self_attn.sinks", KV);
            // the granite MoE forms: the block_sparse_moe router, the merged expert stacks
            // (tensor_patterns.tsv:450/413/512) and the fused shared expert
            // (granite.py:164-177 shared_mlp.input_linear/output_linear)
            b.bf16(p + "block_sparse_moe.gate.weight", EXPERTS, DIM);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "block_sparse_moe.experts." + x + ".w1.weight", FF, DIM);
                b.bf16(p + "block_sparse_moe.experts." + x + ".w3.weight", FF, DIM);
                b.bf16(p + "block_sparse_moe.experts." + x + ".w2.weight", DIM, FF);
            }
            b.bf16(p + "shared_mlp.input_linear.weight", 16, DIM);
            b.bf16(p + "shared_mlp.output_linear.weight", DIM, 8);
            // the granite SWA sets carry no FFN_POST_NORM
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code Gemma4UnifiedForConditionalGeneration}: the suppress_tokens of the generation config. */
    public static SyntheticHf.Builder gemma4Unified() {
        SyntheticHf.Builder b = builder("{'architectures': ['Gemma4UnifiedForConditionalGeneration'],"
                + " 'hidden_size': 8, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 1, 'head_dim': 4, 'global_head_dim': 4, 'intermediate_size': 8,"
                + " 'sliding_window': 4, 'vocab_size': 8, 'rms_norm_eps': 1e-06, 'rope_theta': 10000.0,"
                + " 'layer_types': ['sliding_attention', 'full_attention'],"
                + " 'num_kv_shared_layers': 0, 'max_position_embeddings': 64,"
                + " 'rope_parameters': {'full_attention': {'rope_type': 'proportional', 'rope_theta': 10000.0,"
                + " 'partial_rotary_factor': 0.5}}}");
        // the gemma4 vocabulary takes the Llama-HF tokenizer.json form: a BPE model with
        // byte_fallback and a Sequence decoder (HfVocabBase.isLlamaHf)
        b.tokenizerJson(SyntheticHf.json("{'added_tokens':["
                + "{'id':0,'content':'<s>','special':true,'normalized':false},"
                + "{'id':1,'content':'</s>','special':true,'normalized':false},"
                + "{'id':5,'content':'<|begin|>','special':true,'normalized':false},"
                + "{'id':6,'content':'Ġhi','special':false,'normalized':false,'lstrip':true}],"
                + "'normalizer':null,"
                + "'pre_tokenizer':{'type':'ByteLevel','add_prefix_space':false,'trim_offsets':true,'use_regex':true},"
                + "'post_processor':null,"
                + "'decoder':{'type':'Sequence','decoders':[{'type':'ByteLevel','add_prefix_space':true,"
                + "'trim_offsets':true,'use_regex':true}]},"
                + "'model':{'type':'BPE','unk_token':null,'byte_fallback':true,'ignore_merges':false,"
                + "'vocab':{'<s>':0,'</s>':1,'a':2,'b':3,'ab':4},'merges':['a b']}}"));
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            b.bf16(p + "self_attn.q_norm.weight", KV);
            b.bf16(p + "self_attn.k_norm.weight", KV);
            b.bf16(p + "pre_feedforward_layernorm.weight", DIM);
            b.bf16(p + "post_feedforward_layernorm.weight", DIM);
            b.bf16(p + "mlp.gate_proj.weight", 2 * FF, DIM);
            b.bf16(p + "mlp.up_proj.weight", 2 * FF, DIM);
            b.bf16(p + "mlp.down_proj.weight", DIM, 2 * FF);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 7, 8);
        b.file("generation_config.json", SyntheticHf.json("{'suppress_tokens': [5, 6]}"));
        return b;
    }

    /** The Llama-style attention of a decoder layer under {@code prefix}. */
    private static void attention(SyntheticHf.Builder b, String p) {
        b.bf16(p + "input_layernorm.weight", DIM);
        b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
        b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
        b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
        b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
        b.bf16(p + "post_attention_layernorm.weight", DIM);
    }

    /** The dense Granite body. */
    private static void body(SyntheticHf.Builder b, boolean padding) {
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            // the granite_swa graph loads the attention sinks (tensor_patterns.tsv:286)
            b.bf16(p + "self_attn.sinks", KV);
            // the granite sets carry no FFN_POST_NORM
            b.bf16(p + "mlp.gate_proj.weight", 8, DIM);
            b.bf16(p + "mlp.up_proj.weight", 8, DIM);
            b.bf16(p + "mlp.down_proj.weight", DIM, 8);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
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
                named("AfmoeForCausalLM", HfMoeSwaFixturesA::afmoe),
                named("FlintForCausalLM", HfMoeSwaFixturesA::flint),
                named("Kolibri1ForCausalLM", HfMoeSwaFixturesA::kolibri1),
                named("Cohere2MoeForCausalLM", HfMoeSwaFixturesA::cohere2),
                named("GraniteSWAForCausalLM", HfMoeSwaFixturesA::graniteSwa),
                named("GraniteMoeSWAForCausalLM", HfMoeSwaFixturesA::graniteMoeSwa),
                named("Gemma4UnifiedForConditionalGeneration", HfMoeSwaFixturesA::gemma4Unified));
    }

    private static SyntheticHf.Builder builder(String singleQuotedConfig) {
        return SyntheticHf.builder().config(SyntheticHf.json(singleQuotedConfig));
    }
}

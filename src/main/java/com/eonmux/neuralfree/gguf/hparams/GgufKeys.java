package com.eonmux.neuralfree.gguf.hparams;

/**
 * GGUF metadata key names, one constant per llama.cpp {@code LLM_KV_NAMES} entry (src/llama-arch.cpp:175-481)
 * with the {@code LLM_KV_} prefix dropped. Architecture-scoped keys keep the {@code %s} placeholder of the
 * llama.cpp format string; {@link GgufKv} substitutes the architecture name. Engine code names metadata keys
 * only through these constants.
 */
public final class GgufKeys {

    /**
     * Diffusion models: whether logits are shifted by one position (read by llama.cpp
     * examples/diffusion/diffusion-cli.cpp:183; default true).
     */
    public static final String DIFFUSION_SHIFT_LOGITS = "diffusion.shift_logits";

    public static final String GENERAL_TYPE = "general.type";
    public static final String GENERAL_ARCHITECTURE = "general.architecture";
    public static final String GENERAL_QUANTIZATION_VERSION = "general.quantization_version";
    public static final String GENERAL_ALIGNMENT = "general.alignment";
    public static final String GENERAL_FILE_TYPE = "general.file_type";
    public static final String GENERAL_SAMPLING_SEQUENCE = "general.sampling.sequence";
    public static final String GENERAL_SAMPLING_TOP_K = "general.sampling.top_k";
    public static final String GENERAL_SAMPLING_TOP_P = "general.sampling.top_p";
    public static final String GENERAL_SAMPLING_MIN_P = "general.sampling.min_p";
    public static final String GENERAL_SAMPLING_XTC_PROBABILITY = "general.sampling.xtc_probability";
    public static final String GENERAL_SAMPLING_XTC_THRESHOLD = "general.sampling.xtc_threshold";
    public static final String GENERAL_SAMPLING_TEMP = "general.sampling.temp";
    public static final String GENERAL_SAMPLING_PENALTY_LAST_N = "general.sampling.penalty_last_n";
    public static final String GENERAL_SAMPLING_PENALTY_REPEAT = "general.sampling.penalty_repeat";
    public static final String GENERAL_SAMPLING_MIROSTAT = "general.sampling.mirostat";
    public static final String GENERAL_SAMPLING_MIROSTAT_TAU = "general.sampling.mirostat_tau";
    public static final String GENERAL_SAMPLING_MIROSTAT_ETA = "general.sampling.mirostat_eta";
    public static final String GENERAL_NAME = "general.name";
    public static final String GENERAL_AUTHOR = "general.author";
    public static final String GENERAL_VERSION = "general.version";
    public static final String GENERAL_URL = "general.url";
    public static final String GENERAL_DESCRIPTION = "general.description";
    public static final String GENERAL_LICENSE = "general.license";
    public static final String GENERAL_SOURCE_URL = "general.source.url";
    public static final String GENERAL_SOURCE_HF_REPO = "general.source.huggingface.repository";
    public static final String GENERAL_TENSOR_EXTRA_NAME = "general.tensor_extra.name";
    public static final String GENERAL_TENSOR_EXTRA_PREC_A4 = "general.tensor_extra.prec_a4";

    public static final String VOCAB_SIZE = "%s.vocab_size";
    public static final String CONTEXT_LENGTH = "%s.context_length";
    public static final String EMBEDDING_LENGTH = "%s.embedding_length";
    public static final String EMBEDDING_LENGTH_OUT = "%s.embedding_length_out";
    public static final String EMBEDDING_LENGTH_PER_LAYER = "%s.embedding_length_per_layer_input";
    public static final String FEATURES_LENGTH = "%s.features_length";
    public static final String BLOCK_COUNT = "%s.block_count";
    public static final String LEADING_DENSE_BLOCK_COUNT = "%s.leading_dense_block_count";
    public static final String ATTN_RES_BLOCK_SIZE = "%s.attn_res.block_size";
    public static final String ACTIVATION_SITU_BETA = "%s.activation.situ_beta";
    public static final String ACTIVATION_SITU_LINEAR_BETA = "%s.activation.situ_linear_beta";
    public static final String FEED_FORWARD_LENGTH = "%s.feed_forward_length";
    public static final String EXPERT_FEED_FORWARD_LENGTH = "%s.expert_feed_forward_length";
    public static final String EXPERT_SHARED_FEED_FORWARD_LENGTH = "%s.expert_shared_feed_forward_length";
    public static final String EXPERT_CHUNK_FEED_FORWARD_LENGTH = "%s.expert_chunk_feed_forward_length";
    public static final String SWIGLU_CLAMP_EXP = "%s.swiglu_clamp_exp";
    public static final String SWIGLU_CLAMP_SHEXP = "%s.swiglu_clamp_shexp";
    public static final String USE_PARALLEL_RESIDUAL = "%s.use_parallel_residual";
    public static final String TENSOR_DATA_LAYOUT = "%s.tensor_data_layout";
    public static final String EXPERT_COUNT = "%s.expert_count";
    public static final String EXPERT_USED_COUNT = "%s.expert_used_count";
    public static final String EXPERT_SHARED_COUNT = "%s.expert_shared_count";
    public static final String EXPERT_GROUP_COUNT = "%s.expert_group_count";
    public static final String EXPERT_GROUP_USED_COUNT = "%s.expert_group_used_count";
    public static final String EXPERT_WEIGHTS_SCALE = "%s.expert_weights_scale";
    public static final String EXPERT_WEIGHTS_NORM = "%s.expert_weights_norm";
    public static final String EXPERT_LATENT_LENGTH = "%s.expert_latent_length";
    public static final String EXPERT_GATING_FUNC = "%s.expert_gating_func";
    public static final String EXPERT_GROUP_SCALE = "%s.expert_group_scale";
    public static final String EXPERTS_PER_GROUP = "%s.experts_per_group";
    public static final String MOE_EVERY_N_LAYERS = "%s.moe_every_n_layers";
    public static final String MOE_LATENT_SIZE = "%s.moe_latent_size";
    public static final String POST_NORM_EPS = "%s.post_norm_epsilon";
    public static final String NEXTN_PREDICT_LAYERS = "%s.nextn_predict_layers";
    public static final String NUM_DEEPSTACK_LAYERS = "%s.n_deepstack_layers";
    public static final String DEEPSTACK_MAPPING = "%s.deepstack_mapping";
    public static final String HIDDEN_ACT = "%s.hidden_activation";
    public static final String POOLING_TYPE = "%s.pooling_type";
    public static final String LOGIT_SCALE = "%s.logit_scale";
    public static final String DECODER_START_TOKEN_ID = "%s.decoder_start_token_id";
    public static final String DECODER_BLOCK_COUNT = "%s.decoder_block_count";
    public static final String ATTN_LOGIT_SOFTCAPPING = "%s.attn_logit_softcapping";
    public static final String ROUTER_LOGIT_SOFTCAPPING = "%s.router_logit_softcapping";
    public static final String FINAL_LOGIT_SOFTCAPPING = "%s.final_logit_softcapping";
    public static final String SWIN_NORM = "%s.swin_norm";
    public static final String RESCALE_EVERY_N_LAYERS = "%s.rescale_every_n_layers";
    public static final String TIME_MIX_EXTRA_DIM = "%s.time_mix_extra_dim";
    public static final String TIME_DECAY_EXTRA_DIM = "%s.time_decay_extra_dim";
    public static final String RESIDUAL_SCALE = "%s.residual_scale";
    public static final String EMBEDDING_SCALE = "%s.embedding_scale";
    public static final String ADAPTER_COUNT = "%s.adapters.count";
    public static final String ADAPTER_TOKEN_IDS_ACTIVATE = "%s.adapters.token_ids_activate";
    public static final String ADAPTER_TOKEN_IDS_SUBSTITUTE = "%s.adapters.token_ids_substitute";
    public static final String ADAPTER_LORA_RANK = "%s.adapters.lora_rank";
    public static final String ADAPTER_ROUTER_GAIN = "%s.adapters.router_gain";
    public static final String TOKEN_SHIFT_COUNT = "%s.token_shift_count";
    public static final String INTERLEAVE_MOE_LAYER_STEP = "%s.interleave_moe_layer_step";
    public static final String FULL_ATTENTION_INTERVAL = "%s.full_attention_interval";
    public static final String NUM_LOOPS = "%s.num_loops";
    public static final String SKIP_LOOP_FINAL_NORM = "%s.skip_loop_final_norm";
    public static final String HRM_LAYERS_PER_STACK = "%s.hrm.layers_per_stack";
    public static final String HRM_H_CYCLES = "%s.hrm.h_cycles";
    public static final String HRM_L_CYCLES = "%s.hrm.l_cycles";
    public static final String HRM_PREFIX_LM = "%s.hrm.prefix_lm";

    public static final String ATTENTION_HEAD_COUNT = "%s.attention.head_count";
    public static final String ATTENTION_HEAD_COUNT_KV = "%s.attention.head_count_kv";
    public static final String ATTENTION_MAX_ALIBI_BIAS = "%s.attention.max_alibi_bias";
    public static final String ATTENTION_CLAMP_KQV = "%s.attention.clamp_kqv";
    public static final String ATTENTION_KEY_LENGTH = "%s.attention.key_length";
    public static final String ATTENTION_VALUE_LENGTH = "%s.attention.value_length";
    public static final String ATTENTION_LAYERNORM_EPS = "%s.attention.layer_norm_epsilon";
    public static final String ATTENTION_LAYERNORM_RMS_EPS = "%s.attention.layer_norm_rms_epsilon";
    public static final String ATTENTION_GROUPNORM_EPS = "%s.attention.group_norm_epsilon";
    public static final String ATTENTION_GROUPNORM_GROUPS = "%s.attention.group_norm_groups";
    public static final String ATTENTION_CAUSAL = "%s.attention.causal";
    public static final String ATTENTION_Q_LORA_RANK = "%s.attention.q_lora_rank";
    public static final String ATTENTION_KV_LORA_RANK = "%s.attention.kv_lora_rank";
    public static final String ATTENTION_DECAY_LORA_RANK = "%s.attention.decay_lora_rank";
    public static final String ATTENTION_ICLR_LORA_RANK = "%s.attention.iclr_lora_rank";
    public static final String ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK = "%s.attention.value_residual_mix_lora_rank";
    public static final String ATTENTION_GATE_LORA_RANK = "%s.attention.gate_lora_rank";
    public static final String ATTENTION_RELATIVE_BUCKETS_COUNT = "%s.attention.relative_buckets_count";
    public static final String ATTENTION_SLIDING_WINDOW = "%s.attention.sliding_window";
    public static final String ATTENTION_SLIDING_WINDOW_PATTERN = "%s.attention.sliding_window_pattern";
    public static final String ATTENTION_ROPE_PATTERN = "%s.attention.rope_pattern";
    public static final String ATTENTION_VALUE_EXPERT_COUNT = "%s.attention.value_expert_count";
    public static final String ATTENTION_VALUE_EXPERT_USED_COUNT = "%s.attention.value_expert_used_count";

    public static final String ATTENTION_SCALE = "%s.attention.scale";
    public static final String ATTENTION_OUTPUT_SCALE = "%s.attention.output_scale";
    public static final String ATTENTION_VALUE_SCALE = "%s.attention.value_scale";
    public static final String ATTENTION_TEMPERATURE_LENGTH = "%s.attention.temperature_length";
    public static final String ATTENTION_TEMPERATURE_SCALE = "%s.attention.temperature_scale";
    public static final String REALTIME_DELAY_TOKENS = "%s.realtime_delay_tokens";
    public static final String ATTENTION_KEY_LENGTH_MLA = "%s.attention.key_length_mla";
    public static final String ATTENTION_VALUE_LENGTH_MLA = "%s.attention.value_length_mla";
    public static final String ATTENTION_KEY_LENGTH_SWA = "%s.attention.key_length_swa";
    public static final String ATTENTION_VALUE_LENGTH_SWA = "%s.attention.value_length_swa";
    public static final String ATTENTION_KEY_LENGTH_MLA_SWA = "%s.attention.key_length_mla_swa";
    public static final String ATTENTION_VALUE_LENGTH_MLA_SWA = "%s.attention.value_length_mla_swa";
    public static final String ATTENTION_KV_LORA_RANK_SWA = "%s.attention.kv_lora_rank_swa";
    public static final String ATTENTION_INDEXER_HEAD_COUNT = "%s.attention.indexer.head_count";
    public static final String ATTENTION_INDEXER_KEY_LENGTH = "%s.attention.indexer.key_length";
    public static final String ATTENTION_INDEXER_TOP_K = "%s.attention.indexer.top_k";
    public static final String ATTENTION_INDEXER_BLOCK_SIZE = "%s.attention.indexer.block_size";
    public static final String ATTENTION_INDEXER_LOCAL_BLOCKS = "%s.attention.indexer.local_blocks";
    public static final String ATTENTION_INDEXER_TYPES = "%s.attention.indexer.types";
    public static final String ATTENTION_INDEXER_KPOOL = "%s.attention.indexer.kpool";
    public static final String ATTENTION_INDEXER_KPOOL_SELECT_TAIL = "%s.attention.indexer.kpool_select_tail";
    public static final String ATTENTION_OUTPUT_GROUP_COUNT = "%s.attention.output_group_count";
    public static final String ATTENTION_OUTPUT_LORA_RANK = "%s.attention.output_lora_rank";
    public static final String ATTENTION_COMPRESS_ROPE_FREQ_BASE = "%s.attention.compress_rope_freq_base";
    public static final String ATTENTION_COMPRESS_RATIOS = "%s.attention.compress_ratios";
    public static final String ATTENTION_SHARED_KV_LAYERS = "%s.attention.shared_kv_layers";
    public static final String ATTENTION_RECURRENT_LAYERS = "%s.attention.recurrent_layers";

    public static final String HYPER_CONNECTION_COUNT = "%s.hyper_connection.count";
    public static final String HYPER_CONNECTION_SINKHORN_ITERATIONS = "%s.hyper_connection.sinkhorn_iterations";
    public static final String HYPER_CONNECTION_EPSILON = "%s.hyper_connection.epsilon";
    public static final String HYPER_CONNECTION_MAGNITUDE = "%s.hyper_connection.magnitude";
    public static final String HYPER_CONNECTION_LOW_RANK = "%s.hyper_connection.low_rank";

    public static final String PLE_LAYERS = "%s.ple.layers";
    public static final String PLE_NGRAM_SIZE = "%s.ple.ngram_size";
    public static final String PLE_HEADS_PER_NGRAM = "%s.ple.heads_per_ngram";
    public static final String PLE_CONV_KERNEL = "%s.ple.conv_kernel";
    public static final String PLE_LAYER_MULTIPLIERS = "%s.ple.layer_multipliers";
    public static final String PLE_HEAD_OFFSETS = "%s.ple.head_offsets";
    public static final String PLE_HEAD_VOCAB_SIZES = "%s.ple.head_vocab_sizes";
    public static final String PLE_EOS_TOKEN_ID = "%s.ple.eos_token_id";
    public static final String PLE_IMAGE_TOKEN_ID = "%s.ple.image_token_id";

    public static final String HASH_LAYER_COUNT = "%s.hash_layer_count";

    public static final String ROPE_DIMENSION_COUNT = "%s.rope.dimension_count";
    public static final String ROPE_DIMENSION_COUNT_SWA = "%s.rope.dimension_count_swa";
    public static final String ROPE_DIMENSION_SECTIONS = "%s.rope.dimension_sections";
    public static final String ROPE_SECTION_ORDER = "%s.rope.section_order";
    public static final String ROPE_FREQ_BASE = "%s.rope.freq_base";
    public static final String ROPE_FREQ_BASE_SPATIAL = "%s.rope.freq_base_spatial";
    public static final String ROPE_FREQ_BASE_SWA = "%s.rope.freq_base_swa";
    public static final String ROPE_SCALE_LINEAR = "%s.rope.scale_linear";
    public static final String ROPE_SCALING_TYPE = "%s.rope.scaling.type";
    public static final String ROPE_SCALING_FACTOR = "%s.rope.scaling.factor";
    public static final String ROPE_SCALING_ALPHA = "%s.rope.scaling.alpha";
    public static final String ROPE_SCALING_ATTN_FACTOR = "%s.rope.scaling.attn_factor";
    public static final String ROPE_SCALING_ORIG_CTX_LEN = "%s.rope.scaling.original_context_length";
    public static final String ROPE_SCALING_FINETUNED = "%s.rope.scaling.finetuned";
    public static final String ROPE_SCALING_YARN_LOG_MUL = "%s.rope.scaling.yarn_log_multiplier";
    public static final String ROPE_SCALING_YARN_EXT_FACTOR = "%s.rope.scaling.yarn_ext_factor";
    public static final String ROPE_SCALING_YARN_ATTN_FACTOR = "%s.rope.scaling.yarn_attn_factor";
    public static final String ROPE_SCALING_YARN_BETA_FAST = "%s.rope.scaling.yarn_beta_fast";
    public static final String ROPE_SCALING_YARN_BETA_SLOW = "%s.rope.scaling.yarn_beta_slow";

    public static final String SPLIT_NO = "split.no";
    public static final String SPLIT_COUNT = "split.count";
    public static final String SPLIT_TENSORS_COUNT = "split.tensors.count";

    public static final String SSM_CONV_KERNEL = "%s.ssm.conv_kernel";
    public static final String SSM_INNER_SIZE = "%s.ssm.inner_size";
    public static final String SSM_STATE_SIZE = "%s.ssm.state_size";
    public static final String SSM_TIME_STEP_RANK = "%s.ssm.time_step_rank";
    public static final String SSM_GROUP_COUNT = "%s.ssm.group_count";
    public static final String SSM_DT_B_C_RMS = "%s.ssm.dt_b_c_rms";

    public static final String KDA_HEAD_DIM = "%s.kda.head_dim";
    public static final String KDA_SAFE_GATE = "%s.kda.safe_gate";
    public static final String KDA_GATE_LOWER_BOUND = "%s.kda.gate_lower_bound";

    public static final String WKV_HEAD_SIZE = "%s.wkv.head_size";

    public static final String POSNET_EMBEDDING_LENGTH = "%s.posnet.embedding_length";
    public static final String POSNET_BLOCK_COUNT = "%s.posnet.block_count";

    public static final String CONVNEXT_EMBEDDING_LENGTH = "%s.convnext.embedding_length";
    public static final String CONVNEXT_BLOCK_COUNT = "%s.convnext.block_count";

    public static final String CLASSIFIER_OUTPUT_LABELS = "%s.classifier.output_labels";
    public static final String CLASSIFIER_POOLING_TYPE = "%s.classifier.pooling_type";
    public static final String CLASSIFIER_ACTIVATION = "%s.classifier.activation";

    public static final String DECISION_BLOCK_COUNT = "%s.decision.block_count";
    public static final String DECISION_ROUTING_BLOCK_COUNT = "%s.decision.routing_block_count";
    public static final String DECISION_HEAD_COUNT = "%s.decision.head_count";
    public static final String DECISION_LABELS = "%s.decision.labels";

    public static final String TARGET_LAYERS = "%s.target_layers";
    public static final String TARGET_HIDDEN_SIZE = "%s.target_hidden_size";
    public static final String NORM_BEFORE_RESIDUAL = "%s.norm_before_residual";
    public static final String NORM_BEFORE_FC = "%s.norm_before_fc";

    public static final String DFLASH_BLOCK_SIZE = "%s.block_size";
    public static final String DFLASH_CONV_KERNEL_SIZE = "%s.conv_kernel_size";
    public static final String DFLASH_CONV_GROUP_SIZE = "%s.conv_group_size";
    public static final String DFLASH_SELECTOR_RANK = "%s.selector_rank";
    public static final String DFLASH_SELECTOR_TOP_K = "%s.selector_top_k";

    public static final String LILICORR_HIDDEN_SIZE = "lilicorr.hidden_size";
    public static final String LILICORR_CANDIDATE_TOPK = "lilicorr.candidate_topk";
    public static final String LILICORR_NUM_HEADS = "lilicorr.num_heads";
    public static final String LILICORR_NUM_LAYERS = "lilicorr.num_layers";
    public static final String LILICORR_VECTOR_EPS = "lilicorr.vector_eps";
    public static final String LILICORR_LOGIT_SCALE = "lilicorr.logit_scale";
    public static final String LILICORR_FACTOR_DIM = "lilicorr.factor_dim";
    public static final String LILICORR_MLP_HIDDEN_SIZE = "lilicorr.mlp_hidden_size";

    public static final String SHORTCONV_L_CACHE = "%s.shortconv.l_cache";
    // sentence-transformers dense modules feature dims
    public static final String DENSE_2_FEAT_IN = "%s.dense_2_feat_in";
    public static final String DENSE_2_FEAT_OUT = "%s.dense_2_feat_out";
    public static final String DENSE_3_FEAT_IN = "%s.dense_3_feat_in";
    public static final String DENSE_3_FEAT_OUT = "%s.dense_3_feat_out";

    public static final String TOKENIZER_MODEL = "tokenizer.ggml.model";
    public static final String TOKENIZER_PRE = "tokenizer.ggml.pre";
    public static final String TOKENIZER_LIST = "tokenizer.ggml.tokens";
    public static final String TOKENIZER_TOKEN_TYPE = "tokenizer.ggml.token_type";
    public static final String TOKENIZER_TOKEN_TYPE_COUNT = "tokenizer.ggml.token_type_count";
    public static final String TOKENIZER_SCORES = "tokenizer.ggml.scores";
    public static final String TOKENIZER_MERGES = "tokenizer.ggml.merges";
    public static final String TOKENIZER_BOS_ID = "tokenizer.ggml.bos_token_id";
    public static final String TOKENIZER_EOS_ID = "tokenizer.ggml.eos_token_id";
    public static final String TOKENIZER_EOT_ID = "tokenizer.ggml.eot_token_id";
    public static final String TOKENIZER_EOM_ID = "tokenizer.ggml.eom_token_id";
    public static final String TOKENIZER_UNK_ID = "tokenizer.ggml.unknown_token_id";
    public static final String TOKENIZER_SEP_ID = "tokenizer.ggml.seperator_token_id";
    public static final String TOKENIZER_PAD_ID = "tokenizer.ggml.padding_token_id";
    public static final String TOKENIZER_CLS_ID = "tokenizer.ggml.cls_token_id";
    public static final String TOKENIZER_MASK_ID = "tokenizer.ggml.mask_token_id";
    public static final String TOKENIZER_ADD_BOS = "tokenizer.ggml.add_bos_token";
    public static final String TOKENIZER_ADD_EOS = "tokenizer.ggml.add_eos_token";
    public static final String TOKENIZER_ADD_SEP = "tokenizer.ggml.add_sep_token";
    public static final String TOKENIZER_DNA_AUTO_TAGS = "tokenizer.ggml.dna_auto_tags";
    public static final String TOKENIZER_ADD_PREFIX = "tokenizer.ggml.add_space_prefix";
    public static final String TOKENIZER_REMOVE_EXTRA_WS = "tokenizer.ggml.remove_extra_whitespaces";
    public static final String TOKENIZER_PRECOMPILED_CHARSMAP = "tokenizer.ggml.precompiled_charsmap";
    public static final String TOKENIZER_HF_JSON = "tokenizer.huggingface.json";
    public static final String TOKENIZER_RWKV = "tokenizer.rwkv.world";
    public static final String TOKENIZER_CHAT_TEMPLATE = "tokenizer.chat_template";
    public static final String TOKENIZER_NORMALIZER_NFC = "tokenizer.ggml.normalizer.nfc";
    public static final String TOKENIZER_NORMALIZER_LOWERCASE = "tokenizer.ggml.normalizer.lowercase";
    public static final String TOKENIZER_NORMALIZER_STRIP_ACCENTS = "tokenizer.ggml.normalizer.strip_accents";
    public static final String TOKENIZER_FIM_PRE_ID = "tokenizer.ggml.fim_pre_token_id";
    public static final String TOKENIZER_FIM_SUF_ID = "tokenizer.ggml.fim_suf_token_id";
    public static final String TOKENIZER_FIM_MID_ID = "tokenizer.ggml.fim_mid_token_id";
    public static final String TOKENIZER_FIM_PAD_ID = "tokenizer.ggml.fim_pad_token_id";
    public static final String TOKENIZER_FIM_REP_ID = "tokenizer.ggml.fim_rep_token_id";
    public static final String TOKENIZER_FIM_SEP_ID = "tokenizer.ggml.fim_sep_token_id";
    public static final String TOKENIZER_SUPPRESS_TOKENS = "tokenizer.ggml.suppress_tokens";

    public static final String ADAPTER_TYPE = "adapter.type";
    public static final String ADAPTER_LORA_ALPHA = "adapter.lora.alpha";
    public static final String ADAPTER_LORA_TASK_NAME = "adapter.lora.task_name";
    public static final String ADAPTER_LORA_PROMPT_PREFIX = "adapter.lora.prompt_prefix";
    public static final String ADAPTER_ALORA_INVOCATION_TOKENS = "adapter.alora.invocation_tokens";

    public static final String XIELU_ALPHA_N = "xielu.alpha_n";
    public static final String XIELU_ALPHA_P = "xielu.alpha_p";
    public static final String XIELU_BETA = "xielu.beta";
    public static final String XIELU_EPS = "xielu.eps";

    public static final String PRISM_HADAMARD_VERSION = "prism.hadamard.version";
    public static final String PRISM_HADAMARD_TIED_OUTPUT = "prism.hadamard.tied_output";
    public static final String PRISM_HADAMARD_BLOCK_SIZE = "prism.hadamard.block_size";
    public static final String PRISM_HADAMARD_TRANSFORM = "prism.hadamard.transform";
    public static final String PRISM_HADAMARD_AXIS = "prism.hadamard.axis";
    public static final String PRISM_HADAMARD_SIGN_MODE = "prism.hadamard.sign_mode";
    public static final String PRISM_HADAMARD_SIGN_WIDTHS = "prism.hadamard.sign_widths";
    public static final String PRISM_HADAMARD_SIGN_VALUES = "prism.hadamard.sign_values";
    public static final String PRISM_HADAMARD_WEIGHT_NAMES = "prism.hadamard.weight_names";
    public static final String PRISM_HADAMARD_INVERSE_WEIGHT_NAMES = "prism.hadamard.inverse_weight_names";
    public static final String PRISM_HADAMARD_GDN_V_GROUPED = "prism.hadamard.gdn_v_grouped";

    // deprecated
    public static final String TOKENIZER_PREFIX_ID = "tokenizer.ggml.prefix_token_id";
    public static final String TOKENIZER_SUFFIX_ID = "tokenizer.ggml.suffix_token_id";
    public static final String TOKENIZER_MIDDLE_ID = "tokenizer.ggml.middle_token_id";

    private GgufKeys() {
    }
}

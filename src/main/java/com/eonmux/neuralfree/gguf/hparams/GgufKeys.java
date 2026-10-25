package com.eonmux.neuralfree.gguf.hparams;

/**
 * GGUF metadata key names, one constant per llama.cpp {@code LLM_KV_NAMES} entry (src/llama-arch.cpp:175-481)
 * with the {@code LLM_KV_} prefix dropped. Architecture-scoped keys keep the {@code %s} placeholder of the
 * llama.cpp format string; {@link GgufKv} substitutes the architecture name. The extension keys NeuralFree reads
 * or writes beyond llama.cpp's (the PRISM keys and the {@code neuralfree.*} keys) are listed here too. Engine code
 * names metadata keys only through these constants.
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

    // Model card metadata the converter writes (gguf-py/gguf/constants.py:45-106). The base-model and dataset keys
    // carry a %d index, formatted by the writer.
    public static final String GENERAL_ORGANIZATION = "general.organization";
    public static final String GENERAL_FINETUNE = "general.finetune";
    public static final String GENERAL_BASENAME = "general.basename";
    public static final String GENERAL_QUANTIZED_BY = "general.quantized_by";
    public static final String GENERAL_SIZE_LABEL = "general.size_label";
    public static final String GENERAL_LICENSE_NAME = "general.license.name";
    public static final String GENERAL_LICENSE_LINK = "general.license.link";
    public static final String GENERAL_DOI = "general.doi";
    public static final String GENERAL_UUID = "general.uuid";
    public static final String GENERAL_REPO_URL = "general.repo_url";
    public static final String GENERAL_SOURCE_DOI = "general.source.doi";
    public static final String GENERAL_SOURCE_UUID = "general.source.uuid";
    public static final String GENERAL_SOURCE_REPO_URL = "general.source.repo_url";
    public static final String GENERAL_BASE_MODEL_COUNT = "general.base_model.count";
    public static final String GENERAL_BASE_MODEL_NAME = "general.base_model.%d.name";
    public static final String GENERAL_BASE_MODEL_AUTHOR = "general.base_model.%d.author";
    public static final String GENERAL_BASE_MODEL_VERSION = "general.base_model.%d.version";
    public static final String GENERAL_BASE_MODEL_ORGANIZATION = "general.base_model.%d.organization";
    public static final String GENERAL_BASE_MODEL_DESCRIPTION = "general.base_model.%d.description";
    public static final String GENERAL_BASE_MODEL_URL = "general.base_model.%d.url";
    public static final String GENERAL_BASE_MODEL_DOI = "general.base_model.%d.doi";
    public static final String GENERAL_BASE_MODEL_UUID = "general.base_model.%d.uuid";
    public static final String GENERAL_BASE_MODEL_REPO_URL = "general.base_model.%d.repo_url";
    public static final String GENERAL_DATASET_COUNT = "general.dataset.count";
    public static final String GENERAL_DATASET_NAME = "general.dataset.%d.name";
    public static final String GENERAL_DATASET_AUTHOR = "general.dataset.%d.author";
    public static final String GENERAL_DATASET_VERSION = "general.dataset.%d.version";
    public static final String GENERAL_DATASET_ORGANIZATION = "general.dataset.%d.organization";
    public static final String GENERAL_DATASET_DESCRIPTION = "general.dataset.%d.description";
    public static final String GENERAL_DATASET_URL = "general.dataset.%d.url";
    public static final String GENERAL_DATASET_DOI = "general.dataset.%d.doi";
    public static final String GENERAL_DATASET_UUID = "general.dataset.%d.uuid";
    public static final String GENERAL_DATASET_REPO_URL = "general.dataset.%d.repo_url";
    public static final String GENERAL_TAGS = "general.tags";
    public static final String GENERAL_LANGUAGES = "general.languages";

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
    /** Gemma 3n per-layer activation sparsity (gguf-py/gguf/constants.py:158). */
    public static final String ACTIVATION_SPARSITY_SCALE = "%s.activation_sparsity_scale";
    /** Gemma 3n AltUp (gguf-py/gguf/constants.py:159-160). */
    public static final String ALTUP_ACTIVE_IDX = "%s.altup.active_idx";
    public static final String ALTUP_NUM_INPUTS = "%s.altup.num_inputs";
    /** Draft-model keys (gguf-py/gguf/constants.py:174-175). */
    public static final String SAMPLE_FROM_ANCHOR = "%s.sample_from_anchor";
    public static final String HAS_CONFIDENCE_HEAD = "%s.has_confidence_head";

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
    /** YaRN {@code truncate} (gguf-py/gguf/constants.py:274). */
    public static final String ROPE_SCALING_YARN_TRUNCATE = "%s.rope.scaling.yarn_truncate";
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
    /** Decision-head keys of gguf-py/gguf/constants.py:341-350; the temperature name and bias indexes are formatted. */
    public static final String DECISION_TYPE = "%s.decision.type";
    public static final String DECISION_MAX_HEAD_TOKENS = "%s.decision.max_head_tokens";
    public static final String DECISION_TEMPERATURE = "%s.decision.temperature.%s";
    public static final String DECISION_SCORE_BIAS = "%s.decision.score_bias.%d.%d";

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

    public static final String TOKENIZER_PREFIX = "tokenizer.";
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
    /** A named chat template; {@code %s} is the template name, not the architecture (constants.py:379). */
    public static final String TOKENIZER_CHAT_TEMPLATE_N = "tokenizer.chat_template.%s";
    /** The names of the named chat templates (gguf-py/gguf/constants.py:380). */
    public static final String TOKENIZER_CHAT_TEMPLATES = "tokenizer.chat_templates";
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

    // Projector (mmproj) keys, typed for the projector checkpoints (gguf-py/gguf/constants.py:409-511).
    public static final String CLIP_AUDIO_ATTENTION_HEAD_COUNT = "clip.audio.attention.head_count";
    public static final String CLIP_AUDIO_ATTENTION_HEAD_DIM = "clip.audio.attention.head_dim";
    public static final String CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON = "clip.audio.attention.layer_norm_epsilon";
    public static final String CLIP_AUDIO_BLOCK_COUNT = "clip.audio.block_count";
    public static final String CLIP_AUDIO_CHUNK_SIZE = "clip.audio.chunk_size";
    public static final String CLIP_AUDIO_CONV_KERNEL_SIZE = "clip.audio.conv_kernel_size";
    public static final String CLIP_AUDIO_DEPTHS = "clip.audio.depths";
    public static final String CLIP_AUDIO_EMBEDDING_LENGTH = "clip.audio.embedding_length";
    public static final String CLIP_AUDIO_FEATURE_LAYER = "clip.audio.feature_layer";
    public static final String CLIP_AUDIO_FEED_FORWARD_LENGTH = "clip.audio.feed_forward_length";
    public static final String CLIP_AUDIO_HIDDEN_SIZES = "clip.audio.hidden_sizes";
    public static final String CLIP_AUDIO_LOCAL_BLOCK_COUNT = "clip.audio.local_block_count";
    public static final String CLIP_AUDIO_LOCAL_GROUP_SIZE = "clip.audio.local_group_size";
    public static final String CLIP_AUDIO_MAX_POS_EMB = "clip.audio.max_pos_emb";
    public static final String CLIP_AUDIO_NUM_MEL_BINS = "clip.audio.num_mel_bins";
    public static final String CLIP_AUDIO_PATCH_SIZES = "clip.audio.patch_sizes";
    public static final String CLIP_AUDIO_PROJECTION_DIM = "clip.audio.projection_dim";
    public static final String CLIP_AUDIO_PROJECTOR_DOWNSAMPLE_RATE = "clip.audio.projector.downsample_rate";
    public static final String CLIP_AUDIO_PROJECTOR_HEAD_COUNT = "clip.audio.projector.head_count";
    public static final String CLIP_AUDIO_PROJECTOR_STACK_FACTOR = "clip.audio.projector.stack_factor";
    public static final String CLIP_AUDIO_PROJECTOR_TYPE = "clip.audio.projector_type";
    public static final String CLIP_AUDIO_PROJECTOR_WINDOW_SIZE = "clip.audio.projector.window_size";
    public static final String CLIP_AUDIO_RVQ_CODEBOOK_SIZE = "clip.audio.rvq.codebook_size";
    public static final String CLIP_AUDIO_RVQ_NUM_QUANTIZERS = "clip.audio.rvq.num_quantizers";
    public static final String CLIP_AUDIO_SUBSAMPLING_FACTOR = "clip.audio.subsampling_factor";
    public static final String CLIP_AUDIO_WA_PATTERN_MODE = "clip.audio.wa_pattern_mode";
    public static final String CLIP_AUDIO_WINDOW_SIZE = "clip.audio.window_size";
    public static final String CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT = "clip.gen.audio.attention.head_count";
    public static final String CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT_KV = "clip.gen.audio.attention.head_count_kv";
    public static final String CLIP_GEN_AUDIO_ATTENTION_LAYER_NORM_EPSILON =
            "clip.gen.audio.attention.layer_norm_epsilon";
    public static final String CLIP_GEN_AUDIO_BLOCK_COUNT = "clip.gen.audio.block_count";
    public static final String CLIP_GEN_AUDIO_EMBEDDING_LENGTH = "clip.gen.audio.embedding_length";
    public static final String CLIP_GEN_AUDIO_FEED_FORWARD_LENGTH = "clip.gen.audio.feed_forward_length";
    public static final String CLIP_GEN_AUDIO_MODEL_VARIANT = "clip.gen.audio.model_variant";
    public static final String CLIP_GEN_AUDIO_PROJECTION_DIM = "clip.gen.audio.projection_dim";
    public static final String CLIP_GEN_AUDIO_PROJECTOR_TYPE = "clip.gen.audio.projector_type";
    public static final String CLIP_HAS_AUDIO_ENCODER = "clip.has_audio_encoder";
    public static final String CLIP_HAS_GEN_AUDIO_ENCODER = "clip.has_gen_audio_encoder";
    public static final String CLIP_HAS_VISION_ENCODER = "clip.has_vision_encoder";
    public static final String CLIP_PROJECTOR_TYPE = "clip.projector_type";
    public static final String CLIP_USE_GELU = "clip.use_gelu";
    public static final String CLIP_USE_SILU = "clip.use_silu";
    public static final String CLIP_VISION_ATTENTION_HEAD_COUNT = "clip.vision.attention.head_count";
    public static final String CLIP_VISION_ATTENTION_HEAD_COUNT_KV = "clip.vision.attention.head_count_kv";
    public static final String CLIP_VISION_ATTENTION_HEAD_DIM = "clip.vision.attention.head_dim";
    public static final String CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON = "clip.vision.attention.layer_norm_epsilon";
    public static final String CLIP_VISION_BLOCK_COUNT = "clip.vision.block_count";
    public static final String CLIP_VISION_EMBEDDING_LENGTH = "clip.vision.embedding_length";
    public static final String CLIP_VISION_EXPERT_COUNT_PER_LAYER = "clip.vision.expert_count_per_layer";
    public static final String CLIP_VISION_EXPERT_USED_COUNT = "clip.vision.expert_used_count";
    public static final String CLIP_VISION_FEATURE_LAYER = "clip.vision.feature_layer";
    public static final String CLIP_VISION_FEED_FORWARD_LENGTH = "clip.vision.feed_forward_length";
    public static final String CLIP_VISION_IMAGE_GRID_PINPOINTS = "clip.vision.image_grid_pinpoints";
    public static final String CLIP_VISION_IMAGE_MAX_PIXELS = "clip.vision.image_max_pixels";
    public static final String CLIP_VISION_IMAGE_MEAN = "clip.vision.image_mean";
    public static final String CLIP_VISION_IMAGE_MIN_PIXELS = "clip.vision.image_min_pixels";
    public static final String CLIP_VISION_IMAGE_RESIZE_ALGO = "clip.vision.image_resize_algo";
    public static final String CLIP_VISION_IMAGE_SIZE = "clip.vision.image_size";
    public static final String CLIP_VISION_IMAGE_STD = "clip.vision.image_std";
    public static final String CLIP_VISION_IS_DEEPSTACK_LAYERS = "clip.vision.is_deepstack_layers";
    public static final String CLIP_VISION_MAX_SLICE_NUMS = "clip.vision.max_slice_nums";
    public static final String CLIP_VISION_N_WA_PATTERN = "clip.vision.n_wa_pattern";
    public static final String CLIP_VISION_PATCH_SIZE = "clip.vision.patch_size";
    public static final String CLIP_VISION_PREPROC_IMAGE_SIZE = "clip.vision.preproc_image_size";
    public static final String CLIP_VISION_PREPROC_MAX_PIXELS_TOLERANCE = "clip.vision.preproc_max_pixels_tolerance";
    public static final String CLIP_VISION_PREPROC_MAX_TILES = "clip.vision.preproc_max_tiles";
    public static final String CLIP_VISION_PREPROC_MIN_TILES = "clip.vision.preproc_min_tiles";
    public static final String CLIP_VISION_PREPROC_TILE_SIZE = "clip.vision.preproc_tile_size";
    public static final String CLIP_VISION_PROJECTION_DIM = "clip.vision.projection_dim";
    public static final String CLIP_VISION_PROJECTOR_QUERY_SIDE = "clip.vision.projector.query_side";
    public static final String CLIP_VISION_PROJECTOR_SCALE_FACTOR = "clip.vision.projector.scale_factor";
    public static final String CLIP_VISION_PROJECTOR_SPATIAL_OFFSETS = "clip.vision.projector.spatial_offsets";
    public static final String CLIP_VISION_PROJECTOR_TYPE = "clip.vision.projector_type";
    public static final String CLIP_VISION_PROJECTOR_WINDOW_SIDE = "clip.vision.projector.window_side";
    public static final String CLIP_VISION_SAM_BLOCK_COUNT = "clip.vision.sam.block_count";
    public static final String CLIP_VISION_SAM_EMBEDDING_LENGTH = "clip.vision.sam.embedding_length";
    public static final String CLIP_VISION_SAM_HEAD_COUNT = "clip.vision.sam.head_count";
    public static final String CLIP_VISION_SPATIAL_MERGE_SIZE = "clip.vision.spatial_merge_size";
    public static final String CLIP_VISION_SWIGLU_CLAMP = "clip.vision.swiglu_clamp";
    public static final String CLIP_VISION_WA_LAYER_INDEXES = "clip.vision.wa_layer_indexes";
    public static final String CLIP_VISION_WA_PATTERN_MODE = "clip.vision.wa_pattern_mode";
    public static final String CLIP_VISION_WINDOW_SIZE = "clip.vision.window_size";

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

    // NeuralFree extension keys: written by NeuralFree's model sources and GGUF writer; llama.cpp ignores them.

    /**
     * BOOL: true when a tensor uses a NeuralFree extension weight type (a ggml type id of 1000 or more); a file holding
     * such a tensor must set it. Written by the Hugging Face model source and the GGUF writer.
     */
    public static final String NF_EXTENSION_TYPES = "neuralfree.extension_types";
    /** STRING: the container the GGUF view was built from ({@code safetensors}). Written by the Hugging Face source. */
    public static final String NF_SOURCE_FORMAT = "neuralfree.source.format";
    /**
     * STRING: the normalized quantization method of the checkpoint ({@code quantization_config.quant_method} after
     * alias resolution). Written by the Hugging Face model source.
     */
    public static final String NF_SOURCE_QUANT_METHOD = "neuralfree.source.quant_method";
    /**
     * ARRAY of STRING: the tensor-name prefixes (vision / audio towers, side modules) dropped from a Hugging Face
     * checkpoint, as the converter's filters drop them. Written by the Hugging Face model source.
     */
    public static final String NF_HF_DROPPED_PREFIXES = "neuralfree.hf.dropped_prefixes";
    /**
     * UINT32: version of the activation-quantization metadata, 1 when any weight carries an activation scheme. Written
     * by the Hugging Face model source from the checkpoint's quantization config.
     */
    public static final String NF_ACT_QUANT_VERSION = "neuralfree.act_quant.version";
    /**
     * ARRAY of STRING: the GGUF names of the weights whose inputs carry an activation scheme, sorted; parallel to
     * {@link #NF_ACT_QUANT_SCHEMES}. Written by the Hugging Face model source.
     */
    public static final String NF_ACT_QUANT_WEIGHT_NAMES = "neuralfree.act_quant.weight_names";
    /**
     * ARRAY of INT32: the activation scheme code ({@code ActQuantScheme.code()}) of each weight in
     * {@link #NF_ACT_QUANT_WEIGHT_NAMES}. Written by the Hugging Face model source.
     */
    public static final String NF_ACT_QUANT_SCHEMES = "neuralfree.act_quant.schemes";
    /**
     * FLOAT32: the upper bound on {@code max|x|} of the dynamic FP8 per-token activation scheme
     * ({@code ActQuantScheme.FP8_TOKEN}), finite and positive; absent when the checkpoint bounds nothing (fbgemm_fp8
     * {@code activation_scale_ub}). Written by the Hugging Face model source.
     */
    public static final String NF_ACT_QUANT_SCALE_UB = "neuralfree.act_quant.scale_ub";
    /**
     * FLOAT32: default presence penalty from a checkpoint's {@code generation_config.json}. Written by the Hugging Face
     * model source.
     */
    public static final String NF_SAMPLING_PRESENCE_PENALTY = "neuralfree.sampling.presence_penalty";
    /**
     * FLOAT32: default frequency penalty from a checkpoint's {@code generation_config.json}. Written by the Hugging
     * Face model source.
     */
    public static final String NF_SAMPLING_FREQUENCY_PENALTY = "neuralfree.sampling.frequency_penalty";
    /**
     * INT32: default {@code max_new_tokens} from a checkpoint's {@code generation_config.json}. Written by the Hugging
     * Face model source.
     */
    public static final String NF_SAMPLING_MAX_NEW_TOKENS = "neuralfree.sampling.max_new_tokens";
    /**
     * ARRAY of INT32: per-token attribute bits OR-ed into the vocabulary's token attributes. Written by the Hugging
     * Face tokenizer conversion.
     */
    public static final String NF_TOKENIZER_TOKEN_ATTRS = "neuralfree.tokenizer.token_attrs";
    /**
     * STRING: the {@code tokenizer.json} normalizer as canonical JSON. Written by the Hugging Face tokenizer
     * conversion.
     */
    public static final String NF_TOKENIZER_NORMALIZER = "neuralfree.tokenizer.normalizer";
    /**
     * STRING: the {@code tokenizer.json} pre-tokenizer as canonical JSON. Written by the Hugging Face tokenizer
     * conversion.
     */
    public static final String NF_TOKENIZER_PRE_TOKENIZER = "neuralfree.tokenizer.pre_tokenizer";
    /**
     * STRING: the {@code tokenizer.json} decoder as canonical JSON. Written by the Hugging Face tokenizer conversion.
     */
    public static final String NF_TOKENIZER_DECODER = "neuralfree.tokenizer.decoder";
    /**
     * STRING: the Metaspace prepend scheme, {@code always}, {@code first} or {@code never}. Written by the Hugging Face
     * tokenizer conversion.
     */
    public static final String NF_TOKENIZER_PREPEND_SCHEME = "neuralfree.tokenizer.prepend_scheme";
    /**
     * ARRAY of INT32: token ids the post-processor template puts before the input. Written by the Hugging Face
     * tokenizer conversion.
     */
    public static final String NF_TOKENIZER_PREFIX_IDS = "neuralfree.tokenizer.prefix_ids";
    /**
     * ARRAY of INT32: token ids the post-processor template puts after the input. Written by the Hugging Face
     * tokenizer conversion.
     */
    public static final String NF_TOKENIZER_SUFFIX_IDS = "neuralfree.tokenizer.suffix_ids";
    /**
     * ARRAY of INT32: end-of-generation token ids beyond the vocabulary's EOS / EOT / EOM (from
     * {@code generation_config.json}). Written by the Hugging Face tokenizer conversion.
     */
    public static final String NF_TOKENIZER_EXTRA_EOG_IDS = "neuralfree.tokenizer.extra_eog_ids";
    /**
     * BOOL: {@code clean_up_tokenization_spaces} of the tokenizer. Written by the Hugging Face tokenizer conversion.
     */
    public static final String NF_TOKENIZER_CLEAN_SPACES = "neuralfree.tokenizer.clean_spaces";
    /** BOOL: the BPE model's {@code fuse_unk}. Written by the Hugging Face tokenizer conversion. */
    public static final String NF_TOKENIZER_BPE_UNK_FUSE = "neuralfree.tokenizer.bpe.unk_fuse";
    /** BOOL: the BPE model's {@code ignore_merges}. Written by the Hugging Face tokenizer conversion. */
    public static final String NF_TOKENIZER_BPE_IGNORE_MERGES = "neuralfree.tokenizer.bpe.ignore_merges";
    /** BOOL: the BPE model's {@code add_prefix_space}. Written by the Hugging Face tokenizer conversion. */
    public static final String NF_TOKENIZER_BPE_ADD_PREFIX_SPACE = "neuralfree.tokenizer.bpe.add_prefix_space";
    /** BOOL: the BPE model's {@code byte_fallback}. Written by the Hugging Face tokenizer conversion. */
    public static final String NF_TOKENIZER_BPE_BYTE_FALLBACK = "neuralfree.tokenizer.bpe.byte_fallback";
    /** STRING: the BPE model's {@code continuing_subword_prefix}. Written by the Hugging Face tokenizer conversion. */
    public static final String NF_TOKENIZER_BPE_CONTINUING_SUBWORD_PREFIX =
            "neuralfree.tokenizer.bpe.continuing_subword_prefix";
    /** STRING: the BPE model's {@code end_of_word_suffix}. Written by the Hugging Face tokenizer conversion. */
    public static final String NF_TOKENIZER_BPE_END_OF_WORD_SUFFIX = "neuralfree.tokenizer.bpe.end_of_word_suffix";

    // deprecated
    public static final String TOKENIZER_PREFIX_ID = "tokenizer.ggml.prefix_token_id";
    public static final String TOKENIZER_SUFFIX_ID = "tokenizer.ggml.suffix_token_id";
    public static final String TOKENIZER_MIDDLE_ID = "tokenizer.ggml.middle_token_id";

    private GgufKeys() {
    }
}

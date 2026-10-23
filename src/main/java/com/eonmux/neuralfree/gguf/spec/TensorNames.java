package com.eonmux.neuralfree.gguf.spec;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * GGUF tensor base names, transcribed from llama.cpp {@code LLM_TENSOR_NAMES} (src/llama-arch.cpp:484-806),
 * plus the Hugging Face alias table consulted by {@link TensorCatalog}.
 *
 * <p>Model-level constants are full base names ({@code token_embd}); block constants are the suffix after the
 * stack prefix and layer index ({@code blk.N.attn_q} has suffix {@code attn_q}). A stored tensor name is the
 * base or block name followed by {@code .weight}, {@code .bias} or {@code .scale} ({@link #weight},
 * {@link #bias}, {@link #scale}); llama.cpp's {@code hrm.z_l_init} carries no such suffix.
 *
 * <p>Hugging Face-style names ({@code model.layers.N.self_attn.q_proj.weight}, {@code lm_head.weight}, ...)
 * appear in GGUFs written by older converters; {@link #layerAliases(String)} and {@link #modelAliases(String)}
 * list the alternates of a GGUF name in lookup order. The aliases follow the Llama naming, where
 * {@code post_attention_layernorm} is the pre-FFN norm ({@code ffn_norm}); they are consulted only when the GGUF name
 * is absent.
 */
public final class TensorNames {

    /** Main transformer stack prefix ({@code blk.N.}). */
    public static final String STACK_BLK = "blk";
    /** T5 encoder stack prefix ({@code enc.blk.N.}). */
    public static final String STACK_ENC = "enc.blk";
    /** T5 decoder stack prefix ({@code dec.blk.N.}). */
    public static final String STACK_DEC = "dec.blk";
    /** WavTokenizer position-network stack prefix ({@code posnet.N.}). */
    public static final String STACK_POSNET = "posnet";
    /** WavTokenizer ConvNeXt stack prefix ({@code convnext.N.}). */
    public static final String STACK_CONVNEXT = "convnext";
    /** dflash LiLiCorr stack prefix ({@code lilicorr.layers.N.}). */
    public static final String STACK_LILICORR = "lilicorr.layers";
    /** Hugging Face layer prefix used as an alias of {@link #STACK_BLK}. */
    public static final String HF_LAYER_PREFIX = "model.layers";

    /** Suffix of weight tensors. */
    public static final String WEIGHT = "weight";
    /** Suffix of bias tensors. */
    public static final String BIAS = "bias";
    /** Suffix of per-tensor and per-expert scale tensors (src/llama-model.cpp:1726-1789). */
    public static final String SCALE = "scale";

    // ---- model-level tensors (full base name) ----
    public static final String TOKEN_EMBD = "token_embd";
    public static final String OUTPUT_NORM = "output_norm";
    public static final String OUTPUT_NORM_LFM2 = "token_embd_norm";
    public static final String OUTPUT = "output";
    public static final String ROPE_FREQS = "rope_freqs";
    public static final String HRM_Z_L_INIT = "hrm.z_l_init";
    public static final String POS_EMBD = "position_embd";
    public static final String TOKEN_EMBD_NORM = "token_embd_norm";
    public static final String TOKEN_TYPES = "token_types";
    public static final String CLS = "cls";
    public static final String CLS_OUT = "cls.output";
    public static final String CLS_NORM = "cls.norm";
    public static final String DECISION_HIDDEN_NORM = "decision.hidden_norm";
    public static final String DECISION_PROJ_MEMORY = "decision.proj_memory";
    public static final String DECISION_PROJ_QUESTION = "decision.proj_question";
    public static final String DECISION_PROJ_OPTION_QUESTION = "decision.proj_option_question";
    public static final String DECISION_PROJ_GLOBAL = "decision.proj_global";
    public static final String DECISION_PROJ_OPTION_CONTEXT = "decision.proj_option_context";
    public static final String DECISION_PROJ_OPTION_LEXICAL = "decision.proj_option_lexical";
    public static final String DECISION_OPTION_SUMMARY_NORM = "decision.option_summary_norm";
    public static final String DECISION_FIELD_NORM = "decision.field_norm";
    public static final String DECISION_OPTION_NORM = "decision.option_norm";
    public static final String DECISION_SCALES = "decision.scales";
    public static final String DECISION_SCORER = "decision.scorer";
    public static final String DECISION_SCORER_OUT = "decision.scorer_out";
    public static final String DECISION_QUERY_NORM = "decision.query_norm";
    public static final String DECISION_KEY = "decision.key";
    public static final String DECISION_QUERY = "decision.query";
    public static final String DECISION_OPTION_MLP = "decision.option_mlp";
    public static final String DECISION_QUERY_MLP = "decision.query_mlp";
    public static final String ENC_OUTPUT_NORM = "enc.output_norm";
    public static final String ROPE_FACTORS_LONG = "rope_factors_long";
    public static final String ROPE_FACTORS_SHORT = "rope_factors_short";
    public static final String OUTPUT_RES_SCORE = "output_res_score";
    public static final String HC_HEAD_FN = "output_hc_fn";
    public static final String HC_HEAD_BASE = "output_hc_base";
    public static final String HC_HEAD_SCALE = "output_hc_scale";
    public static final String HC_HEAD_NORM = "output_hc_norm";
    public static final String HC_HEAD_DOWN = "output_hc_down";
    public static final String HC_HEAD_UP = "output_hc_up";
    public static final String PER_LAYER_TOKEN_EMBD = "per_layer_token_embd";
    public static final String PER_LAYER_MODEL_PROJ = "per_layer_model_proj";
    public static final String PER_LAYER_PROJ_NORM = "per_layer_proj_norm";
    public static final String ALTUP_UNEMBD_PROJ = "altup_unembd_proj";
    public static final String ALTUP_PROJ = "altup_proj";
    public static final String DENSE_2_OUT = "dense_2";
    public static final String DENSE_3_OUT = "dense_3";
    public static final String NEXTN_PROJ_PRE = "nextn.pre_projection";
    public static final String NEXTN_PROJ_POST = "nextn.post_projection";
    public static final String DEC_OUTPUT_NORM = "dec.output_norm";
    public static final String CONV1D = "conv1d";
    public static final String MASKED_EMBD_CENTROIDS = "masked_embd_centroids";
    public static final String MASKED_EMBD_ORDERING = "masked_embd_ordering";
    public static final String FC = "fc";
    public static final String D2T = "d2t";
    public static final String DSPARK_MARKOV_W1 = "markov_w1";
    public static final String DSPARK_MARKOV_W2 = "markov_w2";
    public static final String DSPARK_CONF_PROJ = "conf_proj";
    public static final String DFLASH_SELECTOR_PREV = "selector_predecessor";
    public static final String DFLASH_SELECTOR_NEXT = "selector_successor";
    public static final String DFLASH_SELECTOR_HIDDEN = "selector_hidden";
    public static final String LILICORR_TOKEN_PROJ = "lilicorr.token_proj";
    public static final String LILICORR_PASS_HIDDEN_PROJ = "lilicorr.pass_hidden_proj";
    public static final String LILICORR_SLOT_EMBD = "lilicorr.slot_embedding";
    public static final String LILICORR_RANK_EMBD = "lilicorr.rank_embedding";
    public static final String LILICORR_ATTN_BIAS = "lilicorr.attention_bias";
    public static final String LILICORR_CONTEXT_PROJ = "lilicorr.context_proj";
    public static final String LILICORR_OUTPUT_NORM = "lilicorr.output_norm";
    public static final String LILICORR_ANCHOR_NORM = "lilicorr.anchor_norm";
    public static final String LILICORR_FACTOR_INPUT_PROJ = "lilicorr.factor_input_proj";
    public static final String LILICORR_OUT_HEAD = "lilicorr.out_head";
    public static final String LILICORR_IN_HEAD = "lilicorr.in_head";
    public static final String LILICORR_ANCHOR_OUT_HEAD = "lilicorr.anchor_out_head";

    // ---- block tensors (suffix after "blk.N.") ----
    public static final String ATTN_NORM = "attn_norm";
    public static final String ATTN_Q = "attn_q";
    public static final String ATTN_K = "attn_k";
    public static final String ATTN_V = "attn_v";
    public static final String ATTN_OUT = "attn_output";
    public static final String ATTN_ROT_EMBD = "attn_rot_embd";
    public static final String FFN_GATE_INP = "ffn_gate_inp";
    public static final String FFN_NORM = "ffn_norm";
    public static final String FFN_GATE = "ffn_gate";
    public static final String FFN_DOWN = "ffn_down";
    public static final String FFN_UP = "ffn_up";
    public static final String FFN_GATE_EXPS = "ffn_gate_exps";
    public static final String FFN_GATE_UP_EXPS = "ffn_gate_up_exps";
    public static final String FFN_DOWN_EXPS = "ffn_down_exps";
    public static final String FFN_UP_EXPS = "ffn_up_exps";
    public static final String ATTN_POST_NORM = "post_attention_norm";
    public static final String ATTN_Q_NORM = "attn_q_norm";
    public static final String ATTN_K_NORM = "attn_k_norm";
    public static final String ATTN_GATE = "attn_gate";
    public static final String ATTN_RESID_ALPHA = "attn_resid_alpha";
    public static final String ATTN_RESID_BETA = "attn_resid_beta";
    public static final String FFN_POST_NORM = "post_ffw_norm";
    public static final String FFN_POST_NORM_1 = "post_ffw_norm_1";
    public static final String FFN_POST_NORM_2 = "post_ffw_norm_2";
    public static final String FFN_NORM_ADA_DOWN = "ffn_norm_ada_down";
    public static final String FFN_NORM_ADA_UP = "ffn_norm_ada_up";
    public static final String FFN_PRE_NORM_2 = "pre_ffw_norm_2";
    public static final String FFN_GATE_SHEXP = "ffn_gate_shexp";
    public static final String FFN_UP_SHEXP = "ffn_up_shexp";
    public static final String FFN_DOWN_SHEXP = "ffn_down_shexp";
    public static final String FFN_EXP_PROBS_B = "exp_probs_b";
    public static final String FFN_EXP_PROBS_B_VL = "exp_probs_b_vl";
    public static final String FFN_LATENT_DOWN = "ffn_latent_down";
    public static final String FFN_LATENT_UP = "ffn_latent_up";
    public static final String FFN_PRE_EXP_NORM = "ffn_pre_exp_norm";
    public static final String FFN_POST_EXP_NORM = "ffn_post_exp_norm";
    public static final String FFN_RESID_ALPHA = "ffn_resid_alpha";
    public static final String FFN_RESID_BETA = "ffn_resid_beta";
    public static final String ATTN_NORM_2 = "attn_norm_2";
    public static final String ATTN_QKV = "attn_qkv";
    public static final String LAYER_OUT_NORM = "layer_output_norm";
    public static final String LAYER_OUT_SCALE = "layer_output_scale";
    public static final String ATTN_OUT_NORM = "attn_output_norm";
    public static final String FFN_ACT = "ffn.act";
    public static final String FFN_GATE_INP_SHEXP = "ffn_gate_inp_shexp";
    public static final String SSM_A_NOSCAN = "ssm_a";
    public static final String SSM_CONV1D = "ssm_conv1d";
    public static final String SSM_DT = "ssm_dt";
    public static final String SSM_BETA_ALPHA = "ssm_ba";
    public static final String SSM_ALPHA = "ssm_alpha";
    public static final String SSM_IN = "ssm_in";
    public static final String SSM_NORM = "ssm_norm";
    public static final String SSM_OUT = "ssm_out";
    public static final String SSM_X = "ssm_x";
    public static final String SSM_A = "ssm_a";
    public static final String SSM_D = "ssm_d";
    public static final String SSM_DT_NORM = "ssm_dt_norm";
    public static final String SSM_B_NORM = "ssm_b_norm";
    public static final String SSM_C_NORM = "ssm_c_norm";
    public static final String SSM_CONV1D_Q = "ssm_conv1d_q";
    public static final String SSM_CONV1D_K = "ssm_conv1d_k";
    public static final String SSM_CONV1D_V = "ssm_conv1d_v";
    public static final String SSM_F_A = "ssm_f_a";
    public static final String SSM_F_B = "ssm_f_b";
    public static final String SSM_BETA = "ssm_beta";
    public static final String SSM_G_A = "ssm_g_a";
    public static final String SSM_G = "ssm_g";
    public static final String ATTN_RES_SCORE = "attn_res_score";
    public static final String FFN_RES_SCORE = "ffn_res_score";
    public static final String FFN_ROUTED_DOWN = "ffn_routed_down";
    public static final String FFN_ROUTED_UP = "ffn_routed_up";
    public static final String FFN_ROUTED_NORM = "ffn_routed_norm";
    public static final String SSM_G_B = "ssm_g_b";
    public static final String ATTN_Q_A_NORM = "attn_q_a_norm";
    public static final String ATTN_KV_A_NORM = "attn_kv_a_norm";
    public static final String ATTN_Q_A = "attn_q_a";
    public static final String ATTN_Q_B = "attn_q_b";
    public static final String ATTN_KV_A_MQA = "attn_kv_a_mqa";
    public static final String ATTN_KV_B = "attn_kv_b";
    public static final String ATTN_KV = "attn_kv";
    public static final String ATTN_KV_NORM = "attn_kv_a_norm";
    public static final String ATTN_OUT_A = "attn_output_a";
    public static final String ATTN_OUT_B = "attn_output_b";
    public static final String HC_ATTN_FN = "hc_attn_fn";
    public static final String HC_ATTN_BASE = "hc_attn_base";
    public static final String HC_ATTN_SCALE = "hc_attn_scale";
    public static final String HC_FFN_FN = "hc_ffn_fn";
    public static final String HC_FFN_BASE = "hc_ffn_base";
    public static final String HC_FFN_SCALE = "hc_ffn_scale";
    public static final String HC_ATTN_NORM = "hc_attn_norm";
    public static final String HC_ATTN_DOWN = "hc_attn_down";
    public static final String HC_ATTN_UP = "hc_attn_up";
    public static final String HC_ATTN_INJECT = "hc_attn_inject";
    public static final String HC_FFN_NORM = "hc_ffn_norm";
    public static final String HC_FFN_DOWN = "hc_ffn_down";
    public static final String HC_FFN_UP = "hc_ffn_up";
    public static final String HC_FFN_INJECT = "hc_ffn_inject";
    public static final String PLE_KEY = "ple_key";
    public static final String PLE_VALUE = "ple_value";
    public static final String PLE_NORM_KEY = "ple_norm_key";
    public static final String PLE_NORM_QUERY = "ple_norm_query";
    public static final String PLE_NORM_CONV = "ple_norm_conv";
    public static final String PLE_CONV1D = "ple_conv1d";
    public static final String ATTN_COMPRESSOR_WKV = "attn_compressor_kv";
    public static final String ATTN_COMPRESSOR_WGATE = "attn_compressor_gate";
    public static final String ATTN_COMPRESSOR_APE = "attn_compressor_ape";
    public static final String ATTN_COMPRESSOR_NORM = "attn_compressor_norm";
    public static final String PER_LAYER_INP_GATE = "inp_gate";
    public static final String PER_LAYER_PROJ = "proj";
    public static final String PER_LAYER_POST_NORM = "post_norm";
    public static final String ALTUP_CORRECT_COEF = "altup_correct_coef";
    public static final String ALTUP_CORRECT_SCALE = "altup_correct_scale";
    public static final String ALTUP_PREDICT_COEF = "altup_predict_coef";
    public static final String ALTUP_ROUTER = "altup_router";
    public static final String ALTUP_ROUTER_NORM = "altup_router_norm";
    public static final String LAUREL_L = "laurel_l";
    public static final String LAUREL_R = "laurel_r";
    public static final String LAUREL_POST_NORM = "laurel_post_norm";
    public static final String FFN_NORM_EXPS = "ffn_norm_exps";
    public static final String ATTN_K_B = "attn_k_b";
    public static final String ATTN_V_B = "attn_v_b";
    public static final String NEXTN_EH_PROJ = "nextn.eh_proj";
    public static final String NEXTN_EMBED_TOKENS = "nextn.embed_tokens";
    public static final String NEXTN_ENORM = "nextn.enorm";
    public static final String NEXTN_HNORM = "nextn.hnorm";
    public static final String NEXTN_SHARED_HEAD_HEAD = "nextn.shared_head_head";
    public static final String NEXTN_SHARED_HEAD_NORM = "nextn.shared_head_norm";
    public static final String NEXTN_HC_HEAD_NORM = "nextn.hc_head_norm";
    public static final String NEXTN_HC_HEAD_DOWN = "nextn.hc_head_down";
    public static final String NEXTN_HC_HEAD_UP = "nextn.hc_head_up";
    public static final String ATTN_SUB_NORM = "attn_sub_norm";
    public static final String FFN_SUB_NORM = "ffn_sub_norm";
    public static final String TIME_MIX_W1 = "time_mix_w1";
    public static final String TIME_MIX_W2 = "time_mix_w2";
    public static final String TIME_MIX_LERP_X = "time_mix_lerp_x";
    public static final String TIME_MIX_LERP_W = "time_mix_lerp_w";
    public static final String TIME_MIX_LERP_K = "time_mix_lerp_k";
    public static final String TIME_MIX_LERP_V = "time_mix_lerp_v";
    public static final String TIME_MIX_LERP_R = "time_mix_lerp_r";
    public static final String TIME_MIX_LERP_G = "time_mix_lerp_g";
    public static final String TIME_MIX_LERP_FUSED = "time_mix_lerp_fused";
    public static final String TIME_MIX_FIRST = "time_mix_first";
    public static final String TIME_MIX_DECAY = "time_mix_decay";
    public static final String TIME_MIX_DECAY_W1 = "time_mix_decay_w1";
    public static final String TIME_MIX_DECAY_W2 = "time_mix_decay_w2";
    public static final String TIME_MIX_KEY = "time_mix_key";
    public static final String TIME_MIX_VALUE = "time_mix_value";
    public static final String TIME_MIX_RECEPTANCE = "time_mix_receptance";
    public static final String TIME_MIX_GATE = "time_mix_gate";
    public static final String TIME_MIX_LN = "time_mix_ln";
    public static final String TIME_MIX_OUTPUT = "time_mix_output";
    public static final String CHANNEL_MIX_LERP_K = "channel_mix_lerp_k";
    public static final String CHANNEL_MIX_LERP_R = "channel_mix_lerp_r";
    public static final String CHANNEL_MIX_KEY = "channel_mix_key";
    public static final String CHANNEL_MIX_VALUE = "channel_mix_value";
    public static final String CHANNEL_MIX_RECEPTANCE = "channel_mix_receptance";
    public static final String TIME_MIX_W0 = "time_mix_w0";
    public static final String TIME_MIX_A0 = "time_mix_a0";
    public static final String TIME_MIX_A1 = "time_mix_a1";
    public static final String TIME_MIX_A2 = "time_mix_a2";
    public static final String TIME_MIX_V0 = "time_mix_v0";
    public static final String TIME_MIX_V1 = "time_mix_v1";
    public static final String TIME_MIX_V2 = "time_mix_v2";
    public static final String TIME_MIX_G1 = "time_mix_g1";
    public static final String TIME_MIX_G2 = "time_mix_g2";
    public static final String TIME_MIX_K_K = "time_mix_k_k";
    public static final String TIME_MIX_K_A = "time_mix_k_a";
    public static final String TIME_MIX_R_K = "time_mix_r_k";
    public static final String ATTN_SINKS = "attn_sinks";
    public static final String SHORTCONV_CONV = "shortconv.conv";
    public static final String SHORTCONV_INPROJ = "shortconv.in_proj";
    public static final String SHORTCONV_OUTPROJ = "shortconv.out_proj";
    public static final String FFN_GATE_CHEXPS = "ffn_gate_chexps";
    public static final String FFN_DOWN_CHEXPS = "ffn_down_chexps";
    public static final String FFN_UP_CHEXPS = "ffn_up_chexps";
    public static final String VISEXP_ATTN_QKV = "vis_attn_qkv";
    public static final String VISEXP_ATTN_OUT = "vis_attn_output";
    public static final String VISEXP_FFN_GATE = "vis_gate";
    public static final String VISEXP_FFN_DOWN = "vis_down";
    public static final String VISEXP_FFN_UP = "vis_up";
    public static final String INDEXER_K_NORM = "indexer.k_norm";
    public static final String INDEXER_PROJ = "indexer.proj";
    public static final String INDEXER_ATTN_K = "indexer.attn_k";
    public static final String INDEXER_ATTN_Q_B = "indexer.attn_q_b";
    public static final String INDEXER_Q_PROJ = "indexer.q_proj";
    public static final String INDEXER_K_PROJ = "indexer.k_proj";
    public static final String INDEXER_Q_NORM = "indexer.q_norm";
    public static final String INDEXER_COMPRESSOR_WKV = "indexer_compressor_kv";
    public static final String INDEXER_COMPRESSOR_WGATE = "indexer_compressor_gate";
    public static final String INDEXER_COMPRESSOR_APE = "indexer_compressor_ape";
    public static final String INDEXER_COMPRESSOR_NORM = "indexer_compressor_norm";
    public static final String INDEXER_KPOOL_GATE = "indexer_compressor_gate";
    public static final String INDEXER_KPOOL_APE = "indexer_compressor_ape";
    public static final String FFN_GATE_TID2EID = "ffn_gate_tid2eid";
    public static final String DFLASH_ATTN_CONV_BASE = "attn_conv_base";
    public static final String DFLASH_ATTN_CONV_PROJ = "attn_conv_proj";
    public static final String DFLASH_FFN_CONV_BASE = "ffn_conv_base";
    public static final String DFLASH_FFN_CONV_PROJ = "ffn_conv_proj";
    public static final String ATTN_V_GATE = "attn_v_gate";
    public static final String ATTN_V_EXPS = "attn_v_exps";

    // ---- legacy per-expert block tensors (suffix after "blk.N."; the trailing %d is the expert) ----
    public static final String FFN_GATE_EXP = "ffn_gate.%d";
    public static final String FFN_DOWN_EXP = "ffn_down.%d";
    public static final String FFN_UP_EXP = "ffn_up.%d";

    // ---- stack "dec.blk" tensors (suffix after "dec.blk.N.") ----
    public static final String DEC_ATTN_NORM = "attn_norm";
    public static final String DEC_ATTN_Q = "attn_q";
    public static final String DEC_ATTN_K = "attn_k";
    public static final String DEC_ATTN_V = "attn_v";
    public static final String DEC_ATTN_OUT = "attn_o";
    public static final String DEC_ATTN_REL_B = "attn_rel_b";
    public static final String DEC_CROSS_ATTN_NORM = "cross_attn_norm";
    public static final String DEC_CROSS_ATTN_NORM_KV = "cross_attn_norm_kv";
    public static final String DEC_CROSS_ATTN_Q = "cross_attn_q";
    public static final String DEC_CROSS_ATTN_K = "cross_attn_k";
    public static final String DEC_CROSS_ATTN_V = "cross_attn_v";
    public static final String DEC_CROSS_ATTN_OUT = "cross_attn_o";
    public static final String DEC_CROSS_ATTN_REL_B = "cross_attn_rel_b";
    public static final String DEC_FFN_NORM = "ffn_norm";
    public static final String DEC_FFN_GATE = "ffn_gate";
    public static final String DEC_FFN_DOWN = "ffn_down";
    public static final String DEC_FFN_UP = "ffn_up";

    // ---- stack "enc.blk" tensors (suffix after "enc.blk.N.") ----
    public static final String ENC_ATTN_NORM = "attn_norm";
    public static final String ENC_ATTN_Q = "attn_q";
    public static final String ENC_ATTN_K = "attn_k";
    public static final String ENC_ATTN_V = "attn_v";
    public static final String ENC_ATTN_OUT = "attn_o";
    public static final String ENC_ATTN_REL_B = "attn_rel_b";
    public static final String ENC_FFN_NORM = "ffn_norm";
    public static final String ENC_FFN_GATE = "ffn_gate";
    public static final String ENC_FFN_DOWN = "ffn_down";
    public static final String ENC_FFN_UP = "ffn_up";

    // ---- stack "convnext" tensors (suffix after "convnext.N.") ----
    public static final String CONVNEXT_DW = "dw";
    public static final String CONVNEXT_NORM = "norm";
    public static final String CONVNEXT_PW1 = "pw1";
    public static final String CONVNEXT_PW2 = "pw2";
    public static final String CONVNEXT_GAMMA = "gamma";

    // ---- stack "posnet" tensors (suffix after "posnet.N.") ----
    public static final String POS_NET_CONV1 = "conv1";
    public static final String POS_NET_CONV2 = "conv2";
    public static final String POS_NET_NORM = "norm";
    public static final String POS_NET_NORM1 = "norm1";
    public static final String POS_NET_NORM2 = "norm2";
    public static final String POS_NET_ATTN_NORM = "attn_norm";
    public static final String POS_NET_ATTN_Q = "attn_q";
    public static final String POS_NET_ATTN_K = "attn_k";
    public static final String POS_NET_ATTN_V = "attn_v";
    public static final String POS_NET_ATTN_OUT = "attn_output";

    // ---- stack "lilicorr.layers" tensors (suffix after "lilicorr.layers.N.") ----
    public static final String LILICORR_ATTN_IN_PROJ = "attn.in_proj";

    private static final String[][] LAYER_ALIAS_BASES = {
            {ATTN_Q, "self_attn.q_proj"},
            {ATTN_K, "self_attn.k_proj"},
            {ATTN_V, "self_attn.v_proj"},
            {ATTN_OUT, "self_attn.o_proj"},
            {ATTN_Q_NORM, "self_attn.q_norm"},
            {ATTN_K_NORM, "self_attn.k_norm"},
            {FFN_GATE, "mlp.gate_proj"},
            {FFN_UP, "mlp.up_proj"},
            {FFN_DOWN, "mlp.down_proj"},
            {ATTN_NORM, "input_layernorm"},
            {FFN_NORM, "post_attention_layernorm"},
    };

    private static final String[][] MODEL_ALIASES = {
            {TOKEN_EMBD + "." + WEIGHT, "model.embed_tokens.weight", "tok_embeddings.weight"},
            {OUTPUT + "." + WEIGHT, "lm_head.weight"},
            {OUTPUT + "." + BIAS, "lm_head.bias"},
            {OUTPUT_NORM + "." + WEIGHT, "model.norm.weight"},
            {OUTPUT_NORM + "." + BIAS, "model.norm.bias"},
    };

    private static final Map<String, List<String>> LAYER_ALIASES = buildLayerAliases();
    private static final Map<String, List<String>> MODEL_ALIAS_MAP = buildModelAliases();

    private TensorNames() {
    }

    /** {@code base.weight}. */
    public static String weight(String base) {
        return base + "." + WEIGHT;
    }

    /** {@code base.bias}. */
    public static String bias(String base) {
        return base + "." + BIAS;
    }

    /** {@code base.scale}. */
    public static String scale(String base) {
        return base + "." + SCALE;
    }

    /** {@code blk.<il>.<suffix>}. */
    public static String layerName(int il, String suffix) {
        return layerName(STACK_BLK, il, suffix);
    }

    /**
     * {@code <stack>.<il>.<suffix>}.
     *
     * @throws IllegalArgumentException for a negative layer index or an empty stack or suffix
     */
    public static String layerName(String stack, int il, String suffix) {
        if (stack == null || stack.isEmpty()) {
            throw new IllegalArgumentException("tensor stack prefix is empty");
        }
        if (suffix == null || suffix.isEmpty()) {
            throw new IllegalArgumentException("tensor suffix is empty");
        }
        if (il < 0) {
            throw new IllegalArgumentException("layer index " + il + " is negative for tensor " + stack + ".?." + suffix);
        }
        return stack + "." + il + "." + suffix;
    }

    /**
     * Hugging Face alternates of a {@code blk.N.} suffix such as {@code attn_q.weight}, as suffixes under
     * {@link #HF_LAYER_PREFIX}; empty when the suffix has none.
     */
    public static List<String> layerAliases(String suffix) {
        List<String> a = LAYER_ALIASES.get(suffix);
        return a != null ? a : Collections.<String>emptyList();
    }

    /** Alternates of a model-level tensor name such as {@code output.weight}; empty when it has none. */
    public static List<String> modelAliases(String name) {
        List<String> a = MODEL_ALIAS_MAP.get(name);
        return a != null ? a : Collections.<String>emptyList();
    }

    private static Map<String, List<String>> buildLayerAliases() {
        Map<String, List<String>> m = new HashMap<>();
        for (String[] pair : LAYER_ALIAS_BASES) {
            m.put(weight(pair[0]), Collections.singletonList(weight(pair[1])));
            m.put(bias(pair[0]), Collections.singletonList(bias(pair[1])));
        }
        return Collections.unmodifiableMap(m);
    }

    private static Map<String, List<String>> buildModelAliases() {
        Map<String, List<String>> m = new HashMap<>();
        for (String[] row : MODEL_ALIASES) {
            m.put(row[0], Collections.unmodifiableList(Arrays.asList(Arrays.copyOfRange(row, 1, row.length))));
        }
        return Collections.unmodifiableMap(m);
    }
}

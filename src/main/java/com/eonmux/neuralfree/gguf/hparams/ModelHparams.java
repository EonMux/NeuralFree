package com.eonmux.neuralfree.gguf.hparams;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GdnHeadMap;
import com.eonmux.neuralfree.gguf.arch.KdaGateMode;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.RopeScalingType;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SwaType;

/**
 * Model hyperparameters resolved from GGUF metadata: the NeuralFree counterpart of llama.cpp
 * {@code llama_hparams} (src/llama-hparams.h) after {@code load_hparams} and the architecture's
 * {@code load_arch_hparams}. Built by {@link HparamsLoader} through {@link ModelHparamsBuilder}.
 *
 * <p>Immutable and thread-safe; array results are copies. Per-layer values cover all
 * {@link #nLayerAll()} blocks; the trunk is {@code [0, nLayer())} and NextN/MTP blocks follow it. Derived
 * accessors mirror the llama.cpp getters of the same meaning.
 */
public final class ModelHparams {

    /** Maximum block count ({@code LLAMA_MAX_LAYERS}). */
    public static final int MAX_LAYERS = 512;
    /** Maximum expert count ({@code LLAMA_MAX_EXPERTS}). */
    public static final int MAX_EXPERTS = 1024;
    /** Multi-section RoPE positions per token ({@code GGML_MROPE_SECTIONS}). */
    public static final int MROPE_SECTIONS = 4;

    /** {@link #ropeSectionOrder()}: slot i feeds section i ({@code tyxz}). */
    public static final int SECTION_ORDER_TYXZ = 0;
    /** {@link #ropeSectionOrder()}: time last ({@code zyxt}). */
    public static final int SECTION_ORDER_ZYXT = 1;

    /** {@link #nonCausalType()}: all layers non-causal, SWA still applied. */
    public static final int NON_CAUSAL_ALL = 0;
    /** {@link #nonCausalType()}: SWA layers non-causal, dense layers stay causal. */
    public static final int NON_CAUSAL_SWA_ONLY = 1;
    /** {@link #nonCausalType()}: all layers non-causal, SWA not applied within the current ubatch. */
    public static final int NON_CAUSAL_SWA_FULL = 2;

    private final ModelHparamsBuilder d;
    private final int nLayerAll;
    private final int nextnLayers;
    private final boolean[] moe;
    private final int[] kvOwner;

    ModelHparams(ModelHparamsBuilder snapshot, boolean[] moe, int[] kvOwner) {
        this(snapshot, moe, kvOwner, snapshot.nextnLayers());
    }

    private ModelHparams(ModelHparamsBuilder snapshot, boolean[] moe, int[] kvOwner, int nextnLayers) {
        this.d = snapshot;
        this.nLayerAll = snapshot.nLayerAll();
        this.nextnLayers = nextnLayers;
        this.moe = moe;
        this.kvOwner = kvOwner;
    }

    public Arch arch() {
        return d.arch();
    }

    public ArchTraits traits() {
        return d.traits();
    }

    // ---- layer structure ----

    /** All blocks including NextN/MTP blocks ({@code n_layer_all}). */
    public int nLayerAll() {
        return nLayerAll;
    }

    /** NextN/MTP blocks after the trunk ({@code n_layer_nextn}). */
    public int nextnLayers() {
        return nextnLayers;
    }

    /** Trunk blocks, {@code nLayerAll - nextnLayers} (llama-hparams.cpp:347-349). */
    public int nLayer() {
        return nLayerAll - nextnLayers;
    }

    /**
     * These hyperparameters with every block counted as a trunk block ({@link #nextnLayers()} 0, {@link #nLayer()}
     * equal to {@link #nLayerAll()}). Most llama.cpp loaders of MTP architectures create a NextN block with the
     * body of a trunk block (models/glm4-moe.cpp:49-53, models/hy-v3.cpp:74-77), so the trunk tensor conditions
     * evaluated on this view name the decoder tensors of a NextN block.
     */
    public ModelHparams withNextnAsTrunk() {
        return nextnLayers == 0 ? this : new ModelHparams(d, moe, kvOwner, 0);
    }

    /** True when layer {@code il} has a routed-expert FFN. */
    public boolean isMoeLayer(int il) {
        return moe[layer(il)];
    }

    /** Copy of the per-layer {@link #isMoeLayer(int)} values. */
    public boolean[] moeArray() {
        return moe.clone();
    }

    /** Layer whose KV cache layer {@code il} attends to; {@code il} itself when it owns its cache. */
    public int kvOwnerLayer(int il) {
        return kvOwner[layer(il)];
    }

    /** Copy of the per-layer {@link #kvOwnerLayer(int)} values. */
    public int[] kvOwnerLayerArray() {
        return kvOwner.clone();
    }

    // ---- identity ----

    /** {@code general.name}, or null. */
    public String name() { return d.name; }
    /** {@code general.file_type} (llama_ftype), or -1 when absent. */
    public int fileType() { return d.fileType; }

    // ---- sizes ----

    /** Vocabulary size: {@code vocab_size}, else the tokenizer token count. */
    public int nVocab() { return d.nVocab; }
    /** Training context length ({@code n_ctx_train}). */
    public int nCtxTrain() { return d.nCtxTrain; }
    /** Hidden size ({@code n_embd}). */
    public int nEmbd() { return d.nEmbd; }
    /** Input embedding width override, 0 = derived ({@code n_embd_inp_impl}). */
    public int nEmbdInpImpl() { return d.nEmbdInpImpl; }
    /** Encoder input width override, 0 = {@link #nEmbdInp()} ({@code n_embd_inp_enc_impl}). */
    public int nEmbdInpEncImpl() { return d.nEmbdInpEncImpl; }
    /** Output embedding width override, 0 = n_embd ({@code n_embd_out_impl}). */
    public int nEmbdOutImpl() { return d.nEmbdOutImpl; }
    /** Per-layer embedding input width ({@code n_embd_per_layer}). */
    public int nEmbdPerLayer() { return d.nEmbdPerLayer; }
    /** Trailing blocks forming the decision head ({@code n_layer_decision}). */
    public int nLayerDecision() { return d.nLayerDecision; }
    /** Single-head router KV layer index, -1 when absent ({@code router_layer}). */
    public int routerLayer() { return d.routerLayer; }
    /** WavTokenizer posnet width ({@code posnet.n_embd}). */
    public int posnetEmbd() { return d.posnetEmbd; }
    /** WavTokenizer posnet block count ({@code posnet.n_layer}). */
    public int posnetLayers() { return d.posnetLayers; }
    /** WavTokenizer convnext width ({@code convnext.n_embd}). */
    public int convnextEmbd() { return d.convnextEmbd; }
    /** WavTokenizer convnext block count ({@code convnext.n_layer}). */
    public int convnextLayers() { return d.convnextLayers; }
    /** Deepstack embedding count ({@code n_deepstack_layers}). */
    public int nDeepstackLayers() { return d.nDeepstackLayers; }
    /** Realtime output lag in tokens ({@code n_tokens_delay}). */
    public int nTokensDelay() { return d.nTokensDelay; }

    // ---- head dimensions ----

    /** Key head size of full-attention layers ({@code n_embd_head_k_full}). */
    public int headKFull() { return d.headKFull; }
    /** Value head size of full-attention layers ({@code n_embd_head_v_full}). */
    public int headVFull() { return d.headVFull; }
    /** Key head size of SWA layers ({@code n_embd_head_k_swa}). */
    public int headKSwa() { return d.headKSwa; }
    /** Value head size of SWA layers ({@code n_embd_head_v_swa}). */
    public int headVSwa() { return d.headVSwa; }
    /** Rotary dimensions of full-attention layers ({@code n_rot_full}). */
    public int nRotFull() { return d.nRotFull; }
    /** Rotary dimensions of SWA layers ({@code n_rot_swa}). */
    public int nRotSwa() { return d.nRotSwa; }
    /** MLA key head size, 0 = not MLA ({@code n_embd_head_k_mla_impl}). */
    public int headKMlaImpl() { return d.headKMlaImpl; }
    /** MLA value head size, 0 = not MLA ({@code n_embd_head_v_mla_impl}). */
    public int headVMlaImpl() { return d.headVMlaImpl; }
    /** MLA key head size of SWA layers ({@code n_embd_head_k_mla_swa}). */
    public int headKMlaSwa() { return d.headKMlaSwa; }
    /** MLA value head size of SWA layers ({@code n_embd_head_v_mla_swa}). */
    public int headVMlaSwa() { return d.headVMlaSwa; }
    /** MLA query LoRA rank ({@code n_lora_q}). */
    public int nLoraQ() { return d.nLoraQ; }
    /** MLA KV LoRA rank ({@code n_lora_kv}). */
    public int nLoraKv() { return d.nLoraKv; }
    /** MLA KV LoRA rank of SWA layers ({@code n_lora_kv_swa}). */
    public int nLoraKvSwa() { return d.nLoraKvSwa; }

    // ---- norms ----

    /** LayerNorm epsilon ({@code f_norm_eps}). */
    public float normEps() { return d.normEps; }
    /** RMSNorm epsilon ({@code f_norm_rms_eps}). */
    public float normRmsEps() { return d.normRmsEps; }
    /** GroupNorm epsilon ({@code f_norm_group_eps}). */
    public float normGroupEps() { return d.normGroupEps; }
    /** Post-norm epsilon, 0 = same as the block norm ({@code f_post_norm_eps}). */
    public float postNormEps() { return d.postNormEps; }
    /** GroupNorm group count ({@code n_norm_groups}). */
    public int nNormGroups() { return d.nNormGroups; }
    /** Swin-style norm placement ({@code swin_norm}). */
    public boolean swinNorm() { return d.swinNorm; }
    /** {@code norm_before_residual}. */
    public boolean normBeforeResidual() { return d.normBeforeResidual; }
    /** {@code norm_before_fc}. */
    public boolean normBeforeFc() { return d.normBeforeFc; }
    /** Parallel attention/FFN residual ({@code use_par_res}). */
    public boolean useParallelResidual() { return d.useParallelResidual; }

    // ---- attention ----

    /** Causal attention ({@code causal_attn}). */
    public boolean causal() { return d.causal; }
    /** Mask built when causality is switched off ({@code non_causal_type}). */
    public int nonCausalType() { return d.nonCausalType; }
    /** Explicit KQ scale, 0 = 1/sqrt(head_k) ({@code f_attention_scale}). */
    public float attentionScale() { return d.attentionScale; }
    /** Attention output scale ({@code f_attn_out_scale}). */
    public float attnOutScale() { return d.attnOutScale; }
    /** Attention value scale ({@code f_attn_value_scale}). */
    public float attnValueScale() { return d.attnValueScale; }
    /** Attention logit soft-capping enabled ({@code attn_soft_cap}). */
    public boolean attnSoftcapEnabled() { return d.attnSoftcapEnabled; }
    /** Attention logit soft-cap ({@code f_attn_logit_softcapping}). */
    public float attnSoftcap() { return d.attnSoftcap; }
    /** Router logit soft-cap ({@code f_router_logit_softcapping}). */
    public float routerSoftcap() { return d.routerSoftcap; }
    /** Final logit soft-cap ({@code f_final_logit_softcapping}). */
    public float finalSoftcap() { return d.finalSoftcap; }
    /** QKV clamp, 0 = none ({@code f_clamp_kqv}). */
    public float clampKqv() { return d.clampKqv; }
    /** ALiBi maximum bias ({@code f_max_alibi_bias}). */
    public float maxAlibiBias() { return d.maxAlibiBias; }
    /** ALiBi enabled ({@code use_alibi}). */
    public boolean useAlibi() { return d.useAlibi; }
    /** T5 relative attention buckets ({@code n_rel_attn_bkts}). */
    public int nRelAttnBuckets() { return d.nRelAttnBuckets; }
    /** Attention temperature length ({@code attn_temp_length}). */
    public int attnTempLength() { return d.attnTempLength; }
    /** Attention temperature floor scale ({@code n_attn_temp_floor_scale}). */
    public int attnTempFloorScale() { return d.attnTempFloorScale; }
    /** Attention temperature scale ({@code f_attn_temp_scale}). */
    public float attnTempScale() { return d.attnTempScale; }
    /** Attention temperature position offset ({@code f_attn_temp_offset}). */
    public float attnTempOffset() { return d.attnTempOffset; }
    /** Power-form attention temperature ({@code attn_temp_pow}). */
    public boolean attnTempPow() { return d.attnTempPow; }
    /** Every n-th layer is NoPE ({@code n_no_rope_layer_step}). */
    public int noRopeLayerStep() { return d.noRopeLayerStep; }
    /** Llama4 L2 QK norm ({@code use_kq_norm}). */
    public boolean useKqNorm() { return d.useKqNorm; }
    /** First n layers own a KV cache, -1 = all ({@code n_layer_kv_from_start}). */
    public int nLayerKvFromStart() { return d.nLayerKvFromStart; }
    /** Trailing layers reusing earlier KV ({@code attention.shared_kv_layers}). */
    public int sharedKvLayers() { return d.sharedKvLayers; }
    /** MoVA value experts ({@code n_value_expert}). */
    public int nValueExpert() { return d.nValueExpert; }
    /** MoVA value experts used ({@code n_value_expert_used}). */
    public int nValueExpertUsed() { return d.nValueExpertUsed; }

    // ---- sliding window ----

    /** Sliding-window mask type ({@code swa_type}). */
    public SwaType swaType() { return d.swaType; }
    /** Sliding window size, 0 = none ({@code n_swa}). */
    public int nSwa() { return d.nSwa; }

    // ---- rope ----

    /** Resolved RoPE type ({@code rope_type}). */
    public RopeType ropeType() { return d.ropeType; }
    /** RoPE base of full-attention layers ({@code rope_freq_base_train}). */
    public float ropeFreqBase() { return d.ropeFreqBase; }
    /** RoPE base of SWA layers ({@code rope_freq_base_train_swa}). */
    public float ropeFreqBaseSwa() { return d.ropeFreqBaseSwa; }
    /** Spatial RoPE base, 0 = none ({@code rope.freq_base_spatial}). */
    public float ropeFreqBaseSpatial() { return d.ropeFreqBaseSpatial; }
    /** RoPE frequency scale, 1/factor ({@code rope_freq_scale_train}). */
    public float ropeFreqScale() { return d.ropeFreqScale; }
    /** RoPE frequency scale of SWA layers ({@code rope_freq_scale_train_swa}). */
    public float ropeFreqScaleSwa() { return d.ropeFreqScaleSwa; }
    /** RoPE scaling type ({@code rope_scaling_type_train}). */
    public RopeScalingType ropeScalingType() { return d.ropeScalingType; }
    /** NTK alpha ({@code rope_scaling_alpha}). */
    public float ropeScalingAlpha() { return d.ropeScalingAlpha; }
    /** RoPE attention factor ({@code rope_attn_factor}). */
    public float ropeAttnFactor() { return d.ropeAttnFactor; }
    /** Original context of YaRN scaling ({@code n_ctx_orig_yarn}). */
    public int nCtxOrigYarn() { return d.nCtxOrigYarn; }
    /** {@code rope.scaling.finetuned}. */
    public boolean ropeFinetuned() { return d.ropeFinetuned; }
    /** YaRN log multiplier ({@code rope_yarn_log_mul}). */
    public float yarnLogMul() { return d.yarnLogMul; }
    /** YaRN extrapolation factor, negative = derived ({@code yarn_ext_factor}). */
    public float yarnExtFactor() { return d.yarnExtFactor; }
    /** YaRN attention factor ({@code yarn_attn_factor}). */
    public float yarnAttnFactor() { return d.yarnAttnFactor; }
    /** YaRN beta fast ({@code yarn_beta_fast}). */
    public float yarnBetaFast() { return d.yarnBetaFast; }
    /** YaRN beta slow ({@code yarn_beta_slow}). */
    public float yarnBetaSlow() { return d.yarnBetaSlow; }
    /** Multi-section RoPE dimensions, four entries ({@code rope_sections}). */
    public int[] ropeSections() { return d.ropeSections.clone(); }
    /** Position slot feeding each section ({@code rope_section_order}). */
    public int ropeSectionOrder() { return d.ropeSectionOrder; }
    /** Text-token positions of multi-section RoPE. */
    public MropeTextPositions mropeTextPositions() { return d.mropeTextPositions; }

    // ---- recurrent and linear attention ----

    /** SSM convolution kernel ({@code ssm_d_conv}). */
    public int ssmDConv() { return d.ssmDConv; }
    /** SSM inner size ({@code ssm_d_inner}). */
    public int ssmDInner() { return d.ssmDInner; }
    /** SSM state size ({@code ssm_d_state}). */
    public int ssmDState() { return d.ssmDState; }
    /** SSM time-step rank ({@code ssm_dt_rank}). */
    public int ssmDtRank() { return d.ssmDtRank; }
    /** SSM group count ({@code ssm_n_group}). */
    public int ssmNGroup() { return d.ssmNGroup; }
    /** RMS-normalize dt, B and C ({@code ssm_dt_b_c_rms}). */
    public boolean ssmDtBcRms() { return d.ssmDtBcRms; }
    /** Gated delta-net head mapping, null when not GDN. */
    public GdnHeadMap gdnHeadMap() { return d.gdnHeadMap; }
    /** MiniMax linear attention head size ({@code n_embd_head_la}). */
    public int headLinearAttn() { return d.headLinearAttn; }
    /** KDA head size ({@code n_embd_head_kda}). */
    public int kdaHeadDim() { return d.kdaHeadDim; }
    /** KDA safe gate ({@code kda_safe_gate}). */
    public boolean kdaSafeGate() { return d.kdaSafeGate; }
    /** KDA gate lower bound ({@code kda_gate_lower_bound}). */
    public float kdaGateLowerBound() { return d.kdaGateLowerBound; }
    /** KDA gate parameterisation, null when not KDA. */
    public KdaGateMode kdaGateMode() { return d.kdaGateMode; }
    /** RWKV WKV head size ({@code wkv_head_size}). */
    public int wkvHeadSize() { return d.wkvHeadSize; }
    /** RWKV token shift count ({@code token_shift_count}). */
    public int tokenShiftCount() { return d.tokenShiftCount; }
    /** RWKV rescale period ({@code rescale_every_n_layers}). */
    public int rescaleEveryN() { return d.rescaleEveryN; }
    /** RWKV time-mix extra dim ({@code time_mix_extra_dim}). */
    public int timeMixExtraDim() { return d.timeMixExtraDim; }
    /** RWKV time-decay extra dim ({@code time_decay_extra_dim}). */
    public int timeDecayExtraDim() { return d.timeDecayExtraDim; }
    /** RWKV7 decay LoRA rank ({@code n_lora_decay}). */
    public int nLoraDecay() { return d.nLoraDecay; }
    /** RWKV7 ICLR LoRA rank ({@code n_lora_iclr}). */
    public int nLoraIclr() { return d.nLoraIclr; }
    /** RWKV7 value residual mix LoRA rank ({@code n_lora_value_res_mix}). */
    public int nLoraValueResMix() { return d.nLoraValueResMix; }
    /** RWKV7 gate LoRA rank ({@code n_lora_gate}). */
    public int nLoraGate() { return d.nLoraGate; }
    /** LFM2 short convolution cache ({@code n_shortconv_l_cache}). */
    public int shortconvLCache() { return d.shortconvLCache; }
    /** Hybrid full-attention period ({@code full_attention_interval}), 0 when absent. */
    public int fullAttentionInterval() { return d.fullAttentionInterval; }
    /** Cross-layer attention residual block size ({@code attn_res_block_size}). */
    public int attnResBlockSize() { return d.attnResBlockSize; }

    // ---- mixture of experts ----

    /** Expert count ({@code n_expert}). */
    public int nExpert() { return d.nExpert; }
    /** Shared expert count ({@code n_expert_shared}). */
    public int nExpertShared() { return d.nExpertShared; }
    /** Expert group count ({@code n_expert_groups}). */
    public int nExpertGroups() { return d.nExpertGroups; }
    /** Groups used per token ({@code n_group_used}). */
    public int nGroupUsed() { return d.nGroupUsed; }
    /** Experts per group ({@code n_group_experts}). */
    public int nGroupExperts() { return d.nGroupExperts; }
    /** Shared expert FFN size ({@code n_ff_shexp}). */
    public int nFfShexp() { return d.nFfShexp; }
    /** Chunked expert FFN size ({@code n_ff_chexp}). */
    public int nFfChexp() { return d.nFfChexp; }
    /** Leading dense layers ({@code n_layer_dense_lead}). */
    public int nLayerDenseLead() { return d.nLayerDenseLead; }
    /** MoE interleave step ({@code n_moe_layer_step}). */
    public int nMoeLayerStep() { return d.nMoeLayerStep; }
    /** MoE every n layers ({@code moe_every_n_layers}). */
    public int moeEveryN() { return d.moeEveryN; }
    /** MoE latent size ({@code moe_latent_size}). */
    public int moeLatentSize() { return d.moeLatentSize; }
    /** Routed expert hidden size, 0 = n_embd ({@code n_expert_latent}). */
    public int nExpertLatent() { return d.nExpertLatent; }
    /** Expert group scale ({@code expert_group_scale}). */
    public float expertGroupScale() { return d.expertGroupScale; }
    /** Routed weight scale, 0 = none ({@code expert_weights_scale}). */
    public float expertWeightsScale() { return d.expertWeightsScale; }
    /** Normalize routed weights ({@code expert_weights_norm}). */
    public boolean expertWeightsNorm() { return d.expertWeightsNorm; }
    /** Router gating, null for dense models ({@code expert_gating_func}). */
    public GatingFunc gating() { return d.gating; }

    // ---- sparse attention, compressed attention and hyper-connections ----

    /** Indexer head count ({@code indexer_n_head}). */
    public int indexerNHead() { return d.indexerNHead; }
    /** Indexer head size ({@code indexer_head_size}). */
    public int indexerHeadSize() { return d.indexerHeadSize; }
    /** Indexer top-k ({@code indexer_top_k}). */
    public int indexerTopK() { return d.indexerTopK; }
    /** Indexer k-pool size ({@code indexer_kpool}). */
    public int indexerKpool() { return d.indexerKpool; }
    /** {@code indexer_kpool_select_tail}. */
    public boolean indexerKpoolSelectTail() { return d.indexerKpoolSelectTail; }
    /** Head-size slots per cached indexer row ({@code indexer_kpool_row}). */
    public int indexerKpoolRow() { return d.indexerKpoolRow; }
    /** {@code indexer_kpool_by_order}. */
    public boolean indexerKpoolByOrder() { return d.indexerKpoolByOrder; }
    /** MSA block size ({@code indexer_block_size}). */
    public int indexerBlockSize() { return d.indexerBlockSize; }
    /** MSA local blocks ({@code indexer_local_blocks}). */
    public int indexerLocalBlocks() { return d.indexerLocalBlocks; }
    /** Output group count ({@code dsv4_o_group_count}). */
    public int dsv4OGroupCount() { return d.dsv4OGroupCount; }
    /** Output LoRA rank ({@code dsv4_o_lora_rank}). */
    public int dsv4OLoraRank() { return d.dsv4OLoraRank; }
    /** Hyper-connection streams ({@code dsv4_hc_mult}). */
    public int dsv4HcMult() { return d.dsv4HcMult; }
    /** Sinkhorn iterations ({@code dsv4_hc_sinkhorn_iters}). */
    public int dsv4HcSinkhornIters() { return d.dsv4HcSinkhornIters; }
    /** Hash-routed layer count ({@code dsv4_hash_layer_count}). */
    public int dsv4HashLayerCount() { return d.dsv4HashLayerCount; }
    /** Compressed attention RoPE base ({@code dsv4_compress_rope_base}). */
    public float dsv4CompressRopeBase() { return d.dsv4CompressRopeBase; }
    /** Hyper-connection epsilon ({@code dsv4_hc_eps}). */
    public float dsv4HcEps() { return d.dsv4HcEps; }
    /** Hyper-connection rank, 0 = full ({@code hc_low_rank}). */
    public int hcLowRank() { return d.hcLowRank; }
    /** Hyper-connection post-gate scale ({@code hc_magnitude}). */
    public float hcMagnitude() { return d.hcMagnitude; }

    // ---- per-layer embeddings and altup ----

    /** {@code ple_ngram_size}. */
    public int pleNgramSize() { return d.pleNgramSize; }
    /** {@code ple_heads_per_ngram}. */
    public int pleHeadsPerNgram() { return d.pleHeadsPerNgram; }
    /** {@code ple_conv_kernel}. */
    public int pleConvKernel() { return d.pleConvKernel; }
    /** {@code ple_n_heads}. */
    public int pleNHeads() { return d.pleNHeads; }
    /** {@code ple_head_dim}. */
    public int pleHeadDim() { return d.pleHeadDim; }
    /** {@code ple_eos_token_id}. */
    public int pleEosTokenId() { return d.pleEosTokenId; }
    /** {@code ple_image_token_id}. */
    public int pleImageTokenId() { return d.pleImageTokenId; }
    /** N-gram hash multipliers ({@code ple_layer_multipliers}). */
    public long[] pleLayerMultipliers() { return d.pleLayerMultipliers.clone(); }
    /** Token-space head offsets ({@code ple_head_offsets}). */
    public long[] pleHeadOffsets() { return d.pleHeadOffsets.clone(); }
    /** Per-head vocab sizes ({@code ple_head_vocab_sizes}). */
    public long[] pleHeadVocabSizes() { return d.pleHeadVocabSizes.clone(); }
    /** Altup inputs ({@code n_altup}). */
    public int nAltup() { return d.nAltup; }
    /** Altup active index ({@code i_altup_act}). */
    public int iAltupAct() { return d.iAltupAct; }
    /** Laurel rank ({@code laurel_rank}). */
    public int laurelRank() { return d.laurelRank; }
    /** Altup width ({@code n_embd_altup}). */
    public int nEmbdAltup() { return d.nEmbdAltup; }

    // ---- scaling ----

    /** Logit scale, 0 = none ({@code f_logit_scale}). */
    public float logitScale() { return d.logitScale; }
    /** True when logits are divided by {@link #logitScale()} instead of multiplied. */
    public boolean logitScaleDivides() { return d.logitScaleDivides; }
    /** Residual scale, 0 = none ({@code f_residual_scale}). */
    public float residualScale() { return d.residualScale; }
    /** Embedding scale, 0 = none ({@code f_embedding_scale}). */
    public float embeddingScale() { return d.embeddingScale; }
    /** SiTU beta ({@code situ_beta}). */
    public float situBeta() { return d.situBeta; }
    /** SiTU linear beta, 0 = none ({@code situ_linear_beta}). */
    public float situLinearBeta() { return d.situLinearBeta; }

    // ---- control flow ----

    /** Looped trunk repetitions ({@code num_loops}). */
    public int numLoops() { return d.numLoops; }
    /** {@code skip_loop_final_norm}. */
    public boolean skipLoopFinalNorm() { return d.skipLoopFinalNorm; }
    /** {@code n_hrm_layers_per_stack}. */
    public int hrmLayersPerStack() { return d.hrmLayersPerStack; }
    /** {@code n_hrm_h_cycles}. */
    public int hrmHCycles() { return d.hrmHCycles; }
    /** {@code n_hrm_l_cycles}. */
    public int hrmLCycles() { return d.hrmLCycles; }
    /** {@code hrm_prefix_lm}. */
    public boolean hrmPrefixLm() { return d.hrmPrefixLm; }

    // ---- heads, encoders and pooling ----

    /** Embedding pooling ({@code pooling_type}). */
    public PoolingKind pooling() { return d.pooling; }
    /** Pooling before the classifier head, null = unspecified ({@code pooling_type_cls}). */
    public PoolingKind poolingCls() { return d.poolingCls; }
    /** Classifier outputs ({@code n_cls_out}). */
    public int nClsOut() { return d.nClsOut; }
    /** Classifier labels ({@code classifier.output_labels}). */
    public String[] classifierLabels() { return d.classifierLabels.clone(); }
    /** Decision labels ({@code decision.labels}). */
    public String[] decisionLabels() { return d.decisionLabels.clone(); }
    /** {@code decision.routing_block_count}. */
    public int decisionRoutingLayers() { return d.decisionRoutingLayers; }
    /** {@code decision.block_count}. */
    public int decisionJointLayers() { return d.decisionJointLayers; }
    /** {@code decision.head_count}. */
    public int decisionHeadCount() { return d.decisionHeadCount; }
    /** Classifier head activation ({@code act_cls}). */
    public ActOp classifierActivation() { return d.classifierActivation; }
    /** {@code hidden_activation} name, or null. */
    public String hiddenActivation() { return d.hiddenActivation; }
    /** Decoder start token, -1 = none ({@code dec_start_token_id}). */
    public int decStartTokenId() { return d.decStartTokenId; }
    /** Decoder block count ({@code dec_n_layer}). */
    public int decNLayer() { return d.decNLayer; }
    /** {@code dense_2_feat_in}. */
    public int dense2FeatIn() { return d.dense2FeatIn; }
    /** {@code dense_2_feat_out}. */
    public int dense2FeatOut() { return d.dense2FeatOut; }
    /** {@code dense_3_feat_in}. */
    public int dense3FeatIn() { return d.dense3FeatIn; }
    /** {@code dense_3_feat_out}. */
    public int dense3FeatOut() { return d.dense3FeatOut; }
    /** Diffusion logits are shifted by one position ({@code diffusion.shift_logits}). */
    public boolean diffusionShiftLogits() { return d.diffusionShiftLogits; }

    // ---- drafters and adapters ----

    /** Target layers feeding a drafter ({@code target_layers}). */
    public int[] targetLayers() { return d.targetLayers.clone(); }
    /** Target model hidden size ({@code target_hidden_size}). */
    public int targetHiddenSize() { return d.targetHiddenSize; }
    /** {@code dflash_block_size}. */
    public int dflashBlockSize() { return d.dflashBlockSize; }
    /** {@code dflash_conv_kernel_size}. */
    public int dflashConvKernelSize() { return d.dflashConvKernelSize; }
    /** {@code dflash_conv_group_size}. */
    public int dflashConvGroupSize() { return d.dflashConvGroupSize; }
    /** {@code dflash_selector_rank}. */
    public int dflashSelectorRank() { return d.dflashSelectorRank; }
    /** {@code dflash_selector_top_k}. */
    public int dflashSelectorTopK() { return d.dflashSelectorTopK; }
    /** {@code lilicorr_hidden_size}. */
    public int lilicorrHiddenSize() { return d.lilicorrHiddenSize; }
    /** {@code lilicorr_candidate_topk}. */
    public int lilicorrCandidateTopK() { return d.lilicorrCandidateTopK; }
    /** {@code lilicorr_num_heads}. */
    public int lilicorrNumHeads() { return d.lilicorrNumHeads; }
    /** {@code lilicorr_layers}. */
    public int lilicorrLayers() { return d.lilicorrLayers; }
    /** {@code lilicorr_vector_eps}. */
    public float lilicorrVectorEps() { return d.lilicorrVectorEps; }
    /** {@code lilicorr_logit_scale}. */
    public float lilicorrLogitScale() { return d.lilicorrLogitScale; }
    /** {@code lilicorr.factor_dim}. */
    public int lilicorrFactorDim() { return d.lilicorrFactorDim; }
    /** {@code lilicorr.mlp_hidden_size}. */
    public int lilicorrMlpHiddenSize() { return d.lilicorrMlpHiddenSize; }
    /** Granite-switch adapter count ({@code adapters.count}). */
    public int adapterCount() { return d.adapterCount; }
    /** Granite-switch adapter LoRA rank ({@code adapters.lora_rank}). */
    public int adapterLoraRank() { return d.adapterLoraRank; }
    /** Granite-switch router gain ({@code adapters.router_gain}). */
    public float adapterRouterGain() { return d.adapterRouterGain; }
    /** {@code adapters.token_ids_activate}. */
    public int[] adapterTokenIdsActivate() { return d.adapterTokenIdsActivate.clone(); }
    /** {@code adapters.token_ids_substitute}. */
    public int[] adapterTokenIdsSubstitute() { return d.adapterTokenIdsSubstitute.clone(); }

    // ---- per-layer values ----

    /** Layer {@code il}: query heads ({@code n_head_arr}). */
    public int nHead(int il) { return d.nHeadArr[layer(il)]; }
    /** Copy of the per-layer {@link #nHead(int)} values. */
    public int[] nHeadArray() { return d.nHeadArr.clone(); }
    /** Layer {@code il}: KV heads ({@code n_head_kv_arr}). */
    public int nHeadKv(int il) { return d.nHeadKvArr[layer(il)]; }
    /** Copy of the per-layer {@link #nHeadKv(int)} values. */
    public int[] nHeadKvArray() { return d.nHeadKvArr.clone(); }
    /** Layer {@code il}: FFN size ({@code n_ff_arr}). */
    public int nFf(int il) { return d.nFfArr[layer(il)]; }
    /** Copy of the per-layer {@link #nFf(int)} values. */
    public int[] nFfArray() { return d.nFfArr.clone(); }
    /** Layer {@code il}: expert FFN size ({@code n_ff_exp_arr}). */
    public int nFfExp(int il) { return d.nFfExpArr[layer(il)]; }
    /** Copy of the per-layer {@link #nFfExp(int)} values. */
    public int[] nFfExpArray() { return d.nFfExpArr.clone(); }
    /** Layer {@code il}: experts routed per token ({@code n_expert_used_arr}). */
    public int nExpertUsed(int il) { return d.nExpertUsedArr[layer(il)]; }
    /** Copy of the per-layer {@link #nExpertUsed(int)} values. */
    public int[] nExpertUsedArray() { return d.nExpertUsedArr.clone(); }
    /** Layer {@code il}: sliding-window layer ({@code is_swa_impl}). */
    public boolean isSwa(int il) { return d.swaArr[layer(il)]; }
    /** Copy of the per-layer {@link #isSwa(int)} values. */
    public boolean[] isSwaArray() { return d.swaArr.clone(); }
    /** Layer {@code il}: RoPE applied ({@code rope_pattern}). */
    public boolean ropeEnabled(int il) { return d.ropeArr[layer(il)]; }
    /** Copy of the per-layer {@link #ropeEnabled(int)} values. */
    public boolean[] ropeEnabledArray() { return d.ropeArr.clone(); }
    /** Layer {@code il}: recurrent layer ({@code is_recr_impl}). */
    public boolean isRecurrent(int il) { return d.recurrentArr[layer(il)]; }
    /** Copy of the per-layer {@link #isRecurrent(int)} values. */
    public boolean[] isRecurrentArray() { return d.recurrentArr.clone(); }
    /** Layer {@code il}: full (not shared) sparse-attention indexer ({@code is_indexer_full_impl}). */
    public boolean isIndexerFull(int il) { return d.indexerFullArr[layer(il)]; }
    /** Copy of the per-layer {@link #isIndexerFull(int)} values. */
    public boolean[] isIndexerFullArray() { return d.indexerFullArr.clone(); }
    /** Layer {@code il}: per-layer embedding layer ({@code is_ple_impl}). */
    public boolean isPleLayer(int il) { return d.pleArr[layer(il)]; }
    /** Copy of the per-layer {@link #isPleLayer(int)} values. */
    public boolean[] isPleLayerArray() { return d.pleArr.clone(); }
    /** Layer {@code il}: compressed attention ratio ({@code dsv4_compress_ratios}). */
    public int compressRatio(int il) { return d.compressRatioArr[layer(il)]; }
    /** Copy of the per-layer {@link #compressRatio(int)} values. */
    public int[] compressRatioArray() { return d.compressRatioArr.clone(); }
    /** Layer {@code il}: deepstack embedding index, -1 = none ({@code deepstack_mapping_arr}). */
    public int deepstackMapping(int il) { return d.deepstackMappingArr[layer(il)]; }
    /** Copy of the per-layer {@link #deepstackMapping(int)} values. */
    public int[] deepstackMappingArray() { return d.deepstackMappingArr.clone(); }
    /** Layer {@code il}: xIELU alpha_n ({@code xielu_alpha_n}). */
    public float xieluAlphaN(int il) { return d.xieluAlphaNArr[layer(il)]; }
    /** Copy of the per-layer {@link #xieluAlphaN(int)} values. */
    public float[] xieluAlphaNArray() { return d.xieluAlphaNArr.clone(); }
    /** Layer {@code il}: xIELU alpha_p ({@code xielu_alpha_p}). */
    public float xieluAlphaP(int il) { return d.xieluAlphaPArr[layer(il)]; }
    /** Copy of the per-layer {@link #xieluAlphaP(int)} values. */
    public float[] xieluAlphaPArray() { return d.xieluAlphaPArr.clone(); }
    /** Layer {@code il}: xIELU beta ({@code xielu_beta}). */
    public float xieluBeta(int il) { return d.xieluBetaArr[layer(il)]; }
    /** Copy of the per-layer {@link #xieluBeta(int)} values. */
    public float[] xieluBetaArray() { return d.xieluBetaArr.clone(); }
    /** Layer {@code il}: xIELU eps ({@code xielu_eps}). */
    public float xieluEps(int il) { return d.xieluEpsArr[layer(il)]; }
    /** Copy of the per-layer {@link #xieluEps(int)} values. */
    public float[] xieluEpsArray() { return d.xieluEpsArr.clone(); }
    /** Layer {@code il}: expert SwiGLU clamp, 0 = none ({@code swiglu_clamp_exp}). */
    public float swigluClampExp(int il) { return d.swigluClampExpArr[layer(il)]; }
    /** Copy of the per-layer {@link #swigluClampExp(int)} values. */
    public float[] swigluClampExpArray() { return d.swigluClampExpArr.clone(); }
    /** Layer {@code il}: shared expert SwiGLU clamp, 0 = none ({@code swiglu_clamp_shexp}). */
    public float swigluClampShexp(int il) { return d.swigluClampShexpArr[layer(il)]; }
    /** Copy of the per-layer {@link #swigluClampShexp(int)} values. */
    public float[] swigluClampShexpArray() { return d.swigluClampShexpArr.clone(); }

    // ---- derived values (llama-hparams.cpp) ----

    /** Query heads per KV head, 0 when the layer has no KV heads ({@code n_gqa}). */
    public int nGqa(int il) {
        int kv = nHeadKv(il);
        return kv == 0 ? 0 : nHead(il) / kv;
    }

    /** Rotary dimensions of layer {@code il} ({@code n_rot}). */
    public int nRot(int il) {
        return isSwa(il) ? d.nRotSwa : d.nRotFull;
    }

    /** Key head size of layer {@code il} ({@code n_embd_head_k}). */
    public int headK(int il) {
        return isSwa(il) ? d.headKSwa : d.headKFull;
    }

    /** Value head size of layer {@code il} ({@code n_embd_head_v}). */
    public int headV(int il) {
        return isSwa(il) ? d.headVSwa : d.headVFull;
    }

    /** Key width across KV heads ({@code n_embd_k_gqa}). */
    public int nEmbdKGqa(int il) {
        return headK(il) * nHeadKv(il);
    }

    /** Value width across KV heads ({@code n_embd_v_gqa}). */
    public int nEmbdVGqa(int il) {
        return headV(il) * nHeadKv(il);
    }

    /** Largest {@link #nEmbdKGqa(int)} over all layers. */
    public int nEmbdKGqaMax() {
        int max = 0;
        for (int il = 0; il < nLayerAll; il++) {
            max = Math.max(max, nEmbdKGqa(il));
        }
        return max;
    }

    /** Largest {@link #nEmbdVGqa(int)} over all layers. */
    public int nEmbdVGqaMax() {
        int max = 0;
        for (int il = 0; il < nLayerAll; il++) {
            max = Math.max(max, nEmbdVGqa(il));
        }
        return max;
    }

    /** True when the key width differs between layers. */
    public boolean isNEmbdKGqaVariable() {
        for (int il = 1; il < nLayerAll; il++) {
            if (nEmbdKGqa(il) != nEmbdKGqa(0)) {
                return true;
            }
        }
        return false;
    }

    /** True when the value width differs between layers. */
    public boolean isNEmbdVGqaVariable() {
        for (int il = 1; il < nLayerAll; il++) {
            if (nEmbdVGqa(il) != nEmbdVGqa(0)) {
                return true;
            }
        }
        return false;
    }

    /** Input embedding width: main plus deepstack embeddings unless overridden ({@code n_embd_inp}). */
    public int nEmbdInp() {
        if (d.nEmbdInpImpl > 0) {
            return d.nEmbdInpImpl;
        }
        return d.nEmbd + d.nEmbd * d.nDeepstackLayers;
    }

    /** Encoder input width ({@code n_embd_inp_enc}). */
    public int nEmbdInpEnc() {
        return d.nEmbdInpEncImpl > 0 ? d.nEmbdInpEncImpl : nEmbdInp();
    }

    /** Output embedding width ({@code n_embd_out}). */
    public int nEmbdOut() {
        return d.nEmbdOutImpl > 0 ? d.nEmbdOutImpl : d.nEmbd;
    }

    /** Rolling (convolution / token-shift) state width per sequence ({@code n_embd_r}). */
    public long nEmbdR() {
        if (d.wkvHeadSize != 0) {
            return (long) d.tokenShiftCount * d.nEmbd;
        }
        if (d.shortconvLCache != 0) {
            return (long) d.nEmbd * (d.shortconvLCache - 1);
        }
        if (d.kdaHeadDim != 0) {
            long dInner = (long) nHead(0) * d.kdaHeadDim;
            return 3L * (d.ssmDConv > 0 ? d.ssmDConv - 1 : 3) * dInner;
        }
        return (long) (d.ssmDConv > 0 ? d.ssmDConv - 1 : 0) * (d.ssmDInner + 2L * d.ssmNGroup * d.ssmDState);
    }

    /** Recurrent state width per sequence ({@code n_embd_s}). */
    public long nEmbdS() {
        if (d.wkvHeadSize != 0) {
            return (long) d.nEmbd * d.wkvHeadSize;
        }
        if (d.kdaHeadDim != 0) {
            return (long) d.kdaHeadDim * d.kdaHeadDim * nHead(0);
        }
        if (d.headLinearAttn != 0) {
            return (long) d.headLinearAttn * d.headLinearAttn * nHead(0);
        }
        return (long) d.ssmDState * d.ssmDInner;
    }

    /** Per-layer-embedding convolution history rows, 0 without a PLE module ({@code ple_conv_state}). */
    public long pleConvState() {
        if (d.pleNHeads == 0 || d.pleConvKernel == 0) {
            return 0;
        }
        return (long) (d.pleConvKernel - 1) * d.pleNgramSize * d.dsv4HcMult * d.nEmbd;
    }

    /** Positions per token: 4 for MROPE/IMROPE, else 1 ({@code n_pos_per_embd}). */
    public int nPosPerEmbd() {
        return d.ropeType == RopeType.MROPE || d.ropeType == RopeType.IMROPE ? MROPE_SECTIONS : 1;
    }

    /** True when the model uses multi-head latent attention ({@code is_mla}). */
    public boolean isMla() {
        return d.headKMlaImpl != 0 && d.headVMlaImpl != 0;
    }

    /** MLA key head size, or the layer-0 key head size without MLA ({@code n_embd_head_k_mla}). */
    public int headKMla() {
        return isMla() ? d.headKMlaImpl : headK(0);
    }

    /** MLA value head size, or the layer-0 value head size without MLA ({@code n_embd_head_v_mla}). */
    public int headVMla() {
        return isMla() ? d.headVMlaImpl : headV(0);
    }

    /** True when layer {@code il} owns a KV cache ({@code has_kv}). */
    public boolean hasKv(int il) {
        layer(il);
        return d.nLayerKvFromStart < 0 || il < d.nLayerKvFromStart;
    }

    /** True when layer {@code il} applies RoPE; the router layer never does ({@code has_rope}). */
    public boolean hasRope(int il) {
        if (d.routerLayer >= 0 && il == d.routerLayer) {
            return false;
        }
        return ropeEnabled(il);
    }

    /** True when the first two multi-section RoPE sections are set ({@code use_mrope}). */
    public boolean useMrope() {
        return d.ropeSections[0] > 0 && d.ropeSections[1] > 0;
    }

    /** True when any layer is a sliding-window layer ({@code is_swa_any}). */
    public boolean isSwaAny() {
        for (int il = 0; il < nLayerAll; il++) {
            if (d.swaArr[il]) {
                return true;
            }
        }
        return false;
    }

    /** Largest {@link #nExpertUsed(int)} over all layers ({@code n_expert_used_max}). */
    public int nExpertUsedMax() {
        int max = 0;
        for (int il = 0; il < nLayerAll; il++) {
            max = Math.max(max, d.nExpertUsedArr[il]);
        }
        return max;
    }

    /** RoPE base of layer {@code il}: the SWA base on SWA layers (llama-model.cpp:2755-2757). */
    public float ropeFreqBase(int il) {
        return isSwa(il) ? d.ropeFreqBaseSwa : d.ropeFreqBase;
    }

    /** RoPE frequency scale of layer {@code il}: the SWA scale on SWA layers (llama-model.cpp:2759-2761). */
    public float ropeFreqScale(int il) {
        return isSwa(il) ? d.ropeFreqScaleSwa : d.ropeFreqScale;
    }

    private int layer(int il) {
        if (il < 0 || il >= nLayerAll) {
            throw new IndexOutOfBoundsException("layer " + il + " outside [0, " + nLayerAll + ")");
        }
        return il;
    }

    @Override
    public String toString() {
        return "ModelHparams{" + d.arch().ggufName() + ", nEmbd=" + d.nEmbd + ", nLayer=" + nLayer()
                + (nextnLayers > 0 ? "+" + nextnLayers + " nextn" : "") + ", nHead=" + nHead(0)
                + ", nHeadKv=" + nHeadKv(0) + ", nFf=" + nFf(0) + ", nCtxTrain=" + d.nCtxTrain
                + ", rope=" + d.ropeType + ", swa=" + d.swaType + (d.nExpert > 0 ? ", nExpert=" + d.nExpert : "")
                + "}";
    }
}

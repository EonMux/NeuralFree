package com.eonmux.neuralfree.gguf.quant.write;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The tensor-type policy of the quantize tool, the port of llama.cpp src/llama-quant.cpp: the tensor
 * categories ({@code tensor_get_category} :121-163), the eligibility rules ({@code
 * tensor_allows_quantization} :302-421), the mixture recipes ({@code llama_tensor_get_type_impl} :484-760),
 * the shape fallback ({@code tensor_type_fallback} :428-481) and the imatrix requirement ({@code
 * tensor_requires_imatrix} :902-921). {@link Mixture} carries the per-run state the recipes read and
 * mutate: the layer counters advance in tensor-table order, so targets are decided by one
 * {@link Mixture#target} call per tensor in table order.
 * <p>
 * Architecture-conditional rules read {@code general.architecture} and the head/expert/layer counts of
 * {@link ModelInfo}; the 70B rule infers llama.cpp's {@code LLM_TYPE_70B} the way
 * {@code llama_quant_model_from_metadata} does (llama, 80 layers, unequal head counts).
 */
public final class QuantizeTypes {

    private QuantizeTypes() {
    }

    /** The broad tensor categories the mixture rules branch on ({@code tensor_category}). */
    enum Category {
        TOKEN_EMBD, ATTENTION_Q, ATTENTION_V, ATTENTION_K, ATTENTION_QKV, ATTENTION_KV_B,
        ATTENTION_OUTPUT, ATTENTION_GATE, FFN_UP, FFN_GATE, FFN_LATENT, FFN_DOWN, OUTPUT, OTHER
    }

    private static final Pattern BLOCK = Pattern.compile("blk\\.(\\d+)\\.");

    /** The category of {@code name}, in the reference's match order. */
    static Category category(String name) {
        if (name.equals("output.weight")) {
            return Category.OUTPUT;
        }
        if (name.equals("token_embd.weight") || name.equals("per_layer_token_embd.weight")) {
            return Category.TOKEN_EMBD;
        }
        // attn_qkv_gate is [attn_qkv | attn_gate] in one weight, the qkv rows decide its type
        if (name.contains("attn_qkv.weight") || name.contains("attn_qkv_gate.weight")) {
            return Category.ATTENTION_QKV;
        }
        if (name.contains("attn_kv_b.weight")) {
            return Category.ATTENTION_KV_B;
        }
        if (name.contains("attn_v.weight")) {
            return Category.ATTENTION_V;
        }
        if (name.contains("attn_k.weight")) {
            return Category.ATTENTION_K;
        }
        if (name.contains("attn_q.weight")) {
            return Category.ATTENTION_Q;
        }
        if (name.contains("attn_output.weight")) {
            return Category.ATTENTION_OUTPUT;
        }
        if (name.contains("ffn_up")) {
            return Category.FFN_UP;
        }
        if (name.contains("ffn_gate")) {
            return Category.FFN_GATE;
        }
        if (name.contains("ffn_latent")) {
            return Category.FFN_LATENT;
        }
        if (name.contains("attn_gate.weight")) {
            return Category.ATTENTION_GATE;
        }
        if (name.contains("ffn_down")) {
            return Category.FFN_DOWN;
        }
        return Category.OTHER;
    }

    /** True for the MoE expert tensors ({@code tensor_is_expert}). */
    static boolean isExpert(String name) {
        return name.contains("_exps.") || name.contains("_chexps.");
    }

    /** True for the attention-v-like categories, the ones most sensitive to quantization. */
    static boolean isAttnVLike(Category cat) {
        return cat == Category.ATTENTION_V || cat == Category.ATTENTION_QKV || cat == Category.ATTENTION_KV_B;
    }

    /** The model facts the policy reads: architecture name and the counts of the metadata. */
    public static final class ModelInfo {
        /** The {@code general.architecture} value, lower-case; "unknown" when the file carries none. */
        public final String arch;
        public final int nLayer;
        public final int nHead;
        public final int nHeadKv;
        public final int nExpert;
        /** True when importance-matrix weights are in play. */
        public final boolean hasImatrix;

        private ModelInfo(String arch, int nLayer, int nHead, int nHeadKv, int nExpert, boolean hasImatrix) {
            this.arch = arch == null ? "unknown" : arch.toLowerCase(Locale.ROOT);
            this.nLayer = nLayer;
            this.nHead = nHead;
            this.nHeadKv = nHeadKv;
            this.nExpert = nExpert;
            this.hasImatrix = hasImatrix;
        }

        /** The facts from the model metadata values; a missing head-count KV keeps the head count. */
        public static ModelInfo of(String arch, int nLayer, int nHead, Integer nHeadKv, int nExpert,
                                   boolean hasImatrix) {
            int kv = nHeadKv != null && nHeadKv > 0 ? nHeadKv : nHead;
            return new ModelInfo(arch, nLayer, nHead, kv, nExpert, hasImatrix);
        }

        boolean falcon() {
            return arch.equals("falcon");
        }

        boolean glm5Next() {
            return arch.equals("glm5-next");
        }

        boolean museSpark() {
            return arch.equals("muse-spark");
        }

        boolean limite() {
            return arch.equals("limite");
        }

        boolean vllmDecision2() {
            return arch.equals("vllm-decision2");
        }

        /** {@code n_gqa()}: the query heads per KV head. */
        int nGqa() {
            return nHeadKv > 0 ? nHead / nHeadKv : 1;
        }

        /** The reference's {@code LLM_TYPE_70B} inference: llama, 80 layers, unequal head counts. */
        boolean is70B() {
            return arch.equals("llama") && nLayer == 80 && nHead != nHeadKv;
        }
    }

    /**
     * The per-run mixture state: the params and model facts, the layer counters of the recipes and the
     * manual tensor-type patterns. Built over the tensor names in table order; {@link #target} then runs
     * per tensor in the same order.
     */
    public static final class Mixture {
        private final QuantizeParams params;
        private final ModelInfo model;

        private int nAttentionWv;
        private final int nFfnDown;
        private final int nFfnGate;
        private final int nFfnUp;
        private int iAttentionWv;
        private int iFfnDown;
        private int iFfnGate;
        private int iFfnUp;
        private int nFallback;
        private boolean hasTiedEmbeddings = true; // assume tied until we see output.weight

        /** Pass 1: the category counts and the tied-embedding flag over the names in table order. */
        public Mixture(QuantizeParams params, ModelInfo model, List<String> names) {
            this.params = params;
            this.model = model;
            int layers = Math.max(0, model.nLayer);
            this.nFfnDown = layers;
            this.nFfnGate = layers;
            this.nFfnUp = layers;
            for (String name : names) {
                if (isAttnVLike(category(name))) {
                    ++nAttentionWv;
                }
                if (category(name) == Category.OUTPUT) {
                    hasTiedEmbeddings = false;
                }
            }
        }

        /** The tensors the shape fallback demoted, as the reference's {@code n_fallback} warning counts. */
        public int fallbackCount() {
            return nFallback;
        }

        /**
         * Pass 2, one tensor: the target type of {@code name} ({@code llama_tensor_get_type}), the rank
         * and first dimension of the stored tensor, its stored type and the recipe's default type.
         */
        public GGMLType target(String name, long[] ne, GGMLType srcType, GGMLType defaultType) {
            int rank = 1;
            for (long d : ne) {
                if (d > 1) {
                    rank++;
                }
            }
            rank = Math.min(rank, ne.length);
            if (!allowsQuantization(name, rank)) {
                return srcType;
            }
            Category cat = category(name);
            if (params.tokenEmbeddingType != null && cat == Category.TOKEN_EMBD) {
                // per_layer_token_embd follows --token-embedding-type by default, but it is a large
                // separate table, so let an explicit --tensor-type name it
                boolean named = false;
                if (name.equals("per_layer_token_embd.weight")) {
                    for (QuantizeParams.TensorType tt : params.tensorTypes) {
                        if (tt.pattern.matcher(name).find()) {
                            named = true;
                            break;
                        }
                    }
                }
                if (!named) {
                    return params.tokenEmbeddingType;
                }
            }
            if (params.outputTensorType != null && cat == Category.OUTPUT) {
                return params.outputTensorType;
            }

            GGMLType newType = defaultType;
            if (defaultType.isQuantized()) {
                boolean manual = false;
                for (QuantizeParams.TensorType tt : params.tensorTypes) {
                    if (tt.pattern.matcher(name).find()) {
                        if (tt.type != newType) {
                            newType = tt.type;
                        }
                        manual = true;
                        break;
                    }
                }
                if (!manual && !params.pure) {
                    newType = mixtureType(newType, name, ne, cat);
                }
                newType = fallbackType(ne.length == 0 ? 0 : ne[0], newType);
            }
            return newType;
        }

        /**
         * {@code tensor_allows_quantization}: only 2-D+ tensors whose name ends in {@code weight} and
         * holds none of the names the reference keeps, and {@code output.weight} only when asked.
         */
        private boolean allowsQuantization(String name, int rank) {
            if (params.ftype.onlyCopy()) {
                return false;
            }
            if (rank < 2) {
                return false;
            }
            boolean quantize = name.length() >= 6 && name.lastIndexOf("weight") == name.length() - 6;
            quantize &= !name.contains("_norm.weight");
            quantize &= params.quantizeOutputTensor || !name.equals("output.weight");
            quantize &= !name.contains("ffn_gate_inp.weight");
            quantize &= !name.contains("ffn_gate_tid2eid.weight");
            quantize &= !name.contains("engram_gate.weight");
            quantize &= !name.contains("altup") && !name.contains("laurel");
            quantize &= !name.contains("per_layer_model_proj");
            if (model.limite()) {
                quantize &= !name.contains("attn_gate.weight");
                quantize &= !name.contains("attn_ve_gate.weight");
                quantize &= !name.contains("mudd_");
            }
            quantize &= !name.equals("pos_embd.weight") && !name.equals("token_types.weight");
            quantize &= !name.contains("ssm_conv1d");
            quantize &= !name.contains("shortconv.conv.weight");
            quantize &= !name.contains("conv2d_spatial.weight");
            quantize &= !name.startsWith("a.conv1d.");
            quantize &= !name.contains("conv_dw");
            quantize &= !name.startsWith("a.gen.wav.quant.");
            quantize &= !name.contains("indexer.k_proj.weight");
            quantize &= !name.contains("indexer.q_proj.weight");
            if (model.glm5Next()) {
                quantize &= !name.contains("hc_");
                quantize &= !name.contains("indexer.attn_q_b");
                quantize &= !name.contains("indexer.attn_k");
                quantize &= !name.contains("indexer.proj");
                quantize &= !name.contains("indexer_compressor_gate");
                quantize &= !name.contains("indexer_compressor_ape");
                quantize &= !name.contains("ssm_f_a.weight") && !name.contains("ssm_f_b.weight");
                quantize &= !name.contains("ssm_g_a.weight") && !name.contains("ssm_g_b.weight");
                quantize &= !name.contains("ssm_beta.weight");
                quantize &= !name.contains("attn_kv_a_mqa.weight");
                quantize &= !name.contains("attn_k_b.weight") && !name.contains("attn_v_b.weight");
            }
            if (model.vllmDecision2()) {
                quantize &= !name.startsWith("decision.");
            }
            quantize &= !name.contains("time_mix_first.weight") && !name.contains("time_mix_w0.weight")
                    && !name.contains("time_mix_w1.weight") && !name.contains("time_mix_w2.weight")
                    && !name.contains("time_mix_v0.weight") && !name.contains("time_mix_v1.weight")
                    && !name.contains("time_mix_v2.weight") && !name.contains("time_mix_a0.weight")
                    && !name.contains("time_mix_a1.weight") && !name.contains("time_mix_a2.weight")
                    && !name.contains("time_mix_g1.weight") && !name.contains("time_mix_g2.weight")
                    && !name.contains("time_mix_decay_w1.weight") && !name.contains("time_mix_decay_w2.weight")
                    && !name.contains("time_mix_lerp_fused.weight");
            quantize &= !name.contains("attn_rel_b.weight");
            quantize &= !name.contains(".position_embd") && !name.contains("sam.pos_embd")
                    && !name.contains("sam.neck.") && !name.contains("sam.net_")
                    && !name.contains(".rel_pos") && !name.contains(".patch_embd")
                    && !name.contains(".patch_merger");
            quantize &= !name.contains("a.rvq.codebook") && !name.contains("mm.a.code_embd");
            return quantize;
        }

        /** {@code use_more_bits}: the first, last and every third layer take the bigger type. */
        private static boolean useMoreBits(int iLayer, int nLayers) {
            return iLayer < nLayers / 8 || iLayer >= 7 * nLayers / 8 || (iLayer - nLayers / 8) % 3 == 2;
        }

        /** {@code layer_info}: the layer of a tensor name, parsed from the name for expert models. */
        private int layerInfo(int iLayer, int nLayer, String name) {
            if (model.nExpert > 1) {
                Matcher m = BLOCK.matcher(name);
                if (!m.find()) {
                    throw new IllegalStateException("Failed to determine layer for tensor " + name);
                }
                int blk = Integer.parseInt(m.group(1));
                if (blk < 0 || blk >= nLayer) {
                    throw new IllegalStateException("Bad layer " + blk + " for tensor " + name
                            + ". Must be in [0, " + nLayer + ")");
                }
                return blk;
            }
            return iLayer;
        }

        /** Bits per weight of a type, the reference's {@code bits_per_weight}. */
        private static float bitsPerWeight(GGMLType t) {
            return 8.0f * t.getTypeSize() / t.getBlockSize();
        }

        /** {@code llama_tensor_get_type_impl}: the ftype mixture over the category and layer counters. */
        private GGMLType mixtureType(GGMLType newType, String name, long[] ne, Category cat) {
            QuantizeFtype ftype = params.ftype;

            // by default, for glm5-next, don't let these tensors be quantized below Q8_0
            if (model.glm5Next() && (name.contains("attn_q_a") || name.contains("attn_q_b")
                    || name.contains("nextn.eh_proj"))) {
                switch (newType) {
                    case F32:
                    case BF16:
                    case F16:
                        break;
                    default:
                        return GGMLType.Q8_0;
                }
            }

            // bumping to q8_0 is preferred since these tensors are not really big
            if (model.museSpark() && (cat == Category.FFN_LATENT || cat == Category.ATTENTION_GATE)) {
                if (newType.isQuantized() && bitsPerWeight(newType) < bitsPerWeight(GGMLType.Q8_0)) {
                    newType = GGMLType.Q8_0;
                }
            }

            if (cat == Category.OUTPUT || (hasTiedEmbeddings && cat == Category.TOKEN_EMBD)) {
                if (params.outputTensorType != null) {
                    newType = params.outputTensorType;
                } else {
                    long nx = ne.length == 0 ? 0 : ne[0];
                    long qkK = newType.getBlockSize();
                    if (ftype == QuantizeFtype.MXFP4_MOE) {
                        newType = GGMLType.Q8_0;
                    } else if (model.falcon() || nx % qkK != 0) {
                        newType = GGMLType.Q8_0;
                    } else if (ftype == QuantizeFtype.IQ2_XXS || ftype == QuantizeFtype.IQ2_XS
                            || ftype == QuantizeFtype.IQ3_XXS || ftype == QuantizeFtype.IQ1_S
                            || ftype == QuantizeFtype.IQ2_S || ftype == QuantizeFtype.IQ2_M
                            || ftype == QuantizeFtype.IQ1_M) {
                        newType = GGMLType.Q5_K;
                    } else if (newType != GGMLType.Q8_0) {
                        newType = GGMLType.Q6_K;
                    }
                }
            } else if (ftype == QuantizeFtype.MXFP4_MOE) {
                // MoE tensors -> MXFP4, other tensors -> Q8_0
                newType = isExpert(name) ? GGMLType.MXFP4 : GGMLType.Q8_0;
            } else if (cat == Category.TOKEN_EMBD) {
                if (params.tokenEmbeddingType != null) {
                    newType = params.tokenEmbeddingType;
                } else if (ftype == QuantizeFtype.IQ2_XXS || ftype == QuantizeFtype.IQ2_XS
                        || ftype == QuantizeFtype.IQ1_S || ftype == QuantizeFtype.IQ1_M) {
                    newType = GGMLType.Q2_K;
                } else if (ftype == QuantizeFtype.IQ2_S || ftype == QuantizeFtype.IQ2_M) {
                    newType = GGMLType.IQ3_S;
                } else if (ftype == QuantizeFtype.IQ3_XXS) {
                    newType = GGMLType.IQ3_S;
                } else if (ftype == QuantizeFtype.TQ1_0 || ftype == QuantizeFtype.TQ2_0
                        || ftype == QuantizeFtype.Q2_0) {
                    newType = GGMLType.Q4_K;
                }
            } else if (ftype == QuantizeFtype.IQ2_XXS || ftype == QuantizeFtype.IQ2_XS
                    || ftype == QuantizeFtype.IQ1_S || ftype == QuantizeFtype.IQ2_S
                    || ftype == QuantizeFtype.IQ2_M || ftype == QuantizeFtype.IQ1_M) {
                if (isAttnVLike(cat)) {
                    if (model.nGqa() >= 4 || model.nExpert >= 4) {
                        newType = GGMLType.Q4_K;
                    } else {
                        newType = ftype == QuantizeFtype.IQ2_S || ftype == QuantizeFtype.IQ2_M
                                ? GGMLType.IQ3_S : GGMLType.Q2_K;
                    }
                    ++iAttentionWv;
                } else if (model.nExpert == 8 && cat == Category.ATTENTION_K) {
                    newType = GGMLType.Q4_K;
                } else if (cat == Category.FFN_DOWN) {
                    if (iFfnDown < nFfnDown / 8) {
                        newType = ftype == QuantizeFtype.IQ2_S || ftype == QuantizeFtype.IQ2_M
                                ? GGMLType.IQ3_S : GGMLType.Q2_K;
                    }
                    ++iFfnDown;
                } else if (cat == Category.ATTENTION_OUTPUT) {
                    if (model.nExpert == 8) {
                        newType = GGMLType.Q5_K;
                    } else {
                        if (ftype == QuantizeFtype.IQ1_S || ftype == QuantizeFtype.IQ1_M) {
                            newType = GGMLType.IQ2_XXS;
                        } else if (ftype == QuantizeFtype.IQ2_S || ftype == QuantizeFtype.IQ2_M) {
                            newType = GGMLType.IQ3_S;
                        }
                    }
                }
            } else if (isAttnVLike(cat)) {
                if (ftype == QuantizeFtype.Q2_K) {
                    newType = model.nGqa() >= 4 ? GGMLType.Q4_K : GGMLType.Q3_K;
                } else if (ftype == QuantizeFtype.Q2_K_S && model.nGqa() >= 4) {
                    newType = GGMLType.Q4_K;
                } else if (ftype == QuantizeFtype.IQ3_XXS) {
                    newType = model.nGqa() >= 4 ? GGMLType.Q4_K
                            : !model.hasImatrix ? GGMLType.IQ3_S : GGMLType.IQ3_XXS;
                } else if ((ftype == QuantizeFtype.IQ3_XS || ftype == QuantizeFtype.IQ3_S)
                        && model.nGqa() >= 4) {
                    newType = GGMLType.Q4_K;
                } else if (ftype == QuantizeFtype.IQ3_M) {
                    newType = GGMLType.Q4_K;
                } else if (ftype == QuantizeFtype.Q3_K_M) {
                    newType = iAttentionWv < 2 ? GGMLType.Q5_K : GGMLType.Q4_K;
                } else if (ftype == QuantizeFtype.Q3_K_L) {
                    newType = GGMLType.Q5_K;
                } else if ((ftype == QuantizeFtype.IQ4_NL || ftype == QuantizeFtype.IQ4_XS)
                        && model.nGqa() >= 4) {
                    newType = GGMLType.Q5_K;
                } else if ((ftype == QuantizeFtype.Q4_K_M || ftype == QuantizeFtype.Q5_K_M)
                        && useMoreBits(iAttentionWv, nAttentionWv)) {
                    newType = GGMLType.Q6_K;
                } else if (ftype == QuantizeFtype.Q4_K_S && iAttentionWv < 4) {
                    newType = GGMLType.Q5_K;
                }
                if (model.is70B()) {
                    // the 8-head shared attn_v of the 70B models takes the bigger type
                    if (newType == GGMLType.Q3_K || newType == GGMLType.Q4_K) {
                        newType = GGMLType.Q5_K;
                    }
                }
                if (model.nExpert == 8) {
                    // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
                    newType = GGMLType.Q8_0;
                }
                ++iAttentionWv;
            } else if (cat == Category.ATTENTION_K) {
                if (model.nExpert == 8) {
                    newType = GGMLType.Q8_0;
                } else if (ftype == QuantizeFtype.IQ3_XS) {
                    newType = GGMLType.IQ3_XXS;
                } else if (ftype == QuantizeFtype.IQ3_XXS) {
                    newType = GGMLType.IQ2_S;
                }
            } else if (cat == Category.ATTENTION_Q) {
                if (ftype == QuantizeFtype.IQ3_XS) {
                    newType = GGMLType.IQ3_XXS;
                } else if (ftype == QuantizeFtype.IQ3_XXS) {
                    newType = GGMLType.IQ2_S;
                }
            } else if (cat == Category.FFN_DOWN) {
                int iLayer = layerInfo(iFfnDown, nFfnDown, name);
                int nLayer = nFfnDown;
                if (ftype == QuantizeFtype.Q2_K) {
                    newType = GGMLType.Q3_K;
                } else if (ftype == QuantizeFtype.Q2_K_S) {
                    if (iLayer < nLayer / 8) {
                        newType = GGMLType.Q4_K;
                    }
                } else if (ftype == QuantizeFtype.IQ3_XXS && !model.hasImatrix) {
                    newType = iLayer < nLayer / 8 ? GGMLType.Q4_K : GGMLType.Q3_K;
                } else if (ftype == QuantizeFtype.Q3_K_M) {
                    newType = iLayer < nLayer / 16 ? GGMLType.Q5_K
                            : !model.falcon() || useMoreBits(iLayer, nLayer) ? GGMLType.Q4_K
                            : GGMLType.Q3_K;
                } else if (ftype == QuantizeFtype.IQ3_M && (iLayer < nLayer / 8
                        || (model.nExpert == 8 && useMoreBits(iLayer, nLayer)))) {
                    newType = GGMLType.Q4_K;
                } else if (ftype == QuantizeFtype.Q3_K_L) {
                    newType = model.falcon() ? GGMLType.Q4_K : GGMLType.Q5_K;
                } else if (ftype == QuantizeFtype.Q4_K_M) {
                    if (model.falcon()) {
                        newType = iLayer < nLayer / 16 ? GGMLType.Q6_K
                                : useMoreBits(iLayer, nLayer) ? GGMLType.Q5_K : GGMLType.Q4_K;
                    } else {
                        if (useMoreBits(iLayer, nLayer)) {
                            newType = GGMLType.Q6_K;
                        }
                    }
                } else if (iLayer < nLayer / 8 && (ftype == QuantizeFtype.IQ4_NL
                        || ftype == QuantizeFtype.IQ4_XS) && !model.hasImatrix) {
                    newType = GGMLType.Q5_K;
                } else if (ftype == QuantizeFtype.Q5_K_M && useMoreBits(iLayer, nLayer)) {
                    newType = GGMLType.Q6_K;
                } else if (ftype == QuantizeFtype.Q4_K_S && !model.falcon() && iLayer < nLayer / 8) {
                    newType = GGMLType.Q5_K;
                } else if ((ftype == QuantizeFtype.Q4_0 || ftype == QuantizeFtype.Q5_0)
                        && model.hasImatrix && iLayer < nLayer / 8) {
                    // guard against craziness in the first few ffn_down layers, with an imatrix only
                    newType = ftype == QuantizeFtype.Q4_0 ? GGMLType.Q4_1 : GGMLType.Q5_1;
                }
                ++iFfnDown;
            } else if (cat == Category.ATTENTION_OUTPUT) {
                if (!model.falcon()) {
                    if (model.nExpert == 8) {
                        if (ftype == QuantizeFtype.Q2_K || ftype == QuantizeFtype.IQ3_XS
                                || ftype == QuantizeFtype.IQ3_XXS || ftype == QuantizeFtype.Q3_K_S
                                || ftype == QuantizeFtype.Q3_K_M || ftype == QuantizeFtype.IQ4_NL
                                || ftype == QuantizeFtype.Q4_K_S || ftype == QuantizeFtype.Q4_K_M
                                || ftype == QuantizeFtype.IQ3_S || ftype == QuantizeFtype.IQ3_M
                                || ftype == QuantizeFtype.IQ4_XS) {
                            newType = GGMLType.Q5_K;
                        }
                    } else {
                        if (ftype == QuantizeFtype.Q2_K) {
                            newType = GGMLType.Q3_K;
                        } else if (ftype == QuantizeFtype.IQ3_XXS) {
                            newType = GGMLType.IQ3_S;
                        } else if (ftype == QuantizeFtype.Q3_K_M) {
                            newType = GGMLType.Q4_K;
                        } else if (ftype == QuantizeFtype.Q3_K_L) {
                            newType = GGMLType.Q5_K;
                        } else if (ftype == QuantizeFtype.IQ3_M) {
                            newType = GGMLType.Q4_K;
                        }
                    }
                } else {
                    if (ftype == QuantizeFtype.Q3_K_L) {
                        newType = GGMLType.Q4_K;
                    }
                }
            } else if (cat == Category.ATTENTION_QKV) {
                if (ftype == QuantizeFtype.Q3_K_M || ftype == QuantizeFtype.Q3_K_L
                        || ftype == QuantizeFtype.IQ3_M) {
                    newType = GGMLType.Q4_K;
                } else if (ftype == QuantizeFtype.Q4_K_M) {
                    newType = GGMLType.Q5_K;
                } else if (ftype == QuantizeFtype.Q5_K_M) {
                    newType = GGMLType.Q6_K;
                }
            } else if (cat == Category.FFN_GATE) {
                int iLayer = layerInfo(iFfnGate, nFfnGate, name);
                if (ftype == QuantizeFtype.IQ3_XS && (iLayer >= nFfnGate / 8 && iLayer < 7 * nFfnGate / 8)) {
                    newType = GGMLType.IQ3_XXS;
                }
                ++iFfnGate;
            } else if (cat == Category.FFN_UP) {
                int iLayer = layerInfo(iFfnUp, nFfnUp, name);
                if (ftype == QuantizeFtype.IQ3_XS && (iLayer >= nFfnUp / 8 && iLayer < 7 * nFfnUp / 8)) {
                    newType = GGMLType.IQ3_XXS;
                }
                ++iFfnUp;
            }
            return newType;
        }

        /**
         * {@code tensor_type_fallback}: the demotion when the tensor's first dimension does not divide the
         * target's block size, ending in F16 when even the fallback does not fit.
         */
        private GGMLType fallbackType(long ncols, GGMLType targetType) {
            GGMLType returnType = targetType;
            long qkK = targetType.getBlockSize();
            if (ncols % qkK != 0) {
                ++nFallback;
                switch (targetType) {
                    case IQ1_S:
                    case IQ1_M:
                    case IQ2_XXS:
                    case IQ2_XS:
                    case IQ2_S:
                    case IQ3_XXS:
                    case IQ3_S:
                    case IQ4_XS:
                        returnType = GGMLType.IQ4_NL;
                        break;
                    case Q2_0:
                    case Q2_K:
                    case Q3_K:
                    case TQ1_0:
                    case TQ2_0:
                        returnType = GGMLType.Q4_0;
                        break;
                    case Q4_K:
                        returnType = GGMLType.Q5_0;
                        break;
                    case Q5_K:
                        returnType = GGMLType.Q5_1;
                        break;
                    case Q6_K:
                        returnType = GGMLType.Q8_0;
                        break;
                    default:
                        if (qkK <= 32) {
                            // the target is already a 32-block type, so there is no smaller block to
                            // demote to; the check below turns it into F16 as needed
                            returnType = targetType;
                            break;
                        }
                        throw new IllegalStateException("no tensor type fallback is defined for type "
                                + targetType);
                }
                if (ncols % returnType.getBlockSize() != 0) {
                    returnType = GGMLType.F16;
                }
            }
            return returnType;
        }
    }

    /** {@code tensor_requires_imatrix}: the very low-bit types, and Q2_K inside a Q2_K_S file. */
    public static boolean requiresImatrix(String name, GGMLType dstType, QuantizeFtype ftype) {
        if (name.equals("token_embd.weight") || name.equals("per_layer_token_embd.weight")
                || name.equals("output.weight")) {
            return false;
        }
        switch (dstType) {
            case IQ3_XXS:
            case IQ2_XXS:
            case IQ2_XS:
            case IQ2_S:
            case IQ1_M:
            case IQ1_S:
                return true;
            case Q2_K:
                return ftype == QuantizeFtype.Q2_K_S;
            default:
                return false;
        }
    }
}

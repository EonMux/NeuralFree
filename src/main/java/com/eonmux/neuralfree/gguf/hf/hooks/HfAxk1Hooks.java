package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Hugging Face AXK-1 conversion (vLLM's {@code AXK1ForCausalLM}, AXK1.py): the DeepSeek-V2 multi-head latent
 * attention and routed experts ({@link HfDeepseekHooks}) with two additions. Every layer applies vLLM's
 * {@code post_mlp_layernorm} to the branch output, but only on the sparse (mixture-of-experts) layers, so those
 * tensors become {@code blk.N.post_ffw_norm} and the dense layers' copies are dropped. The {@code llama_4_scaling}
 * object becomes the position-dependent query temperature
 * {@code 1 + beta * log(1 + floor(pos / original_max_position_embeddings))}.
 *
 * <p>The mixture's layer types are the checkpoint's {@code mlp_layer_types}, or the contiguous sparse block after
 * {@code first_k_dense_replace} (with {@code moe_layer_freq} 1); the interleaved layouts vLLM allows through
 * {@code mlp_layer_types} cannot be expressed as one dense lead and are refused by name.</p>
 */
public final class HfAxk1Hooks {

    /** {@code model.layers.N.post_mlp_layernorm.weight} (AXK1.py:648, 694-695). */
    private static final Pattern POST_MLP_NORM =
            Pattern.compile("model\\.layers\\.(\\d+)\\.post_mlp_layernorm\\.weight");

    /** The per-layer arrangement and attention temperature of one checkpoint. Immutable. */
    static final class Layout {
        final boolean[] sparse;
        final int leadingDense;
        final int qLoraRank;
        final boolean hasTemperature;
        final double beta;
        final int originalMaxPositions;

        Layout(boolean[] sparse, int leadingDense, int qLoraRank, boolean hasTemperature, double beta,
               int originalMaxPositions) {
            this.sparse = sparse;
            this.leadingDense = leadingDense;
            this.qLoraRank = qLoraRank;
            this.hasTemperature = hasTemperature;
            this.beta = beta;
            this.originalMaxPositions = originalMaxPositions;
        }

        boolean sparse(int il) {
            return il >= 0 && il < sparse.length && sparse[il];
        }
    }

    private HfAxk1Hooks() {
    }

    /** AXK-1 has no llama.cpp converter, so it carries its own GGUF architecture name. */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.AXK1;
    }

    /**
     * The DeepSeek-V2 metadata with the AXK-1 overrides: the computed dense lead, the Q LoRA rank (0 when the
     * checkpoint has no {@code q_a_proj}), sigmoid routing when the configuration names no
     * {@code scoring_func}, and the {@code llama_4_scaling} temperature.
     *
     * @throws GgufFormatException naming the unsupported field
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        Layout l = layout(ctx);
        HfDeepseekHooks.meta(ctx, kv, plan);
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, l.leadingDense);
        kv.set(GgufKeys.ATTENTION_Q_LORA_RANK, l.qLoraRank);
        if (!ctx.config().find("scoring_func").isPresent()) {
            kv.set(GgufKeys.EXPERT_GATING_FUNC, GatingFunc.SIGMOID.ggufCode());
        }
        if (l.hasTemperature) {
            kv.set(GgufKeys.ATTENTION_TEMPERATURE_SCALE, l.beta);
            kv.set(GgufKeys.ATTENTION_TEMPERATURE_LENGTH, l.originalMaxPositions);
        }
    }

    /**
     * The DeepSeek-V2 tensor step with the post-MLP norm routed by layer type: a sparse layer's
     * {@code post_mlp_layernorm} becomes {@code post_ffw_norm}; a dense layer's copy is unused by vLLM's forward and
     * is dropped with that reason.
     *
     * @throws GgufFormatException as the DeepSeek-V2 step does
     * @throws IOException         as the DeepSeek-V2 step does
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        Matcher m = POST_MLP_NORM.matcher(hfName);
        if (m.matches()) {
            int il = Integer.parseInt(m.group(1));
            Layout l = layout(ctx);
            if (il < l.sparse.length && !l.sparse(il)) {
                out.drop(hfName, "AXK-1 applies post_mlp_layernorm only on sparse layers; this dense layer's copy"
                        + " is unused");
                return;
            }
        }
        HfDeepseekHooks.tensors(ctx, hfName, src, out);
    }

    /** The DeepSeek-V2 tensor filter. */
    public static String filter(String name, UnaryOperator<String> base) {
        return HfDeepseekHooks.filter(name, base);
    }

    /** The DeepSeek-V2 completeness check for the routed expert sets. */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        HfDeepseekHooks.extra(ctx, out);
    }

    /** The DeepSeek-V2 vocabulary. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfDeepseekHooks.vocab(ctx, pre);
    }

    /** The parsed layout of {@code ctx}, built once per load. */
    static Layout layout(HfLoadContext ctx) {
        return ctx.state(HfAxk1Hooks.class.getName() + ".layout", Layout.class,
                () -> parse(ctx.config(), ctx.blockCount()));
    }

    /**
     * Parses and validates the AXK-1 configuration.
     *
     * @throws GgufFormatException naming the unsupported field
     */
    static Layout parse(HfConfig cfg, int blocks) {
        if (cfg.findBool(false, "attention_bias")) {
            throw invalid(cfg, "attention_bias", "biased attention projections are unsupported");
        }
        Optional<String> act = cfg.findString("hidden_act");
        if (act.isPresent() && !"silu".equals(act.get()) && !"swish".equals(act.get())) {
            throw invalid(cfg, "hidden_act", "must be 'silu' (vLLM AXK-1 supports no other activation), got '"
                    + act.get() + "'");
        }
        int nope = cfg.findInt(0, "qk_nope_head_dim");
        if (nope <= 0) {
            throw invalid(cfg, "qk_nope_head_dim", "must be positive for the MLA graph, got " + nope
                    + " (the plain multi-head AXK-1 variant is unsupported)");
        }
        int rope = cfg.findInt(0, "qk_rope_head_dim");
        if (rope <= 0) {
            throw invalid(cfg, "qk_rope_head_dim", "must be positive for the MLA graph, got " + rope
                    + " (the plain multi-head AXK-1 variant is unsupported)");
        }

        int nRouted = cfg.requireInt("n_routed_experts");
        if (nRouted <= 0) {
            throw invalid(cfg, "n_routed_experts", "must be positive, got " + nRouted);
        }
        int topK = cfg.requireInt("num_experts_per_tok");
        if (topK <= 0 || topK > nRouted) {
            throw invalid(cfg, "num_experts_per_tok", "must be in [1, " + nRouted + "], got " + topK);
        }
        int nGroup = cfg.requireInt("n_group");
        int topkGroup = cfg.requireInt("topk_group");
        if (nGroup <= 0 || nRouted % nGroup != 0) {
            throw invalid(cfg, "n_group", "must be a positive divisor of n_routed_experts " + nRouted + ", got "
                    + nGroup);
        }
        if (topkGroup <= 0 || topkGroup > nGroup) {
            throw invalid(cfg, "topk_group", "must be in [1, " + nGroup + "], got " + topkGroup);
        }

        boolean[] sparse = layerTypes(cfg, blocks);

        int qLoraRank = 0;
        OptionalInt qLora = cfg.findInt("q_lora_rank");
        if (qLora.isPresent()) {
            qLoraRank = qLora.getAsInt();
            if (qLoraRank <= 0) {
                throw invalid(cfg, "q_lora_rank", "must be positive when present, got " + qLoraRank);
            }
        }

        Object scalingRaw = cfg.find("llama_4_scaling").orElse(null);
        boolean hasTemperature = scalingRaw != null;
        double beta = 0;
        int originalMaxPositions = 0;
        if (hasTemperature) {
            if (!(scalingRaw instanceof Map)) {
                throw invalid(cfg, "llama_4_scaling",
                        "must be an object with beta and original_max_position_embeddings");
            }
            Map<?, ?> scaling = (Map<?, ?>) scalingRaw;
            Object betaRaw = scaling.get("beta");
            if (!(betaRaw instanceof Number) || !Double.isFinite(((Number) betaRaw).doubleValue())) {
                throw invalid(cfg, "llama_4_scaling.beta", "must be a finite number, got " + betaRaw);
            }
            beta = ((Number) betaRaw).doubleValue();
            Object origRaw = scaling.get("original_max_position_embeddings");
            if (!(origRaw instanceof Number) || ((Number) origRaw).doubleValue() != ((Number) origRaw).longValue()
                    || ((Number) origRaw).longValue() <= 0
                    || ((Number) origRaw).longValue() > Integer.MAX_VALUE) {
                throw invalid(cfg, "llama_4_scaling.original_max_position_embeddings",
                        "must be a positive integer, got " + origRaw);
            }
            originalMaxPositions = ((Number) origRaw).intValue();
        }

        return new Layout(sparse, leadingDense(sparse), qLoraRank, hasTemperature, beta, originalMaxPositions);
    }

    /**
     * vLLM's {@code _get_mlp_layer_types} (AXK1.py:98-107): {@code mlp_layer_types} when present, else a sparse
     * layer at every index past {@code first_k_dense_replace} that {@code moe_layer_freq} selects. Only the
     * contiguous dense lead the engine models is accepted.
     */
    private static boolean[] layerTypes(HfConfig cfg, int blocks) {
        boolean[] sparse = new boolean[blocks];
        Object raw = cfg.find("mlp_layer_types").orElse(null);
        if (raw != null) {
            if (!(raw instanceof List)) {
                throw invalid(cfg, "mlp_layer_types", "must be an array of 'dense' and 'sparse' entries");
            }
            List<?> types = (List<?>) raw;
            if (types.size() < blocks) {
                throw invalid(cfg, "mlp_layer_types", "has " + types.size() + " entries, fewer than the " + blocks
                        + " layers");
            }
            boolean seenSparse = false;
            for (int il = 0; il < blocks; il++) {
                Object type = types.get(il);
                if ("sparse".equals(type)) {
                    seenSparse = true;
                    sparse[il] = true;
                } else if ("dense".equals(type)) {
                    if (seenSparse) {
                        throw invalid(cfg, "mlp_layer_types", "layer " + il
                                + " is dense after a sparse layer; only one dense lead is supported");
                    }
                } else {
                    throw invalid(cfg, "mlp_layer_types", "layer " + il + " is '" + type
                            + "'; only 'dense' and 'sparse' are supported");
                }
            }
        } else {
            int firstDense = cfg.requireInt("first_k_dense_replace");
            int freq = cfg.findInt(1, "moe_layer_freq");
            if (freq != 1) {
                throw invalid(cfg, "moe_layer_freq", "must be 1 (every layer after the dense lead is sparse), got "
                        + freq);
            }
            if (firstDense < 0 || firstDense >= blocks) {
                throw invalid(cfg, "first_k_dense_replace", "must be in [0, " + (blocks - 1) + "], got " + firstDense);
            }
            for (int il = firstDense; il < blocks; il++) {
                sparse[il] = true;
            }
        }
        int leading = leadingDense(sparse);
        if (leading >= blocks) {
            throw invalid(cfg, "mlp_layer_types", "every layer is dense, so the routed experts have no sparse layer"
                    + " to run on");
        }
        return sparse;
    }

    private static int leadingDense(boolean[] sparse) {
        int n = 0;
        while (n < sparse.length && !sparse[n]) {
            n++;
        }
        return n;
    }

    private static GgufFormatException invalid(HfConfig cfg, String field, String message) {
        return new GgufFormatException(cfg.source() + ": '" + field + "': " + message);
    }
}

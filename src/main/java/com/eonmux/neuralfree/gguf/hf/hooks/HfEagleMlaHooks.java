package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.QkvLayouts;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The DeepSeek-MTP-shaped EAGLE draft vLLM serves with {@code deepseek_eagle.EagleDeepseekV3ForCausalLM}
 * (V/model_executor/models/registry.py:724, the {@code EagleDeepSeekMTPModel} name): the single decoder layer and the
 * {@code [enorm(embedding) | hnorm(target hidden)] -> fc} fusion of vLLM's {@code deepseek_eagle.DeepseekV2Model},
 * whose attention is DeepSeek's multi-head latent attention.
 *
 * <p>The draft runs on the {@link com.eonmux.neuralfree.gguf.arch.Arch#EAGLE} row through
 * {@link HfEagle1Hooks}; the name is shared with the llama- and mistral-shaped EAGLE-1 drafts, so
 * {@link HfEagle1Hooks} routes a checkpoint that declares the MLA geometry ({@link #isMla}) here and keeps the
 * other drafts on its own mapping. This class converts the DeepSeek tensors:
 *
 * <ul>
 *   <li>the low-rank query path {@code q_a_proj -> q_a_layernorm -> q_b_proj}, or a plain {@code q_proj} when the
 *       configuration has no {@code q_lora_rank};</li>
 *   <li>the latent KV path {@code kv_a_proj_with_mqa -> kv_a_layernorm}, whose {@code kv_b_proj} the converter
 *       splits into the absorbed per-head {@code attn_k_b} / {@code attn_v_b} ({@code deepseek.py:434-450});</li>
 *   <li>the fusion input norms {@code enorm} / {@code hnorm} and the fixed {@code fc} fusion weight;</li>
 *   <li>the routed experts of the draft layer ({@code mlp.experts.<x>.*} stacked into the per-layer expert
 *       tensors), its shared expert and its router, before the remaining layer tensors map as the llama-shaped
 *       ones do;</li>
 *   <li>the DeepSeek MLA metadata: {@code kv_lora_rank}, the decompressed key and value head widths,
 *       {@code rope.dimension_count = qk_rope_head_dim}, the YaRN log multiplier {@code 0.1 * mscale_all_dim} and
 *       the MoE keys of an expert draft.</li>
 * </ul>
 *
 * <p>Weight names follow vLLM's loader, which prefixes {@code model.} to every weight that is not the LM head
 * (deepseek_eagle.py:248-254); the checkpoint stores them without that prefix. The layer index of the draft's one
 * decoder layer becomes block 0. The variants the engine does not run are refused by name: a tied output head, a
 * checkpoint whose attending head count differs from its KV head count, and a query path that disagrees with the
 * configured Q LoRA rank (a plain {@code q_proj} in a checkpoint that declares {@code q_lora_rank}, or the low-rank
 * projections without one).
 */
public final class HfEagleMlaHooks {

    /** The query and attention output projections of the draft layer, without the Q/K permutation of a Llama draft. */
    private static final Map<String, String> MLA_TENSORS = mlaTensors();

    /** The draft's own tensors whose GGUF names this class writes directly. */
    private static final String FC_WEIGHT = "model.fc.weight";
    private static final String ENORM_WEIGHT = "model.enorm.weight";
    private static final String HNORM_WEIGHT = "model.hnorm.weight";
    private static final String EMBED_WEIGHT = "model.embed_tokens.weight";
    private static final String LM_HEAD_WEIGHT = "lm_head.weight";

    /** A decoder-layer tensor of the draft: {@code model.layers.<n>.<tail>}. */
    private static final Pattern LAYER = Pattern.compile("^model\\.layers\\.(\\d{1,9})\\.(.+)$");

    /** The routed-expert members the converter stacks, in its order (deepseek.py:416). */
    private static final List<String> EXPERT_PARTS = Collections.unmodifiableList(Arrays.asList(
            "down_proj", "gate_proj", "up_proj"));
    private static final String EXPERTS_STATE = HfEagleMlaHooks.class.getName() + ".experts";

    /** The KV LoRA rank DeepSeek's configurations default to (deepseek.py:317). */
    private static final int DEFAULT_KV_LORA_RANK = 512;

    private HfEagleMlaHooks() {
    }

    private static Map<String, String> mlaTensors() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("q_a_proj", TensorNames.ATTN_Q_A);
        m.put("q_a_layernorm", TensorNames.ATTN_Q_A_NORM);
        m.put("q_b_proj", TensorNames.ATTN_Q_B);
        m.put("q_proj", TensorNames.ATTN_Q);
        m.put("kv_a_proj_with_mqa", TensorNames.ATTN_KV_A_MQA);
        m.put("kv_a_layernorm", TensorNames.ATTN_KV_A_NORM);
        m.put("o_proj", TensorNames.ATTN_OUT);
        return Collections.unmodifiableMap(m);
    }

    // ------------------------------------------------------------------ configuration

    /**
     * True when the draft configuration declares DeepSeek multi-head latent attention: it carries the MLA key
     * geometry ({@code kv_lora_rank}, or the {@code qk_nope_head_dim} / {@code qk_rope_head_dim} / {@code v_head_dim}
     * triple the latent attention needs). DeepSeek's configuration classes default {@code kv_lora_rank} to
     * {@value #DEFAULT_KV_LORA_RANK} (deepseek.py:317), so a checkpoint may omit it; the llama- and mistral-shaped
     * EAGLE-1 configurations carry none of these keys.
     */
    public static boolean isMla(HfConfig cfg) {
        return cfg.findInt("kv_lora_rank").isPresent() || cfg.findInt("qk_nope_head_dim").isPresent()
                || cfg.findInt("qk_rope_head_dim").isPresent() || cfg.findInt("v_head_dim").isPresent();
    }

    /**
     * The checks of a DeepSeek-MTP-shaped draft configuration: the MLA head geometry the metadata writes, one
     * attending head count shared by the query and the KV decompression, and no tied output head.
     *
     * @throws GgufFormatException when a key is missing or inconsistent with the layout
     */
    public static void check(HfConfig cfg) {
        String source = cfg.source() + ": " + cfg.architecture() + " ";
        cfg.requireInt("qk_rope_head_dim");
        cfg.requireInt("qk_nope_head_dim");
        cfg.requireInt("v_head_dim");
        if (cfg.findInt(DEFAULT_KV_LORA_RANK, "kv_lora_rank") <= 0) {
            throw new GgufFormatException(source + "needs a positive 'kv_lora_rank'");
        }
        if (cfg.findBool(false, "tie_word_embeddings")) {
            throw new GgufFormatException(source + "ties its output head to the token embedding; the DeepSeek"
                    + " MTP-shaped EAGLE draft keeps its own LM head (vLLM's deepseek_eagle)");
        }
        int nHead = cfg.requireInt("num_attention_heads");
        OptionalInt nKv = cfg.findInt("num_key_value_heads");
        if (nKv.isPresent() && nKv.getAsInt() != nHead) {
            throw new GgufFormatException(source + "has " + nKv.getAsInt() + " KV heads for " + nHead
                    + " attention heads; multi-head latent attention decompresses one KV latent per attention head");
        }
    }

    // ------------------------------------------------------------------ metadata

    /**
     * The generic keys of {@link HfGenericMetadata}, the EAGLE-1 target keys of
     * {@link HfEagle1Hooks#targetKeys}, the DeepSeek MLA geometry and the MoE keys of an expert draft. The MLA keys
     * are written after the generic pass so the head geometry describes the latent attention rather than the
     * {@code hidden_size / num_attention_heads} default.
     *
     * @throws IOException         when the target cannot be opened
     * @throws GgufFormatException when a value has the wrong type, an MLA dimension is missing, or the target lacks
     *                             what the keys need
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfGenericMetadata.apply(ctx, kv);
        HfLlamaHooks.llamaKv(ctx, kv, true);
        int kvLora = cfg.findInt(DEFAULT_KV_LORA_RANK, "kv_lora_rank");
        int qkRope = cfg.requireInt("qk_rope_head_dim");
        int qkNope = cfg.requireInt("qk_nope_head_dim");
        int vHead = cfg.requireInt("v_head_dim");
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, 1);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, kvLora + qkRope);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, kvLora);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH_MLA, qkNope + qkRope);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_MLA, vHead);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, qkRope);
        int qLora = cfg.findInt(0, "q_lora_rank");
        if (qLora > 0) {
            kv.set(GgufKeys.ATTENTION_Q_LORA_RANK, qLora);
        }
        kv.set(GgufKeys.ATTENTION_KV_LORA_RANK, kvLora);
        writeMoe(ctx, kv);
        Object mscaleAllDim = cfg.rope().raw().get("mscale_all_dim");
        if (mscaleAllDim != null) {
            if (!(mscaleAllDim instanceof Number) || mscaleAllDim instanceof Boolean) {
                throw new GgufFormatException(cfg.source() + ": 'rope_parameters.mscale_all_dim' must be a number");
            }
            kv.set(GgufKeys.ROPE_SCALING_YARN_LOG_MUL, 0.1 * ((Number) mscaleAllDim).doubleValue());
        }
        HfEagle1Hooks.targetKeys(ctx, kv);
    }

    /**
     * The MoE keys of a draft layer with routed experts (deepseek.py:360-379): the expert width, the dense lead of a
     * draft whose single layer is MoE, the shared-expert count, the routing scale and the weight normalization.
     */
    private static void writeMoe(HfLoadContext ctx, HfMetadataBuilder kv) {
        HfConfig cfg = ctx.config();
        OptionalInt routed = cfg.findInt("n_routed_experts");
        if (!routed.isPresent() || routed.getAsInt() <= 0) {
            return;
        }
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, cfg.findInt(0, "first_k_dense_replace"));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH,
                cfg.requireInt("moe_intermediate_size", "intermediate_size"));
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, cfg.findInt(0, "n_shared_experts"));
        if (cfg.findFloat("routed_scaling_factor").isPresent()) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, cfg.findFloat("routed_scaling_factor").getAsDouble());
        }
        if (cfg.findBool(false, "norm_topk_prob")) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, true);
        }
    }

    // ------------------------------------------------------------------ tensors

    /**
     * The tensors of the DeepSeek-MTP-shaped layout: the fusion weight and its two input norms, the MLA projections
     * with {@code kv_b_proj} split into the absorbed {@code attn_k_b} / {@code attn_v_b}, the stacked routed experts,
     * and the name map of the remaining tensors after the layer index is reduced to block 0.
     *
     * @throws IOException         when an expert tensor or the target cannot be read
     * @throws GgufFormatException when a tensor has no mapping or {@code kv_b_proj} does not have the MLA shape
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = rename(hfName);
        if (name.equals(FC_WEIGHT)) {
            out.add(ctx.names().ggufName("FC", -1, ".weight"), src);
            return;
        }
        if (name.equals(ENORM_WEIGHT)) {
            out.add(TensorNames.weight(TensorNames.ENC_ENORM), src);
            return;
        }
        if (name.equals(HNORM_WEIGHT)) {
            out.add(TensorNames.weight(TensorNames.ENC_HNORM), src);
            return;
        }
        if (name.equals(EMBED_WEIGHT)) {
            out.add(ctx.names().ggufName("TOKEN_EMBD", -1, ".weight"), src);
            return;
        }
        if (name.equals(LM_HEAD_WEIGHT)) {
            out.add(ctx.names().ggufName("OUTPUT", -1, ".weight"), src);
            return;
        }
        Matcher layer = LAYER.matcher(name);
        String tail = layer.matches() ? layer.group(2) : "";
        if (tail.startsWith("self_attn.kv_b_proj.")) {
            splitKvB(ctx, name, src, out);
            return;
        }
        Matcher projection = MLA_PROJECTION.matcher(tail);
        if (projection.matches()) {
            requireQueryPath(ctx, projection.group(1), hfName);
            out.add(TensorNames.layerName(0, MLA_TENSORS.get(projection.group(1)) + "." + projection.group(2)), src);
            return;
        }
        if (tail.contains("mlp.experts.")) {
            if (!experts(ctx).offer(ctx, hfName, out)) {
                throw new GgufFormatException("Unprocessed experts: tensor '" + hfName + "' is not an expert member"
                        + " <prefix>.experts.<x>.<part> with a part in " + EXPERT_PARTS);
            }
            return;
        }
        out.add(ctx.names().mapOrThrow(name), src);
    }

    /** {@code model.layers.<n>.self_attn.<projection>.(weight|bias)} after the layer index is reduced to block 0. */
    private static final Pattern MLA_PROJECTION = Pattern.compile(
            "^self_attn\\.(q_a_proj|q_a_layernorm|q_b_proj|q_proj|kv_a_proj_with_mqa|kv_a_layernorm|o_proj)"
                    + "\\.(weight|bias)$");

    /**
     * The checkpoint's query path must match the configured Q LoRA rank: the low-rank {@code q_a_proj} /
     * {@code q_a_layernorm} / {@code q_b_proj} exist exactly when {@code q_lora_rank} is positive, and a plain
     * {@code q_proj} exists otherwise (deepseek.py:317-322).
     *
     * @throws GgufFormatException naming the tensor and the configuration key that disagrees with it
     */
    private static void requireQueryPath(HfLoadContext ctx, String projection, String hfName) {
        int rank = ctx.config().findInt(0, "q_lora_rank");
        boolean lowRank = projection.equals("q_a_proj") || projection.equals("q_a_layernorm")
                || projection.equals("q_b_proj");
        if (lowRank && rank <= 0) {
            throw new GgufFormatException("tensor '" + hfName + "' stores the low-rank query path, but the"
                    + " configuration has no positive 'q_lora_rank'");
        }
        if (!lowRank && projection.equals("q_proj") && rank > 0) {
            throw new GgufFormatException("tensor '" + hfName + "' is a plain query projection, but the"
                    + " configuration declares 'q_lora_rank' = " + rank + "; the checkpoint needs q_a_proj,"
                    + " q_a_layernorm and q_b_proj");
        }
    }

    /**
     * vLLM's weight transform (deepseek_eagle.py:248-254) with the single decoder layer as block 0: every weight but
     * the LM head takes the {@code model.} prefix, and the draft's one layer is {@code model.layers.0}.
     */
    static String rename(String hfName) {
        String name = hfName.equals(LM_HEAD_WEIGHT) || hfName.equals("lm_head.bias")
                ? hfName : hfName.startsWith("model.") ? hfName : "model." + hfName;
        Matcher layer = LAYER.matcher(name);
        if (layer.matches() && !"0".equals(layer.group(1))) {
            return "model.layers.0." + layer.group(2);
        }
        return name;
    }

    /**
     * {@code DeepseekV2Model.modify_tensors}' {@code kv_b_proj} split (deepseek.py:434-450): the checkpoint's
     * {@code [n_head * (qk_nope + v_head), kv_lora]} tensor to the absorbed per-head {@code attn_k_b} and
     * {@code attn_v_b} the MLA mixer reads.
     *
     * @throws GgufFormatException when the tensor is not rank 2 or does not split into the configured heads
     */
    private static void splitKvB(HfLoadContext ctx, String name, RowSource src, TensorSink out) {
        HfConfig cfg = ctx.config();
        long[] shape = src.shape();
        if (shape.length != 2) {
            throw new GgufFormatException("tensor '" + name + "' has shape " + Arrays.toString(shape)
                    + "; kv_b_proj must have rank 2");
        }
        int nHead = cfg.requireInt("num_attention_heads");
        int qkNope = cfg.requireInt("qk_nope_head_dim");
        int vHead = cfg.requireInt("v_head_dim");
        RowSource[] kvB;
        try {
            kvB = QkvLayouts.mlaKvB(src, nHead, qkNope, vHead, Math.toIntExact(shape[1]));
        } catch (IllegalArgumentException | ArithmeticException e) {
            throw new GgufFormatException("tensor '" + name + "' does not split into " + nHead + " heads of "
                    + qkNope + " key and " + vHead + " value rows: " + e.getMessage(), e);
        }
        out.add(TensorNames.layerName(0, TensorNames.weight(TensorNames.ATTN_K_B)), kvB[0]);
        out.add(TensorNames.layerName(0, TensorNames.weight(TensorNames.ATTN_V_B)), kvB[1]);
    }

    /**
     * The per-load routed-expert collector: the parts of every expert, stacked into the merged layer tensors under
     * the {@code model.}-prefixed name the map resolves ({@link #rename} of the collector's merged name).
     */
    private static HfHookSupport.ExpertCollector experts(HfLoadContext ctx) {
        return ctx.state(EXPERTS_STATE, HfHookSupport.ExpertCollector.class, () -> new HfHookSupport.ExpertCollector(
                ctx.config().requireInt("n_routed_experts"), EXPERT_PARTS,
                (prefix, part, tail) -> rename(prefix + "." + part + tail)));
    }

    /**
     * {@code prepare_tensors}' expert completeness check: every expert set that was started must have been stacked.
     *
     * @throws GgufFormatException naming the first incomplete expert set
     */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        if (ctx.config().findInt(0, "n_routed_experts") > 0) {
            experts(ctx).requireComplete();
        }
    }
}

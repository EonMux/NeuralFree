package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The hooks of vLLM's Step3 text decoder ({@code vllm/model_executor/models/step3_text.py}). The attention fuses Q,
 * K and V into one {@code qkv_proj} whose query part is only {@code share_q_dim} wide; an RMS post-norm
 * ({@code inter_norm}) then widens it per head ({@code wq}) before RoPE. Every MoE layer adds a plain shared expert
 * to the routed experts. The checkpoint stores the routed experts already stacked, so the standard expert tensor
 * patterns apply.
 */
public final class HfStep3TextHooks {

    private static final Pattern LAYER = Pattern.compile("^model\\.layers\\.(\\d+)\\..*$");

    private HfStep3TextHooks() {
    }

    /**
     * The values the vLLM class reads but the checkpoint omits or names differently: the single key/value head
     * ({@code num_attention_groups} is one and the class requires one KV head) and the MoE counts under their
     * llama.cpp keys. A non-{@code silu} activation is refused by name.
     */
    public static HfConfig fixup(HfConfig cfg) {
        String act = cfg.findString("hidden_act", "activation_function").orElse("silu");
        if (!"silu".equalsIgnoreCase(act)) {
            throw new GgufFormatException(cfg.source() + ": 'hidden_act' = '" + act
                    + "' is not supported: Step3 text uses the gated SwiGLU feed-forward");
        }
        int groups = cfg.findInt(1, "num_attention_groups", "num_key_value_heads");
        if (groups != 1) {
            throw new GgufFormatException(cfg.source() + ": 'num_attention_groups' is " + groups
                    + "; Step3 text attention requires a single key/value head");
        }
        Map<String, Object> derived = new LinkedHashMap<>();
        derived.put("num_key_value_heads", 1L);
        if (!cfg.find("n_routed_experts").isPresent() && cfg.find("moe_num_experts").isPresent()) {
            derived.put("n_routed_experts", cfg.find("moe_num_experts").get());
        }
        if (!cfg.find("num_experts_per_tok").isPresent() && cfg.find("moe_top_k").isPresent()) {
            derived.put("num_experts_per_tok", cfg.find("moe_top_k").get());
        }
        return cfg.withFlat(derived);
    }

    /** The entry of the Step3 text checkpoints. */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.STEP3_TEXT;
    }

    /**
     * {@code Step3TextModel} metadata: the generic keys, the exact vocabulary, the per-head key/value widths of
     * {@code head_dim}, the shared query width and, when the checkpoint has routed experts, the routed and shared
     * expert counts.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        int ctxLength = cfg.findInt(0, "max_seq_len", "max_position_embeddings", "max_position_embedding");
        if (ctxLength > 0 && !kv.has(GgufKeys.CONTEXT_LENGTH)) {
            kv.set(GgufKeys.CONTEXT_LENGTH, ctxLength);
        }
        int headDim = cfg.requireInt("head_dim");
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_Q_LORA_RANK, cfg.requireInt("share_q_dim"));
        if (cfg.findInt(0, "moe_num_experts", "n_routed_experts") <= 0) {
            return;
        }
        kv.set(GgufKeys.EXPERT_COUNT, cfg.requireInt("moe_num_experts", "n_routed_experts"));
        kv.set(GgufKeys.EXPERT_USED_COUNT, cfg.requireInt("moe_top_k", "num_experts_per_tok"));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("moe_intermediate_size"));
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, 1);
        kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, cfg.requireInt("share_expert_dim"));
        if (cfg.findBool(false, "norm_expert_weight")) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, true);
        }
    }

    /**
     * {@code Step3TextModel.load_weights}: the checkpoint's {@code q_proj} is the shared query projection, while
     * {@code k_proj} and {@code v_proj} map directly; {@code inter_norm} becomes the shared query norm;
     * {@code wq} widens the query per head. A fused {@code qkv_proj} splits into the same three projections, and a
     * fused {@code gate_up_proj} of a dense layer splits into its gate and up halves. Every other tensor maps with
     * its biases.
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        Matcher m = LAYER.matcher(hfName);
        if (!m.matches()) {
            HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
            return;
        }
        int bid = Integer.parseInt(m.group(1));
        HfConfig cfg = ctx.config();
        if (hfName.endsWith(".self_attn.q_proj.weight")) {
            out.add(ctx.names().ggufName("ATTN_Q_A", bid, ".weight"), src);
            return;
        }
        if (hfName.endsWith(".self_attn.qkv_proj.weight")) {
            splitQkv(ctx, cfg, bid, src, out);
            return;
        }
        if (hfName.endsWith(".self_attn.inter_norm.weight")) {
            out.add(ctx.names().ggufName("ATTN_Q_A_NORM", bid, ".weight"), src);
            return;
        }
        if (hfName.endsWith(".self_attn.wq.weight")) {
            out.add(ctx.names().ggufName("ATTN_Q_B", bid, ".weight"), src);
            return;
        }
        if (hfName.endsWith(".mlp.gate_up_proj.weight")) {
            splitGateUp(ctx, bid, src, out);
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /**
     * The fused {@code [shared query; K; V]} projection, split on its output rows.
     *
     * @throws GgufFormatException when the tensor's output width is not {@code share_q_dim + 2 * head_dim}
     */
    private static void splitQkv(HfLoadContext ctx, HfConfig cfg, int bid, RowSource src, TensorSink out) {
        int share = cfg.requireInt("share_q_dim");
        int headDim = cfg.requireInt("head_dim");
        RowSource[] parts;
        try {
            parts = RowTransforms.splitRows(src, share, headDim, headDim);
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException("tensor 'self_attn.qkv_proj.weight' of layer " + bid
                    + " does not hold share_q_dim " + share + " + 2 x head_dim " + headDim + " rows: "
                    + e.getMessage(), e);
        }
        out.add(ctx.names().ggufName("ATTN_Q_A", bid, ".weight"), parts[0]);
        out.add(ctx.names().ggufName("ATTN_K", bid, ".weight"), parts[1]);
        out.add(ctx.names().ggufName("ATTN_V", bid, ".weight"), parts[2]);
    }

    /** The dense layer's {@code gate_up_proj}, split into its two equal halves. */
    private static void splitGateUp(HfLoadContext ctx, int bid, RowSource src, TensorSink out) {
        long[] shape = src.shape();
        long rows = shape[0];
        if (shape.length != 2 || rows % 2 != 0) {
            throw new GgufFormatException("tensor 'mlp.gate_up_proj.weight' of layer " + bid + " has " + rows
                    + " output rows, which is not two equal gate and up halves");
        }
        RowSource[] parts = RowTransforms.splitRows(src, rows / 2, rows / 2);
        out.add(ctx.names().ggufName("FFN_GATE", bid, ".weight"), parts[0]);
        out.add(ctx.names().ggufName("FFN_UP", bid, ".weight"), parts[1]);
    }
}

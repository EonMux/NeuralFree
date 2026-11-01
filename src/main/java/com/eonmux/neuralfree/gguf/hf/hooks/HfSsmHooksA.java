package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * Hooks of the plain hybrid tail state-space-model converter classes: {@code FalconH1Model}
 * (LC/conversion/falcon_h1.py:16-119), {@code JambaModel} (jamba.py:15-120) and {@code Plamo2Model}
 * (plamo.py:63-148), the second half of the {@link HfSsmHooks} sub-family shared with the Mamba classes
 * (mamba.py) there.
 *
 * <p>Every method is a public static hook reference that subclasses compose where the converter calls
 * {@code super()}. The {@code mamba_}-prefixed hyperparameter lookups live on {@link HfSsmHooks}.</p>
 */
public final class HfSsmHooksA {

    private HfSsmHooksA() {
    }

    // ------------------------------------------------------------------ FalconH1Model (falcon_h1.py:16)

    /** {@link Arch#FALCON_H1}. */
    public static Arch falconH1Arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.FALCON_H1;
    }

    /**
     * {@code FalconH1Model.set_gguf_parameters} (falcon_h1.py:98-119) over the Mamba-2 rows it runs via
     * {@code super()}: the vocabulary size, the overridden context length ({@code max_position_embeddings}, 0 when
     * absent) and feed-forward length, the attention heads and widths, the silu-only activation check, the SSM head
     * divisibility and the RoPE base of the rope parameters. The expansion check of Mamba-2 does not run here, as
     * the converter skips it for FALCON_H1 (mamba.py:151).
     */
    public static void falconH1Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        long dModel = cfg.requireInt("hidden_size", "d_model", "mamba_hidden_size", "mamba_d_model",
                "mamba_dim");
        long dInner = fusedInt(ctx, 0, "mamba_d_ssm", "intermediate_size", "d_inner");
        if (dInner == 0) {
            dInner = 2L * dModel;
        }
        long headDim = fusedInt(ctx, 64, "mamba_d_head", "head_dim");
        long dHead = fusedRequireInt(ctx, "d_head");
        long[] group = HfSsmHooks.groupShape(cfg, dInner, "n_groups", "mamba_n_groups");
        OptionalDouble eps = cfg.findFloat("layer_norm_epsilon", "rms_norm_eps");
        kv.set(GgufKeys.CONTEXT_LENGTH, HfSsmHooks.DEFAULT_CONTEXT_LENGTH);
        kv.set(GgufKeys.EMBEDDING_LENGTH, dModel);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, 0L);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, 0L);
        kv.set(GgufKeys.BLOCK_COUNT, (long) ctx.blockCount());
        kv.set(GgufKeys.SSM_CONV_KERNEL, (long) fusedInt(ctx, 4, "conv_kernel", "d_conv"));
        kv.set(GgufKeys.SSM_INNER_SIZE, dInner);
        kv.set(GgufKeys.SSM_STATE_SIZE, (long) fusedInt(ctx, 128, "state_size", "d_state"));
        kv.set(GgufKeys.SSM_TIME_STEP_RANK, dInner / headDim);
        kv.set(GgufKeys.SSM_GROUP_COUNT, group[0]);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, eps.isPresent() ? eps.getAsDouble() : 1e-5);
        kv.set(GgufKeys.VOCAB_SIZE, (long) cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.CONTEXT_LENGTH, (long) cfg.findInt(0, "max_position_embeddings"));
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, (long) fusedRequireInt(ctx, "intermediate_size"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, (long) cfg.requireInt("num_attention_heads"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, (long) cfg.requireInt("num_key_value_heads"));
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, headDim);
        Optional<String> act = cfg.findString("hidden_act");
        if (act.isPresent() && !"silu".equals(act.get())) {
            throw new GgufFormatException(cfg.source() + ": Only SILU activation supported, hidden_act '"
                    + act.get() + "' (falcon_h1.py:115)");
        }
        if (dInner % dHead != 0) {
            throw new GgufFormatException(cfg.source() + ": SSM inner size " + dInner + " is not a multiple of"
                    + " head dim " + dHead + " (falcon_h1.py:116)");
        }
        OptionalDouble theta = cfg.rope().global().theta();
        if (!theta.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": rope_parameters.rope_theta is required"
                    + " (falcon_h1.py:119)");
        }
        kv.set(GgufKeys.ROPE_FREQ_BASE, theta.getAsDouble());
    }

    /** The first present of {@code keys} and their {@code mamba_}-prefixed spellings, or {@code dflt}. */
    static int fusedInt(HfLoadContext ctx, int dflt, String... keys) {
        return HfSsmHooks.prefixedInt(ctx.config(), dflt, keys);
    }

    /** As {@link #fusedInt}, required. */
    static int fusedRequireInt(HfLoadContext ctx, String... keys) {
        return HfSsmHooks.prefixedRequireInt(ctx.config(), keys);
    }

    /**
     * {@code FalconH1Model.modify_tensors} (falcon_h1.py:60-96) over {@link HfSsmHooks#mamba2Tensor}: the folded
     * multipliers, whose names are checked in the converter's order ({@code down_proj} beats {@code gate_proj}
     * beats {@code q}/{@code k}/{@code v}/{@code o_proj} beats {@code out_proj}/{@code in_proj} beats the
     * embedding/output heads), the {@code in_proj} row segments of {@code ssm_multipliers}, and the grouped Mamba
     * norm reshape, which the Mamba-2 path computed with the sibling sizes already.
     */
    public static void falconH1Tensor(HfLoadContext ctx, String name, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(name)) {
            return;
        }
        RowSource s = src;
        if (name.contains("down_proj")) {
            s = ElementTransforms.mulScalar(s, fusedMultipliers(ctx, "mlp_multipliers")[1]);
        } else if (name.contains("gate_proj")) {
            s = ElementTransforms.mulScalar(s, fusedMultipliers(ctx, "mlp_multipliers")[0]);
        } else if (name.contains("k_proj")) {
            s = ElementTransforms.mulScalar(s, fusedFloat(ctx, "key_multiplier")
                    * fusedFloat(ctx, "attention_in_multiplier"));
        } else if (name.contains("q_proj")) {
            s = ElementTransforms.mulScalar(s, fusedFloat(ctx, "attention_in_multiplier"));
        } else if (name.contains("v_proj")) {
            s = ElementTransforms.mulScalar(s, fusedFloat(ctx, "attention_in_multiplier"));
        } else if (name.contains("o_proj")) {
            s = ElementTransforms.mulScalar(s, fusedFloat(ctx, "attention_out_multiplier"));
        } else if (name.contains("out_proj")) {
            s = ElementTransforms.mulScalar(s, fusedFloat(ctx, "ssm_out_multiplier"));
        } else if (name.contains("in_proj")) {
            s = ElementTransforms.mulScalar(s, fusedFloat(ctx, "ssm_in_multiplier"));
            s = ElementTransforms.mulRowsBy(s, ssmRowSegments(ctx, name, s));
        } else if (name.contains("lm_head")) {
            s = ElementTransforms.mulScalar(s, requiredFloat(ctx, "lm_head_multiplier"));
        } else if (name.contains("embed_tokens")) {
            s = ElementTransforms.mulScalar(s, requiredFloat(ctx, "embedding_multiplier"));
        } else if (name.contains("mamba.norm")) {
            s = ElementTransforms.reshape(s, falconNormShape(ctx));
        }
        HfSsmHooks.mamba2Tensor(ctx, name, s, out);
    }

    /** One configured float value; absent is neutral (1) as the converter's optional {@code find_hparam}. */
    private static float fusedFloat(HfLoadContext ctx, String name) {
        OptionalDouble d = ctx.config().findFloat(HfSsmHooks.prefixedAliases("mamba", name).toArray(new String[0]));
        return d.isPresent() ? (float) d.getAsDouble() : 1f;
    }

    /** One configured float value, required. */
    private static float requiredFloat(HfLoadContext ctx, String name) {
        OptionalDouble d = ctx.config().findFloat(name);
        if (!d.isPresent()) {
            throw new GgufFormatException(ctx.config().source() + ": '" + name + "' is required (falcon_h1.py:89)");
        }
        return (float) d.getAsDouble();
    }

    /** The multipliers list {@code name}, required when its tensors are folded (falcon_h1.py:42-43). */
    private static float[] fusedMultipliers(HfLoadContext ctx, String name) {
        Object v = ctx.config().find(name).orElse(null);
        if (!(v instanceof List)) {
            throw new GgufFormatException(ctx.config().source() + ": '" + name
                    + "' is a required list of multipliers");
        }
        List<?> items = (List<?>) v;
        float[] out = new float[items.size()];
        for (int i = 0; i < out.length; i++) {
            if (!(items.get(i) instanceof Number)) {
                throw new GgufFormatException(ctx.config().source() + ": '" + name + "[" + i + "]' must be a"
                        + " number, got " + items.get(i));
            }
            out[i] = (float) ((Number) items.get(i)).doubleValue();
        }
        return out;
    }

    /**
     * The five {@code ssm_multipliers} row segments of {@code in_proj} (falcon_h1.py:80-87), as
     * {@code mamba_d_ssm}, {@code mamba_n_groups * mamba_d_state} counts of rows each.
     */
    private static float[] ssmRowSegments(HfLoadContext ctx, String name, RowSource src) {
        HfConfig cfg = ctx.config();
        float[] m = fusedMultipliers(ctx, "ssm_multipliers");
        long dInner = cfg.findInt(0, "mamba_d_ssm");
        long gts = cfg.findInt(0, "mamba_n_groups") * cfg.findInt(0, "mamba_d_state");
        long d0 = src.shape()[0];
        long tail = d0 - (2L * dInner + 2L * gts);
        if ((2L * dInner + 2L * gts) > d0 || tail < 0) {
            throw new GgufFormatException(cfg.source() + ": tensor '" + name + "' has " + d0
                    + " rows, never matching 2 * mamba_d_ssm " + dInner + " and 2 * groups*state " + gts);
        }
        float[] factors = new float[(int) d0];
        fill(factors, 0, dInner, m[0]);
        fill(factors, dInner, dInner, m[1]);
        fill(factors, 2L * dInner, gts, m[2]);
        fill(factors, 2L * dInner + gts, gts, m[3]);
        fill(factors, 2L * dInner + 2L * gts, tail, m[4]);
        return factors;
    }

    private static void fill(float[] a, long from, long count, float value) {
        for (int i = 0; i < count; i++) {
            a[(int) (from + i)] = value;
        }
    }

    /** The {@code (mamba_n_groups, mamba_d_ssm/mamba_n_groups)} grouped-norm shape of Falcon-H1. */
    private static long[] falconNormShape(HfLoadContext ctx) {
        HfConfig cfg = ctx.config();
        long dInner = cfg.findInt(0, "mamba_d_ssm");
        long[] group = HfSsmHooks.groupShape(cfg, dInner, "n_groups", "mamba_n_groups");
        return new long[]{group[0], dInner / group[0]};
    }

    // ------------------------------------------------------------------ JambaModel (jamba.py:15)

    /** {@link Arch#JAMBA}. */
    public static Arch jambaArch(HfConfig cfg, HfTensorIndex index) {
        return Arch.JAMBA;
    }

    /**
     * {@code JambaModel.set_gguf_parameters} (jamba.py:25-55): the Mamba sizes ({@code d_inner} is {@code
     * mamba_expand * d_model}, both required), the attention layers' head count from {@code attn_layer_offset}/
     * {@code attn_layer_period} (:38-40), and the expert counts with the Jamba defaults (d_conv 4, d_state 16,
     * {@code dt_rank} {@code ceil(d_model / 16)}, epsilon {@code 1e-6}). The class does not extend the generic
     * metadata.
     */
    public static void jambaMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        long dModel = cfg.requireInt("hidden_size", "mamba_d_model");
        long dInner = (long) cfg.requireInt("mamba_expand") * dModel;
        long dtRank = cfg.findInt(0, "mamba_dt_rank");
        if (dtRank == 0) {
            dtRank = HfSsmHooks.ceilDiv((int) dModel, 16);
        }
        OptionalDouble eps = cfg.findFloat("layer_norm_epsilon", "rms_norm_eps");
        kv.set(GgufKeys.BLOCK_COUNT, (long) ctx.blockCount());
        kv.set(GgufKeys.CONTEXT_LENGTH, (long) cfg.requireInt("max_position_embeddings", "n_ctx"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, dModel);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, (long) cfg.requireInt("intermediate_size"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, (long) cfg.requireInt("num_attention_heads"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, attentionLayers(cfg, ctx.blockCount()));
        kv.set(GgufKeys.SSM_CONV_KERNEL, (long) cfg.findInt(4, "mamba_d_conv"));
        kv.set(GgufKeys.SSM_INNER_SIZE, dInner);
        kv.set(GgufKeys.SSM_STATE_SIZE, (long) cfg.findInt(16, "mamba_d_state"));
        kv.set(GgufKeys.SSM_TIME_STEP_RANK, dtRank);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, eps.isPresent() ? eps.getAsDouble() : 1e-6);
        kv.set(GgufKeys.EXPERT_COUNT, (long) cfg.requireInt("num_local_experts", "num_experts"));
        kv.set(GgufKeys.EXPERT_USED_COUNT, (long) cfg.requireInt("num_experts_per_tok", "num_experts_per_token"));
    }

    /** {@code n_kv_vec} of jamba.py:38-40: zero off the periodic attention layers. */
    private static List<Integer> attentionLayers(HfConfig cfg, int blockCount) {
        int offset = cfg.requireInt("attn_layer_offset");
        int period = cfg.requireInt("attn_layer_period");
        long nKv = cfg.requireInt("num_key_value_heads");
        if (nKv > Integer.MAX_VALUE) {
            throw new GgufFormatException(cfg.source() + ": num_key_value_heads " + nKv + " is out of range");
        }
        List<Integer> out = new ArrayList<>(blockCount);
        for (int i = 0; i < Math.min(offset, blockCount); i++) {
            out.add(0);
        }
        for (int i = offset; i < blockCount; i++) {
            out.add((i - offset) % period == 0 ? (int) nKv : 0);
        }
        return out;
    }

    /**
     * {@code JambaModel.modify_tensors} (jamba.py:59-111): the Mini-Jamba {@code .moe.} → {@code .feed_forward.}
     * rename and, on dense layers, {@code .experts.0.} stripped (jamba.py:62-68); routed experts collected per layer
     * and stacked into {@code mlp.experts.<part>}; the conv squeeze and {@code .A_log} → {@code -exp}, applied by
     * the mapped name's tensor id (jamba.py:104/-107).
     */
    public static void jambaTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        String name = hfName.replace(".moe.", ".feed_forward.");
        int bid = HfLoadContext.bidOf(hfName);
        if (bid >= 0 && !isMoeLayer(cfg, bid)) {
            name = name.replace(".experts.0.", ".");
        }
        if (name.contains(".feed_forward.experts.")) {
            int nExperts = cfg.requireInt("num_local_experts", "num_experts");
            jambaExperts(ctx, nExperts).offer(ctx, name, out);
            return;
        }
        if (out.isConsumed(name)) {
            return;
        }
        RowSource s = name.endsWith(".A_log") ? ElementTransforms.negExp(src) : src;
        String gguf = ctx.names().mapOrThrow(name);
        if (ctx.names().matches(gguf, "SSM_CONV1D", bid, HfSsmHooks.WEIGHT)) {
            s = ElementTransforms.squeezeConv(s);
        }
        out.add(gguf, s);
    }

    /** {@code bid} is an MoE layer when it passes {@code expert_layer_offset}/{@code period} (jamba.py:64-67). */
    private static boolean isMoeLayer(HfConfig cfg, int bid) {
        int offset = cfg.requireInt("expert_layer_offset");
        int period = cfg.requireInt("expert_layer_period");
        return bid >= offset && ((long) bid - offset) % period == 0;
    }

    /** The per-load Jamba expert collector (the {@code _experts} buffers, jamba.py:57-100). */
    private static HfHookSupport.ExpertCollector jambaExperts(HfLoadContext ctx, int nExperts) {
        return ctx.state(HfSsmHooks.class.getName() + ".jambaExperts", HfHookSupport.ExpertCollector.class,
                () -> new HfHookSupport.ExpertCollector(nExperts, HfSsmHooks.JAMBA_EXPERT_PARTS,
                        (prefix, part, tail) -> "model.layers." + HfHookSupport.layerOf(prefix)
                                + ".mlp.experts." + part + tail));
    }

    /** {@code prepare_tensors}' unprocessed-experts check (jamba.py:113-120). */
    public static void jambaExtra(HfLoadContext ctx, TensorSink out) {
        if (!ctx.config().flat().containsKey("num_local_experts")
                && !ctx.config().flat().containsKey("num_experts")) {
            return;
        }
        jambaExperts(ctx, ctx.config().requireInt("num_local_experts", "num_experts")).requireComplete();
    }

    /**
     * {@code JambaModel.set_vocab} (jamba.py:18-23): SentencePiece for {@code tokenizer.model}, else the Llama-HF
     * tokenizer with the space prefix off.
     */
    public static Map<String, Object> jambaVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx);
        if (ctx.tokenizerFiles().sentencePieceModel() != null) {
            return HfVocabWriters.sentencePiece(in, pre);
        }
        Map<String, Object> md = new LinkedHashMap<>(HfVocabWriters.llamaHf(in, pre));
        md.put(GgufKeys.TOKENIZER_ADD_PREFIX, false);
        return Collections.unmodifiableMap(md);
    }

    // ------------------------------------------------------------------ Plamo2Model (plamo.py:63)

    /** {@link Arch#PLAMO2}. */
    public static Arch plamo2Arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.PLAMO2;
    }

    /**
     * {@code Plamo2Model.set_gguf_parameters} (plamo.py:69-117): the vocabulary size, the attention/Mamba layer
     * pattern of {@code mamba_step} (default 2, mamba_enabled default true, plamo.py:76-97) written as the per-layer
     * head-count arrays, the PLaMo-2 defaults (context 2048, width 4096, per-head size 128, rms epsilon 1e-6, RoPE
     * base 10000, state size 64, conv 4, 64 Mamba heads, FFN 13312), and zero SSM B/C groups.
     */
    public static void plamo2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.VOCAB_SIZE, (long) cfg.requireInt("vocab_size"));
        int blockCount = ctx.blockCount();
        int mambaStep = cfg.findInt(2, "mamba_step");
        if (cfg.findBool(true, "mamba_enabled")) {
            List<Integer> kvHeads = new ArrayList<>(blockCount);
            List<Integer> qHeads = new ArrayList<>(blockCount);
            for (int i = 0; i < blockCount; i++) {
                boolean isMamba;
                if (blockCount <= mambaStep / 2) {
                    isMamba = i != blockCount - 1;
                } else {
                    isMamba = i % mambaStep != mambaStep / 2;
                }
                if (isMamba) {
                    kvHeads.add(0);
                    qHeads.add(0);
                } else {
                    kvHeads.add(cfg.findInt(4, "num_key_value_heads"));
                    qHeads.add(cfg.findInt(32, "num_attention_heads"));
                }
            }
            if (mambaStep > 0) {
                kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, kvHeads);
                kv.set(GgufKeys.ATTENTION_HEAD_COUNT, qHeads);
            }
        }
        kv.set(GgufKeys.CONTEXT_LENGTH, (long) cfg.findInt(2048, "max_position_embeddings"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, (long) cfg.findInt(4096, "hidden_size"));
        long perHead = cfg.findInt(128, "hidden_size_per_head");
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, perHead);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, perHead);
        kv.set(GgufKeys.BLOCK_COUNT, (long) blockCount);
        OptionalDouble eps = cfg.findFloat("rms_norm_eps");
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, eps.isPresent() ? eps.getAsDouble() : 1e-6);
        OptionalDouble theta = cfg.rope().global().theta();
        kv.set(GgufKeys.ROPE_FREQ_BASE, theta.isPresent() ? theta.getAsDouble() : 10000.0);
        kv.set(GgufKeys.SSM_STATE_SIZE, (long) cfg.findInt(64, "mamba_d_state"));
        kv.set(GgufKeys.SSM_CONV_KERNEL, (long) cfg.findInt(4, "mamba_d_conv"));
        long heads = (long) cfg.findInt(64, "mamba_num_heads");
        kv.set(GgufKeys.SSM_TIME_STEP_RANK, heads);
        kv.set(GgufKeys.SSM_INNER_SIZE, heads * perHead);
        kv.set(GgufKeys.SSM_GROUP_COUNT, 0L);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, (long) cfg.findInt(13312, "intermediate_size"));
    }

    /**
     * {@code Plamo2Model.modify_tensors} (plamo.py:119-148): the runtime name tails, the Mamba conv squeeze and
     * {@code A_log}, and the Plamo norm offsets, exactly at plamo.py:136-145: pre-mixer and pre-FFN +1, post-mixer
     * +1/5, post-FFN +1/5^1.5, other norms +1, and the {@code dt_bias} to {@code dt_proj.bias} renames.
     */
    public static void plamo2Tensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = hfName;
        RowSource s = src;
        if (name.endsWith(".A_log")) {
            s = ElementTransforms.negExp(s);
        } else if (name.endsWith(".dt_bias")) {
            name = HfHookSupport.renameTail(name, ".dt_bias", ".dt_proj.bias");
        } else if (name.endsWith(".dt_norm_weight")) {
            name = HfHookSupport.renameTail(name, ".dt_norm_weight", ".dt_norm.weight");
        } else if (name.endsWith(".B_norm_weight")) {
            name = HfHookSupport.renameTail(name, ".B_norm_weight", ".B_norm.weight");
        } else if (name.endsWith(".C_norm_weight")) {
            name = HfHookSupport.renameTail(name, ".C_norm_weight", ".C_norm.weight");
        } else if (name.endsWith(".k_weight")) {
            name = HfHookSupport.renameTail(name, ".k_weight", ".k.weight");
        } else if (name.endsWith(".q_weight")) {
            name = HfHookSupport.renameTail(name, ".q_weight", ".q.weight");
        } else if (name.endsWith(".conv1d.weight")) {
            s = ElementTransforms.squeezeConv(s);
            if (s.shape().length != 2) {
                throw new GgufFormatException("tensor '" + hfName + "' " + Arrays.toString(s.shape())
                        + ": a PLaMo 2 conv1d must squeeze to two dimensions (plamo.py:136)");
            }
        } else if (name.endsWith(".pre_mixer_norm.weight")) {
            s = HfHookSupport.normPlusOne(s);
        } else if (name.endsWith(".post_mixer_norm.weight")) {
            s = ElementTransforms.addScalar(s, 1f / 5f);
        } else if (name.endsWith(".pre_mlp_norm.weight")) {
            s = HfHookSupport.normPlusOne(s);
        } else if (name.endsWith(".post_mlp_norm.weight")) {
            s = ElementTransforms.addScalar(s, (float) (1.0 / Math.pow(5.0, 1.5)));
        } else if (name.endsWith(".norm.weight")) {
            s = HfHookSupport.normPlusOne(s);
        }
        HfTensorHook.DEFAULT.apply(ctx, name, s, out);
    }
    /** {@code FalconH1Model.set_vocab} (falcon_h1.py:64-65): the plain GPT-2 tokenizer vocabulary. */
    public static Map<String, Object> falconH1Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    /**
     * {@code Plamo2Model.set_vocab} (plamo.py:66-68): the {@code tokenizer.jsonl} PLaMo2 vocabulary with the
     * {@code <|plamo:op|>} end-of-turn token.
     */
    public static Map<String, Object> plamo2Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.plamo(HfHookSupport.vocabInput(ctx), "<|plamo:op|>", java.util.List.of());
    }
}

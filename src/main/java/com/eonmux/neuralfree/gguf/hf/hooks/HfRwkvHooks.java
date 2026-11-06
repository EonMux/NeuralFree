package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Pattern;

/**
 * The four RWKV converter classes (LC/conversion/rwkv.py:15-303): {@code Rwkv6Model}, {@code RWKV6Qwen2Model},
 * {@code Rwkv7Model} and {@code ARwkv7Model}. The RWKV6 classes transpose the low-rank factor matrices
 * ({@code time_mix_w1}, {@code time_mix_decay_w1}, {@code time_mix_decay_w2}; {@code time_mix_w2} also swaps its
 * last two dimensions), squeeze the decay and lerp weights, halve the time-mix output and channel-mix value weights
 * once per {@code rescale_every} block, and stack the five {w,k,v,r,g} lerp weights of a block into
 * {@code time_mix_lerp_fused}. RWKV6Qwen2 additionally reorders the stored (w, k, v, r, g) groups of
 * {@code time_mix_w1}/{@code time_mix_w2} to the tiled (r, k, v, w, g) order. The RWKV7 classes unify the fla-hub
 * tensor names in the class filter, rename the {@code _lora.lora} sub-bottlenecks, stop transposing the low-rank
 * weights when a checkpoint ships such names, flatten {@code time_mix_r_k}, ignore the dummy layer-0
 * {@code attention.v&lt;digit&gt;} heads and synthesize the layer-0 {@code time_mix_v*} rows from the layer-0
 * {@code time_mix_a*}, and stack the {@code attention.x_*} lerp parts (an already-concatenated {@code x_x} reshaped
 * directly). RWKV6 and RWKV7 read the RWKV world vocabulary with the bundled {@value #RWKV_WORLD_TEMPLATE} template;
 * RWKV6Qwen2 and ARWKV7 read the tokenizer's sentence-piece vocabulary, falling back to GPT-2 BPE. RWKV is not
 * context limited: every class writes the context length {@value #NO_CONTEXT_LIMIT} and the required-but-unused head
 * count of 0.
 */
public final class HfRwkvHooks {

    /** The bundled RWKV world chat template, added when the checkpoint has none (base.py:2450-2456). */
    static final String RWKV_WORLD_TEMPLATE = "llama-cpp-rwkv-world.jinja";

    /** RWKV has no context limit; the converter writes a fixed one (rwkv.py:33, :128, :215, :283). */
    static final long NO_CONTEXT_LIMIT = 1048576L;

    /** The {w,k,v,r,g} lerp order of RWKV6 (rwkv.py:106-110). */
    private static final List<String> RWKV6_LERP = Arrays.asList("w", "k", "v", "r", "g");
    /** The {r,w,k,v,a,g} lerp order of RWKV7 (rwkv.py:178). */
    private static final List<String> RWKV7_LERP = Arrays.asList("r", "w", "k", "v", "a", "g");
    /** The gate-less lerp order of RWKV7 (rwkv.py:178). */
    private static final List<String> RWKV7_LERP_NO_GATE = Arrays.asList("r", "w", "k", "v", "a");
    /** The stored {w,k,v,r,g} groups read as {r,w,...}: RWKV6Qwen2's reorder (rwkv.py:168-174). */
    private static final int[] RKVWG_REORDER = {3, 1, 2, 0, 4};
    /** The low-rank factors RWKV7 transposes when the checkpoint keeps dense names (rwkv.py:206-216). */
    private static final String[] LORA_TRANSPOSE_TAILS = {
            "time_mix_w1.weight", "time_mix_w2.weight", "time_mix_a1.weight", "time_mix_a2.weight",
            "time_mix_v1.weight", "time_mix_v2.weight", "time_mix_g1.weight", "time_mix_g2.weight"};

    /** The dummy layer-0 value heads of some fla-hub checkpoints (rwkv.py:186-189). */
    private static final Pattern LAYER0_DUMMY_VALUE = Pattern.compile("attention\\.v\\d");

    private static final String W1_TAIL = "time_mix_w1.weight";
    private static final String W2_TAIL = "time_mix_w2.weight";
    private static final String LERP_TAIL = ".time_mix_lerp_";
    private static final String LERP_STATE = "rwkv.lerpParts";
    private static final String TRANSPOSE_STATE = "rwkv7.loraTranspose";
    private static final String WEIGHT = ".weight";
    /** Converter tensor name ({@code MODEL_TENSOR}) of the fused token-shift lerp weights. */
    private static final String TIME_MIX_LERP_FUSED = "TIME_MIX_LERP_FUSED";

    private HfRwkvHooks() {
    }

    /** A resolver to {@link Arch#RWKV6}. */
    public static HfArchResolver rwkv6Arch() {
        return HfArchResolver.of(Arch.RWKV6);
    }

    /** A resolver to {@link Arch#RWKV6QWEN2}. */
    public static HfArchResolver rwkv6Qwen2Arch() {
        return HfArchResolver.of(Arch.RWKV6QWEN2);
    }

    /** A resolver to {@link Arch#RWKV7}. */
    public static HfArchResolver rwkv7Arch() {
        return HfArchResolver.of(Arch.RWKV7);
    }

    /** A resolver to {@link Arch#ARWKV7}. */
    public static HfArchResolver arwkv7Arch() {
        return HfArchResolver.of(Arch.ARWKV7);
    }

    // ------------------------------------------------------------------ Rwkv6Model

    /**
     * {@code Rwkv6Model.set_gguf_parameters} (rwkv.py:21-51): no context limit, the layer-norm epsilon, the rescale
     * interval, the WKV head size and the time-mix / time-decay extra ranks (64 / 32, or 128 / 64 for a 4096-wide
     * model), the feed-forward length with its {@code (hidden * 3.5) // 32 * 32} fallback, and the required —
     * unused — head count of 0.
     */
    public static void rwkv6Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int hidden = cfg.requireInt("hidden_size");
        kv.set(GgufKeys.CONTEXT_LENGTH, NO_CONTEXT_LIMIT);
        kv.set(GgufKeys.EMBEDDING_LENGTH, hidden);
        kv.set(GgufKeys.BLOCK_COUNT, blockCount(ctx, plan));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, eps(cfg, "layer_norm_epsilon"));
        kv.set(GgufKeys.RESCALE_EVERY_N_LAYERS, cfg.requireInt("rescale_every"));
        kv.set(GgufKeys.WKV_HEAD_SIZE, cfg.requireInt("head_size"));
        kv.set(GgufKeys.TIME_MIX_EXTRA_DIM, hidden == 4096 ? 64 : 32);
        kv.set(GgufKeys.TIME_DECAY_EXTRA_DIM, hidden == 4096 ? 128 : 64);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, intermediateOf(cfg, hidden * 7L / 2L / 32L * 32L));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, 0);
    }

    /** {@code Rwkv6Model.modify_tensors} (rwkv.py:53-116), without RWKV6Qwen2's grouped reorder. */
    public static void rwkv6Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource s = rwkv6Transform(ctx, hfName, src);
        if (rwkv6Lerp(ctx, hfName, s, out)) {
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, s, out);
    }

    // ------------------------------------------------------------------ RWKV6Qwen2Model

    /**
     * {@code RWKV6Qwen2Model.set_gguf_parameters} (rwkv.py:119-152): the Qwen sizes, the WKV head size
     * {@code hidden_size / num_attention_heads}, the RMS-norm epsilon, the grouped-query value head count, the
     * token shift, and the class's extra ranks ({@code lora_rank_tokenshift}, {@code lora_rank_decay} with the
     * 4096-dependent defaults).
     */
    public static void rwkv6Qwen2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int hidden = cfg.requireInt("hidden_size");
        int heads = cfg.requireInt("num_attention_heads");
        kv.set(GgufKeys.CONTEXT_LENGTH, NO_CONTEXT_LIMIT);
        kv.set(GgufKeys.EMBEDDING_LENGTH, hidden);
        kv.set(GgufKeys.BLOCK_COUNT, blockCount(ctx, plan));
        kv.set(GgufKeys.WKV_HEAD_SIZE, hidden / heads);
        kv.set(GgufKeys.TIME_MIX_EXTRA_DIM, cfg.findInt(64, "lora_rank_tokenshift"));
        kv.set(GgufKeys.TIME_DECAY_EXTRA_DIM, cfg.findInt(128, "lora_rank_decay"));
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, cfg.requireInt("intermediate_size"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, eps(cfg, "rms_norm_eps"));
        kv.set(GgufKeys.TOKEN_SHIFT_COUNT, 1);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, cfg.requireInt("num_key_value_heads"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, 0);
    }

    /**
     * {@code RWKV6Qwen2Model.modify_tensors} (rwkv.py:154-183): the shared RWKV6 chain, then the reorder of the
     * stored (w, k, v, r, g) groups of {@code time_mix_w1}/{@code time_mix_w2} to the tiled (r, k, v, w, g) order.
     */
    public static void rwkv6Qwen2Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource s = rwkv6Transform(ctx, hfName, src);
        if (rwkv6Lerp(ctx, hfName, s, out)) {
            return;
        }
        if (hfName.endsWith(W1_TAIL) || hfName.endsWith(W2_TAIL)) {
            s = reorderRkvwg(s, hfName);
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, s, out);
    }

    // ------------------------------------------------------------------ Rwkv7Model

    /**
     * {@code Rwkv7Model.filter_tensors} (rwkv.py:171-181) composed before the inherited filters: the fla-hub
     * {@code blocks}/{@code ffn}/{@code self_attn}/{@code attn}/{@code time_mixer}/{@code feed_forward_norm}/
     * {@code g_norm} names unified to the converter's canonical form.
     */
    public static HfNameFilter rwkv7Filter() {
        return (name, base) -> base.apply(unifyRwkv7(name));
    }

    private static String unifyRwkv7(String name) {
        String n = name.replace("blocks", "layers").replace("ffn", "feed_forward");
        n = n.replace("self_attn", "attention").replace("attn", "attention");
        n = n.replace("time_mixer.", "");
        return n.replace("feed_forward_norm", "ln2").replace("g_norm", "ln_x");
    }

    /**
     * {@code Rwkv7Model.set_gguf_parameters} (rwkv.py:182-217): the {@code lora_rank_*} In-Context-Learning-Rates
     * with {@code calc_lora_rank} (rwkv.py:153-156) over {@code decay_low_rank_dim} and friends, the layer-norm
     * epsilon ({@code layer_norm_epsilon} or {@code norm_eps}), the WKV head size and the feed-forward length with
     * the {@code hidden * 4} fallback.
     */
    public static void rwkv7Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int hidden = cfg.requireInt("hidden_size");
        kv.set(GgufKeys.CONTEXT_LENGTH, NO_CONTEXT_LIMIT);
        kv.set(GgufKeys.EMBEDDING_LENGTH, hidden);
        kv.set(GgufKeys.BLOCK_COUNT, blockCount(ctx, plan));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, eps(cfg, "layer_norm_epsilon", "norm_eps"));
        kv.set(GgufKeys.WKV_HEAD_SIZE, headSize(cfg));
        loraRanks(kv, cfg, hidden);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, intermediateOf(cfg, hidden * 4L));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, 0);
    }

    /** The {@code Rwkv7Model.modify_tensors} chain (rwkv.py:222-266) with the six-part lerp stack. */
    public static void rwkv7Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        rwkv7Common(ctx, hfName, src, out, RWKV7_LERP);
    }

    // ------------------------------------------------------------------ ARwkv7Model

    /**
     * {@code ARwkv7Model.set_gguf_parameters} (rwkv.py:268-303) with the fixed ranks 64 / 64 / 32 / (128 or 0),
     * RMS norm and the token shift; the gate rank is 0 without {@code wkv_has_gate}.
     */
    public static void arwkv7Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        Object version = cfg.find("wkv_version").orElse(null);
        if (!Objects.equals(version, 7) && !"7".equals(String.valueOf(version))) {
            throw new GgufFormatException(cfg.source() + ": ARWKV7 requires wkv_version 7, got " + version);
        }
        kv.set(GgufKeys.CONTEXT_LENGTH, NO_CONTEXT_LIMIT);
        kv.set(GgufKeys.EMBEDDING_LENGTH, cfg.requireInt("hidden_size"));
        kv.set(GgufKeys.BLOCK_COUNT, blockCount(ctx, plan));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, eps(cfg, "rms_norm_eps"));
        kv.set(GgufKeys.WKV_HEAD_SIZE, cfg.requireInt("head_size"));
        kv.set(GgufKeys.ATTENTION_DECAY_LORA_RANK, 64);
        kv.set(GgufKeys.ATTENTION_ICLR_LORA_RANK, 64);
        kv.set(GgufKeys.ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK, 32);
        kv.set(GgufKeys.ATTENTION_GATE_LORA_RANK, hasGate(cfg) ? 128 : 0);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, cfg.requireInt("intermediate_size"));
        kv.set(GgufKeys.TOKEN_SHIFT_COUNT, 1);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, 0);
    }

    /** The {@code ARwkv7Model} tensors (rwkv.py:222-266 inherited): the lerp stack drops its gate without one. */
    public static void arwkv7Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        rwkv7Common(ctx, hfName, src, out, hasGate(ctx.config()) ? RWKV7_LERP : RWKV7_LERP_NO_GATE);
    }

    // ------------------------------------------------------------------ vocabularies

    /** {@code _set_vocab_rwkv_world} (base.py:2420-2462): the world tokens, the bundled template, EOT 261. */
    public static Map<String, Object> worldVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx)
                .withFallbackChatTemplate(HfChatTemplates.bundled(RWKV_WORLD_TEMPLATE));
        return HfVocabWriters.rwkvWorld(in, pre);
    }

    /** {@code _set_vocab_sentencepiece} falling back to {@code _set_vocab_gpt2} (rwkv.py:89-91, :272-276). */
    public static Map<String, Object> sentencePieceOrGpt2(HfLoadContext ctx, PreNameResolver pre)
            throws IOException {
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx);
        return HfVocabWriters.canSentencePiece(in)
                ? HfVocabWriters.sentencePiece(in, pre) : HfVocabWriters.gpt2(in, pre);
    }

    // ------------------------------------------------------------------ shared RWKV6 chain

    private static RowSource rwkv6Transform(HfLoadContext ctx, String hfName, RowSource src) {
        RowSource s = src;
        if (hfName.endsWith(W1_TAIL) || hfName.endsWith("time_mix_decay_w1.weight")
                || hfName.endsWith("time_mix_decay_w2.weight")) {
            s = ExpertTransforms.transposeLast2(s);
        }
        if (hfName.endsWith(W2_TAIL)) {
            s = ExpertTransforms.transposeLast2(s);
        }
        if (hfName.endsWith("time_mix_decay.weight") || hfName.contains("lerp")) {
            s = ElementTransforms.squeezeConv(s);
        }
        OptionalInt rescale = ctx.config().findInt("rescale_every");
        if (rescale.isPresent() && rescale.getAsInt() > 0
                && (hfName.endsWith("time_mix_output.weight") || hfName.endsWith("channel_mix_value.weight"))) {
            int bid = HfHookSupport.layerOf(hfName);
            s = ElementTransforms.mulScalar(s, (float) Math.pow(2.0, -(bid / rescale.getAsInt())));
        }
        return s;
    }

    /** Reorders the five stored (w, k, v, r, g) groups of {@code src} to the tiled (r, k, v, w, g) order. */
    private static RowSource reorderRkvwg(RowSource src, String hfName) {
        long[] shape = src.shape();
        long rows = src.rows();
        if (rows < 5 || rows % 5 != 0) {
            throw new GgufFormatException("tensor '" + hfName + "' of shape " + Arrays.toString(shape)
                    + " does not hold a multiple of five rkvwg groups");
        }
        long unit = rows / 5;
        if (unit > Integer.MAX_VALUE) {
            throw new GgufFormatException("tensor '" + hfName + "': one rkvwg group of " + unit
                    + " rows is too large to reorder");
        }
        return RowTransforms.gatherRows(src, (int) unit, RKVWG_REORDER);
    }

    /**
     * The lerp stacking of RWKV6 (rwkv.py:99-112): every {@code time_mix_lerp_<g>.weight} of the five groups but
     * {@code time_mix_lerp_x} buffers per block and the complete block emits the stacked
     * {@code time_mix_lerp_fused.weight} in the {w,k,v,r,g} order.
     */
    private static boolean rwkv6Lerp(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        int dot = hfName.indexOf(LERP_TAIL);
        if (dot < 0) {
            return false;
        }
        String group = hfName.substring(dot + LERP_TAIL.length());
        int weight = group.lastIndexOf(WEIGHT);
        if (weight < 0 || !RWKV6_LERP.contains(group.substring(0, weight))) {
            return false;
        }
        Lerp lerp = ctx.state(LERP_STATE, Lerp.class, Lerp::new);
        synchronized (lerp) {
            int bid = HfHookSupport.layerOf(hfName);
            lerp.add(RWKV6_LERP.indexOf(group.substring(0, weight)), squeeze(src), bid);
            if (lerp.pending(bid, RWKV6_LERP)) {
                return true;
            }
            List<RowSource> stack = lerp.ordered(bid, RWKV6_LERP);
            lerp.clear(bid);
            // torch.stack([w.unsqueeze(0) for w in squeezed], dim=0).unsqueeze(1): [5, 1, 1, n_embd] (rwkv.py:79)
            long inner = innerTotal(stack.get(0));
            out.add(ctx.names().ggufName(TIME_MIX_LERP_FUSED, bid, WEIGHT),
                    stacked(stack, 1, 1, inner));
            return true;
        }
    }

    // ------------------------------------------------------------------ shared RWKV7 chain

    private static void rwkv7Common(HfLoadContext ctx, String hfName, RowSource src, TensorSink out,
                                    List<String> groups) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = hfName;
        RowSource s = src;
        if (name.contains("_lora.lora")) {
            // the names of a fla-hub checkpoint stop the class's dense-name transposes (rwkv.py:228-229)
            ctx.state(TRANSPOSE_STATE, AtomicBoolean.class, AtomicBoolean::new).set(false);
            // rename the low-rank Sequential parts (rwkv.py:224-227)
            name = removeLoraDecorations(name);
        }
        int bid = HfLoadContext.bidOf(name);
        if (name.contains("attention.x_")) {
            rwkv7Lerp(ctx, name, s, out, groups);
            return;
        }
        if (bid == 0 && LAYER0_DUMMY_VALUE.matcher(name).find()) {
            out.drop(hfName, "RWKV7: the dummy layer-0 value heads are not used (rwkv.py:186-189)");
            return;
        }
        s = ElementTransforms.squeezeConv(s);
        boolean transpose = !ctx.state(TRANSPOSE_STATE, AtomicBoolean.class, AtomicBoolean::new).get();
        if (transpose) {
            for (String tail : LORA_TRANSPOSE_TAILS) {
                if (name.endsWith(tail)) {
                    s = ExpertTransforms.transposeLast2(s);
                    break;
                }
            }
        }
        if (name.contains("r_k")) {
            s = ElementTransforms.reshape(s, innerTotal(s));
        }
        if (bid == 0 && name.contains("time_mix_a")) {
            // rwkv.py:262-265: the dummy layer-0 v0/v1/v2 come from the a0/a1/a2 weights
            out.add(ctx.names().mapOrThrow(name.replace("time_mix_a", "time_mix_v")), s);
        }
        HfTensorHook.DEFAULT.apply(ctx, name, s, out);
    }

    private static String removeLoraDecorations(String name) {
        return name.replace("_lora.lora.0.weight", "1.weight")
                .replace("_lora.lora.2.weight", "2.weight")
                .replace("_lora.lora.2.bias", "0.weight");
    }

    /**
     * The RWKV7 lerp stack (rwkv.py:190-204): the already-concatenated {@code attention.x_x} becomes
     * {@code time_mix_lerp_fused} by reshaping; otherwise the {@code attention.x_<g>} parts buffer per block and
     * the complete block emits the fused stack in the class's order.
     */
    private static void rwkv7Lerp(HfLoadContext ctx, String name, RowSource src, TensorSink out,
                                  List<String> groups) throws IOException {
        List<String> tagged = Arrays.asList(name.split("\\."));
        String group = null;
        for (String part : tagged) {
            if (part.startsWith("x_") && !"x_x".equals(part) && part.length() > 2) {
                String g = part.substring(2);
                if (g.endsWith(WEIGHT)) {
                    g = g.substring(0, g.length() - WEIGHT.length());
                }
                if (groups.contains(g)) {
                    group = g;
                    break;
                }
            }
        }
        if (group != null) {
            Lerp lerp = ctx.state(LERP_STATE, Lerp.class, Lerp::new);
            int bid = HfHookSupport.layerOf(name);
            synchronized (lerp) {
                lerp.add(groups.indexOf(group), squeeze(src), bid);
                if (lerp.pending(bid, groups)) {
                    return;
                }
                List<RowSource> stack = lerp.ordered(bid, groups);
                lerp.clear(bid);
                // torch.stack of the unsqueezed x_<g> tensors, dim 0 (rwkv.py:235)
                out.add(ctx.names().ggufName(TIME_MIX_LERP_FUSED, bid, WEIGHT), stacked(stack, src.shape()));
            }
            return;
        }
        lerpFused(ctx, name, src, out, groups.size());
    }

    private static void lerpFused(HfLoadContext ctx, String name, RowSource src, TensorSink out, int n)
            throws IOException {
        long rows = src.rows();
        long nRows = n;
        long inner = innerTotal(src) / nRows;
        RowSource fused = ElementTransforms.reshape(src, nRows, 1, 1, inner);
        int bid = HfHookSupport.layerOf(name);
        out.add(ctx.names().ggufName(TIME_MIX_LERP_FUSED, bid, WEIGHT), fused);
    }

    /**
     * {@code torch.stack(parts, dim=0)} of equally shaped parts whose own shape is {@code partShape}: the parts'
     * elements in order, shaped {@code [parts.size(), partShape...]}.
     */
    private static RowSource stacked(List<RowSource> parts, long... partShape) {
        List<RowSource> flat = new ArrayList<>(parts.size());
        for (RowSource p : parts) {
            flat.add(ElementTransforms.reshape(p, innerTotal(p)));
        }
        long[] shape = new long[partShape.length + 1];
        shape[0] = parts.size();
        System.arraycopy(partShape, 0, shape, 1, partShape.length);
        return ElementTransforms.reshape(RowTransforms.concatRows(flat), shape);
    }

    /** The element count of {@code src}. */
    private static long innerTotal(RowSource src) {
        long n = 1;
        for (long d : src.shape()) {
            n = Math.multiplyExact(n, d);
        }
        return n;
    }

    // ------------------------------------------------------------------ shared small helpers

    private static RowSource squeeze(RowSource src) {
        return ElementTransforms.squeezeConv(src);
    }

    private static long blockCount(HfLoadContext ctx, HfPlanView plan) {
        return ctx.blockCount();
    }

    private static double eps(HfConfig cfg, String... aliases) {
        return cfg.findFloat(aliases).orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": the layer-norm epsilon (one of "
                        + Arrays.toString(aliases) + ") is missing"));
    }

    private static boolean hasGate(HfConfig cfg) {
        return truthy(cfg.find("wkv_has_gate").orElse(null));
    }

    private static boolean truthy(Object v) {
        if (v instanceof Boolean) {
            return (Boolean) v;
        }
        if (v instanceof Number) {
            return ((Number) v).doubleValue() != 0.0;
        }
        if (v instanceof String) {
            return !((String) v).isEmpty();
        }
        return v != null;
    }

    private static long intermediateOf(HfConfig cfg, long fallback) {
        Object v = cfg.find("intermediate_size").orElse(null);
        return v instanceof Number ? ((Number) v).longValue() : fallback;
    }

    private static int headSize(HfConfig cfg) {
        OptionalInt head = cfg.findInt("head_size", "head_dim");
        if (!head.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": required key 'head_size' is missing");
        }
        return head.getAsInt();
    }

    /**
     * {@code calc_lora_rank} (rwkv.py:153-156) applied to {@code max(1, round(hidden^exp * mult / 32)) * 32} for
     * the ranks whose configuration value is absent, over the fla-hub {@code *_low_rank_dim} names and the
     * upstream {@code lora_rank_*} names (rwkv.py:160-176).
     */
    private static void loraRanks(HfMetadataBuilder kv, HfConfig cfg, int hidden) {
        kv.set(GgufKeys.ATTENTION_DECAY_LORA_RANK,
                loraRank(cfg, "lora_rank_decay", "decay_low_rank_dim", hidden, 0.5, 1.8));
        kv.set(GgufKeys.ATTENTION_ICLR_LORA_RANK,
                loraRank(cfg, "lora_rank_iclr", "a_low_rank_dim", hidden, 0.5, 1.8));
        kv.set(GgufKeys.ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK,
                loraRank(cfg, "lora_rank_value_residual_mix", "v_low_rank_dim", hidden, 0.5, 1.3));
        kv.set(GgufKeys.ATTENTION_GATE_LORA_RANK,
                loraRank(cfg, "lora_rank_gate", "gate_low_rank_dim", hidden, 0.8, 0.6));
    }

    private static int loraRank(HfConfig cfg, String key, String lowRankDim, int hidden, double exp,
                                double multiplier) {
        Object v = cfg.find(key, lowRankDim).orElse(null);
        if (v instanceof Number) {
            return ((Number) v).intValue();
        }
        if (v == null) {
            // calc_lora_rank: max(1, round(hidden ** exp * multiplier / 32)) * 32
            double raw = Math.pow(hidden, exp) * multiplier / 32.0;
            long rounded = Math.max(1L, Math.round(raw));
            long rank = Math.multiplyExact(rounded, 32L);
            if (rank > Integer.MAX_VALUE) {
                throw new GgufFormatException(cfg.source() + ": the computed " + key + " rank " + rank
                        + " does not fit an int");
            }
            return (int) rank;
        }
        throw new GgufFormatException(cfg.source() + ": '" + key + " / " + lowRankDim + " must be an integer, got "
                + v);
    }

    // ------------------------------------------------------------------ the per-load lerp stack

    /**
     * The buffered lerp parts of one load (rwkv.py:69-74, :99-110, :191-200): the sources of every block's groups
     * keyed by their group-order index.
     */
    private static final class Lerp {
        /** group order index to (block to source). */
        final Map<Integer, Map<Integer, RowSource>> byBlock = new LinkedHashMap<>();

        void add(int order, RowSource src, int bid) {
            Map<Integer, RowSource> parts = byBlock.computeIfAbsent(order, k -> new LinkedHashMap<>());
            if (parts.containsKey(bid)) {
                throw new GgufFormatException("time_mix lerp group " + order + " repeats in block " + bid);
            }
            parts.put(bid, src);
        }

        /** True while some group of the block's stack is still missing. */
        boolean pending(int bid, List<String> groups) {
            return count(bid, groups) < groups.size();
        }

        /** The sources of the block's stack in the class's order. */
        List<RowSource> ordered(int bid, List<String> groups) {
            List<RowSource> out = new ArrayList<>(groups.size());
            for (int i = 0; i < groups.size(); i++) {
                Map<Integer, RowSource> parts = byBlock.get(i);
                RowSource s = parts == null ? null : parts.get(bid);
                if (s == null) {
                    throw new GgufFormatException("time_mix lerp group " + i + " has no tensor in block " + bid);
                }
                out.add(s);
            }
            return out;
        }

        /** The stack of the block is emitted once per lerp set, which builds only one at a time. */
        void clear(int bid) {
            for (Map<Integer, RowSource> parts : byBlock.values()) {
                parts.remove(bid);
            }
        }

        int count(int bid, List<String> groups) {
            int n = 0;
            for (int i = 0; i < groups.size(); i++) {
                Map<Integer, RowSource> parts = byBlock.get(i);
                if (parts != null && parts.containsKey(bid)) {
                    n++;
                }
            }
            return n;
        }
    }
}

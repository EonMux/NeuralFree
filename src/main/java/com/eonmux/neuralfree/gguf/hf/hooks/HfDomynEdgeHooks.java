package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.UnaryOperator;

/**
 * Hugging Face DomynEdge conversion (vLLM's {@code domyn_edge.py}, whose configuration lives with the checkpoint):
 * a Qwen2 decoder whose layers switch RoPE and sliding-window attention on individually and whose head width is per
 * layer (the sliding-window layers of the shipped checkpoints are the narrow ones). Q and K pass a per-head RMS norm
 * before RoPE; Q's norm has no weight, so Q is normalized weightlessly and only K's tensor is stored. Every sub-layer
 * output passes a post norm before it joins the residual stream (sandwich norms).
 */
public final class HfDomynEdgeHooks {
    private static final String LAYERS = "model.layers.";
    private static final String WEIGHT = ".weight";
    private static final String LAYOUT_STATE = HfDomynEdgeHooks.class.getName() + ".layout";

    private HfDomynEdgeHooks() {
    }

    /** The per-layer geometry of one checkpoint. Immutable. */
    static final class Layout {
        final int[] headSize;
        final boolean[] swa;
        final boolean[] rope;
        final int fullHead;
        final int swaHead;
        final int ropeDim;
        /** Keys each sliding-window query attends to, the current one included. */
        final int nSwa;
        final boolean anySwa;

        Layout(int[] headSize, boolean[] swa, boolean[] rope, int fullHead, int swaHead, int ropeDim, int nSwa) {
            this.headSize = headSize;
            this.swa = swa;
            this.rope = rope;
            this.fullHead = fullHead;
            this.swaHead = swaHead;
            this.ropeDim = ropeDim;
            this.nSwa = nSwa;
            this.anySwa = nSwa > 0;
        }

        boolean isSwa(int il) {
            return il >= 0 && il < swa.length && swa[il];
        }
    }

    /**
     * Writes the DomynEdge dimensions: the two head widths with their key and value lengths, the rotary width, the
     * shared sliding window with its per-layer switches and the per-layer RoPE switches.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        Layout l = layout(ctx);
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, l.fullHead);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, l.fullHead);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH_SWA, l.swaHead);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_SWA, l.swaHead);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, l.ropeDim);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT_SWA, l.ropeDim);
        if (l.anySwa) {
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, l.nSwa);
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, l.swa);
        if (!allTrue(l.rope)) {
            kv.set(GgufKeys.ATTENTION_ROPE_PATTERN, l.rope);
        }
    }

    private static boolean allTrue(boolean[] flags) {
        for (boolean f : flags) {
            if (!f) {
                return false;
            }
        }
        return true;
    }

    /** No class-level renames: the layer tensors map by their own names. */
    public static String filter(String name, UnaryOperator<String> base) {
        return base.apply(name);
    }

    /**
     * Maps the DomynEdge weights; {@code attention.q_norm} has no weight in the checkpoint, so only K's norm is read
     * and the graph normalizes Q weightlessly.
     *
     * @throws GgufFormatException when the checkpoint carries a tensor the layout does not define
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        out.add(ggufName(ctx, hfName), src);
    }

    private static String ggufName(HfLoadContext ctx, String hfName) {
        if ("model.embed_tokens.weight".equals(hfName)) {
            return TensorNames.weight(TensorNames.TOKEN_EMBD);
        }
        if ("model.norm.weight".equals(hfName)) {
            return TensorNames.weight(TensorNames.OUTPUT_NORM);
        }
        if ("lm_head.weight".equals(hfName)) {
            return TensorNames.weight(TensorNames.OUTPUT);
        }
        int bid = HfLoadContext.bidOf(hfName);
        String base = layerBase(hfName);
        String mapped = base == null ? null : tensorOf(base);
        if (bid < 0 || mapped == null) {
            throw new GgufFormatException("DomynEdge does not support tensor '" + hfName + "'");
        }
        Layout l = layout(ctx);
        if (bid >= l.headSize.length) {
            throw new GgufFormatException("DomynEdge does not support tensor '" + hfName + "': layer " + bid
                    + " is outside the " + l.headSize.length + " configured layers");
        }
        return TensorNames.layerName(bid, mapped + WEIGHT);
    }

    /** The part of a layer tensor name between {@code model.layers.<n>.} and its {@code .weight} suffix. */
    private static String layerBase(String hfName) {
        if (!hfName.startsWith(LAYERS) || !hfName.endsWith(WEIGHT)) {
            return null;
        }
        int start = LAYERS.length();
        int dot = hfName.indexOf('.', start);
        if (dot < 0 || dot == start) {
            return null;
        }
        for (int i = start; i < dot; i++) {
            if (!Character.isDigit(hfName.charAt(i))) {
                return null;
            }
        }
        return hfName.substring(dot + 1, hfName.length() - WEIGHT.length());
    }

    private static String tensorOf(String base) {
        switch (base) {
            case "attention.q_proj": return TensorNames.ATTN_Q;
            case "attention.k_proj": return TensorNames.ATTN_K;
            case "attention.v_proj": return TensorNames.ATTN_V;
            case "attention.o_proj": return TensorNames.ATTN_OUT;
            case "attention.k_norm": return TensorNames.ATTN_K_NORM;
            case "pre_attention_norm": return TensorNames.ATTN_NORM;
            case "post_attention_norm": return TensorNames.ATTN_POST_NORM;
            case "pre_feedforward_norm": return TensorNames.FFN_NORM;
            case "post_feedforward_norm": return TensorNames.FFN_POST_NORM;
            case "feedforward.gate_proj": return TensorNames.FFN_GATE;
            case "feedforward.up_proj": return TensorNames.FFN_UP;
            case "feedforward.down_proj": return TensorNames.FFN_DOWN;
            default: return null;
        }
    }

    /** The parsed geometry of {@code ctx}, built once per load. */
    static Layout layout(HfLoadContext ctx) {
        return ctx.state(LAYOUT_STATE, Layout.class, () -> parse(ctx.config(), ctx.blockCount()));
    }

    /**
     * Parses and validates {@code head_sizes}, {@code window_sizes} and {@code position_layers}.
     *
     * @throws GgufFormatException when a list is missing or short, a position type or window is unsupported, the head
     *                             sizes of a layer class differ, or the rotary width cannot be shared by the RoPE
     *                             layers
     */
    static Layout parse(HfConfig cfg, int blocks) {
        if (cfg.findBool(false, "attention_bias")) {
            throw invalid(cfg, "attention_bias", "biased attention projections are unsupported");
        }
        List<Integer> heads = cfg.findIntList("head_sizes").orElse(null);
        if (heads == null) {
            throw invalid(cfg, "head_sizes", "is required");
        }
        if (heads.size() < blocks) {
            throw invalid(cfg, "head_sizes", "has " + heads.size() + " entries, fewer than the " + blocks + " layers");
        }
        Object windowsRaw = cfg.find("window_sizes").orElse(null);
        Object positionsRaw = cfg.find("position_layers").orElse(null);
        if (positionsRaw != null && !(positionsRaw instanceof List)) {
            throw invalid(cfg, "position_layers", "must be an array");
        }
        List<?> positions = positionsRaw == null ? null : (List<?>) positionsRaw;
        if (positions != null && positions.size() < blocks) {
            throw invalid(cfg, "position_layers",
                    "has " + positions.size() + " entries, fewer than the " + blocks + " layers");
        }
        List<?> windows = null;
        if (windowsRaw != null) {
            if (!(windowsRaw instanceof List)) {
                throw invalid(cfg, "window_sizes", "must be an array");
            }
            windows = (List<?>) windowsRaw;
            if (windows.size() < blocks) {
                throw invalid(cfg, "window_sizes",
                        "has " + windows.size() + " entries, fewer than the " + blocks + " layers");
            }
        }
        int[] headSize = new int[blocks];
        boolean[] swa = new boolean[blocks];
        boolean[] rope = new boolean[blocks];
        Set<Integer> fullHeads = new HashSet<>();
        Set<Integer> swaHeads = new HashSet<>();
        Set<Integer> ropeHeads = new HashSet<>();
        int window = 0;
        for (int il = 0; il < blocks; il++) {
            int head = heads.get(il);
            if (head <= 0) {
                throw invalid(cfg, "head_sizes", "layer " + il + " has a non-positive head size " + head);
            }
            headSize[il] = head;
            int w = windowOf(cfg, windows, il);
            swa[il] = w > 0;
            if (w > 0) {
                if (window == 0) {
                    window = w;
                } else if (window != w) {
                    throw invalid(cfg, "window_sizes", "layer " + il + " has window " + w + " where earlier sliding"
                            + " layers have " + window + "; a single sliding window width is supported");
                }
            }
            (swa[il] ? swaHeads : fullHeads).add(head);
            rope[il] = positionOf(cfg, positions, il);
            if (rope[il]) {
                ropeHeads.add(head);
            }
        }
        if (fullHeads.size() > 1) {
            throw invalid(cfg, "head_sizes", "full-attention layers must share one head size, got " + fullHeads);
        }
        if (swaHeads.size() > 1) {
            throw invalid(cfg, "head_sizes", "sliding-window layers must share one head size, got " + swaHeads);
        }
        if (ropeHeads.size() > 1) {
            throw invalid(cfg, "head_sizes", "RoPE layers must share one head size, got " + ropeHeads);
        }
        int swaHead = swaHeads.isEmpty() ? 0 : swaHeads.iterator().next();
        int fullHead = fullHeads.isEmpty() ? swaHead : fullHeads.iterator().next();
        if (fullHead <= 0) {
            throw invalid(cfg, "head_sizes", "no layer has a positive head size");
        }
        if (swaHead <= 0) {
            swaHead = fullHead;
        }
        int ropeDim = ropeHeads.isEmpty() ? swaHead : ropeHeads.iterator().next();
        if (!ropeHeads.isEmpty()) {
            if (ropeDim % 2 != 0) {
                throw invalid(cfg, "head_sizes", "the rotary width " + ropeDim + " must be even");
            }
            for (int il = 0; il < blocks; il++) {
                if (rope[il] && headSize[il] != ropeDim) {
                    throw invalid(cfg, "head_sizes", "RoPE layer " + il + " has head size " + headSize[il]
                            + " but the rotary width is " + ropeDim);
                }
            }
        }
        return new Layout(headSize, swa, rope, fullHead, swaHead, ropeDim, window > 0 ? window + 1 : 0);
    }

    /**
     * The window of layer {@code il}: the left width {@code w} of its {@code [w, 0]} pair, 0 for none. The
     * configuration's {@code w} counts the keys before the query, so a layer attends to {@code w + 1} keys.
     */
    private static int windowOf(HfConfig cfg, List<?> windows, int il) {
        if (windows == null) {
            return 0;
        }
        Object entry = windows.get(il);
        if (entry instanceof Number && !(entry instanceof Boolean)) {
            return Math.max(number(cfg, "window_sizes", il, entry), 0);
        }
        if (!(entry instanceof List) || ((List<?>) entry).isEmpty()) {
            throw invalid(cfg, "window_sizes", "layer " + il + " must be a [left, right] pair");
        }
        List<?> pair = (List<?>) entry;
        int left = number(cfg, "window_sizes", il, pair.get(0));
        int right = pair.size() > 1 ? number(cfg, "window_sizes", il, pair.get(1)) : 0;
        if (right != 0) {
            throw invalid(cfg, "window_sizes", "layer " + il + " has a non-zero right window " + right
                    + "; only causal left windows are supported");
        }
        return Math.max(left, 0);
    }

    private static int number(HfConfig cfg, String key, int il, Object value) {
        if (!(value instanceof Number) || value instanceof Boolean) {
            throw invalid(cfg, key, "layer " + il + " must hold integers");
        }
        long v = ((Number) value).longValue();
        if (v < Integer.MIN_VALUE || v > Integer.MAX_VALUE) {
            throw invalid(cfg, key, "layer " + il + " value " + v + " is out of range");
        }
        return (int) v;
    }

    /** Whether layer {@code il} rotates: {@code position_layers[il]} is {@code "rope"}, not {@code "nope"}. */
    private static boolean positionOf(HfConfig cfg, List<?> positions, int il) {
        if (positions == null) {
            return true;
        }
        Object entry = positions.get(il);
        if ("rope".equals(entry)) {
            return true;
        }
        if ("nope".equals(entry)) {
            return false;
        }
        throw invalid(cfg, "position_layers", "layer " + il + " must be 'rope' or 'nope', got " + entry);
    }

    private static GgufFormatException invalid(HfConfig cfg, String field, String message) {
        return new GgufFormatException(cfg.source() + ": '" + field + "': " + message);
    }

}

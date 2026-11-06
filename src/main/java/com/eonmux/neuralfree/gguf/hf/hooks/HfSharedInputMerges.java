package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * {@code Qwen4ExpTextModel._merge_shared_input} (LC/conversion/qwen4exp.py:177-208): the matmuls that read one input
 * are written as one {@code [a | b]} weight so the runtime does one matmul. Each half the inherited tensor chain plans
 * is held until its partner arrives, then the two are planned as the merged tensor, rows of {@code a} first. The
 * hyper-connection pairs merge in every block; the linear-attention pairs only in the blocks {@code layer_types}
 * marks {@code linear_attention}. A pre-quantized checkpoint ({@code quantization_config} set) merges nothing, since
 * its per-tensor scales cannot follow one merged weight.
 */
final class HfSharedInputMerges {

    private static final String STATE = "qwen4exp.sharedInputMerges";
    private static final String LINEAR_ATTENTION = "linear_attention";
    private static final String WEIGHT = ".weight";

    /** One merge: the merged tensor, its two halves in row order, and whether it is linear-attention only. */
    private static final class Merge {
        final String merged;
        final List<String> parts;
        final boolean linearOnly;

        Merge(String merged, String a, String b, boolean linearOnly) {
            this.merged = merged;
            this.parts = Collections.unmodifiableList(Arrays.asList(a, b));
            this.linearOnly = linearOnly;
        }
    }

    /** {@code _SHARED_INPUT_MERGES} (qwen4exp.py:179-184), in the converter's order. */
    private static final List<Merge> MERGES = Collections.unmodifiableList(Arrays.asList(
            new Merge("HC_ATTN_DOWN_INJECT", "HC_ATTN_DOWN", "HC_ATTN_INJECT", false),
            new Merge("HC_FFN_DOWN_INJECT", "HC_FFN_DOWN", "HC_FFN_INJECT", false),
            new Merge("ATTN_QKV_GATE", "ATTN_QKV", "ATTN_GATE", true),
            new Merge("SSM_BETA_ALPHA_CAT", "SSM_BETA", "SSM_ALPHA", true)));

    /** The halves waiting for their partner, keyed by {@code "<bid>:<merged tensor>"}. */
    private static final class Pending {
        final Map<String, RowSource[]> halves = new TreeMap<>();
    }

    private HfSharedInputMerges() {
    }

    /**
     * The sink the inherited tensor chain of the checkpoint tensor {@code hfName} plans into: {@code out} itself
     * for a model-level tensor or a pre-quantized checkpoint, else a sink that holds and merges the halves.
     */
    static TensorSink sink(HfLoadContext ctx, String hfName, TensorSink out) {
        Objects.requireNonNull(out, "out");
        int bid = HfLoadContext.bidOf(hfName);
        if (bid < 0 || quantized(ctx.config())) {
            return out;
        }
        return new MergingSink(ctx, bid, isLinear(ctx.config(), bid), out);
    }

    /**
     * The {@code prepare_tensors} check (qwen4exp.py:254-255): no half may be left without its partner.
     *
     * @throws GgufFormatException naming the halves left over
     */
    static void requireComplete(HfLoadContext ctx) {
        Pending p = pending(ctx);
        synchronized (p) {
            if (!p.halves.isEmpty()) {
                throw new GgufFormatException("merged weights missing a half: " + p.halves.keySet());
            }
        }
    }

    private static Pending pending(HfLoadContext ctx) {
        return ctx.state(STATE, Pending.class, Pending::new);
    }

    /** {@code bid < len(layer_types) and layer_types[bid] == "linear_attention"} (qwen4exp.py:193-194). */
    private static boolean isLinear(HfConfig cfg, int bid) {
        Object types = cfg.find("layer_types").orElse(null);
        if (!(types instanceof List)) {
            return false;
        }
        List<?> list = (List<?>) types;
        return bid < list.size() && LINEAR_ATTENTION.equals(list.get(bid));
    }

    /** Python truthiness of {@code hparams.get("quantization_config")} (qwen4exp.py:189). */
    private static boolean quantized(HfConfig cfg) {
        Object q = cfg.find("quantization_config").orElse(null);
        if (q == null) {
            return false;
        }
        if (q instanceof Map) {
            return !((Map<?, ?>) q).isEmpty();
        }
        if (q instanceof List) {
            return !((List<?>) q).isEmpty();
        }
        if (q instanceof String) {
            return !((String) q).isEmpty();
        }
        if (q instanceof Boolean) {
            return (Boolean) q;
        }
        if (q instanceof Number) {
            return ((Number) q).doubleValue() != 0.0;
        }
        return true;
    }

    /** Forwards everything but the merge halves, which it holds until both are planned. */
    private static final class MergingSink implements TensorSink {
        private final HfLoadContext ctx;
        private final int bid;
        private final boolean linear;
        private final TensorSink out;

        MergingSink(HfLoadContext ctx, int bid, boolean linear, TensorSink out) {
            this.ctx = ctx;
            this.bid = bid;
            this.linear = linear;
            this.out = out;
        }

        @Override
        public void add(String ggufName, RowSource src) {
            for (Merge m : MERGES) {
                if (m.linearOnly && !linear) {
                    continue;
                }
                for (int idx = 0; idx < m.parts.size(); idx++) {
                    if (ctx.names().matches(ggufName, m.parts.get(idx), bid, WEIGHT)) {
                        hold(m, idx, src);
                        return;
                    }
                }
            }
            out.add(ggufName, src);
        }

        private void hold(Merge m, int idx, RowSource src) {
            Pending p = pending(ctx);
            List<RowSource> both;
            synchronized (p) {
                String key = bid + ":" + m.merged;
                RowSource[] halves = p.halves.computeIfAbsent(key, k -> new RowSource[m.parts.size()]);
                if (halves[idx] != null) {
                    throw new GgufFormatException("block " + bid + " plans " + m.parts.get(idx) + " twice");
                }
                halves[idx] = src;
                for (RowSource h : halves) {
                    if (h == null) {
                        return;
                    }
                }
                p.halves.remove(key);
                both = new ArrayList<>(Arrays.asList(halves));
            }
            out.add(ctx.names().ggufName(m.merged, bid, WEIGHT), RowTransforms.concatRows(both));
        }

        @Override
        public void addKv(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
            out.addKv(key, type, elementType, value);
        }

        @Override
        public void drop(String hfName, String reason) {
            out.drop(hfName, reason);
        }

        @Override
        public boolean isConsumed(String hfName) {
            return out.isConsumed(hfName);
        }

        @Override
        public void consume(String hfName) {
            out.consume(hfName);
        }
    }
}

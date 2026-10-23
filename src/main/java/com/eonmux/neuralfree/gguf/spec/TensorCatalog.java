package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Presence and shape index over the tensors of a GGUF file, built once and never touching tensor data.
 *
 * <p>Layer lookups take a suffix including the kind, e.g. {@code layer(3, "attn_q.weight")} resolves
 * {@code blk.3.attn_q.weight}; for the main stack the Hugging Face alias table of {@link TensorNames} is
 * consulted when the GGUF name is absent ({@code model.layers.3.self_attn.q_proj.weight}). Model-level lookups
 * through {@link #get(String)} use the model alias table the same way. The returned {@link TensorRef} carries
 * the name actually stored in the file.
 *
 * <p>{@link #fusedQkv(int)} and {@link #gateUpExps(int)} implement llama.cpp {@code create_tensor_qkv} and
 * {@code create_tensor_gate_up_exps} (src/llama-model.cpp:3846-3900) for every architecture. Immutable.
 */
public final class TensorCatalog {

    private final String archName;
    private final Map<String, GGUFTensorInfo> infos;
    private final Map<String, TensorRef> refs;
    private final int blockCount;

    private TensorCatalog(String archName, Map<String, GGUFTensorInfo> infos, Map<String, TensorRef> refs, int blockCount) {
        this.archName = archName;
        this.infos = infos;
        this.refs = refs;
        this.blockCount = blockCount;
    }

    /**
     * Catalog of a parsed GGUF file; the architecture name comes from {@code general.architecture}.
     *
     * @throws IllegalArgumentException when the file lists the same tensor name twice
     */
    public static TensorCatalog of(GGUFFileParser parser) {
        Objects.requireNonNull(parser, "parser");
        String arch = parser.getString("general.architecture");
        return of(arch != null ? arch : "unknown", parser.getTensors());
    }

    /**
     * Catalog over explicit tensor descriptions.
     *
     * @throws IllegalArgumentException when a name repeats or a description is malformed
     */
    public static TensorCatalog of(String archName, Collection<GGUFTensorInfo> tensors) {
        Objects.requireNonNull(tensors, "tensors");
        Map<String, GGUFTensorInfo> infos = new LinkedHashMap<>();
        Map<String, TensorRef> refs = new LinkedHashMap<>();
        int maxBlock = -1;
        String prefix = TensorNames.STACK_BLK + ".";
        String hfPrefix = TensorNames.HF_LAYER_PREFIX + ".";
        for (GGUFTensorInfo info : tensors) {
            TensorRef ref = TensorRef.of(info);
            if (infos.put(ref.name(), info) != null) {
                throw new IllegalArgumentException("architecture '" + archName + "': tensor '" + ref.name()
                        + "' is listed twice in the GGUF file");
            }
            refs.put(ref.name(), ref);
            if (ref.name().startsWith(prefix)) {
                maxBlock = Math.max(maxBlock, blockIndex(ref.name(), prefix.length()));
            } else if (ref.name().startsWith(hfPrefix)) {
                maxBlock = Math.max(maxBlock, blockIndex(ref.name(), hfPrefix.length()));
            }
        }
        return new TensorCatalog(archName != null ? archName : "unknown", Collections.unmodifiableMap(infos),
                Collections.unmodifiableMap(refs), maxBlock + 1);
    }

    private static int blockIndex(String name, int start) {
        int end = name.indexOf('.', start);
        if (end <= start) {
            return -1;
        }
        int v = 0;
        for (int i = start; i < end; i++) {
            char c = name.charAt(i);
            if (c < '0' || c > '9' || v > (Integer.MAX_VALUE - 9) / 10) {
                return -1;
            }
            v = v * 10 + (c - '0');
        }
        return v;
    }

    /** {@code general.architecture} of the file, or {@code "unknown"}. */
    public String archName() {
        return archName;
    }

    /** Number of tensors in the file. */
    public int size() {
        return refs.size();
    }

    /** All tensor names in file order (unmodifiable). */
    public Set<String> names() {
        return refs.keySet();
    }

    /**
     * One plus the highest {@code N} of any main-stack tensor ({@code blk.N.*} or its Hugging Face alias
     * {@code model.layers.N.*}); 0 when the file has none.
     */
    public int blockCount() {
        return blockCount;
    }

    /** True when a tensor with exactly this name exists (no aliases). */
    public boolean has(String name) {
        return refs.containsKey(name);
    }

    /**
     * Tensor with this name, or its model-level alias (e.g. {@code lm_head.weight} for {@code output.weight}); null
     * when absent.
     */
    public TensorRef get(String name) {
        TensorRef r = refs.get(name);
        if (r != null) {
            return r;
        }
        for (String alias : TensorNames.modelAliases(name)) {
            r = refs.get(alias);
            if (r != null) {
                return r;
            }
        }
        return null;
    }

    /**
     * Like {@link #get(String)} but required.
     *
     * @throws MissingTensorException naming the architecture and tensor when absent
     */
    public TensorRef require(String name) {
        TensorRef r = get(name);
        if (r == null) {
            throw new MissingTensorException(archName, -1, name);
        }
        return r;
    }

    /** True when {@link #layer(int, String)} finds the tensor. */
    public boolean hasLayer(int il, String suffix) {
        return layer(il, suffix) != null;
    }

    /** Main-stack tensor {@code blk.<il>.<suffix>} or its Hugging Face alias; null when absent. */
    public TensorRef layer(int il, String suffix) {
        TensorRef r = refs.get(TensorNames.layerName(il, suffix));
        if (r != null) {
            return r;
        }
        for (String alias : TensorNames.layerAliases(suffix)) {
            r = refs.get(TensorNames.layerName(TensorNames.HF_LAYER_PREFIX, il, alias));
            if (r != null) {
                return r;
            }
        }
        return null;
    }

    /**
     * Tensor {@code <stack>.<il>.<suffix>} of any stack ({@link TensorNames#STACK_ENC}, ...); the main stack
     * also consults the alias table. Null when absent.
     */
    public TensorRef layer(String stack, int il, String suffix) {
        if (TensorNames.STACK_BLK.equals(stack)) {
            return layer(il, suffix);
        }
        return refs.get(TensorNames.layerName(stack, il, suffix));
    }

    /**
     * Required main-stack tensor.
     *
     * @throws MissingTensorException naming the architecture, layer and tensor when absent
     */
    public TensorRef require(int il, String suffix) {
        return require(TensorNames.STACK_BLK, il, suffix);
    }

    /**
     * Required tensor of any stack.
     *
     * @throws MissingTensorException naming the architecture, layer and tensor when absent
     */
    public TensorRef require(String stack, int il, String suffix) {
        TensorRef r = layer(stack, il, suffix);
        if (r == null) {
            throw new MissingTensorException(archName, il, TensorNames.layerName(stack, il, suffix));
        }
        return r;
    }

    /** Parser description of a tensor for loading, by its stored name; null when absent. */
    public GGUFTensorInfo info(String storedName) {
        return infos.get(storedName);
    }

    /** Parser description of a referenced tensor; null when the reference is null. */
    public GGUFTensorInfo info(TensorRef ref) {
        return ref != null ? infos.get(ref.name()) : null;
    }

    /**
     * The fused {@code attn_qkv} projection of a main-stack layer, or null when the layer stores separate
     * Q/K/V weights. Mirrors {@code create_tensor_qkv}: a fused bias is used when present; otherwise separate
     * Q/K/V biases may accompany the fused weight.
     */
    public FusedQkv fusedQkv(int il) {
        TensorRef w = layer(il, TensorNames.weight(TensorNames.ATTN_QKV));
        if (w == null) {
            return null;
        }
        TensorRef b = layer(il, TensorNames.bias(TensorNames.ATTN_QKV));
        TensorRef qb = null;
        TensorRef kb = null;
        TensorRef vb = null;
        if (b == null) {
            qb = layer(il, TensorNames.bias(TensorNames.ATTN_Q));
            kb = layer(il, TensorNames.bias(TensorNames.ATTN_K));
            vb = layer(il, TensorNames.bias(TensorNames.ATTN_V));
        }
        TensorRef s = layer(il, TensorNames.scale(TensorNames.ATTN_QKV));
        return new FusedQkv(archName, il, w, b, qb, kb, vb, s);
    }

    /**
     * Routed-expert gate/up weights of a main-stack layer, or null when the layer has neither
     * {@code ffn_gate_up_exps} nor {@code ffn_up_exps}. Mirrors {@code create_tensor_gate_up_exps}: the fused
     * tensor wins; otherwise separate gate and up tensors are used, where a missing gate means ungated
     * experts.
     *
     * @throws MissingTensorException when {@code ffn_gate_exps} exists without {@code ffn_up_exps}
     */
    public GateUpExps gateUpExps(int il) {
        TensorRef gateUp = layer(il, TensorNames.weight(TensorNames.FFN_GATE_UP_EXPS));
        TensorRef gate = layer(il, TensorNames.weight(TensorNames.FFN_GATE_EXPS));
        TensorRef up = layer(il, TensorNames.weight(TensorNames.FFN_UP_EXPS));
        TensorRef gateScale = layer(il, TensorNames.scale(TensorNames.FFN_GATE_EXPS));
        TensorRef upScale = layer(il, TensorNames.scale(TensorNames.FFN_UP_EXPS));
        if (gateUp != null) {
            return new GateUpExps(archName, il, gateUp, layer(il, TensorNames.bias(TensorNames.FFN_GATE_UP_EXPS)),
                    null, null, null, null, null, upScale);
        }
        if (up == null) {
            if (gate != null) {
                throw new MissingTensorException(archName, il,
                        TensorNames.layerName(il, TensorNames.weight(TensorNames.FFN_UP_EXPS)));
            }
            return null;
        }
        return new GateUpExps(archName, il, null, null, gate, up,
                layer(il, TensorNames.bias(TensorNames.FFN_GATE_EXPS)), layer(il, TensorNames.bias(TensorNames.FFN_UP_EXPS)),
                gateScale, upScale);
    }

    /**
     * True when a dense FFN packs gate and up in {@code ffn_up} ({@code ne[1] == 2 * nFf}) and has no
     * {@code ffn_gate} (llama.cpp {@code LLM_FFN_SWIGLU} with contiguous halves, src/llama-graph.cpp:2075).
     */
    public boolean denseUpFused(int il, long nFf) {
        if (nFf <= 0) {
            throw new IllegalArgumentException("n_ff " + nFf + " must be positive for layer " + il);
        }
        TensorRef up = layer(il, TensorNames.weight(TensorNames.FFN_UP));
        return up != null && !hasLayer(il, TensorNames.weight(TensorNames.FFN_GATE)) && up.ne(1) == 2 * nFf;
    }

    /** Fused Q/K/V projection of one layer. Rows: Q {@code [0, q)}, K {@code [q, q+k)}, V {@code [q+k, q+k+v)}. */
    public static final class FusedQkv {
        private final String arch;
        private final int layer;
        private final TensorRef weight;
        private final TensorRef bias;
        private final TensorRef qBias;
        private final TensorRef kBias;
        private final TensorRef vBias;
        private final TensorRef weightScale;

        FusedQkv(String arch, int layer, TensorRef weight, TensorRef bias, TensorRef qBias, TensorRef kBias,
                 TensorRef vBias, TensorRef weightScale) {
            this.arch = arch;
            this.layer = layer;
            this.weight = weight;
            this.bias = bias;
            this.qBias = qBias;
            this.kBias = kBias;
            this.vBias = vBias;
            this.weightScale = weightScale;
        }

        public int layer() {
            return layer;
        }

        /** {@code attn_qkv.weight}, {@code ne = [n_embd, q+k+v]}. */
        public TensorRef weight() {
            return weight;
        }

        /** {@code attn_qkv.bias}, or null. */
        public TensorRef bias() {
            return bias;
        }

        /** Separate {@code attn_q.bias} accompanying the fused weight when no fused bias exists, or null. */
        public TensorRef qBias() {
            return qBias;
        }

        public TensorRef kBias() {
            return kBias;
        }

        public TensorRef vBias() {
            return vBias;
        }

        /** {@code attn_qkv.scale} ({@code {1}}, single-value per-tensor scale), or null. */
        public TensorRef weightScale() {
            return weightScale;
        }

        /** Output rows of the fused projection ({@code ne[1]}). */
        public long rows() {
            return weight.ne(1);
        }

        /**
         * Checks the fused shape against the per-layer projection widths.
         *
         * @throws IllegalStateException when {@code ne[1] != qDim + kDim + vDim} or a bias length differs
         */
        public void validate(long qDim, long kDim, long vDim) {
            long total = qDim + kDim + vDim;
            if (qDim <= 0 || kDim <= 0 || vDim <= 0) {
                throw new IllegalArgumentException("layer " + layer + ": Q/K/V widths must be positive, got "
                        + qDim + "/" + kDim + "/" + vDim);
            }
            if (rows() != total) {
                throw new IllegalStateException("architecture '" + arch + "': " + weight.name() + " has " + rows()
                        + " output rows, expected q+k+v = " + qDim + "+" + kDim + "+" + vDim + " = " + total);
            }
            checkLength(bias, total);
            checkLength(qBias, qDim);
            checkLength(kBias, kDim);
            checkLength(vBias, vDim);
        }

        private void checkLength(TensorRef b, long expected) {
            if (b != null && b.numElements() != expected) {
                throw new IllegalStateException("architecture '" + arch + "': " + b.name() + " has "
                        + b.numElements() + " elements, expected " + expected);
            }
        }
    }

    /**
     * Routed-expert gate/up weights of one layer: either the fused {@code ffn_gate_up_exps}
     * ({@code ne = [n_embd, 2*n_ff, n_expert]}, gate rows first) or separate gate and up tensors.
     */
    public static final class GateUpExps {
        private final String arch;
        private final int layer;
        private final TensorRef gateUp;
        private final TensorRef gateUpBias;
        private final TensorRef gate;
        private final TensorRef up;
        private final TensorRef gateBias;
        private final TensorRef upBias;
        private final TensorRef gateScale;
        private final TensorRef upScale;

        GateUpExps(String arch, int layer, TensorRef gateUp, TensorRef gateUpBias, TensorRef gate, TensorRef up,
                   TensorRef gateBias, TensorRef upBias, TensorRef gateScale, TensorRef upScale) {
            this.arch = arch;
            this.layer = layer;
            this.gateUp = gateUp;
            this.gateUpBias = gateUpBias;
            this.gate = gate;
            this.up = up;
            this.gateBias = gateBias;
            this.upBias = upBias;
            this.gateScale = gateScale;
            this.upScale = upScale;
        }

        public int layer() {
            return layer;
        }

        /** True for the fused {@code ffn_gate_up_exps} form. */
        public boolean fused() {
            return gateUp != null;
        }

        /** True when the experts have a gate (fused, or a separate {@code ffn_gate_exps}). */
        public boolean gated() {
            return gateUp != null || gate != null;
        }

        /** Fused tensor, or null. */
        public TensorRef gateUp() {
            return gateUp;
        }

        /** {@code ffn_gate_up_exps.bias}, or null. */
        public TensorRef gateUpBias() {
            return gateUpBias;
        }

        /** Separate gate tensor, or null (fused form or ungated experts). */
        public TensorRef gate() {
            return gate;
        }

        /** Separate up tensor, or null in the fused form. */
        public TensorRef up() {
            return up;
        }

        public TensorRef gateBias() {
            return gateBias;
        }

        public TensorRef upBias() {
            return upBias;
        }

        /** {@code ffn_gate_exps.scale} (separate form only), or null. */
        public TensorRef gateScale() {
            return gateScale;
        }

        /**
         * {@code ffn_up_exps.scale}, or null. In the fused form it scales the whole gate_up product
         * (src/llama-graph.cpp:2372) and no gate scale applies.
         */
        public TensorRef upScale() {
            return upScale;
        }

        /** Expert FFN width: half the fused rows, or the up rows. */
        public long nFf() {
            if (gateUp != null) {
                long rows = gateUp.ne(1);
                if (rows % 2 != 0) {
                    throw new IllegalStateException("architecture '" + arch + "': " + gateUp.name() + " has an odd row count "
                            + rows + "; gate and up halves must be equal");
                }
                return rows / 2;
            }
            return up.ne(1);
        }

        /** Number of experts ({@code ne[2]}). */
        public long nExpert() {
            return gateUp != null ? gateUp.ne(2) : up.ne(2);
        }
    }
}

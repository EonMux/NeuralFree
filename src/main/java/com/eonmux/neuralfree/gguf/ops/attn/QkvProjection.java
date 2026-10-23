package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Projection plan of one softmax-attention layer, shared by {@link GpuAttentionMixer} and {@link CpuAttentionMixer}:
 * which tensors produce Q, K, V and the output gate from the mixer input, and which produce the branch output from
 * the attention result. Tensor names are layer suffixes ({@code attn_q.weight}), looked up in the layer's own stack
 * at its weight layer. Immutable.
 *
 * <p>Input projections follow llama.cpp {@code build_qkv} (src/llama-graph.cpp:1807-1912):
 * <ul>
 *   <li>{@link Source#FUSED}: one {@code attn_qkv} product whose rows hold {@code [Q | K | V]} per token
 *       ({@link #qOffset()}, {@link #kOffset()}, {@link #vOffset()} within a row of {@link #fusedWidth()}), then
 *       {@code attn_qkv.bias}, or the separate {@code attn_q/k/v.bias} when all three exist (llama-graph.cpp:
 *       1826-1834);</li>
 *   <li>{@link Source#SEPARATE}: {@code attn_q}, {@code attn_k}, {@code attn_v}, each with its optional bias; on a
 *       k2-horizon MoVA layer V comes from the routed value experts instead ({@link #valueExperts()}: router
 *       {@code attn_v_gate} with its selection bias, experts {@code attn_v_exps}, models/k2-horizon.cpp:137-205,
 *       248);</li>
 *   <li>the KQV clamp applies after the biases (llama-graph.cpp:1835-1838, 1873-1903).</li>
 * </ul>
 * Graphs that project without {@code build_qkv} add none of these biases ({@link AttentionSpec#inBias()}: mimo2,
 * qwen4exp).
 * A {@link AttnGateKind#Q_INTERLEAVED} gate widens the Q projection to {@code 2 * n_head * head_dim} with
 * {@code [q_h | gate_h]} per head (models/qwen35.cpp:277-305); the other gate kinds project {@code attn_gate} from the mixer
 * input (models/beam.cpp:180-188, models/laguna.cpp:211-259). A layer that reads another layer's cache projects no
 * K/V and adds no input bias: Q is {@code attn_q @ x}, or the first {@link #qProjWidth()} values of each
 * {@code attn_qkv} row (models/gemma3n.cpp:150, 173-182; models/gemma4.cpp:216-226, 277-281). In gemma4 and dflash a
 * layer without {@code attn_v} uses the raw K projection as V, before the K norm and RoPE (models/gemma4.cpp:253-255,
 * models/dflash.cpp:864-865). The output is {@code attn_output} ({@code attn_o} in the T5 stacks, models/t5.cpp) with
 * its optional bias, unless the graph passes no bias although the file may carry one
 * ({@link AttentionSpec#outBias()}: falcon-h1, ernie4_5, ..., models/falcon-h1.cpp:151-153,
 * models/ernie4-5.cpp:118-120), preceded by bitnet's
 * {@code attn_sub_norm} (models/bitnet.cpp:100-108) or by muse-spark's weightless per-head RMS norm, which runs before
 * the output gate (models/muse-spark.cpp:142-151). The linear replacement of deci is {@code wo @ x} alone
 * (models/deci.cpp:114-117).</p>
 */
public final class QkvProjection {

    /** Architectures whose graphs take V from the K projection when {@code attn_v} is absent. */
    private static final Set<Arch> V_FROM_K_ARCHS = Collections.unmodifiableSet(EnumSet.of(Arch.GEMMA4, Arch.DFLASH));

    /** Architectures that RMS-normalize each head of the attention output, without a weight, before the gate. */
    private static final Set<Arch> OUT_HEAD_NORM_ARCHS = Collections.unmodifiableSet(EnumSet.of(Arch.MUSE_SPARK));

    /** How Q, K and V are produced. */
    public enum Source {
        /** Three products {@code attn_q}, {@code attn_k}, {@code attn_v}. */
        SEPARATE,
        /** One {@code attn_qkv} product split by row offsets. */
        FUSED,
        /** No Q/K/V: deci's linear replacement. */
        NONE
    }

    private final int layer;
    private final Source source;
    private final boolean projectsKv;
    private final boolean vFromK;
    private final boolean interleavedGate;
    private final int qDim;
    private final int kDim;
    private final int vDim;
    private final int outDim;
    private final int qProjWidth;
    private final String qWeight;
    private final String kWeight;
    private final String vWeight;
    private final String qkvWeight;
    private final String qkvBias;
    private final String qBias;
    private final String kBias;
    private final String vBias;
    private final String gateWeight;
    private final String gateBias;
    private final String outWeight;
    private final String outBias;
    private final String subNormWeight;
    private final String subNormBias;
    private final boolean outHeadNorm;
    private final boolean valueExperts;
    private final String vExpsWeight;
    private final String vGateWeight;
    private final String vGateBias;

    private QkvProjection(Builder b) {
        this.layer = b.layer;
        this.source = b.source;
        this.projectsKv = b.projectsKv;
        this.vFromK = b.vFromK;
        this.interleavedGate = b.interleavedGate;
        this.qDim = b.qDim;
        this.kDim = b.kDim;
        this.vDim = b.vDim;
        this.outDim = b.outDim;
        this.qProjWidth = b.qProjWidth;
        this.qWeight = b.qWeight;
        this.kWeight = b.kWeight;
        this.vWeight = b.vWeight;
        this.qkvWeight = b.qkvWeight;
        this.qkvBias = b.qkvBias;
        this.qBias = b.qBias;
        this.kBias = b.kBias;
        this.vBias = b.vBias;
        this.gateWeight = b.gateWeight;
        this.gateBias = b.gateBias;
        this.outWeight = b.outWeight;
        this.outBias = b.outBias;
        this.subNormWeight = b.subNormWeight;
        this.subNormBias = b.subNormBias;
        this.outHeadNorm = b.outHeadNorm;
        this.valueExperts = b.valueExperts;
        this.vExpsWeight = b.vExpsWeight;
        this.vGateWeight = b.vGateWeight;
        this.vGateBias = b.vGateBias;
    }

    /**
     * Plan of {@code layer}'s attention.
     *
     * @param has  true when the layer has a tensor of the given suffix (in its stack at its weight layer)
     * @param arch the model architecture (selects the graphs that take V from the K projection)
     * @throws IllegalArgumentException      when the layer has no attention spec
     * @throws UnsupportedOperationException for value experts on a fused {@code attn_qkv} layer (no llama.cpp graph
     *                                       has one), or an interleaved gate or a V taken from K whose Q/K and V
     *                                       head dims differ
     */
    public static QkvProjection of(LayerSpec layer, Predicate<String> has, Arch arch) {
        Objects.requireNonNull(layer, "layer");
        Objects.requireNonNull(has, "has");
        Objects.requireNonNull(arch, "arch");
        AttentionSpec a = layer.attention();
        if (a == null) {
            throw new IllegalArgumentException("layer " + layer.il() + " (" + layer.mixerKind()
                    + ") has no attention spec");
        }
        Builder b = new Builder(layer.il());
        b.qDim = a.qDim();
        b.kDim = a.kDim();
        b.vDim = a.vDim();
        b.outDim = a.outDim();
        boolean t5Stack = TensorNames.STACK_ENC.equals(layer.stack()) || TensorNames.STACK_DEC.equals(layer.stack());
        String outBase = t5Stack ? TensorNames.DEC_ATTN_OUT : TensorNames.ATTN_OUT;
        b.outWeight = TensorNames.weight(outBase);
        boolean linear = a.linearReplace() || layer.mixerKind() == MixerKind.ATTENTION_LINEAR_REPLACE;
        if (linear) {
            b.source = Source.NONE;
            return new QkvProjection(b);
        }
        b.outBias = a.outBias() ? optional(has, TensorNames.bias(outBase)) : null;
        NormSpec norm = layer.norm();
        if (norm != null && norm.attnSubNorm()) {
            b.subNormWeight = TensorNames.weight(TensorNames.ATTN_SUB_NORM);
            b.subNormBias = optional(has, TensorNames.bias(TensorNames.ATTN_SUB_NORM));
        }
        b.outHeadNorm = OUT_HEAD_NORM_ARCHS.contains(arch);
        b.projectsKv = !a.usesMemory() || !a.sharesKv(layer.il());
        boolean inBias = b.projectsKv && a.inBias();
        b.interleavedGate = a.gate() == AttnGateKind.Q_INTERLEAVED;
        if (b.interleavedGate && a.headDimK() != a.headDimV()) {
            throw new UnsupportedOperationException("layer " + layer.il() + ": a Q-interleaved gate needs equal Q and"
                    + " V head dims, got " + a.headDimK() + " and " + a.headDimV());
        }
        b.qProjWidth = b.interleavedGate ? 2 * b.qDim : b.qDim;
        if (a.gate() != AttnGateKind.NONE && !b.interleavedGate) {
            b.gateWeight = TensorNames.weight(TensorNames.ATTN_GATE);
            b.gateBias = optional(has, TensorNames.bias(TensorNames.ATTN_GATE));
        }
        String qkvW = TensorNames.weight(TensorNames.ATTN_QKV);
        boolean fused = a.fusedQkv() || has.test(qkvW);
        if (fused && a.valueExperts() != null) {
            throw new UnsupportedOperationException("layer " + layer.il() + " routes V through value experts but has"
                    + " a fused attn_qkv projection");
        }
        if (fused) {
            b.source = Source.FUSED;
            b.qkvWeight = qkvW;
            b.qkvBias = inBias ? optional(has, TensorNames.bias(TensorNames.ATTN_QKV)) : null;
            if (inBias && b.qkvBias == null) {
                String qb = TensorNames.bias(TensorNames.ATTN_Q);
                String kb = TensorNames.bias(TensorNames.ATTN_K);
                String vb = TensorNames.bias(TensorNames.ATTN_V);
                if (has.test(qb) && has.test(kb) && has.test(vb)) {
                    b.qBias = qb;
                    b.kBias = kb;
                    b.vBias = vb;
                }
            }
        } else {
            b.source = Source.SEPARATE;
            b.qWeight = TensorNames.weight(TensorNames.ATTN_Q);
            b.qBias = inBias ? optional(has, TensorNames.bias(TensorNames.ATTN_Q)) : null;
            if (b.projectsKv) {
                b.kWeight = TensorNames.weight(TensorNames.ATTN_K);
                b.kBias = inBias ? optional(has, TensorNames.bias(TensorNames.ATTN_K)) : null;
                String vw = TensorNames.weight(TensorNames.ATTN_V);
                if (a.valueExperts() != null) {
                    b.valueExperts = true;
                    b.vExpsWeight = TensorNames.weight(TensorNames.ATTN_V_EXPS);
                    b.vGateWeight = TensorNames.weight(TensorNames.ATTN_V_GATE);
                    b.vGateBias = a.valueExperts().probBias() ? TensorNames.bias(TensorNames.ATTN_V_GATE) : null;
                } else if (!has.test(vw) && V_FROM_K_ARCHS.contains(arch)) {
                    if (a.headDimK() != a.headDimV()) {
                        throw new UnsupportedOperationException("layer " + layer.il() + " takes V from the K"
                                + " projection but its K and V head dims differ: " + a.headDimK() + " and "
                                + a.headDimV());
                    }
                    b.vFromK = true;
                } else {
                    b.vWeight = vw;
                    b.vBias = inBias ? optional(has, TensorNames.bias(TensorNames.ATTN_V)) : null;
                }
            }
        }
        return new QkvProjection(b);
    }

    private static String optional(Predicate<String> has, String suffix) {
        return has.test(suffix) ? suffix : null;
    }

    /**
     * Validates a projection weight's shape: {@code cols} inputs and {@code rows} outputs.
     *
     * @throws IllegalArgumentException naming the tensor when the shape differs
     */
    public static void checkShape(String name, long actualCols, long actualRows, long cols, long rows) {
        if (actualCols != cols || actualRows != rows) {
            throw new IllegalArgumentException("attention tensor " + name + " is [" + actualCols + " x " + actualRows
                    + "], expected [" + cols + " x " + rows + "]");
        }
    }

    /**
     * Validates a vector tensor (bias, norm weight, gate bias) holding {@code n} values.
     *
     * @throws IllegalArgumentException naming the tensor when the element count differs
     */
    public static void checkLength(String name, long elements, long n) {
        if (elements != n) {
            throw new IllegalArgumentException("attention tensor " + name + " holds " + elements + " values, expected "
                    + n);
        }
    }

    /**
     * True when a per-head Q/K norm tensor of shape {@code {cols, rows}} holds one scalar per head
     * ({@code {1, nHeads}}, talkie's q_norm, models/talkie.cpp:26) rather than {@code rowLen} values per row; ggml
     * broadcasts it over each head's {@code rowLen} normalized values, so the mixers expand it to
     * {@code [nHeads][rowLen]}.
     */
    public static boolean isPerHeadScalar(QkNormKind kind, long cols, long rows, int nHeads, int rowLen) {
        return (kind == QkNormKind.RMS_HEAD || kind == QkNormKind.LN_HEAD) && rowLen > 1 && cols == 1
                && rows == nHeads;
    }

    /** Main-stack layer index of the plan. */
    public int layer() { return layer; }
    public Source source() { return source; }
    /** True when the layer computes and stores its own K and V (false for a shared-KV reader). */
    public boolean projectsKv() { return projectsKv; }
    /** True when V is a copy of the raw K projection (no {@code attn_v}; gemma4, dflash). */
    public boolean vFromK() { return vFromK; }
    /** True when the Q projection carries the interleaved output gate. */
    public boolean interleavedGate() { return interleavedGate; }
    public int qDim() { return qDim; }
    public int kDim() { return kDim; }
    public int vDim() { return vDim; }
    public int outDim() { return outDim; }
    /** Width of the Q projection output: {@code 2 * qDim} with an interleaved gate, else {@code qDim}. */
    public int qProjWidth() { return qProjWidth; }
    /** Row width of the fused product: {@code qProjWidth + kDim + vDim}. */
    public int fusedWidth() { return qProjWidth + kDim + vDim; }
    /** Offset of Q in a fused row. */
    public int qOffset() { return 0; }
    /** Offset of K in a fused row. */
    public int kOffset() { return qProjWidth; }
    /** Offset of V in a fused row. */
    public int vOffset() { return qProjWidth + kDim; }
    /** {@code attn_q.weight} for {@link Source#SEPARATE}; null otherwise. */
    public String qWeight() { return qWeight; }
    /** {@code attn_k.weight} when a separate K is projected; null otherwise. */
    public String kWeight() { return kWeight; }
    /**
     * {@code attn_v.weight} when a separate V is projected; null otherwise (also when {@link #vFromK()} or
     * {@link #valueExperts()}).
     */
    public String vWeight() { return vWeight; }
    /** {@code attn_qkv.weight} for {@link Source#FUSED}; null otherwise. */
    public String qkvWeight() { return qkvWeight; }
    /** {@code attn_qkv.bias} when present and added ({@link AttentionSpec#inBias()}, K/V projected); else null. */
    public String qkvBias() { return qkvBias; }
    /**
     * Q bias ({@link #qProjWidth()} values) applied after the projection or split; null when absent, when the graph
     * adds no input bias ({@link AttentionSpec#inBias()}) and on a layer that reads another layer's cache.
     */
    public String qBias() { return qBias; }
    public String kBias() { return kBias; }
    public String vBias() { return vBias; }
    /** {@code attn_gate.weight} for the gate kinds projected from the input; null otherwise. */
    public String gateWeight() { return gateWeight; }
    /** Per-head {@code attn_gate.bias} when present; null otherwise. */
    public String gateBias() { return gateBias; }
    /** {@code attn_output.weight} or {@code attn_o.weight}. */
    public String outWeight() { return outWeight; }
    public String outBias() { return outBias; }
    /** {@code attn_sub_norm.weight} when the layer has a sub-norm before the output projection; null otherwise. */
    public String subNormWeight() { return subNormWeight; }
    public String subNormBias() { return subNormBias; }
    /**
     * True when each {@code head_dim_v} row of the attention output is RMS-normalized without a weight
     * ({@code f_norm_rms_eps}) before the output gate and {@code wo} (muse-spark).
     */
    public boolean outHeadNorm() { return outHeadNorm; }
    /**
     * True when V is the weighted sum of routed value experts ({@link AttentionSpec#valueExperts()}, k2-horizon
     * MoVA); the layer then has no {@code attn_v}.
     */
    public boolean valueExperts() { return valueExperts; }
    /** {@code attn_v_exps.weight} ({@code [n_embd, v_dim, n_value_expert]}) of a MoVA layer; null otherwise. */
    public String vExpsWeight() { return vExpsWeight; }
    /** {@code attn_v_gate.weight}, the value router ({@code [n_embd, n_value_expert]}); null without MoVA. */
    public String vGateWeight() { return vGateWeight; }
    /** {@code attn_v_gate.bias}, which biases the expert selection only; null when the router has none. */
    public String vGateBias() { return vGateBias; }

    @Override
    public String toString() {
        return "QkvProjection{layer=" + layer + ", " + source + (projectsKv ? "" : ", shared KV")
                + (vFromK ? ", V from K" : "") + (valueExperts ? ", value experts" : "")
                + (interleavedGate ? ", interleaved gate" : "") + ", q=" + qProjWidth + ", k=" + kDim + ", v=" + vDim
                + ", out=" + outWeight + '}';
    }

    private static final class Builder {
        private final int layer;
        private Source source = Source.SEPARATE;
        private boolean projectsKv;
        private boolean vFromK;
        private boolean interleavedGate;
        private int qDim;
        private int kDim;
        private int vDim;
        private int outDim;
        private int qProjWidth;
        private String qWeight;
        private String kWeight;
        private String vWeight;
        private String qkvWeight;
        private String qkvBias;
        private String qBias;
        private String kBias;
        private String vBias;
        private String gateWeight;
        private String gateBias;
        private String outWeight;
        private String outBias;
        private String subNormWeight;
        private String subNormBias;
        private boolean outHeadNorm;
        private boolean valueExperts;
        private String vExpsWeight;
        private String vGateWeight;
        private String vGateBias;

        private Builder(int layer) {
            this.layer = layer;
        }
    }
}

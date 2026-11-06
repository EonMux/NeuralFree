package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkVector;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * Validated tensors of one block of the pre-norm ViT tower every clip encoder shares (llama.cpp
 * {@code clip_layer} as {@code build_vit} reads it): layer norm 1, fused or separate Q/K/V (K and V
 * {@code n_head_kv} heads wide), the optional Q/K norms, the output projection and its optional layer scale,
 * layer norm 2, the gated or plain FFN and its optional layer scale, under the modality's tensor prefix
 * ({@code v.blk.N.} or {@code a.blk.N.}). A block that carries
 * {@code ffn_gate_exps} replaces the dense FFN with the sigmoid-routed experts of clip.cpp {@code build_moe_ffn}
 * (dots3note): router {@code ffn_gate_inp}, the stacked gate/up/down experts and the optional selection bias
 * {@code exp_probs_b}. The {@code build_vit} encoders also read the Gemma 4 block tensors when present: the
 * post-attention and post-FFN norms {@code attn_post_norm} / {@code ffn_post_norm} and the block output scale
 * {@code out_scale}; a {@link Spec} adds the clippable-linear bounds ({@link ClipClamp}) of Gemma 4 and the
 * per-head attention sinks {@code attn_sinks} of MiMo-VL, and fixes the head width of a grouped-query tower whose
 * heads are not {@code n_embd / n_head} wide (Q is {@code n_head * d_head} wide, K and V
 * {@code n_head_kv * d_head}).
 *
 * <p>The legacy swap of old llava / mobilevlm / glm-edge / qwen-vl / exaone4_5 / gemma3 / idefics3 / minicpmv
 * conversions and of the CLIP two-tower files ({@link ClipHparams#twoTower()}) applies here: when a file's
 * {@code ffn_down.weight} holds the {@code [n_embd, n_ff]} shape, the up/down pair is swapped
 * (clip.cpp {@code is_ffn_swapped}). Absent optionals are null. Immutable; the references view the file.
 */
public final class ClipLayer {

    final boolean fusedQkv;
    final TensorRef qkvW;
    final TensorRef qkvB;
    final TensorRef qW;
    final TensorRef qB;
    final TensorRef kW;
    final TensorRef kB;
    final TensorRef vW;
    final TensorRef vB;
    final TensorRef oW;
    final TensorRef oB;
    final TensorRef ln1W;
    final TensorRef ln1B;
    final TensorRef ln2W;
    final TensorRef ln2B;
    final TensorRef ffnUpW;
    final TensorRef ffnUpB;
    final TensorRef ffnGateW;
    final TensorRef ffnGateB;
    final TensorRef ffnDownW;
    final TensorRef ffnDownB;
    final TensorRef ls1W;
    final TensorRef ls2W;
    final TensorRef qNormW;
    final TensorRef kNormW;
    final TensorRef moeRouterW;
    final TensorRef moeGateW;
    final TensorRef moeUpW;
    final TensorRef moeDownW;
    final TensorRef moeSelectBias;
    final int nExpert;
    final int nFfExpert;
    final TensorRef attnPostNormW;
    final TensorRef ffnPostNormW;
    final TensorRef outScaleW;
    final TensorRef attnSinks;
    final int dHead;
    private final Map<String, ClipClamp> clamps;

    /** What a tower reads beyond the plain block: see {@link #resolve(TensorCatalog, ClipHparams, int, Spec)}. */
    static final class Spec {
        /** The plain block. */
        static final Spec PLAIN = new Spec(false, false, false, false, 0);
        /** The plain block plus the {@code build_vit} extras. */
        static final Spec EXTRAS = new Spec(true, true, false, false, 0);
        /** The plain block plus only the layer scales {@code ls1} and {@code ls2} (MiniCPM-V 4.6). */
        static final Spec LAYER_SCALES = new Spec(false, true, false, false, 0);

        final boolean extras;
        final boolean layerScales;
        final boolean clamps;
        final boolean sinks;
        final int dHead;

        private Spec(boolean extras, boolean layerScales, boolean clamps, boolean sinks, int dHead) {
            this.extras = extras;
            this.layerScales = layerScales;
            this.clamps = clamps;
            this.sinks = sinks;
            this.dHead = dHead;
        }

        /** Also resolves the clippable-linear bounds of every block matrix. */
        Spec withClamps() {
            return new Spec(extras, layerScales, true, sinks, dHead);
        }

        /** Also resolves the per-head attention sinks ({@code n_head} values) when the block carries them. */
        Spec withSinks() {
            return new Spec(extras, layerScales, clamps, true, dHead);
        }

        /** Heads {@code dHead} wide instead of {@code n_embd / n_head}. */
        Spec withHeadDim(int headDim) {
            if (headDim <= 0) {
                throw new IllegalArgumentException("head width " + headDim + " is not positive");
            }
            return new Spec(extras, layerScales, clamps, sinks, headDim);
        }
    }

    /**
     * Weight/bias slot pairs in order: qkv, q, k, v, attn_out, ln1, ln2, ffn_up, ffn_gate, ffn_down; then ls1,
     * ls2, the Q norm and the K norm; then the router, the gate/up/down experts and the selection bias.
     */
    private ClipLayer(TensorRef[] t, int dHead, Map<String, ClipClamp> clamps) {
        this.qkvW = t[0];
        this.qkvB = t[1];
        this.fusedQkv = qkvW != null;
        this.qW = t[2];
        this.qB = t[3];
        this.kW = t[4];
        this.kB = t[5];
        this.vW = t[6];
        this.vB = t[7];
        this.oW = t[8];
        this.oB = t[9];
        this.ln1W = t[10];
        this.ln1B = t[11];
        this.ln2W = t[12];
        this.ln2B = t[13];
        this.ffnUpW = t[14];
        this.ffnUpB = t[15];
        this.ffnGateW = t[16];
        this.ffnGateB = t[17];
        this.ffnDownW = t[18];
        this.ffnDownB = t[19];
        this.ls1W = t[20];
        this.ls2W = t[21];
        this.qNormW = t[22];
        this.kNormW = t[23];
        this.moeRouterW = t[24];
        this.moeGateW = t[25];
        this.moeUpW = t[26];
        this.moeDownW = t[27];
        this.moeSelectBias = t[28];
        this.nExpert = moeGateW == null ? 0 : (int) moeGateW.ne(2);
        this.nFfExpert = moeGateW == null ? 0 : (int) moeGateW.ne(1);
        this.attnPostNormW = t[29];
        this.ffnPostNormW = t[30];
        this.outScaleW = t[31];
        this.attnSinks = t[32];
        this.dHead = dHead;
        this.clamps = clamps;
    }

    /**
     * Block {@code il} of the encoder {@code hp} describes, shape-checked against its widths. With
     * {@code extras} the layer scales and Q/K norms resolve when the file carries them (the encoders
     * {@link MmprojTensorTable#readsVitExtras} names); without, they stay null.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     * @throws IllegalArgumentException                               when a shape differs from the loader's
     */
    static ClipLayer resolve(TensorCatalog tc, ClipHparams hp, int il, boolean extras) {
        return resolve(tc, hp, il, extras ? Spec.EXTRAS : Spec.PLAIN);
    }

    /**
     * Block {@code il} as {@code spec} reads it: the {@link #resolve(TensorCatalog, ClipHparams, int, boolean)}
     * tensors, the clamp bounds and attention sinks when the spec asks for them, Q/K/V shaped for its head width.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     * @throws IllegalArgumentException                               when a shape differs from the loader's
     */
    static ClipLayer resolve(TensorCatalog tc, ClipHparams hp, int il, Spec spec) {
        String p = hp.modality().tensorPrefix() + ".blk." + il + ".";
        int nEmbd = hp.nEmbd();
        int dh = spec.dHead > 0 ? spec.dHead : hp.dHead();
        int qWidth = hp.nHead() * dh;
        int kvWidth = hp.nHeadKv() * dh;
        TensorRef[] t = new TensorRef[33]; // the constructor's slot order
        t[0] = tc.get(p + "attn_qkv.weight");
        if (t[0] != null) {
            t[1] = tc.get(p + "attn_qkv.bias");
            shape(tc, t[0], nEmbd, qWidth + 2 * kvWidth);
            checkVector(tc, t[1], qWidth + 2 * kvWidth);
        } else {
            // K and V project to the n_head_kv heads of a grouped-query encoder (build_vit)
            for (int i = 0; i < 3; i++) {
                String name = p + "attn_" + "qkv".charAt(i);
                int width = i == 0 ? qWidth : kvWidth;
                t[2 + 2 * i] = tc.require(name + ".weight");
                t[3 + 2 * i] = tc.get(name + ".bias");
                shape(tc, t[2 + 2 * i], nEmbd, width);
                checkVector(tc, t[3 + 2 * i], width);
            }
        }
        t[8] = tc.require(p + "attn_out.weight");
        t[9] = tc.get(p + "attn_out.bias");
        shape(tc, t[8], qWidth, nEmbd);
        checkVector(tc, t[9], nEmbd);
        for (int n = 1; n <= 2; n++) {
            t[8 + 2 * n] = tc.require(p + "ln" + n + ".weight");
            t[9 + 2 * n] = tc.get(p + "ln" + n + ".bias");
            shape(tc, t[8 + 2 * n], nEmbd);
            checkVector(tc, t[9 + 2 * n], nEmbd);
        }
        if (tc.get(p + "ffn_gate_exps.weight") != null) {
            moe(tc, hp, p, t);
        } else {
            ffn(tc, hp, p, t);
        }
        if (spec.extras) {
            extras(tc, p, nEmbd, dh, qWidth, kvWidth, t);
        } else if (spec.layerScales) {
            layerScales(tc, p, nEmbd, t);
        }
        if (spec.sinks) {
            t[32] = tc.get(p + "attn_sinks");
            checkVector(tc, t[32], hp.nHead());
        }
        return new ClipLayer(t, dh, spec.clamps ? clamps(tc, t) : Collections.<String, ClipClamp>emptyMap());
    }

    /** The layer scales, Q/K norms, post norms and output scale {@code build_vit} applies when present. */
    private static void extras(TensorCatalog tc, String p, int nEmbd, int dh, int qWidth, int kvWidth,
                               TensorRef[] t) {
        layerScales(tc, p, nEmbd, t);
        t[22] = qkNorm(tc, tc.get(p + "attn_q_norm.weight"), t[0] != null, dh, qWidth);
        t[23] = qkNorm(tc, tc.get(p + "attn_k_norm.weight"), t[0] != null, dh, kvWidth);
        t[29] = tc.get(p + "attn_post_norm.weight");
        t[30] = tc.get(p + "ffn_post_norm.weight");
        t[31] = tc.get(p + "out_scale.weight");
        checkVector(tc, t[29], nEmbd);
        checkVector(tc, t[30], nEmbd);
        checkVector(tc, t[31], nEmbd);
    }

    /** The layer scales {@code ls1} and {@code ls2}, when present. */
    private static void layerScales(TensorCatalog tc, String p, int nEmbd, TensorRef[] t) {
        t[20] = tc.get(p + "ls1.weight");
        t[21] = tc.get(p + "ls2.weight");
        checkVector(tc, t[20], nEmbd);
        checkVector(tc, t[21], nEmbd);
    }

    /** The clamp bounds of every matrix slot that carries any (Gemma 4 clippable linears). */
    private static Map<String, ClipClamp> clamps(TensorCatalog tc, TensorRef[] t) {
        Map<String, ClipClamp> m = new LinkedHashMap<>();
        for (int i : new int[]{0, 2, 4, 6, 8, 14, 16, 18}) {
            if (t[i] != null) {
                ClipClamp c = ClipClamp.resolve(tc, t[i].name());
                if (c != null) {
                    m.put(t[i].name(), c);
                }
            }
        }
        return Collections.unmodifiableMap(m);
    }

    /**
     * A Q/K norm is per head for a fused projection, per head or over the projection's whole {@code rowWidth}
     * for separate ones (build_vit).
     */
    private static TensorRef qkNorm(TensorCatalog tc, TensorRef r, boolean fused, int dh, int rowWidth) {
        if (r == null) {
            return null;
        }
        if (fused || r.ne(0) == dh) {
            shape(tc, r, dh);
        } else {
            shape(tc, r, rowWidth);
        }
        return r;
    }

    /** The routed-expert FFN: shapes from the gate experts, checked against the router and the other experts. */
    private static void moe(TensorCatalog tc, ClipHparams hp, String p, TensorRef[] t) {
        TensorRef gate = tc.require(p + "ffn_gate_exps.weight");
        long nFfExp = gate.ne(1);
        long nExp = gate.ne(2);
        if (gate.rank() != 3 || nFfExp <= 0 || nExp <= 0) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + gate.name()
                    + " has shape " + gate + ", the loader expects {" + hp.nEmbd() + ", n_ff, n_expert}");
        }
        shape(tc, gate, hp.nEmbd(), nFfExp, nExp);
        t[24] = tc.require(p + "ffn_gate_inp.weight");
        shape(tc, t[24], hp.nEmbd(), nExp);
        t[25] = gate;
        t[26] = tc.require(p + "ffn_up_exps.weight");
        shape(tc, t[26], hp.nEmbd(), nFfExp, nExp);
        t[27] = tc.require(p + "ffn_down_exps.weight");
        shape(tc, t[27], nFfExp, hp.nEmbd(), nExp);
        t[28] = tc.get(p + "exp_probs_b.weight");
        checkVector(tc, t[28], (int) nExp);
    }

    private static void ffn(TensorCatalog tc, ClipHparams hp, String p, TensorRef[] t) {
        TensorRef upW = tc.require(p + "ffn_up.weight");
        TensorRef upB = tc.get(p + "ffn_up.bias");
        TensorRef downW = tc.require(p + "ffn_down.weight");
        TensorRef downB = tc.get(p + "ffn_down.bias");
        if (hp.twoTower() ? downW.ne(0) == hp.nEmbd() : isLegacyFfnSwapped(hp.projectorType(), downW, hp.nEmbd())) {
            TensorRef w = upW;
            upW = downW;
            downW = w;
            TensorRef b = upB;
            upB = downB;
            downB = b;
        }
        shape(tc, upW, hp.nEmbd(), hp.nFf());
        shape(tc, downW, hp.nFf(), hp.nEmbd());
        checkVector(tc, upB, hp.nFf());
        checkVector(tc, downB, hp.nEmbd());
        TensorRef gateW = tc.get(p + "ffn_gate.weight");
        TensorRef gateB = tc.get(p + "ffn_gate.bias");
        if (gateW != null) {
            shape(tc, gateW, hp.nEmbd(), hp.nFf());
        }
        checkVector(tc, gateB, hp.nFf());
        t[14] = upW;
        t[15] = upB;
        t[16] = gateW;
        t[17] = gateB;
        t[18] = downW;
        t[19] = downB;
    }

    /** The old conversions of these families store ffn up/down swapped (clip.cpp is_ffn_swapped). */
    private static boolean isLegacyFfnSwapped(ProjectorType type, TensorRef downW, int nEmbd) {
        switch (type) {
            case MLP:
            case MLP_NORM:
            case LDP:
            case LDPV2:
            case GLM_EDGE:
            case QWEN2VL:
            case QWEN25VL:
            case EXAONE4_5:
            case GEMMA3:
            case IDEFICS3:
            case MINICPMV:
            case MINICPMV4_6:
            case MINICPMV4_7:
            case MINICPM_OCR_1_0:
                return downW.ne(0) == nEmbd;
            default:
                return false;
        }
    }

    public boolean fusedQkv() {
        return fusedQkv;
    }

    public TensorRef qkvWeight() {
        return qkvW;
    }

    public TensorRef qkvBias() {
        return qkvB;
    }

    public TensorRef qWeight() {
        return qW;
    }

    public TensorRef qBias() {
        return qB;
    }

    public TensorRef kWeight() {
        return kW;
    }

    public TensorRef kBias() {
        return kB;
    }

    public TensorRef vWeight() {
        return vW;
    }

    public TensorRef vBias() {
        return vB;
    }

    public TensorRef oWeight() {
        return oW;
    }

    public TensorRef oBias() {
        return oB;
    }

    public TensorRef ln1Weight() {
        return ln1W;
    }

    public TensorRef ln1Bias() {
        return ln1B;
    }

    public TensorRef ln2Weight() {
        return ln2W;
    }

    public TensorRef ln2Bias() {
        return ln2B;
    }

    public TensorRef ffnUpWeight() {
        return ffnUpW;
    }

    public TensorRef ffnUpBias() {
        return ffnUpB;
    }

    public TensorRef ffnGateWeight() {
        return ffnGateW;
    }

    public TensorRef ffnGateBias() {
        return ffnGateB;
    }

    public TensorRef ffnDownWeight() {
        return ffnDownW;
    }

    public TensorRef ffnDownBias() {
        return ffnDownB;
    }

    /** Layer scale of the attention output ({@code ls1}), or null. */
    public TensorRef ls1Weight() {
        return ls1W;
    }

    /** Layer scale of the FFN output ({@code ls2}), or null. */
    public TensorRef ls2Weight() {
        return ls2W;
    }

    /** Q norm weight, per head ({@code d_head} wide) or over the whole row ({@code n_embd}), or null. */
    public TensorRef qNormWeight() {
        return qNormW;
    }

    /** K norm weight, per head or over the whole row, or null. */
    public TensorRef kNormWeight() {
        return kNormW;
    }

    /** The post-attention norm weight ({@code attn_post_norm}, no bias), or null. */
    public TensorRef attnPostNormWeight() {
        return attnPostNormW;
    }

    /** The post-FFN norm weight ({@code ffn_post_norm}, no bias), or null. */
    public TensorRef ffnPostNormWeight() {
        return ffnPostNormW;
    }

    /** The block output scale ({@code out_scale}) multiplied into the stream after the second residual, or null. */
    public TensorRef outScaleWeight() {
        return outScaleW;
    }

    /** The per-head attention sink logits ({@code attn_sinks}, {@code n_head} values), or null. */
    public TensorRef attnSinks() {
        return attnSinks;
    }

    /** Head width of the block's attention. */
    public int dHead() {
        return dHead;
    }

    /** The clamp bounds of the block matrix {@code weight}, or null when it has none. */
    ClipClamp clamp(TensorRef weight) {
        return weight == null ? null : clamps.get(weight.name());
    }

    /** True when the block runs the routed experts instead of the dense FFN. */
    public boolean moe() {
        return moeGateW != null;
    }

    /** The router {@code ffn_gate_inp} ({@code [n_embd, n_expert]}), or null for a dense block. */
    public TensorRef moeRouterWeight() {
        return moeRouterW;
    }

    /** The stacked gate experts {@code [n_embd, n_ff_exp, n_expert]}, or null. */
    public TensorRef moeGateWeight() {
        return moeGateW;
    }

    /** The stacked up experts {@code [n_embd, n_ff_exp, n_expert]}, or null. */
    public TensorRef moeUpWeight() {
        return moeUpW;
    }

    /** The stacked down experts {@code [n_ff_exp, n_embd, n_expert]}, or null. */
    public TensorRef moeDownWeight() {
        return moeDownW;
    }

    /** The selection bias {@code exp_probs_b} ({@code [n_expert]}), or null. */
    public TensorRef moeSelectBias() {
        return moeSelectBias;
    }

    /** Experts of a MoE block, 0 for a dense block. */
    public int nExpert() {
        return nExpert;
    }

    /** Hidden width of one expert, 0 for a dense block. */
    public int nFfExpert() {
        return nFfExpert;
    }
}

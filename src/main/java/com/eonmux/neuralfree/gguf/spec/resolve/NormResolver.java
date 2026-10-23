package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.PostNormRole;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;

/**
 * Resolves the normalizations of each layer into a {@link NormSpec}. Tensor
 * presence decides placement:
 * <ul>
 *   <li>POST_ONLY when the architecture's trait says so (olmo2, exaone4: no {@code attn_norm} / {@code ffn_norm},
 *       norms on the branch outputs);</li>
 *   <li>PRE_FFN only when {@code attn_norm} and {@code post_attention_norm} exist and {@code ffn_norm} is absent
 *       ({@code post_attention_norm} is then the FFN input norm: gpt-oss, glm4moe, seed_oss, qwen3next, qwen35,
 *       qwen4exp, ...);</li>
 *   <li>SANDWICH when {@code post_attention_norm} or {@code post_ffw_norm} exists otherwise (gemma2/3/3n/4, glm4,
 *       afmoe, ...), and for grok, whose post-FFN norm is {@code layer_output_norm} or {@code post_ffw_norm}
 *       (models/grok.cpp:75-77, 185-188);</li>
 *   <li>NONE otherwise.</li>
 * </ul>
 * {@code attn_norm_2} ({@link NormSpec#attnNorm2()}) is the FFN input norm of RWKV6/7 and the attention input norm
 * of falcon-40b, whose FFN reads {@code attn_norm} of the same layer input (models/falcon.cpp:70-86, 125); the FFN
 * reuses the {@code attn_norm} output ({@link NormSpec#sharedAttnNormForFfn()}) when a layer with an FFN has
 * {@code attn_norm} but no other FFN-side norm (falcon-7b, phi2, plamo, command-r, cohere2, stablelm-2-12b,
 * Nemotron-H FFN layers). olmo, talkie and hrm_text apply weightless norms. {@code attn_output_norm} and
 * {@code layer_output_norm} are block norms only in the BERT graph (post-LN after each residual add, with no
 * {@code attn_norm}), dbrx ({@code attn_output_norm} is its FFN input norm) and grok ({@code attn_output_norm} on
 * the attention branch, {@code layer_output_norm} as the post-FFN norm); elsewhere {@code layer_output_norm} is a
 * NextN shared-head norm (bailingmoe2, mimo2) and is not part of the block. chameleon with {@code swin_norm} applies
 * {@code attn_norm} and {@code ffn_norm} to the branch outputs instead of the inputs (POST_ONLY,
 * models/chameleon.cpp:75-83, 132-136, 142-148, 157-162). A norm tensor is found under its {@code .weight} name or
 * under its bare name, which falcon-h1 ({@code ffn_norm}, models/falcon-h1.cpp:96) and plamo2/plamo3
 * ({@code post_attention_norm}, {@code post_ffw_norm}, models/plamo2.cpp:98-102, plamo3.cpp:50-53) use. Use
 * {@link #ffnNormTensor(int)}, {@link #postAttnNormTensor(int)} and {@link #postFfnNormTensor(int)} for the tensors
 * of the branch norms, which are not always named {@code ffn_norm} / {@code post_attention_norm} /
 * {@code post_ffw_norm}. Immutable.
 */
public final class NormResolver {

    /** Graphs whose block norms carry no weight tensor. */
    private static final Set<Arch> WEIGHTLESS = Collections.unmodifiableSet(EnumSet.of(
            Arch.OLMO, Arch.TALKIE, Arch.HRM_TEXT));

    /** Q/K norms that are LayerNorm ({@code LLM_NORM}) per head. */
    private static final Set<Arch> LN_QK = Collections.unmodifiableSet(EnumSet.of(
            Arch.CHAMELEON, Arch.COMMAND_R, Arch.STABLELM));

    /**
     * Graphs that normalize Q and K after RoPE: hunyuan-moe (models/hunyuan-moe.cpp:94-117), hunyuan_vl and
     * hunyuan-dense, which reuses its graph (models/hunyuan-vl.cpp:105-142, models.h:2259-2264), maincoder
     * (models/maincoder.cpp:80-95) and talkie (models/talkie.cpp:76-92).
     */
    private static final Set<Arch> QK_NORM_AFTER_ROPE = Collections.unmodifiableSet(EnumSet.of(
            Arch.HUNYUAN_MOE, Arch.HUNYUAN_DENSE, Arch.HUNYUAN_VL, Arch.MAINCODER, Arch.TALKIE));

    /** Graphs that load {@code attn_q_norm.bias} / {@code attn_k_norm.bias} but never apply them (apertus.cpp:50-52, 93-96). */
    private static final Set<Arch> QK_NORM_BIAS_UNUSED = Collections.unmodifiableSet(EnumSet.of(Arch.APERTUS));

    private final ModelHparams hp;
    private final TensorCatalog tc;
    private final Arch arch;
    private final NormKind kind;
    private final float eps;
    private final boolean bertGraph;

    /**
     * @throws IllegalStateException when the architecture's norm kind is not a row norm or its epsilon is unset
     */
    public NormResolver(ModelHparams hp, TensorCatalog tc) {
        this.hp = Objects.requireNonNull(hp, "hp");
        this.tc = Objects.requireNonNull(tc, "tc");
        this.arch = hp.arch();
        this.kind = blockNormKind(hp);
        this.bertGraph = hp.traits().graphFamily() == Arch.BERT;
        this.eps = kind == NormKind.RMS ? hp.normRmsEps() : hp.normEps();
        if (!(eps > 0f)) {
            throw new IllegalStateException(arch.ggufName() + ": " + kind + " norm epsilon is not set (got " + eps + ")");
        }
    }

    /**
     * Block norm kind: the trait's, except cohere2moe, which uses LayerNorm when the file has no RMS epsilon
     * (stored as 0) and RMS otherwise (models/cohere2moe.cpp:149).
     */
    private static NormKind blockNormKind(ModelHparams hp) {
        NormKind k = hp.traits().normKind();
        if (hp.arch() == Arch.COHERE2MOE) {
            k = hp.normRmsEps() == 0f ? NormKind.LAYER : NormKind.RMS;
        }
        if (!k.isRowNorm()) {
            throw new IllegalStateException(hp.arch().ggufName() + ": block norm kind " + k + " is not a row norm");
        }
        return k;
    }

    /** Block norm kind. */
    public NormKind kind() { return kind; }

    /** Block norm epsilon ({@code f_norm_rms_eps} for RMS, {@code f_norm_eps} for LayerNorm). */
    public float eps() { return eps; }

    /**
     * Norms of main-stack layer {@code il}.
     *
     * @throws IllegalStateException when the tensors describe a placement llama.cpp does not build
     */
    public NormSpec resolve(int il) {
        checkLayer(il);
        return resolve(TensorNames.STACK_BLK, il);
    }

    /**
     * Norms of layer {@code il} of {@code stack} (the T5 {@link TensorNames#STACK_ENC} / {@link TensorNames#STACK_DEC}
     * stacks carry {@code attn_norm} and {@code ffn_norm}; the decoder's {@code cross_attn_norm} belongs to the
     * cross-attention step).
     *
     * @throws IllegalStateException when the tensors describe a placement llama.cpp does not build
     */
    public NormSpec resolve(String stack, int il) {
        Objects.requireNonNull(stack, "stack");
        boolean main = TensorNames.STACK_BLK.equals(stack);
        boolean attnNorm = has(stack, il, TensorNames.ATTN_NORM);
        boolean ffnNorm = has(stack, il, TensorNames.FFN_NORM);
        boolean postAttn = has(stack, il, TensorNames.ATTN_POST_NORM);
        boolean postFfn = has(stack, il, TensorNames.FFN_POST_NORM);
        boolean attnOut = has(stack, il, TensorNames.ATTN_OUT_NORM);
        boolean layerOut = has(stack, il, TensorNames.LAYER_OUT_NORM);
        boolean weightless = WEIGHTLESS.contains(arch);
        NormSpec.Builder b = NormSpec.builder(kind, eps).weightless(weightless).groups(main ? normGroups(hp) : 1);
        QkNormKind qk = main ? qkNorm(il) : QkNormKind.NONE;
        b.qkNorm(qk);
        b.attnSubNorm(has(stack, il, TensorNames.ATTN_SUB_NORM));
        b.ffnSubNorm(has(stack, il, TensorNames.FFN_SUB_NORM));
        if (main && arch != Arch.TALKIE) {
            b.layerOutputScale(tc.layer(il, TensorNames.weight(TensorNames.LAYER_OUT_SCALE)));
        }

        if (weightless) {
            return b.attnNorm(true).ffnNorm(true).postRole(PostNormRole.NONE).build();
        }
        if (bertGraph) {
            return b.attnNorm(false).ffnNorm(false).postRole(PostNormRole.NONE)
                    .attnOutputNorm(attnOut).layerOutputNorm(layerOut).build();
        }
        if (arch == Arch.DBRX) {
            return b.attnNorm(attnNorm).ffnNorm(attnOut).postRole(PostNormRole.NONE).build();
        }
        if (arch == Arch.GROK) {
            boolean post = postFfn || layerOut;
            return b.attnNorm(attnNorm).ffnNorm(ffnNorm).attnOutputNorm(attnOut)
                    .postRole(post ? PostNormRole.SANDWICH : PostNormRole.NONE).postFfnNorm(post).build();
        }
        if (main && swinNorm()) {
            if (!attnNorm || !ffnNorm) {
                throw new IllegalStateException(arch.ggufName() + " layer " + il + ": swin_norm applies attn_norm and"
                        + " ffn_norm to the branch outputs, but the layer lacks " + (attnNorm ? "ffn_norm" : "attn_norm"));
            }
            return b.attnNorm(false).ffnNorm(false).postRole(PostNormRole.POST_ONLY)
                    .postAttnNorm(true).postFfnNorm(true).build();
        }
        if (hp.traits().postNormRole() == PostNormRole.POST_ONLY) {
            if (attnNorm || ffnNorm) {
                throw new IllegalStateException(arch.ggufName() + " layer " + il + ": post-only norms but the layer has "
                        + (attnNorm ? "attn_norm" : "ffn_norm"));
            }
            return b.attnNorm(false).ffnNorm(false).postRole(PostNormRole.POST_ONLY)
                    .postAttnNorm(postAttn).postFfnNorm(postFfn).build();
        }
        b.attnNorm(attnNorm);
        boolean attnNorm2 = attnNorm && has(stack, il, TensorNames.ATTN_NORM_2) && usesAttnNorm2AsFfnNorm();
        b.attnNorm2(attnNorm2);
        if (attnNorm && postAttn && !ffnNorm && !postFfn) {
            return b.ffnNorm(false).postRole(PostNormRole.PRE_FFN).postAttnNorm(true).build();
        }
        b.ffnNorm(ffnNorm);
        if (arch == Arch.MUSE_SPARK && main) {
            postAttn = true;
            b.postAttnWeightless(true).postFfnWeightFirst(postFfn);
        }
        if (postAttn || postFfn) {
            b.postRole(PostNormRole.SANDWICH).postAttnNorm(postAttn).postFfnNorm(postFfn).postEps(hp.postNormEps());
        } else {
            b.postRole(PostNormRole.NONE);
        }
        boolean shared = main && attnNorm && !ffnNorm && !attnNorm2 && !postAttn && !postFfn && hasFfn(il);
        if (main && ffnNorm) {
            ffnAda(b, il);
        }
        return b.sharedAttnNormForFfn(shared).build();
    }

    /**
     * mistral3 (Voxtral realtime): when layer 0 carries {@code ffn_norm_ada_down}, every layer without a router
     * conditions its {@code ffn_norm} output by the ada projections of that layer (models/mistral3.cpp:57-62, 154,
     * 220-236).
     *
     * @throws MissingTensorException when such a layer lacks one of its ada projections
     */
    private void ffnAda(NormSpec.Builder b, int il) {
        if (arch != Arch.MISTRAL3 || tc.layer(0, TensorNames.weight(TensorNames.FFN_NORM_ADA_DOWN)) == null
                || has(TensorNames.STACK_BLK, il, TensorNames.FFN_GATE_INP)) {
            return;
        }
        String downName = TensorNames.weight(TensorNames.FFN_NORM_ADA_DOWN);
        String upName = TensorNames.weight(TensorNames.FFN_NORM_ADA_UP);
        TensorRef down = tc.layer(il, downName);
        TensorRef up = tc.layer(il, upName);
        if (down == null || up == null) {
            throw new MissingTensorException(arch.ggufName(), il,
                    TensorNames.layerName(il, down == null ? downName : upName));
        }
        b.ffnAda(down, up, hp.nTokensDelay());
    }

    /**
     * Slices of k2-horizon's grouped block and output norms ({@code attention.group_norm_groups}, 0 meaning 1,
     * models/k2-horizon.cpp:7-10, 124-135); 1 for every other graph.
     *
     * @throws IllegalStateException when {@code n_embd} is not a multiple of the group count
     */
    public static int normGroups(ModelHparams hp) {
        if (hp.arch() != Arch.K2_HORIZON || hp.nNormGroups() <= 1) {
            return 1;
        }
        int g = hp.nNormGroups();
        if (hp.nEmbd() % g != 0) {
            throw new IllegalStateException(hp.arch().ggufName() + ": n_embd " + hp.nEmbd() + " is not a multiple of"
                    + " the " + g + " norm groups (k2-horizon.cpp:125)");
        }
        return g;
    }

    /** chameleon {@code swin_norm}: block norms on the branch outputs (models/chameleon.cpp:75, 132, 142, 157). */
    private boolean swinNorm() {
        return arch == Arch.CHAMELEON && hp.swinNorm();
    }

    /**
     * {@code attn_norm_2} is a second input norm only in falcon-40b, where it feeds the attention and {@code attn_norm}
     * feeds the FFN (models/falcon.cpp:70-86, 125), and in RWKV6/7, where it feeds the channel mix (rwkv6.cpp, rwkv7.cpp).
     */
    private boolean usesAttnNorm2AsFfnNorm() {
        return arch == Arch.FALCON || arch == Arch.RWKV6 || arch == Arch.RWKV7;
    }

    /**
     * Q/K normalization of main-stack layer {@code il}:
     * <ul>
     *   <li>sensenova_u1: RMS over each half of the head (models/sensenova-u1.cpp:103-107);</li>
     *   <li>{@code use_kq_norm} (llama, and llama4 on its RoPE layers): weightless RMS per head
     *       (models/llama4.cpp:178-184), applied after RoPE ({@link #qkNormAfterRope(int)});</li>
     *   <li>{@code attn_q_norm} / {@code attn_k_norm} present: LayerNorm with bias over the whole projection for
     *       the BERT graph (jina-bert-v2, models/bert.cpp:115-129) and mpt (models/mpt.cpp:106-118); LayerNorm per
     *       head for chameleon, command-r and stablelm; RMS per head for k2-horizon, whose flat
     *       {@code [head_dim * n_head]} weight is one row per head (models/k2-horizon.cpp:73-76, 250-261); RMS over
     *       the whole projection when the weight spans {@code n_head * head_dim} (olmo2, olmoe, minimax-m2); RMS per
     *       head otherwise (a {@code [head_dim, n_head]} or {@code [1, n_head]} weight carries one row per head).</li>
     * </ul>
     *
     * @throws IllegalStateException for a per-head LayerNorm architecture whose weight spans the whole projection,
     *                               a shape llama.cpp does not load
     */
    public QkNormKind qkNorm(int il) {
        checkLayer(il);
        if (arch == Arch.SENSENOVA_U1) {
            return QkNormKind.RMS_SPLIT_HALF;
        }
        TensorRef q = tc.layer(il, TensorNames.weight(TensorNames.ATTN_Q_NORM));
        TensorRef k = tc.layer(il, TensorNames.weight(TensorNames.ATTN_K_NORM));
        if (q == null && k == null) {
            return kqNormWithoutWeights(il) ? QkNormKind.RMS_HEAD : QkNormKind.NONE;
        }
        if (bertGraph || arch == Arch.MPT) {
            return QkNormKind.LN_FULL;
        }
        if (arch == Arch.K2_HORIZON) {
            return QkNormKind.RMS_HEAD;
        }
        TensorRef w = q != null ? q : k;
        int heads = q != null ? hp.nHead(il) : hp.nHeadKv(il);
        boolean full = w.rank() == 1 && heads > 1 && w.ne(0) == (long) heads * hp.headK(il);
        if (LN_QK.contains(arch)) {
            if (full) {
                throw new IllegalStateException(arch.ggufName() + " layer " + il + ": per-head Q/K LayerNorm weight "
                        + w + " spans the whole projection");
            }
            return QkNormKind.LN_HEAD;
        }
        return full ? QkNormKind.RMS_FULL : QkNormKind.RMS_HEAD;
    }

    /**
     * True when main-stack layer {@code il} normalizes Q and K after RoPE rather than before: the graphs of
     * {@link #QK_NORM_AFTER_ROPE} and the weightless {@code use_kq_norm} of llama / llama4, which llama.cpp applies
     * to the rotated heads (models/llama.cpp:146-170, models/llama4.cpp:162-184). False when the layer has no Q/K
     * norm.
     */
    public boolean qkNormAfterRope(int il) {
        QkNormKind qk = qkNorm(il);
        if (qk == QkNormKind.NONE) {
            return false;
        }
        if (QK_NORM_AFTER_ROPE.contains(arch)) {
            return true;
        }
        return tc.layer(il, TensorNames.weight(TensorNames.ATTN_Q_NORM)) == null
                && tc.layer(il, TensorNames.weight(TensorNames.ATTN_K_NORM)) == null
                && kqNormWithoutWeights(il);
    }

    /**
     * True when the Q/K norm of main-stack layer {@code il} adds {@code attn_q_norm.bias} / {@code attn_k_norm.bias}
     * when the file carries them; apertus loads them and normalizes without them (models/apertus.cpp:50-52, 93-96).
     */
    public boolean qkNormBias(int il) {
        checkLayer(il);
        return !QK_NORM_BIAS_UNUSED.contains(arch);
    }

    /** {@code hparams.use_kq_norm} in the llama graph (all layers) and the llama4 graph (RoPE layers only). */
    private boolean kqNormWithoutWeights(int il) {
        if (!hp.useKqNorm()) {
            return false;
        }
        if (arch == Arch.LLAMA4) {
            return hp.hasRope(il);
        }
        return arch == Arch.LLAMA || arch == Arch.LLAMA_EMBED;
    }

    /**
     * Tensor of the FFN input norm of main-stack layer {@code il}: {@code ffn_norm}; {@code post_attention_norm}
     * for PRE_FFN layers; {@code attn_norm_2} for RWKV; {@code attn_norm} for falcon-40b, whose attention reads
     * {@code attn_norm_2} instead (models/falcon.cpp:70-86, 125); dbrx's {@code attn_output_norm}, applied to
     * {@code x + attn} (models/dbrx.cpp); null when the FFN input is un-normalized, shares the attention norm
     * output, or the norms are weightless.
     */
    public TensorRef ffnNormTensor(int il) {
        NormSpec spec = resolve(il);
        if (spec.weightless() || spec.sharedAttnNormForFfn()) {
            return null;
        }
        if (spec.ffnInputIsPostAttnNorm()) {
            return weight(il, TensorNames.ATTN_POST_NORM);
        }
        if (spec.attnNorm2()) {
            return weight(il, arch == Arch.FALCON ? TensorNames.ATTN_NORM : TensorNames.ATTN_NORM_2);
        }
        if (arch == Arch.DBRX) {
            return weight(il, TensorNames.ATTN_OUT_NORM);
        }
        return spec.ffnNorm() ? weight(il, TensorNames.FFN_NORM) : null;
    }

    /**
     * Tensor of the norm on the attention branch output of main-stack layer {@code il}: {@code post_attention_norm}
     * ({@code .weight} or bare name); chameleon's {@code attn_norm} under {@code swin_norm}; null when the layer has
     * none.
     */
    public TensorRef postAttnNormTensor(int il) {
        checkLayer(il);
        if (swinNorm()) {
            return weight(il, TensorNames.ATTN_NORM);
        }
        return weight(il, TensorNames.ATTN_POST_NORM);
    }

    /**
     * Tensor of the norm on the FFN branch output of main-stack layer {@code il}: {@code post_ffw_norm}; for grok,
     * {@code layer_output_norm} when present, else {@code post_ffw_norm} (models/grok.cpp:75-77); chameleon's
     * {@code ffn_norm} under {@code swin_norm}; null when the layer has none.
     */
    public TensorRef postFfnNormTensor(int il) {
        checkLayer(il);
        if (swinNorm()) {
            return weight(il, TensorNames.FFN_NORM);
        }
        if (arch == Arch.GROK) {
            TensorRef lo = weight(il, TensorNames.LAYER_OUT_NORM);
            if (lo != null) {
                return lo;
            }
        }
        return weight(il, TensorNames.FFN_POST_NORM);
    }

    /** True when main-stack layer {@code il} carries a dense, routed or channel-mix FFN. */
    private boolean hasFfn(int il) {
        return has(TensorNames.STACK_BLK, il, TensorNames.FFN_DOWN)
                || has(TensorNames.STACK_BLK, il, TensorNames.FFN_DOWN_EXPS)
                || has(TensorNames.STACK_BLK, il, TensorNames.FFN_GATE_INP)
                || has(TensorNames.STACK_BLK, il, TensorNames.CHANNEL_MIX_VALUE);
    }

    private boolean has(String stack, int il, String base) {
        return tc.layer(stack, il, TensorNames.weight(base)) != null || tc.layer(stack, il, base) != null;
    }

    /** Norm weight {@code blk.<il>.<base>.weight}, else the bare {@code blk.<il>.<base>}; null when neither exists. */
    private TensorRef weight(int il, String base) {
        TensorRef r = tc.layer(il, TensorNames.weight(base));
        return r != null ? r : tc.layer(il, base);
    }

    private void checkLayer(int il) {
        if (il < 0 || il >= hp.nLayerAll()) {
            throw new IndexOutOfBoundsException(arch.ggufName() + ": layer " + il + " outside [0, " + hp.nLayerAll() + ")");
        }
    }

    @Override
    public String toString() {
        return "NormResolver{" + arch.ggufName() + ", " + kind + ", eps=" + eps + '}';
    }
}

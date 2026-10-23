package com.eonmux.neuralfree.gguf.gpu.block;

import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.gpu.forward.BuiltInGpuExtras;
import com.eonmux.neuralfree.gguf.gpu.head.GpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.ops.act.FfnBlocks;
import com.eonmux.neuralfree.gguf.ops.act.GpuDenseFfn;
import com.eonmux.neuralfree.gguf.ops.attn.AttnBlocks;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.deltanet.DeltaNetBlocks;
import com.eonmux.neuralfree.gguf.ops.deltanet.DeltaNetKernels;
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Blocks;
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Kernels;
import com.eonmux.neuralfree.gguf.ops.encdec.EncDecBlocks;
import com.eonmux.neuralfree.gguf.ops.gemma3n.Gemma3nBlocks;
import com.eonmux.neuralfree.gguf.ops.gemma3n.Gemma3nKernels;
import com.eonmux.neuralfree.gguf.ops.hc.HcBlocks;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels;
import com.eonmux.neuralfree.gguf.ops.heads.HeadsBlocks;
import com.eonmux.neuralfree.gguf.ops.heads.HeadsKernels;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraExtra;
import com.eonmux.neuralfree.gguf.ops.mla.MlaBlocks;
import com.eonmux.neuralfree.gguf.ops.moe.MoeBlocks;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.rwkv.RwkvBlocks;
import com.eonmux.neuralfree.gguf.ops.sparse.MsaBlocks;
import com.eonmux.neuralfree.gguf.ops.sparse.MsaKernels;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseBlocks;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseKernels;
import com.eonmux.neuralfree.gguf.ops.ssm.SsmBlocks;
import com.eonmux.neuralfree.gguf.ops.ssm.SsmKernels;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * The GPU block factories of every family in one {@link GpuBlockRegistry}: Attn, Mla, Ffn, Moe, Ssm, Rwkv, DeltaNet,
 * Sparse, Msa, Hc, Gemma3n, EncDec, Heads and Dsv4, plus {@link BuiltInGpuExtras#factories()}. Every
 * {@link MixerKind} except {@code NONE} and every FFN kind is served by exactly one family.
 *
 * <p>Routing of the sparse mixers: {@code ModelSpecBuilder} resolves a DSA indexer with a pooled key cache
 * ({@code DsaSpec.kpool() > 0}, QSA) to {@link MixerKind#MSA}, served by {@link MsaBlocks} (whose dense layers bind
 * the {@link AttnBlocks} attention factory), and a lightning indexer without pooling to {@link MixerKind#DSA_MLA},
 * served by {@link SparseBlocks}. {@link #checkRouting} rejects a spec that breaks this split.</p>
 *
 * <p>The switched-LoRA extra ({@link HeadsBlocks#gpuExtras(SwitchedLoraExtra.Router)}) reads the adapter slots
 * of the model's one {@link SwitchedLoraExtra.Router}; whoever drives the forward passes feeds that router.</p>
 */
public final class GpuFactoryCatalog {

    private static final List<String> GDN_KERNELS = Collections.unmodifiableList(Arrays.asList(
            SsmKernels.SSM_CONV, DeltaNetKernels.GDN_GATE_PREP, DeltaNetKernels.GDN_SCAN));
    private static final List<String> HC_KERNELS = Collections.unmodifiableList(Arrays.asList(
            HcKernels.HC_PRE_POST_COMB, HcKernels.HC_PRE, HcKernels.HC_POST, HcKernels.RES_STACK_SCORE,
            HcKernels.RES_STACK_MIX));
    private static final List<String> MOE_KERNELS = Collections.unmodifiableList(Arrays.asList(
            MoeKernels.ROUTER, MoeKernels.HASH_IDS, MoeKernels.WEIGHTED_SUM, MoeKernels.ADD_ID,
            MoeKernels.BUCKETIZE, MoeKernels.SCALE_INPUT));

    private GpuFactoryCatalog() {
    }

    /** Every family's factories, without a switched-LoRA router (a switched-LoRA layer then fails to bind). */
    public static GpuBlockRegistry defaults() {
        return defaults(null);
    }

    /**
     * Every family's factories.
     *
     * @param router the model's switched-LoRA router, or null for a model without switched-LoRA layers
     * @throws IllegalStateException when the attention family serves no {@link MixerKind#ATTENTION} factory
     */
    public static GpuBlockRegistry defaults(SwitchedLoraExtra.Router router) {
        GpuMixerFactory attention = attentionFactory();
        List<GpuMixerFactory> mixers = new ArrayList<>();
        mixers.addAll(AttnBlocks.gpuMixers());
        mixers.addAll(MlaBlocks.gpuMixers());
        mixers.addAll(FfnBlocks.gpuMixers());
        mixers.addAll(MoeBlocks.gpuMixers());
        mixers.addAll(SsmBlocks.gpuMixers());
        mixers.addAll(RwkvBlocks.gpuMixers());
        mixers.addAll(DeltaNetBlocks.gpuMixers());
        mixers.addAll(SparseBlocks.gpuMixers());
        mixers.addAll(MsaBlocks.gpuMixers(attention));
        mixers.addAll(HcBlocks.gpuMixers());
        mixers.addAll(Gemma3nBlocks.gpuMixers());
        mixers.addAll(EncDecBlocks.gpuMixers());
        mixers.addAll(HeadsBlocks.gpuMixers());
        mixers.addAll(Dsv4Blocks.gpuMixers());

        List<GpuFfnFactory> ffns = new ArrayList<>();
        ffns.addAll(AttnBlocks.gpuFfns());
        ffns.addAll(MlaBlocks.gpuFfns());
        ffns.addAll(FfnBlocks.gpuFfns());
        ffns.addAll(MoeBlocks.gpuFfns());
        ffns.addAll(SsmBlocks.gpuFfns());
        ffns.addAll(RwkvBlocks.gpuFfns());
        ffns.addAll(DeltaNetBlocks.gpuFfns());
        ffns.addAll(SparseBlocks.gpuFfns());
        ffns.addAll(MsaBlocks.gpuFfns());
        ffns.addAll(HcBlocks.gpuFfns());
        ffns.addAll(Gemma3nBlocks.gpuFfns());
        ffns.addAll(EncDecBlocks.gpuFfns());
        ffns.addAll(HeadsBlocks.gpuFfns());
        ffns.addAll(Dsv4Blocks.gpuFfns());

        List<GpuLayerExtraFactory> extras = new ArrayList<>();
        extras.addAll(AttnBlocks.gpuExtras());
        extras.addAll(MlaBlocks.gpuExtras());
        extras.addAll(FfnBlocks.gpuExtras());
        extras.addAll(MoeBlocks.gpuExtras());
        extras.addAll(SsmBlocks.gpuExtras());
        extras.addAll(RwkvBlocks.gpuExtras());
        extras.addAll(DeltaNetBlocks.gpuExtras());
        extras.addAll(SparseBlocks.gpuExtras());
        extras.addAll(MsaBlocks.gpuExtras());
        extras.addAll(HcBlocks.gpuExtras());
        extras.addAll(Gemma3nBlocks.gpuExtras());
        extras.addAll(EncDecBlocks.gpuExtras());
        extras.addAll(HeadsBlocks.gpuExtras(router));
        extras.addAll(Dsv4Blocks.gpuExtras());
        extras.addAll(BuiltInGpuExtras.factories());
        return new GpuBlockRegistry(mixers, ffns, extras);
    }

    /** Fresh head preprocessors of every family; the backend that binds them closes them. */
    public static List<GpuHeadPreprocessor> headPreprocessors() {
        return new ArrayList<>(HcBlocks.gpuHeadPreprocessors());
    }

    /** True when a layer of {@code spec} carries the switched-LoRA extra, so the model needs a router. */
    public static boolean usesSwitchedLora(ModelSpec spec) {
        for (LayerSpec l : allLayers(spec)) {
            if (l.hasExtra(LayerSpec.Extras.SWITCHED_LORA)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Checks the sparse-mixer routing described in the class comment.
     *
     * @throws IllegalStateException when a {@link MixerKind#DSA_MLA} layer has a pooled (QSA) indexer
     */
    public static void checkRouting(ModelSpec spec) {
        for (LayerSpec l : allLayers(spec)) {
            if (l.mixerKind() == MixerKind.DSA_MLA && l.dsa() != null && l.dsa().kpool() > 0) {
                throw new IllegalStateException(spec.arch().ggufName() + " layer " + l.il() + " (" + l.stack()
                        + ") has a pooled QSA indexer (kpool " + l.dsa().kpool() + ") but is planned as DSA_MLA;"
                        + " pooled indexers are served by the MSA family");
            }
        }
    }

    /**
     * Kernels the block families of {@code spec} launch besides the weight-type dependent projections and
     * {@code {T}_MatVecId} products, for prewarming: each family's list for the mixers, FFNs, extras and head the spec
     * uses (DSV4 layers also take the sparse indexer kernels).
     */
    public static List<String> kernels(ModelSpec spec) {
        Set<String> k = new LinkedHashSet<>();
        for (LayerSpec l : allLayers(spec)) {
            addMixerKernels(l.mixerKind(), k);
            if (l.parallelAttention() || l.crossAttention() != null) {
                k.addAll(AttnBlocks.KERNELS);
            }
            switch (l.ffnKind()) {
                case DENSE:
                    k.addAll(GpuDenseFfn.KERNELS);
                    break;
                case MOE:
                case MOE_LATENT:
                case MOE_CHUNKED:
                case MOE_PARALLEL_DENSE:
                    k.addAll(MOE_KERNELS);
                    k.addAll(GpuDenseFfn.KERNELS);
                    break;
                case RWKV_CHANNEL_MIX:
                    k.addAll(RwkvBlocks.KERNELS);
                    break;
                default:
                    break;
            }
            addExtraKernels(l, k);
        }
        if (spec.head() != null && spec.head().hcHead()) {
            k.addAll(HC_KERNELS);
        }
        return Collections.unmodifiableList(new ArrayList<>(k));
    }

    private static void addMixerKernels(MixerKind kind, Collection<String> k) {
        switch (kind) {
            case ATTENTION:
            case ATTENTION_LINEAR_REPLACE:
            case CROSS_ATTENTION_T5:
            case MLA:
                k.addAll(AttnBlocks.KERNELS);
                break;
            case DSA_MLA:
                k.addAll(AttnBlocks.KERNELS);
                k.addAll(SparseKernels.ALL);
                break;
            case MSA:
                k.addAll(AttnBlocks.KERNELS);
                k.addAll(MsaKernels.ALL);
                break;
            case DSV4:
                k.addAll(AttnKernels.all());
                k.addAll(Dsv4Kernels.ALL);
                k.addAll(SparseKernels.ALL);
                break;
            case MAMBA1:
            case MAMBA2:
            case PLAMO2_MAMBA:
            case SHORTCONV:
            case SHORTCONV_CENTERED:
                k.addAll(SsmKernels.ALL);
                break;
            case RWKV6:
            case RWKV6_QWEN2:
            case RWKV7:
                k.addAll(RwkvBlocks.KERNELS);
                break;
            case GDN:
            case KDA:
                k.addAll(GDN_KERNELS);
                break;
            case LIGHTNING:
                k.add(DeltaNetKernels.LIGHTNING_ATTN);
                break;
            default:
                break;
        }
    }

    private static void addExtraKernels(LayerSpec l, Collection<String> k) {
        if (l.hasExtra(LayerSpec.Extras.HC) || l.hasExtra(LayerSpec.Extras.ATTN_RES)) {
            k.addAll(HC_KERNELS);
        }
        if (l.hasExtra(LayerSpec.Extras.ALTUP) || l.hasExtra(LayerSpec.Extras.LAUREL)
                || l.hasExtra(LayerSpec.Extras.PLE) || l.hasExtra(LayerSpec.Extras.NGRAM_PLE)) {
            k.addAll(Gemma3nKernels.ALL);
        }
        if (l.hasExtra(LayerSpec.Extras.SWITCHED_LORA)) {
            k.addAll(HeadsKernels.ALL);
        }
    }

    private static List<LayerSpec> allLayers(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        List<LayerSpec> all = new ArrayList<>(spec.layers());
        all.addAll(spec.nextnLayers());
        all.addAll(spec.encoderLayers());
        return all;
    }

    private static GpuMixerFactory attentionFactory() {
        for (GpuMixerFactory f : AttnBlocks.gpuMixers()) {
            if (f.kind() == MixerKind.ATTENTION) {
                return f;
            }
        }
        throw new IllegalStateException("AttnBlocks serves no " + MixerKind.ATTENTION + " factory for the dense"
                + " layers of the MSA family");
    }
}

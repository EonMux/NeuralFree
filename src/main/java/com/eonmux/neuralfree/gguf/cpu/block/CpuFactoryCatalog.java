package com.eonmux.neuralfree.gguf.cpu.block;

import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.cpu.forward.BuiltInCpuExtras;
import com.eonmux.neuralfree.gguf.cpu.head.CpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.ops.act.FfnBlocks;
import com.eonmux.neuralfree.gguf.ops.attn.AttnBlocks;
import com.eonmux.neuralfree.gguf.ops.deltanet.DeltaNetBlocks;
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Blocks;
import com.eonmux.neuralfree.gguf.ops.encdec.EncDecBlocks;
import com.eonmux.neuralfree.gguf.ops.gemma3n.Gemma3nBlocks;
import com.eonmux.neuralfree.gguf.ops.hc.HcBlocks;
import com.eonmux.neuralfree.gguf.ops.heads.HeadsBlocks;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraExtra;
import com.eonmux.neuralfree.gguf.ops.mla.MlaBlocks;
import com.eonmux.neuralfree.gguf.ops.moe.MoeBlocks;
import com.eonmux.neuralfree.gguf.ops.rwkv.RwkvBlocks;
import com.eonmux.neuralfree.gguf.ops.sparse.MsaBlocks;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseBlocks;
import com.eonmux.neuralfree.gguf.ops.ssm.SsmBlocks;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The single list of every concrete CPU block factory: the mixers, FFN blocks and layer extras each
 * {@code ops/<family>/<Family>Blocks} class exposes, plus the built-in residual extras and the head preprocessors.
 * The twin of {@code GpuFactoryCatalog}.
 *
 * <p>Families: attention (ATTENTION, ATTENTION_LINEAR_REPLACE), MLA, dense FFN, MoE (routed FFNs and the
 * {@code moe_layer_input} extra), SSM (Mamba 1/2, PLaMo-2 Mamba, short convolutions), RWKV (6, 6-Qwen2, 7 and the
 * channel mix), DeltaNet (GDN, KDA, Lightning), DSA (DSA_MLA), MSA (MSA and QSA, whose dense layers reuse the
 * attention factory), hyper-connections (HC, attention residuals), Gemma-3n (AltUp, Laurel, PLE, n-gram PLE),
 * encoder-decoder (T5 cross attention), heads (graniteswitch switched LoRA), DeepSeek-V4 and the built-in extras
 * (embedding skip, alpha/beta residual, loop norm). Stateless; every call returns fresh lists.</p>
 *
 * <p>Routing of the sparse mixers: {@code ModelSpecBuilder} resolves a DSA indexer with a pooled key cache
 * ({@code DsaSpec.kpool() > 0}, QSA) to {@link MixerKind#MSA}, served by {@link MsaBlocks}, and a lightning indexer
 * without pooling to {@link MixerKind#DSA_MLA}, served by {@link SparseBlocks}; {@link #checkRouting} rejects a spec
 * that breaks this split. The switched-LoRA extras read the slots of the model's one
 * {@link SwitchedLoraExtra.Router}; whoever drives the forward passes feeds that router.</p>
 */
public final class CpuFactoryCatalog {

    private CpuFactoryCatalog() {
    }

    /**
     * The registry of every CPU factory without a switched-LoRA router (a graniteswitch layer then fails to
     * create its extra).
     */
    public static CpuBlockRegistry defaults() {
        return defaults(null);
    }

    /**
     * The registry of every CPU factory.
     *
     * @param router the graniteswitch adapter router shared by the switched-LoRA extras; null for models without
     *               switched LoRA
     * @throws IllegalArgumentException when two families register the same kind or feature id
     */
    public static CpuBlockRegistry defaults(SwitchedLoraExtra.Router router) {
        return new CpuBlockRegistry(mixers(), ffns(), extras(router));
    }

    /** Every mixer factory, one per {@link MixerKind} the CPU runs (unmodifiable). */
    public static List<CpuMixerFactory> mixers() {
        List<CpuMixerFactory> attention = AttnBlocks.cpuMixers();
        List<CpuMixerFactory> out = new ArrayList<>(attention);
        out.addAll(MlaBlocks.cpuMixers());
        out.addAll(FfnBlocks.cpuMixers());
        out.addAll(MoeBlocks.cpuMixers());
        out.addAll(SsmBlocks.cpuMixers());
        out.addAll(RwkvBlocks.cpuMixers());
        out.addAll(DeltaNetBlocks.cpuMixers());
        out.addAll(SparseBlocks.cpuMixers());
        out.addAll(MsaBlocks.cpuMixers(denseAttention(attention)));
        out.addAll(HcBlocks.cpuMixers());
        out.addAll(Gemma3nBlocks.cpuMixers());
        out.addAll(EncDecBlocks.cpuMixers());
        out.addAll(HeadsBlocks.cpuMixers());
        out.addAll(Dsv4Blocks.cpuMixers());
        return Collections.unmodifiableList(out);
    }

    /** Every FFN block factory (unmodifiable). */
    public static List<CpuFfnFactory> ffns() {
        List<CpuFfnFactory> out = new ArrayList<>();
        out.addAll(AttnBlocks.cpuFfns());
        out.addAll(MlaBlocks.cpuFfns());
        out.addAll(FfnBlocks.cpuFfns());
        out.addAll(MoeBlocks.cpuFfns());
        out.addAll(SsmBlocks.cpuFfns());
        out.addAll(RwkvBlocks.cpuFfns());
        out.addAll(DeltaNetBlocks.cpuFfns());
        out.addAll(SparseBlocks.cpuFfns());
        out.addAll(MsaBlocks.cpuFfns());
        out.addAll(HcBlocks.cpuFfns());
        out.addAll(Gemma3nBlocks.cpuFfns());
        out.addAll(EncDecBlocks.cpuFfns());
        out.addAll(HeadsBlocks.cpuFfns());
        out.addAll(Dsv4Blocks.cpuFfns());
        return Collections.unmodifiableList(out);
    }

    /**
     * Every layer-extra factory, one per feature id (unmodifiable).
     *
     * @param router the graniteswitch adapter router, or null
     */
    public static List<CpuLayerExtraFactory> extras(SwitchedLoraExtra.Router router) {
        List<CpuLayerExtraFactory> out = new ArrayList<>();
        out.addAll(AttnBlocks.cpuExtras());
        out.addAll(MlaBlocks.cpuExtras());
        out.addAll(FfnBlocks.cpuExtras());
        out.addAll(MoeBlocks.cpuExtras());
        out.addAll(SsmBlocks.cpuExtras());
        out.addAll(RwkvBlocks.cpuExtras());
        out.addAll(DeltaNetBlocks.cpuExtras());
        out.addAll(SparseBlocks.cpuExtras());
        out.addAll(MsaBlocks.cpuExtras());
        out.addAll(HcBlocks.cpuExtras());
        out.addAll(Gemma3nBlocks.cpuExtras());
        out.addAll(EncDecBlocks.cpuExtras());
        out.addAll(HeadsBlocks.cpuExtras(router));
        out.addAll(Dsv4Blocks.cpuExtras());
        out.addAll(BuiltInCpuExtras.cpuExtras());
        return Collections.unmodifiableList(out);
    }

    /**
     * Fresh head preprocessors for one backend (the head binds and owns those that apply to its model), in
     * registration order (unmodifiable).
     */
    public static List<CpuHeadPreprocessor> headPreprocessors() {
        return Collections.unmodifiableList(new ArrayList<>(HcBlocks.cpuHeadPreprocessors()));
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

    private static List<LayerSpec> allLayers(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        List<LayerSpec> all = new ArrayList<>(spec.layers());
        all.addAll(spec.nextnLayers());
        all.addAll(spec.encoderLayers());
        return all;
    }

    /** The ATTENTION factory of the attention family, which MSA's dense layers delegate to. */
    private static CpuMixerFactory denseAttention(List<CpuMixerFactory> attention) {
        for (CpuMixerFactory f : attention) {
            if (f.kind() == MixerKind.ATTENTION) {
                return f;
            }
        }
        throw new IllegalStateException("AttnBlocks.cpuMixers() serves no " + MixerKind.ATTENTION + " factory");
    }
}

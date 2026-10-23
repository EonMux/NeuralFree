package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Block factories of the softmax-attention family: {@link MixerKind#ATTENTION} and
 * {@link MixerKind#ATTENTION_LINEAR_REPLACE} (deci's {@code wo @ x}, models/deci.cpp:114-117) on the GPU
 * ({@link GpuAttentionMixer}) and the CPU ({@link CpuAttentionMixer}). The family has no FFNs and no layer extras.
 */
public final class AttnBlocks {

    /**
     * Every kernel the attention mixers launch besides the projections' GEMM kernels and the MoVA value-expert
     * kernels, for prewarming.
     */
    public static final List<String> KERNELS = kernels();

    /**
     * The type-independent kernels of a MoVA layer's value experts ({@link GpuValueExperts}) besides
     * {@link #KERNELS}; the expert products are the weight-type dependent {@code {T}_MatVecId} /
     * {@code {T}_BatchMatMulId}.
     */
    public static final List<String> VALUE_EXPERT_KERNELS = Collections.unmodifiableList(Arrays.asList(
            MoeKernels.ROUTER, MoeKernels.BUCKETIZE, MoeKernels.WEIGHTED_SUM));

    private AttnBlocks() {
    }

    private static List<String> kernels() {
        List<String> k = new ArrayList<>(AttnKernels.all());
        k.addAll(RopeKernels.KV_STORE_KERNELS);
        k.addAll(RopeKernels.KV_ROPE_SHIFT_KERNELS);
        k.addAll(Arrays.asList(AttnExtKernels.TEMP_SCALE, AttnExtKernels.OUT_GATE, RopeKernels.ROPE_EXT,
                NormKernels.ROW_NORM, ActKernels.UNARY, HadamardSpec.KERNEL, GpuOps.COPY2D, GpuOps.AXPBY,
                GpuAttentionMixer.BIAS_ADD));
        return Collections.unmodifiableList(k);
    }

    /** GPU attention mixers for both kinds. */
    public static List<GpuMixerFactory> gpuMixers() {
        return Collections.unmodifiableList(Arrays.<GpuMixerFactory>asList(
                new GpuFactory(MixerKind.ATTENTION), new GpuFactory(MixerKind.ATTENTION_LINEAR_REPLACE)));
    }

    /** CPU attention mixers, the twins of {@link #gpuMixers()}. */
    public static List<CpuMixerFactory> cpuMixers() {
        return Collections.unmodifiableList(Arrays.<CpuMixerFactory>asList(
                new CpuFactory(MixerKind.ATTENTION), new CpuFactory(MixerKind.ATTENTION_LINEAR_REPLACE)));
    }

    /** No FFNs in this family. */
    public static List<GpuFfnFactory> gpuFfns() {
        return Collections.emptyList();
    }

    /** No FFNs in this family. */
    public static List<CpuFfnFactory> cpuFfns() {
        return Collections.emptyList();
    }

    /** No layer extras in this family. */
    public static List<GpuLayerExtraFactory> gpuExtras() {
        return Collections.emptyList();
    }

    /** No layer extras in this family. */
    public static List<CpuLayerExtraFactory> cpuExtras() {
        return Collections.emptyList();
    }

    /**
     * Checks that a layer belongs to {@code kind} and carries an attention spec whose linear-replace flag matches.
     *
     * @throws IllegalArgumentException otherwise
     */
    static void checkLayer(MixerKind kind, LayerSpec layer) {
        Objects.requireNonNull(layer, "layer");
        if (layer.mixerKind() != kind) {
            throw new IllegalArgumentException("layer " + layer.stack() + "." + layer.il() + " has mixer kind "
                    + layer.mixerKind() + ", not " + kind);
        }
        if (layer.attention() == null) {
            throw new IllegalArgumentException("layer " + layer.stack() + "." + layer.il() + " has no attention spec");
        }
        boolean replace = kind == MixerKind.ATTENTION_LINEAR_REPLACE;
        if (layer.attention().linearReplace() != replace) {
            throw new IllegalArgumentException("layer " + layer.stack() + "." + layer.il() + " of kind " + kind
                    + " has linearReplace=" + layer.attention().linearReplace());
        }
    }

    /** Creates {@link GpuAttentionMixer}s for one kind. */
    private static final class GpuFactory implements GpuMixerFactory {
        private final MixerKind kind;

        GpuFactory(MixerKind kind) {
            this.kind = kind;
        }

        @Override
        public MixerKind kind() {
            return kind;
        }

        @Override
        public GpuMixer create(LayerSpec layer) {
            checkLayer(kind, layer);
            return new GpuAttentionMixer();
        }

        @Override
        public String toString() {
            return "AttnBlocks.gpu" + kind;
        }
    }

    /** Creates {@link CpuAttentionMixer}s for one kind. */
    private static final class CpuFactory implements CpuMixerFactory {
        private final MixerKind kind;

        CpuFactory(MixerKind kind) {
            this.kind = kind;
        }

        @Override
        public MixerKind kind() {
            return kind;
        }

        @Override
        public CpuMixer create(LayerSpec layer) {
            checkLayer(kind, layer);
            return new CpuAttentionMixer();
        }

        @Override
        public String toString() {
            return "AttnBlocks.cpu" + kind;
        }
    }
}

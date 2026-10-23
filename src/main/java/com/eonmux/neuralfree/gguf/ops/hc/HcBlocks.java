package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.cpu.head.CpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.head.GpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Factory exposure of the hyper-connection family for the block catalogs: the
 * {@code LayerSpec.Extras.HC} and {@code LayerSpec.Extras.ATTN_RES} layer extras and the hyper-connection head
 * preprocessors. The family has no mixer and no FFN.
 *
 * <p>Ordering contract for the spec builder: a layer that carries both {@code ngram_ple} and {@code hc}
 * (qwen4exp) lists {@code ngram_ple} first, because the n-gram embeddings are added to the stream before the
 * attention collapse (qwen4exp.cpp:452-461). The input of the first hyper-connection layer is the embedding row
 * repeated over the {@code hc} streams, written by the embedding stage ({@code hc_init}, deepseek4.cpp:1240-1242;
 * {@link CpuHyperConnections#expandStreams} on the CPU).</p>
 */
public final class HcBlocks {

    private static final GpuLayerExtraFactory GPU_HC = new GpuLayerExtraFactory() {
        @Override
        public String featureId() {
            return LayerSpec.Extras.HC;
        }

        @Override
        public GpuLayerExtra create(LayerSpec layer) {
            requireHyperConnections(layer);
            return new GpuHcExtra();
        }

        @Override
        public String toString() {
            return "HcBlocks.gpu(" + LayerSpec.Extras.HC + ")";
        }
    };

    private static final GpuLayerExtraFactory GPU_ATTN_RES = new GpuLayerExtraFactory() {
        @Override
        public String featureId() {
            return LayerSpec.Extras.ATTN_RES;
        }

        @Override
        public GpuLayerExtra create(LayerSpec layer) {
            Objects.requireNonNull(layer, "layer");
            return new GpuAttnResidualExtra();
        }

        @Override
        public String toString() {
            return "HcBlocks.gpu(" + LayerSpec.Extras.ATTN_RES + ")";
        }
    };

    private static final CpuLayerExtraFactory CPU_HC = new CpuLayerExtraFactory() {
        @Override
        public String featureId() {
            return LayerSpec.Extras.HC;
        }

        @Override
        public CpuLayerExtra create(LayerSpec layer) {
            requireHyperConnections(layer);
            return new CpuHcExtra();
        }

        @Override
        public String toString() {
            return "HcBlocks.cpu(" + LayerSpec.Extras.HC + ")";
        }
    };

    private static final CpuLayerExtraFactory CPU_ATTN_RES = new CpuLayerExtraFactory() {
        @Override
        public String featureId() {
            return LayerSpec.Extras.ATTN_RES;
        }

        @Override
        public CpuLayerExtra create(LayerSpec layer) {
            Objects.requireNonNull(layer, "layer");
            return new CpuAttnResidualExtra();
        }

        @Override
        public String toString() {
            return "HcBlocks.cpu(" + LayerSpec.Extras.ATTN_RES + ")";
        }
    };

    private HcBlocks() {
    }

    /** No mixers in this family. */
    public static List<GpuMixerFactory> gpuMixers() {
        return Collections.emptyList();
    }

    /** No mixers in this family. */
    public static List<CpuMixerFactory> cpuMixers() {
        return Collections.emptyList();
    }

    /** No FFN blocks in this family. */
    public static List<GpuFfnFactory> gpuFfns() {
        return Collections.emptyList();
    }

    /** No FFN blocks in this family. */
    public static List<CpuFfnFactory> cpuFfns() {
        return Collections.emptyList();
    }

    /** The {@code hc} and {@code attn_res} GPU layer extras (unmodifiable). */
    public static List<GpuLayerExtraFactory> gpuExtras() {
        return Collections.unmodifiableList(Arrays.asList(GPU_HC, GPU_ATTN_RES));
    }

    /** The {@code hc} and {@code attn_res} CPU layer extras (unmodifiable). */
    public static List<CpuLayerExtraFactory> cpuExtras() {
        return Collections.unmodifiableList(Arrays.asList(CPU_HC, CPU_ATTN_RES));
    }

    /** A fresh, unbound hyper-connection head collapse for one model (unmodifiable list). */
    public static List<GpuHeadPreprocessor> gpuHeadPreprocessors() {
        return Collections.singletonList(new GpuHcHeadCollapse());
    }

    /** A fresh, unbound hyper-connection head collapse for one model (unmodifiable list). */
    public static List<CpuHeadPreprocessor> cpuHeadPreprocessors() {
        return Collections.singletonList(new CpuHcHeadCollapse());
    }

    /**
     * @throws UnsupportedOperationException when the layer's residual is not per-sub-block hyper-connections
     */
    private static void requireHyperConnections(LayerSpec layer) {
        Objects.requireNonNull(layer, "layer");
        ResidualSpec r = layer.residual();
        if (r.kind() != ResidualKind.HYPER_CONNECTIONS || !r.hcPerSubBlock()) {
            throw new UnsupportedOperationException("layer " + layer.il() + " lists the '" + LayerSpec.Extras.HC
                    + "' extra but its residual is " + r + "; hyper-connections run per sub-block only");
        }
    }
}

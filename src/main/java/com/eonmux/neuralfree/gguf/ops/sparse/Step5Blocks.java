package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Collections;
import java.util.List;

/**
 * Block factories of the Step5 sparse region attention family, and the tensor helpers its CPU mixer shares.
 *
 * <p>The family serves {@link MixerKind#STEP5}: {@link CpuStep5Mixer} runs the region indexer on full-attention layers
 * and dense windowed attention on sliding blocks; {@link GpuStep5Mixer} runs the same steps with the sparse-region
 * kernels of {@link Step5Kernels} on the device. The family has no FFN blocks or layer extras: the dense and
 * routed-plus-shared-expert FFNs are the ordinary {@code FfnKind} blocks.
 */
public final class Step5Blocks {

    /** The kernels every Step5 layer launches besides the weight-type dependent projections. */
    public static final List<String> KERNELS = Step5Kernels.ALL;

    private Step5Blocks() {
    }

    /** The CPU mixer factory of {@link MixerKind#STEP5}. */
    public static List<CpuMixerFactory> cpuMixers() {
        return Collections.singletonList(new CpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return MixerKind.STEP5;
            }

            @Override
            public CpuMixer create(LayerSpec layer) {
                if (layer.step5() == null) {
                    throw new IllegalArgumentException("layer " + describe(layer) + " has no Step5Spec");
                }
                return new CpuStep5Mixer();
            }

            @Override
            public String toString() {
                return "Step5Blocks.cpu(STEP5)";
            }
        });
    }

    /** The GPU mixer factory of {@link MixerKind#STEP5}. */
    public static List<GpuMixerFactory> gpuMixers() {
        return Collections.singletonList(new GpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return MixerKind.STEP5;
            }

            @Override
            public GpuMixer create(LayerSpec layer) {
                if (layer.step5() == null) {
                    throw new IllegalArgumentException("layer " + describe(layer) + " has no Step5Spec");
                }
                return new GpuStep5Mixer();
            }

            @Override
            public String toString() {
                return "Step5Blocks.gpu(STEP5)";
            }
        });
    }

    /** The family has no FFN blocks. */
    public static List<GpuFfnFactory> gpuFfns() {
        return Collections.emptyList();
    }

    /** The family has no FFN blocks. */
    public static List<CpuFfnFactory> cpuFfns() {
        return Collections.emptyList();
    }

    /** The family has no layer extras. */
    public static List<GpuLayerExtraFactory> gpuExtras() {
        return Collections.emptyList();
    }

    /** The family has no layer extras. */
    public static List<CpuLayerExtraFactory> cpuExtras() {
        return Collections.emptyList();
    }

    /** {@code stack.layer} of a spec, for messages. */
    static String describe(LayerSpec layer) {
        return layer.stack() + "." + layer.il();
    }

    /** The embedding width the layer's mixer reads ({@code n_embd} or the hyper-connection stream width). */
    static int mixerWidth(LayerSpec layer) {
        return layer.residual().streamWidth();
    }

    /**
     * Required matrix {@code <base>.weight} read in place, with {@code cols} inputs and {@code rows} outputs.
     *
     * @throws MissingTensorException when it is absent
     * @throws IllegalStateException  when its shape differs
     */
    static CpuTensorRef cpuMatrix(CpuWeights ws, LayerSpec layer, String base, long cols, long rows) {
        String suffix = TensorNames.weight(base);
        CpuTensorRef w = ws.layer(layer, suffix);
        if (w.cols() != cols || w.rows() != rows) {
            throw new IllegalStateException("layer " + describe(layer) + ": " + layer.tensorName(suffix) + " is "
                    + w.cols() + "x" + w.rows() + ", the Step5 mixer needs " + cols + "x" + rows);
        }
        return w;
    }

    /**
     * F32 values of tensor {@code name} ({@code width} values), or null when it is absent and optional.
     *
     * @param name full layer suffix ({@code attn_q_norm.weight})
     * @throws MissingTensorException when a required tensor is absent
     * @throws IllegalStateException  when it holds another number of values
     */
    static float[] cpuVector(CpuWeights ws, LayerSpec layer, String name, int width, boolean required) {
        float[] v = ws.layerFloatsOrNull(layer, name);
        if (v == null) {
            if (required) {
                throw new MissingTensorException(ws.archName(), layer.il(), layer.tensorName(name));
            }
            return null;
        }
        if (v.length != width) {
            throw new IllegalStateException("layer " + describe(layer) + ": " + layer.tensorName(name) + " holds "
                    + v.length + " values, the Step5 mixer needs " + width);
        }
        return v;
    }

    /** The F32 frequency factors of a RoPE spec, or null when the spec has none. */
    static float[] cpuFreqFactors(CpuWeights ws, RopeSpec rope) {
        return rope == null || rope.freqFactors() == null ? null : ws.floats(rope.freqFactors());
    }

    /** Writes {@code qPos[t] = pos0 + t} for {@code t < nTokens}. */
    static void fillQueryPositions(int pos0, int nTokens, int[] qPos) {
        for (int t = 0; t < nTokens; t++) {
            qPos[t] = pos0 + t;
        }
    }

    /**
     * The position &rarr; row map of a contiguous cache: {@code rowOfPos[p] = layout.rowOf(p)} when that row of the
     * group holds {@code p}, -1 otherwise, for {@code p < nPos}.
     */
    static void fillContiguousRows(CpuMemory memory, int group, KvLayout layout, int nPos, int[] rowOfPos) {
        for (int p = 0; p < nPos; p++) {
            int row = layout.rowOf(p);
            rowOfPos[p] = memory.positionOfRow(group, row) == p ? row : -1;
        }
    }
}

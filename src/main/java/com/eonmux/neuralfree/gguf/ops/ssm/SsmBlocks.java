package com.eonmux.neuralfree.gguf.ops.ssm;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
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
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Block factories of the SSM family, exposed to the block catalogs: the Mamba-1, Mamba-2 and PLaMo-2 selective-scan
 * mixers ({@link GpuMambaMixer}, {@link CpuMambaMixer}) and the LFM2 causal and centered short-convolution mixers
 * ({@link GpuShortConvMixer}, {@link CpuShortConvMixer}). The family has no FFN blocks and no layer extras.
 *
 * <p>A falcon-h1 layer ({@link LayerSpec#parallelAttention()}) is served by the plain Mamba-2 mixer: the layer
 * executors bind the attention branch through the ATTENTION factory, run it on the same normed input and add it to
 * the mixer output (models/falcon-h1.cpp:133-163; {@code GpuLayerExecutor} / {@code CpuLayerExecutor}).</p>
 *
 * <p>The package-private helpers below resolve the family's tensors (some are stored without a {@code .weight}
 * suffix) and their F32 forms, shared by the GPU and CPU mixers.</p>
 */
public final class SsmBlocks {

    private SsmBlocks() {
    }

    // ================================================================== factory exposure

    /** GPU mixers of every SSM kind. */
    public static List<GpuMixerFactory> gpuMixers() {
        return Collections.unmodifiableList(Arrays.asList(
                gpuMamba(MixerKind.MAMBA1),
                gpuMamba(MixerKind.MAMBA2),
                gpuMamba(MixerKind.PLAMO2_MAMBA),
                gpuShortConv(MixerKind.SHORTCONV),
                gpuShortConv(MixerKind.SHORTCONV_CENTERED)));
    }

    /** CPU mixers of every SSM kind. */
    public static List<CpuMixerFactory> cpuMixers() {
        return Collections.unmodifiableList(Arrays.asList(
                cpuMamba(MixerKind.MAMBA1),
                cpuMamba(MixerKind.MAMBA2),
                cpuMamba(MixerKind.PLAMO2_MAMBA),
                cpuShortConv(MixerKind.SHORTCONV),
                cpuShortConv(MixerKind.SHORTCONV_CENTERED)));
    }

    /** The SSM family has no FFN blocks. */
    public static List<GpuFfnFactory> gpuFfns() {
        return Collections.emptyList();
    }

    /** The SSM family has no FFN blocks. */
    public static List<CpuFfnFactory> cpuFfns() {
        return Collections.emptyList();
    }

    /** The SSM family has no layer extras. */
    public static List<GpuLayerExtraFactory> gpuExtras() {
        return Collections.emptyList();
    }

    /** The SSM family has no layer extras. */
    public static List<CpuLayerExtraFactory> cpuExtras() {
        return Collections.emptyList();
    }

    private static GpuMixerFactory gpuMamba(MixerKind kind) {
        return new GpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return kind;
            }

            @Override
            public GpuMixer create(LayerSpec layer) {
                checkKind(kind, layer);
                return new GpuMambaMixer(kind);
            }

            @Override
            public String toString() {
                return "SsmBlocks.gpu(" + kind + ")";
            }
        };
    }

    private static CpuMixerFactory cpuMamba(MixerKind kind) {
        return new CpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return kind;
            }

            @Override
            public CpuMixer create(LayerSpec layer) {
                checkKind(kind, layer);
                return new CpuMambaMixer(kind);
            }

            @Override
            public String toString() {
                return "SsmBlocks.cpu(" + kind + ")";
            }
        };
    }

    private static GpuMixerFactory gpuShortConv(MixerKind kind) {
        return new GpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return kind;
            }

            @Override
            public GpuMixer create(LayerSpec layer) {
                checkKind(kind, layer);
                return new GpuShortConvMixer(kind);
            }

            @Override
            public String toString() {
                return "SsmBlocks.gpu(" + kind + ")";
            }
        };
    }

    private static CpuMixerFactory cpuShortConv(MixerKind kind) {
        return new CpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return kind;
            }

            @Override
            public CpuMixer create(LayerSpec layer) {
                checkKind(kind, layer);
                return new CpuShortConvMixer(kind);
            }

            @Override
            public String toString() {
                return "SsmBlocks.cpu(" + kind + ")";
            }
        };
    }

    // ================================================================== layer checks

    /**
     * Checks that a layer is of the mixer kind a factory or mixer serves and carries its recurrent spec.
     *
     * @throws IllegalArgumentException otherwise
     */
    static void checkKind(MixerKind kind, LayerSpec layer) {
        if (layer.mixerKind() != kind || layer.recurrent() == null || layer.recurrent().kind() != kind) {
            throw new IllegalArgumentException(kind + " mixer cannot run layer " + layer.stack() + "." + layer.il()
                    + " (" + layer.mixerKind() + ")");
        }
    }

    /** Layer name used in messages. */
    static String describe(LayerSpec layer) {
        return layer.stack() + "." + layer.il();
    }

    /**
     * Floats per sequence of a recurrent state tensor of the layer's plan.
     *
     * @throws IllegalStateException when the tensor is missing or its size differs from {@code expected}
     */
    static long stateFloats(List<LayerMemoryPlan.StateTensor> tensors, String name, long expected, LayerSpec layer) {
        for (LayerMemoryPlan.StateTensor t : tensors) {
            if (t.name().equals(name)) {
                if (t.floats() != expected) {
                    throw new IllegalStateException("layer " + describe(layer) + ": recurrent state '" + name
                            + "' holds " + t.floats() + " floats, the " + layer.mixerKind() + " mixer needs "
                            + expected);
                }
                if (expected > Integer.MAX_VALUE) {
                    throw new IllegalStateException("layer " + describe(layer) + ": recurrent state '" + name
                            + "' of " + expected + " floats exceeds the kernels' 32-bit index range");
                }
                return t.floats();
            }
        }
        throw new IllegalStateException("layer " + describe(layer) + " has no recurrent state '" + name + "'");
    }

    // ================================================================== GPU tensors

    /**
     * Required matrix {@code <suffix>.weight} with {@code cols} inputs and {@code rows} outputs.
     *
     * @throws MissingTensorException when it is absent
     * @throws IllegalStateException  when its shape differs
     */
    static GpuWeight gpuMatrix(GpuWeights ws, LayerSpec layer, String base, long cols, long rows) {
        GpuWeight w = ws.layer(layer, TensorNames.weight(base));
        if (w.cols() != cols || w.rows() != rows) {
            throw new IllegalStateException("layer " + describe(layer) + ": " + w.name() + " is " + w.cols() + "x"
                    + w.rows() + ", the " + layer.mixerKind() + " mixer needs " + cols + "x" + rows);
        }
        return w;
    }

    /**
     * Tensor stored as {@code <base>.weight} or as bare {@code <base>} (llama.cpp creates {@code ssm_a},
     * {@code ssm_d} and the PLaMo-2 dt/B/C norms without the suffix), or null when neither exists.
     */
    static GpuWeight gpuEitherOrNull(GpuWeights ws, LayerSpec layer, String base) {
        GpuWeight w = ws.layerOrNull(layer, TensorNames.weight(base));
        return w != null ? w : ws.layerOrNull(layer, base);
    }

    /**
     * Required tensor as {@link #gpuEitherOrNull}.
     *
     * @throws MissingTensorException naming the bare tensor when neither form exists
     */
    static GpuWeight gpuEither(GpuWeights ws, LayerSpec layer, String base) {
        GpuWeight w = gpuEitherOrNull(ws, layer, base);
        if (w == null) {
            throw new MissingTensorException(ws.archName(), layer.il(), layer.tensorName(base));
        }
        return w;
    }

    /**
     * The tensor as a dense F32 device buffer of {@code floats} values in row order: its own buffer when it is a
     * single-buffer F32 tensor, otherwise a buffer filled by looking up every row with {@code GpuOps.embed}
     * (the row dequantization of the lookup kernels), added to {@code owned}.
     *
     * @throws IllegalStateException         when the tensor does not hold exactly {@code floats} values
     * @throws UnsupportedOperationException when no lookup kernel exists for its type
     */
    static GpuBuffer gpuF32(GpuForwardContext ctx, GpuWeight w, long floats, List<GpuBuffer> owned) {
        long n = w.rows() * w.cols();
        if (n != floats) {
            throw new IllegalStateException("tensor " + w.name() + " holds " + n + " values, " + floats + " expected");
        }
        if (w.type() == GGMLType.F32 && w.isSingleShard()) {
            return w.shard(0);
        }
        if (w.rows() > Integer.MAX_VALUE) {
            throw new IllegalStateException("tensor " + w.name() + " has too many rows to convert to F32");
        }
        int[] rows = new int[(int) w.rows()];
        for (int r = 0; r < rows.length; r++) {
            rows[r] = r;
        }
        GpuBuffer b = ctx.backend().allocateBuffer(Math.multiplyExact(floats, (long) Float.BYTES));
        owned.add(b);
        ctx.ops().embed(w, rows, b, 0);
        return b;
    }

    // ================================================================== CPU tensors

    /**
     * Required matrix {@code <suffix>.weight} read in place, with {@code cols} inputs and {@code rows} outputs.
     *
     * @throws MissingTensorException when it is absent
     * @throws IllegalStateException  when its shape differs
     */
    static CpuTensorRef cpuMatrix(CpuWeights ws, LayerSpec layer, String base, long cols, long rows) {
        String suffix = TensorNames.weight(base);
        CpuTensorRef w = ws.layer(layer, suffix);
        if (w.cols() != cols || w.rows() != rows) {
            throw new IllegalStateException("layer " + describe(layer) + ": " + layer.tensorName(suffix) + " is "
                    + w.cols() + "x" + w.rows() + ", the " + layer.mixerKind() + " mixer needs " + cols + "x" + rows);
        }
        return w;
    }

    /** F32 values of {@code <base>.weight} or bare {@code <base>} (shared, read only), or null when absent. */
    static float[] cpuEitherOrNull(CpuWeights ws, LayerSpec layer, String base, long floats) {
        float[] v = ws.layerFloatsOrNull(layer, TensorNames.weight(base));
        String name = TensorNames.weight(base);
        if (v == null) {
            v = ws.layerFloatsOrNull(layer, base);
            name = base;
        }
        if (v != null && v.length != floats) {
            throw new IllegalStateException("layer " + describe(layer) + ": " + layer.tensorName(name) + " holds "
                    + v.length + " values, " + floats + " expected");
        }
        return v;
    }

    /**
     * Required F32 values as {@link #cpuEitherOrNull}.
     *
     * @throws MissingTensorException naming the bare tensor when neither form exists
     */
    static float[] cpuEither(CpuWeights ws, LayerSpec layer, String base, long floats) {
        float[] v = cpuEitherOrNull(ws, layer, base, floats);
        if (v == null) {
            throw new MissingTensorException(ws.archName(), layer.il(), layer.tensorName(base));
        }
        return v;
    }

    /**
     * F32 values of the exact tensor {@code suffix} (shared, read only), or null when {@code required} is false and
     * it is absent.
     *
     * @throws MissingTensorException when a required tensor is absent
     * @throws IllegalStateException  when it does not hold {@code floats} values
     */
    static float[] cpuFloats(CpuWeights ws, LayerSpec layer, String suffix, long floats, boolean required) {
        float[] v = ws.layerFloatsOrNull(layer, suffix);
        if (v == null) {
            if (required) {
                throw new MissingTensorException(ws.archName(), layer.il(), layer.tensorName(suffix));
            }
            return null;
        }
        if (v.length != floats) {
            throw new IllegalStateException("layer " + describe(layer) + ": " + layer.tensorName(suffix) + " holds "
                    + v.length + " values, " + floats + " expected");
        }
        return v;
    }
}

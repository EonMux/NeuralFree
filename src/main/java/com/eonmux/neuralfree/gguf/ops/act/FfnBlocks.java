package com.eonmux.neuralfree.gguf.ops.act;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnBlock;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Block factories of the activation family: the dense feed-forward network ({@link FfnKind#DENSE}) on the GPU
 * ({@link GpuDenseFfn}) and the CPU ({@link CpuDenseFfn}). The family has no mixers and no layer extras.
 *
 * <p>{@link DenseLayout} is the shape contract both twins share: which form of llama.cpp {@code build_ffn}
 * (src/llama-graph.cpp:1922-2139) a layer uses, its widths, and the optional per-layer stages, validated against
 * the tensors the layer actually holds.</p>
 */
public final class FfnBlocks {

    private FfnBlocks() {
    }

    /** No mixers in this family. */
    public static List<GpuMixerFactory> gpuMixers() {
        return Collections.emptyList();
    }

    /** No mixers in this family. */
    public static List<CpuMixerFactory> cpuMixers() {
        return Collections.emptyList();
    }

    /** The GPU dense FFN factory. */
    public static List<GpuFfnFactory> gpuFfns() {
        return Collections.<GpuFfnFactory>singletonList(GpuDenseFactory.INSTANCE);
    }

    /** The CPU dense FFN factory. */
    public static List<CpuFfnFactory> cpuFfns() {
        return Collections.<CpuFfnFactory>singletonList(CpuDenseFactory.INSTANCE);
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
     * The dense FFN spec of a layer.
     *
     * @throws IllegalArgumentException when the layer's FFN kind is not {@link FfnKind#DENSE} or it carries no
     *                                  dense {@link FfnSpec}
     */
    static FfnSpec denseSpec(LayerSpec layer) {
        Objects.requireNonNull(layer, "layer");
        if (layer.ffnKind() != FfnKind.DENSE) {
            throw new IllegalArgumentException("layer " + layer.stack() + "." + layer.il() + " has FFN kind "
                    + layer.ffnKind() + ", not " + FfnKind.DENSE);
        }
        FfnSpec spec = layer.ffn();
        if (spec == null || spec.role() != FfnSpec.Role.DENSE) {
            throw new IllegalArgumentException("layer " + layer.stack() + "." + layer.il()
                    + " needs a DENSE FfnSpec, got " + spec);
        }
        return spec;
    }

    /** Creates {@link GpuDenseFfn} blocks. */
    private static final class GpuDenseFactory implements GpuFfnFactory {
        static final GpuDenseFactory INSTANCE = new GpuDenseFactory();

        @Override
        public FfnKind kind() {
            return FfnKind.DENSE;
        }

        @Override
        public GpuFfnBlock create(LayerSpec layer) {
            denseSpec(layer);
            return new GpuDenseFfn();
        }

        @Override
        public String toString() {
            return "FfnBlocks.gpuDense";
        }
    }

    /** Creates {@link CpuDenseFfn} blocks. */
    private static final class CpuDenseFactory implements CpuFfnFactory {
        static final CpuDenseFactory INSTANCE = new CpuDenseFactory();

        @Override
        public FfnKind kind() {
            return FfnKind.DENSE;
        }

        @Override
        public CpuFfnBlock create(LayerSpec layer) {
            denseSpec(layer);
            return new CpuDenseFfn();
        }

        @Override
        public String toString() {
            return "FfnBlocks.cpuDense";
        }
    }

    /**
     * Shape and stage plan of one dense FFN, shared by the GPU and CPU blocks. Immutable.
     *
     * <p>Every form works on token-major rows. The FFN working rows ({@link #gateUpWidth()} floats per token) start at
     * element 0 of the {@code FFN_GATE_UP} scratch slot; a separate up projection goes to the {@link #TEMP_UP}
     * temporary and the fused-up GLU output to {@link #TEMP_ACT}, each from element 0. The activated rows of
     * {@link #nFf()} floats then feed the down projection.</p>
     */
    static final class DenseLayout {

        /** Temporary holding the up projection beside a separate gate. */
        static final String TEMP_UP = "ffn.dense.up";
        /** Temporary holding the GLU output of a fused gate+up projection. */
        static final String TEMP_ACT = "ffn.dense.act";
        /** Temporary holding a biased or tensor-scaled down projection bound for an output view at an offset. */
        static final String TEMP_DOWN = "ffn.dense.down";
        /** {@code ffn_sub_norm.weight} (bitnet). */
        static final String SUB_NORM = TensorNames.weight(TensorNames.FFN_SUB_NORM);
        /** {@code ffn.act.scales} (mpt): GELU outputs divided per column (llama-graph.cpp:2032-2057). */
        static final String ACT_SCALES = TensorNames.FFN_ACT + ".scales";

        /** The four {@code build_ffn} forms. */
        enum Form {
            /** {@code down(glu(gate(x), up(x)))}: {@code LLM_FFN_PAR} with a separate gate. */
            PARALLEL_GLU,
            /** {@code down(glu(up(x)[:, :n_ff], up(x)[:, n_ff:]))}: gate half first in one {@code ffn_up}. */
            FUSED_GLU,
            /** {@code down(act(gate(up(x))))}: {@code LLM_FFN_SEQ} with a gate. */
            SEQUENTIAL,
            /** {@code down(act(up(x)))}. */
            UNGATED
        }

        private final FfnSpec spec;
        private final Form form;
        private final int nIn;
        private final int nFf;
        private final int nUp;
        private final int nOut;
        private final boolean subNorm;
        private final float subNormEps;

        private DenseLayout(FfnSpec spec, Form form, int nIn, int nFf, int nUp, int nOut, boolean subNorm,
                            float subNormEps) {
            this.spec = spec;
            this.form = form;
            this.nIn = nIn;
            this.nFf = nFf;
            this.nUp = nUp;
            this.nOut = nOut;
            this.subNorm = subNorm;
            this.subNormEps = subNormEps;
        }

        /**
         * The layout of a layer's dense FFN, validated against its tensor shapes.
         *
         * @param up   {@code {cols, rows}} of the up projection
         * @param gate {@code {cols, rows}} of the gate projection, or null when the layer has none
         * @param down {@code {cols, rows}} of the down projection
         * @throws IllegalArgumentException when the layer has no dense FFN spec
         * @throws IllegalStateException    when the tensors contradict the spec's form or widths
         */
        static DenseLayout of(LayerSpec layer, long[] up, long[] gate, long[] down) {
            FfnSpec spec = denseSpec(layer);
            String where = layer.stack() + "." + layer.il();
            Objects.requireNonNull(up, "up shape");
            Objects.requireNonNull(down, "down shape");
            Form form = formOf(spec);
            boolean needsGate = form == Form.PARALLEL_GLU || form == Form.SEQUENTIAL;
            if (needsGate != (gate != null)) {
                throw new IllegalStateException("layer " + where + ": " + form + " FFN " + (needsGate
                        ? "needs a gate projection " + spec.gateWeight() + " but the layer has none"
                        : "takes no gate projection but the layer holds " + spec.gateWeight()));
            }
            int nFf = spec.nFf();
            int nIn = width(up[0], where, "up input");
            int nUp = width(up[1], where, "up output");
            int nOut = width(down[1], where, "down output");
            expect(down[0] == nFf, where, "down projection reads " + down[0] + " columns, n_ff is " + nFf);
            switch (form) {
                case PARALLEL_GLU:
                    expect(nUp == nFf, where, "up projection has " + nUp + " rows, n_ff is " + nFf);
                    expect(gate[1] == nFf, where, "gate projection has " + gate[1] + " rows, n_ff is " + nFf);
                    expect(gate[0] == nIn, where, "gate reads " + gate[0] + " columns, up reads " + nIn);
                    break;
                case FUSED_GLU:
                    expect(nUp == 2L * nFf, where, "fused up projection has " + nUp + " rows, 2 * n_ff is "
                            + 2L * nFf);
                    break;
                case SEQUENTIAL:
                    expect(gate[0] == nUp, where, "gate reads " + gate[0] + " columns, the up projection has "
                            + nUp + " rows");
                    expect(gate[1] == nFf, where, "gate projection has " + gate[1] + " rows, n_ff is " + nFf);
                    break;
                default:
                    expect(nUp == nFf, where, "up projection has " + nUp + " rows, n_ff is " + nFf);
                    break;
            }
            NormSpec norm = layer.norm();
            boolean subNorm = norm != null && norm.ffnSubNorm();
            return new DenseLayout(spec, form, nIn, nFf, nUp, nOut, subNorm, subNorm ? norm.eps() : 0f);
        }

        private static Form formOf(FfnSpec spec) {
            if (spec.gating() == FfnGating.FUSED_UP) {
                return Form.FUSED_GLU;
            }
            if (spec.gating() == FfnGating.PAR) {
                return Form.PARALLEL_GLU;
            }
            return spec.gated() ? Form.SEQUENTIAL : Form.UNGATED;
        }

        private static int width(long v, String where, String what) {
            if (v <= 0 || v > Integer.MAX_VALUE) {
                throw new IllegalStateException("layer " + where + ": " + what + " width " + v + " is out of range");
            }
            return (int) v;
        }

        private static void expect(boolean ok, String where, String message) {
            if (!ok) {
                throw new IllegalStateException("layer " + where + ": " + message);
            }
        }

        FfnSpec spec() { return spec; }
        Form form() { return form; }
        /** Input width (the up projection's columns). */
        int nIn() { return nIn; }
        /** Activated width, read by the down projection. */
        int nFf() { return nFf; }
        /** Rows of the up projection ({@code 2 * n_ff} when fused). */
        int nUp() { return nUp; }
        /** Output width (the down projection's rows). */
        int nOut() { return nOut; }
        boolean hasGate() { return form == Form.PARALLEL_GLU || form == Form.SEQUENTIAL; }
        boolean usesGlu() { return form == Form.PARALLEL_GLU || form == Form.FUSED_GLU; }
        GluOp glu() { return spec.glu(); }
        ActOp act() { return spec.act(); }
        ActParams params() { return spec.params(); }

        /** Floats per token used at the start of {@code FFN_GATE_UP}: the gate rows, or the fused gate+up rows. */
        int gateUpWidth() {
            return form == Form.FUSED_GLU ? 2 * nFf : nFf;
        }

        /** Floats per token of {@link #TEMP_UP}; 0 when the up projection writes into {@code FFN_GATE_UP}. */
        int upTempWidth() {
            if (form == Form.PARALLEL_GLU) {
                return nFf;
            }
            return form == Form.SEQUENTIAL ? nUp : 0;
        }

        /** Row stride of the gate values in {@code FFN_GATE_UP}. */
        int gateStride() {
            return gateUpWidth();
        }

        /** Gemma3n Gaussian top-k multiplier on the gate rows; 0 when the layer has no activation sparsity. */
        float sparsity() {
            return spec.activationSparsityStdMul();
        }

        /** The layer applies {@code ffn_sub_norm} (RMS) to the activated rows before the down projection. */
        boolean subNorm() { return subNorm; }
        float subNormEps() { return subNormEps; }

        /**
         * Whether {@code ffn.act.scales} divides the activation: only after a GELU or GELU-erf activation that is not
         * a GLU ({@code build_ffn} applies {@code act_scales} on those paths only, llama-graph.cpp:2032-2057).
         */
        boolean actScalesApply() {
            if (form != Form.UNGATED && form != Form.SEQUENTIAL) {
                return false;
            }
            ActOp a = spec.act();
            return a == ActOp.GELU_TANH || a == ActOp.GELU_ERF;
        }

        /** Base tensor names of the gate, up and down projections ({@code ffn_gate}, ...). */
        String gateBase() { return spec.role().gate(); }
        String upBase() { return spec.role().up(); }
        String downBase() { return spec.role().down(); }

        @Override
        public String toString() {
            return "DenseLayout{" + form + ", in=" + nIn + ", ff=" + nFf + ", up=" + nUp + ", out=" + nOut
                    + ", op=" + (usesGlu() ? glu() : act()) + (sparsity() > 0f ? ", sparsity=" + sparsity() : "")
                    + (subNorm ? ", subNorm" : "") + '}';
        }
    }
}

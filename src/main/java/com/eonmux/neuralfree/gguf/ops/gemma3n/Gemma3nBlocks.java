package com.eonmux.neuralfree.gguf.ops.gemma3n;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Factory exposure of the Gemma3n family for the block catalogs: the layer extras
 * {@code altup} and {@code laurel} (gemma3n), {@code ple} (gemma4, gemma-embedding2) and {@code ngram_ple}
 * (qwen4exp), plus the model-level layouts their GPU and CPU forms share. The family has no mixer and no FFN; the
 * gemma3n activation sparsity belongs to the FFN blocks.
 *
 * <p>Extras contract for the spec builder:
 * <ul>
 *   <li>gemma3n: every layer lists {@code altup} then {@code laurel}. The per-layer embedding of gemma3n is part of
 *       the AltUp correction (gemma3n.cpp:232-258), so gemma3n layers do not list {@code ple}; the {@code ple}
 *       extra rejects a layer that also lists {@code altup}.</li>
 *   <li>gemma4 with {@code per_layer_token_embd} and gemma-embedding2: every layer lists {@code ple}, before any
 *       extra that rescales the layer output ({@code layer_out_scale}), because the per-layer embedding is added
 *       first (gemma4.cpp:376-403, gemma-embedding2.cpp:172-197).</li>
 *   <li>qwen4exp: the PLE layer lists {@code ngram_ple} before {@code hc}: the n-gram embeddings are added to the
 *       stream before the attention collapse (qwen4exp.cpp:452-461).</li>
 * </ul>
 * Layer 0 must list {@code altup} (gemma3n) or {@code ple} (gemma4, gemma-embedding2): its extra computes the
 * per-layer inputs of every layer from the embedding output.</p>
 */
public final class Gemma3nBlocks {

    private static final GpuLayerExtraFactory GPU_ALTUP = gpu(LayerSpec.Extras.ALTUP);
    private static final GpuLayerExtraFactory GPU_LAUREL = gpu(LayerSpec.Extras.LAUREL);
    private static final GpuLayerExtraFactory GPU_PLE = gpu(LayerSpec.Extras.PLE);
    private static final GpuLayerExtraFactory GPU_NGRAM_PLE = gpu(LayerSpec.Extras.NGRAM_PLE);
    private static final CpuLayerExtraFactory CPU_ALTUP = cpu(LayerSpec.Extras.ALTUP);
    private static final CpuLayerExtraFactory CPU_LAUREL = cpu(LayerSpec.Extras.LAUREL);
    private static final CpuLayerExtraFactory CPU_PLE = cpu(LayerSpec.Extras.PLE);
    private static final CpuLayerExtraFactory CPU_NGRAM_PLE = cpu(LayerSpec.Extras.NGRAM_PLE);

    private Gemma3nBlocks() {
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

    /** The {@code altup}, {@code laurel}, {@code ple} and {@code ngram_ple} GPU layer extras (unmodifiable). */
    public static List<GpuLayerExtraFactory> gpuExtras() {
        return Collections.unmodifiableList(Arrays.asList(GPU_ALTUP, GPU_LAUREL, GPU_PLE, GPU_NGRAM_PLE));
    }

    /** The {@code altup}, {@code laurel}, {@code ple} and {@code ngram_ple} CPU layer extras (unmodifiable). */
    public static List<CpuLayerExtraFactory> cpuExtras() {
        return Collections.unmodifiableList(Arrays.asList(CPU_ALTUP, CPU_LAUREL, CPU_PLE, CPU_NGRAM_PLE));
    }

    private static GpuLayerExtraFactory gpu(String id) {
        return new GpuLayerExtraFactory() {
            @Override
            public String featureId() {
                return id;
            }

            @Override
            public GpuLayerExtra create(LayerSpec layer) {
                Objects.requireNonNull(layer, "layer");
                switch (id) {
                    case LayerSpec.Extras.ALTUP:
                        return new GpuAltUpExtra();
                    case LayerSpec.Extras.LAUREL:
                        return new GpuAltUpExtra.Laurel();
                    case LayerSpec.Extras.PLE:
                        return new GpuPleExtra();
                    default:
                        return new GpuNgramPleExtra();
                }
            }

            @Override
            public String toString() {
                return "Gemma3nBlocks.gpu(" + id + ")";
            }
        };
    }

    private static CpuLayerExtraFactory cpu(String id) {
        return new CpuLayerExtraFactory() {
            @Override
            public String featureId() {
                return id;
            }

            @Override
            public CpuLayerExtra create(LayerSpec layer) {
                Objects.requireNonNull(layer, "layer");
                switch (id) {
                    case LayerSpec.Extras.ALTUP:
                        return new CpuAltUpExtra();
                    case LayerSpec.Extras.LAUREL:
                        return new CpuAltUpExtra.Laurel();
                    case LayerSpec.Extras.PLE:
                        return new CpuPleExtra();
                    default:
                        return new CpuNgramPleExtra();
                }
            }

            @Override
            public String toString() {
                return "Gemma3nBlocks.cpu(" + id + ")";
            }
        };
    }

    /**
     * Requires a tensor to have exactly the ggml shape {@code {cols, rows}} (any rank whose trailing dimensions
     * multiply to {@code rows}).
     *
     * @throws IllegalArgumentException naming the tensor and both shapes otherwise
     */
    static void requireShape(String arch, TensorRef t, long cols, long rows) {
        if (t.cols() != cols || t.rows() != rows) {
            throw new IllegalArgumentException(arch + ": tensor " + t.name() + " has shape " + Arrays.toString(t.ne())
                    + ", expected " + cols + " columns and " + rows + " rows");
        }
    }

    /**
     * Requires a vector tensor of exactly {@code n} elements (stored as {@code {n}} or reshaped, for example
     * {@code {n_embd, hc}}).
     *
     * @throws IllegalArgumentException naming the tensor otherwise
     */
    static void requireElements(String arch, TensorRef t, long n) {
        if (t.numElements() != n) {
            throw new IllegalArgumentException(arch + ": tensor " + t.name() + " has " + t.numElements()
                    + " elements, expected " + n);
        }
    }

    /**
     * The per-layer embedding inputs of gemma3n, gemma4 and gemma-embedding2: {@code inp_per_layer[t][l][p]} for
     * token {@code t < n_tokens}, layer {@code l < n_layer} and {@code p < width}, computed once per step from the
     * embedding output {@code inpL} (llama.cpp's {@code [width, n_layer, n_tokens]} tensor before its permute):
     * <pre>
     *   proj = rms_norm(per_layer_model_proj . inpL * (1/sqrt(n_embd))) * per_layer_proj_norm      (per width row)
     *   gemma3n, gemma4:   inp = (proj + per_layer_token_embd[token] * sqrt(width)) * (1/sqrt(2))
     *   gemma-embedding2:  inp = proj
     * </pre>
     * (gemma3n.cpp:320-373, gemma4.cpp:182-190 and 480-533, gemma-embedding2.cpp:219-232). Layer {@code il} reads the
     * {@code [n_tokens][width]} view at offset {@code il*width} with row stride {@code n_layer*width}. The width is
     * {@code n_embd_altup} for gemma3n and {@code n_embd_per_layer} otherwise; gemma-embedding2 has no token table.
     * Immutable.
     */
    static final class PleLayout {
        private final String arch;
        private final int nEmbd;
        private final int nLayer;
        private final int width;
        private final float eps;
        private final TensorRef tokenTable;
        private final TensorRef modelProj;
        private final TensorRef projNorm;

        private PleLayout(String arch, int nEmbd, int nLayer, int width, float eps, TensorRef tokenTable,
                          TensorRef modelProj, TensorRef projNorm) {
            this.arch = arch;
            this.nEmbd = nEmbd;
            this.nLayer = nLayer;
            this.width = width;
            this.eps = eps;
            this.tokenTable = tokenTable;
            this.modelProj = modelProj;
            this.projNorm = projNorm;
        }

        /**
         * Resolves the layout from the model's hparams and tensors.
         *
         * @throws UnsupportedOperationException when the architecture has no per-layer embedding inputs
         * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
         * @throws IllegalArgumentException      when the width is not positive or a tensor shape does not fit
         */
        static PleLayout resolve(ModelSpec spec) {
            Objects.requireNonNull(spec, "spec");
            ModelHparams hp = spec.hparams();
            String arch = spec.arch().ggufName();
            TensorCatalog tc = spec.tensors();
            int width;
            boolean tokens;
            if (spec.arch() == Arch.GEMMA3N) {
                width = hp.nEmbdAltup();
                tokens = true;
            } else if (spec.arch() == Arch.GEMMA4) {
                width = hp.nEmbdPerLayer();
                tokens = true;
            } else if (spec.arch() == Arch.GEMMA_EMBEDDING2) {
                width = hp.nEmbdPerLayer();
                tokens = false;
            } else {
                throw new UnsupportedOperationException(arch + " has no per-layer embedding inputs");
            }
            if (width <= 0) {
                throw new IllegalArgumentException(arch + ": the per-layer embedding width must be positive, got "
                        + width);
            }
            int nEmbd = hp.nEmbd();
            int nLayer = spec.nLayer();
            long all = (long) width * nLayer;
            TensorRef modelProj = tc.require(TensorNames.weight(TensorNames.PER_LAYER_MODEL_PROJ));
            requireShape(arch, modelProj, nEmbd, all);
            TensorRef projNorm = tc.require(TensorNames.weight(TensorNames.PER_LAYER_PROJ_NORM));
            requireElements(arch, projNorm, width);
            TensorRef table = null;
            if (tokens) {
                table = tc.require(TensorNames.weight(TensorNames.PER_LAYER_TOKEN_EMBD));
                if (table.cols() != all) {
                    throw new IllegalArgumentException(arch + ": " + table.name() + " rows hold " + table.cols()
                            + " floats, expected n_layer*width = " + all);
                }
            }
            return new PleLayout(arch, nEmbd, nLayer, width, hp.normRmsEps(), table, modelProj, projNorm);
        }

        String arch() { return arch; }
        int nEmbd() { return nEmbd; }
        int nLayer() { return nLayer; }
        /** Floats per layer and token ({@code n_embd_altup} or {@code n_embd_per_layer}). */
        int width() { return width; }
        /** Floats per token: {@code n_layer * width}. */
        int rowStride() { return nLayer * width; }

        /**
         * Floats of the inputs of {@code ubatch} tokens.
         *
         * @throws IllegalArgumentException when they exceed the 32-bit index range of the kernels
         */
        int floats(int ubatch) {
            long n = (long) ubatch * nLayer * width;
            if (n > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(arch + ": per-layer inputs of " + ubatch + " tokens (" + n
                        + " floats) exceed the 32-bit index range; reduce the ubatch");
            }
            return (int) n;
        }
        /** Offset of layer {@code il}'s view within a token row. */
        int layerOffset(int il) { return il * width; }
        float eps() { return eps; }
        /** {@code per_layer_token_embd}, or null for gemma-embedding2. */
        TensorRef tokenTable() { return tokenTable; }
        TensorRef modelProj() { return modelProj; }
        TensorRef projNorm() { return projNorm; }
        /** {@code 1/sqrtf(n_embd)} on the projection (gemma3n.cpp:353). */
        float projScale() { return 1.0f / (float) Math.sqrt((float) nEmbd); }
        /** {@code sqrtf(width)} on the token rows (gemma3n.cpp:323). */
        float tokenScale() { return (float) Math.sqrt((float) width); }
        /** {@code 1/sqrtf(2)} on the sum (gemma3n.cpp:354). */
        float inputScale() { return 1.0f / (float) Math.sqrt(2.0f); }

        /**
         * Validates a layer's per-layer embedding gate: {@code inp_gate {n_embd, width}}, {@code proj {width, n_embd}}
         * and {@code post_norm {n_embd}} at the layer's weight layer.
         *
         * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when one is absent
         * @throws IllegalArgumentException      when a shape does not fit
         */
        void requireLayerTensors(ModelSpec spec, LayerSpec layer) {
            TensorCatalog tc = spec.tensors();
            String stack = layer.stack();
            int wl = layer.weightLayer();
            requireShape(arch, tc.require(stack, wl, TensorNames.weight(TensorNames.PER_LAYER_INP_GATE)), nEmbd, width);
            requireShape(arch, tc.require(stack, wl, TensorNames.weight(TensorNames.PER_LAYER_PROJ)), width, nEmbd);
            requireElements(arch, tc.require(stack, wl, TensorNames.weight(TensorNames.PER_LAYER_POST_NORM)), nEmbd);
        }

        @Override
        public String toString() {
            return "PleLayout{" + arch + ", width=" + width + ", layers=" + nLayer + (tokenTable != null ? ", tokens" : "")
                    + '}';
        }
    }

    /**
     * The qwen4exp n-gram per-layer embedding of one PLE layer (qwen4exp.cpp:1203-1420): the hash constants, the
     * shapes and the depthwise convolution geometry, plus the host-side hash. Immutable.
     *
     * <p>Shapes: the residual stream holds {@code hc} streams of {@code n_embd} ({@code D = hc*n_embd} per token);
     * {@code per_layer_token_embd} rows hold {@code ple_head_dim} floats and {@code ple_n_heads*ple_head_dim = n_embd};
     * {@code ple_key {n_embd, D}}, {@code ple_value {n_embd, n_embd}}, {@code ple_norm_* } hold {@code D} floats as
     * {@code [hc][n_embd]}; {@code ple_conv1d {K, D}} with {@code K = ple_conv_kernel}. The convolution is dilated by
     * {@code ple_ngram_size} and keeps {@code hist = (K-1)*ple_ngram_size} steps of history per channel in the
     * {@code ple_conv} recurrent state, laid out {@code [D][hist]} with time innermost, oldest first.</p>
     */
    static final class NgramLayout {
        private final String arch;
        private final int hc;
        private final int nEmbd;
        private final int nGram;
        private final int perGram;
        private final int nHeads;
        private final int headDim;
        private final int eos;
        private final long[] multipliers;
        private final long[] offsets;
        private final long[] vocab;
        private final int kernel;
        private final float eps;

        private NgramLayout(String arch, int hc, int nEmbd, int nGram, int perGram, int nHeads, int headDim, int eos,
                            long[] multipliers, long[] offsets, long[] vocab, int kernel, float eps) {
            this.arch = arch;
            this.hc = hc;
            this.nEmbd = nEmbd;
            this.nGram = nGram;
            this.perGram = perGram;
            this.nHeads = nHeads;
            this.headDim = headDim;
            this.eos = eos;
            this.multipliers = multipliers;
            this.offsets = offsets;
            this.vocab = vocab;
            this.kernel = kernel;
            this.eps = eps;
        }

        /**
         * Resolves and validates the layout of a PLE layer.
         *
         * @throws IllegalArgumentException when the layer is not a PLE layer, the residual is not a hyper-connection
         *                                  stream, an hparam is out of range or a tensor shape does not fit
         * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a tensor is absent
         */
        static NgramLayout resolve(ModelSpec spec, LayerSpec layer) {
            Objects.requireNonNull(spec, "spec");
            Objects.requireNonNull(layer, "layer");
            ModelHparams hp = spec.hparams();
            String arch = spec.arch().ggufName();
            if (!hp.isPleLayer(layer.weightLayer())) {
                throw new IllegalArgumentException(arch + " layer " + layer.il() + " lists '"
                        + LayerSpec.Extras.NGRAM_PLE + "' but is not a PLE layer of the model");
            }
            ResidualSpec r = layer.residual();
            int hc = r.hcMult();
            int nEmbd = hp.nEmbd();
            if (hc < 1 || hc > HcKernels.MAX_HC || r.streamWidth() != hc * nEmbd) {
                throw new IllegalArgumentException(arch + " layer " + layer.il() + ": the n-gram embedding needs a "
                        + "residual of hc*n_embd with 1 <= hc <= " + HcKernels.MAX_HC + ", got " + r);
            }
            int nGram = hp.pleNgramSize();
            int perGram = hp.pleHeadsPerNgram();
            int nHeads = hp.pleNHeads();
            int headDim = hp.pleHeadDim();
            int kernel = hp.pleConvKernel();
            if (nGram < 2 || perGram < 1 || nHeads != (nGram - 1) * perGram || kernel < 1 || headDim < 1) {
                throw new IllegalArgumentException(arch + ": PLE hparams out of range: ngram=" + nGram + " per_ngram="
                        + perGram + " heads=" + nHeads + " head_dim=" + headDim + " conv_kernel=" + kernel);
            }
            if ((long) nHeads * headDim != nEmbd) {
                throw new IllegalArgumentException(arch + ": " + nHeads + " PLE heads of " + headDim
                        + " do not span n_embd " + nEmbd);
            }
            long[] mult = hp.pleLayerMultipliers();
            long[] off = hp.pleHeadOffsets();
            long[] voc = hp.pleHeadVocabSizes();
            if (mult.length < nGram || off.length < nHeads || voc.length < nHeads) {
                throw new IllegalArgumentException(arch + ": PLE hash arrays are shorter than ngram " + nGram
                        + " / heads " + nHeads);
            }
            TensorCatalog tc = spec.tensors();
            TensorRef table = tc.require(TensorNames.weight(TensorNames.PER_LAYER_TOKEN_EMBD));
            if (table.cols() != headDim) {
                throw new IllegalArgumentException(arch + ": " + table.name() + " rows hold " + table.cols()
                        + " floats, expected ple_head_dim " + headDim);
            }
            for (int h = 0; h < nHeads; h++) {
                if (voc[h] <= 0 || off[h] < 0 || off[h] + voc[h] > table.rows()) {
                    throw new IllegalArgumentException(arch + ": PLE head " + h + " rows [" + off[h] + ", "
                            + (off[h] + voc[h]) + ") fall outside the " + table.rows() + " rows of " + table.name());
                }
            }
            long d = (long) hc * nEmbd;
            String stack = layer.stack();
            int wl = layer.weightLayer();
            requireShape(arch, tc.require(stack, wl, TensorNames.weight(TensorNames.PLE_KEY)), nEmbd, d);
            requireShape(arch, tc.require(stack, wl, TensorNames.weight(TensorNames.PLE_VALUE)), nEmbd, nEmbd);
            requireElements(arch, tc.require(stack, wl, TensorNames.weight(TensorNames.PLE_NORM_KEY)), d);
            requireElements(arch, tc.require(stack, wl, TensorNames.weight(TensorNames.PLE_NORM_QUERY)), d);
            requireElements(arch, tc.require(stack, wl, TensorNames.weight(TensorNames.PLE_NORM_CONV)), d);
            requireShape(arch, tc.require(stack, wl, TensorNames.weight(TensorNames.PLE_CONV1D)), kernel, d);
            return new NgramLayout(arch, hc, nEmbd, nGram, perGram, nHeads, headDim, hp.pleEosTokenId(),
                    Arrays.copyOf(mult, nGram), Arrays.copyOf(off, nHeads), Arrays.copyOf(voc, nHeads), kernel,
                    hp.normRmsEps());
        }

        String arch() { return arch; }
        int hc() { return hc; }
        int nEmbd() { return nEmbd; }
        /** Floats per token of the hyper-connection stream: {@code hc*n_embd}. */
        int streamWidth() { return hc * nEmbd; }
        int nGram() { return nGram; }
        int nHeads() { return nHeads; }
        int headDim() { return headDim; }
        /** Convolution taps {@code K}. */
        int kernel() { return kernel; }
        /** Convolution dilation: the n-gram size. */
        int dilation() { return nGram; }
        /** History steps per channel: {@code (K-1)*dilation}. */
        int history() { return (kernel - 1) * nGram; }
        float eps() { return eps; }
        /** {@code 1/sqrtf(n_embd)} on the per-stream key-query dot product (qwen4exp.cpp:1355). */
        float scoreScale() { return 1.0f / (float) Math.sqrt((float) nEmbd); }
        /** Positions before the step that the hash reads: {@code ngram - 1}. */
        int lookBehind() { return nGram - 1; }

        /**
         * The n-gram row indices of {@code nTokens} tokens (qwen4exp.cpp:1240-1290). {@code window[w]} holds the
         * token at position {@code pos0 - lookBehind() + w}, or a negative value when that position has no token
         * (before the sequence start); token {@code i} of the step is {@code window[i + lookBehind()]}. For each
         * token the context {@code ctx[0] = token}, {@code ctx[s]} = the token {@code s} positions back, is cut at the
         * first missing or EOS predecessor (every later entry reads as EOS); for {@code n = 2 .. ngram} the hash
         * {@code mixed = XOR_{j<n} ctx[j]*mult[j]} (unsigned 64-bit) selects, for each of the {@code per_ngram} heads
         * {@code h} of that n, row {@code mixed mod vocab[h] + offset[h]} into {@code idx[i*n_heads + h]}.
         *
         * @throws IllegalArgumentException when an array is too short for {@code nTokens} tokens
         */
        void hash(int[] window, int nTokens, int[] idx) {
            int back = lookBehind();
            if (nTokens < 0 || window.length < nTokens + back || idx.length < (long) nTokens * nHeads) {
                throw new IllegalArgumentException("n-gram hash of " + nTokens + " tokens needs a window of "
                        + (nTokens + back) + " and " + ((long) nTokens * nHeads) + " indices, got " + window.length
                        + " and " + idx.length);
            }
            for (int i = 0; i < nTokens; i++) {
                int at = i + back;
                long mixed = (long) window[at] * multipliers[0];
                boolean cut = false;
                for (int n = 2; n <= nGram; n++) {
                    int s = n - 1;
                    int t = cut ? -1 : window[at - s];
                    cut = cut || t < 0 || t == eos;
                    long c = cut ? eos : t;
                    mixed ^= c * multipliers[s];
                    int base = i * nHeads + (n - 2) * perGram;
                    for (int g = 0; g < perGram; g++) {
                        int h = (n - 2) * perGram + g;
                        idx[base + g] = (int) (Long.remainderUnsigned(mixed, vocab[h]) + offsets[h]);
                    }
                }
            }
        }

        @Override
        public String toString() {
            return "NgramLayout{" + arch + ", hc=" + hc + ", ngram=" + nGram + ", heads=" + nHeads + "x" + headDim
                    + ", conv=" + kernel + '}';
        }
    }
}

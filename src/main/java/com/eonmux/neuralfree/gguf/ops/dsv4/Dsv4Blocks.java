package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.CompressedKvPlan;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Kernels.CompMode;
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Kernels.RopeParams;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseBlocks.IndexerConfig;
import com.eonmux.neuralfree.gguf.spec.DsaSpec;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Block factories of the DeepSeek-V4 attention family ({@link MixerKind#DSV4}): {@link GpuDsv4Mixer} on the GPU and
 * {@link CpuDsv4Mixer} on the CPU (llama.cpp src/models/deepseek4.cpp:886-1229). The family has no FFN blocks and no
 * layer extras: the MoE FFN (including hash routing) belongs to the MoE family and the hyper-connection pre/post mixes
 * to the HC family's extra, which collapses the {@code hc} streams before the attention norm and expands them after the
 * branch (deepseek4.cpp:1252-1268). A mixer therefore receives {@code n_embd}-wide, {@code attn_norm}-ed rows and
 * returns {@code n_embd}-wide branch rows.
 *
 * <p>{@link Dsv4Layout} is the shape and RoPE contract both mixers share, validated against the tensors a layer
 * holds.</p>
 */
public final class Dsv4Blocks {

    private Dsv4Blocks() {
    }

    /** The GPU DeepSeek-V4 mixer factory. */
    public static List<GpuMixerFactory> gpuMixers() {
        return Collections.<GpuMixerFactory>singletonList(GpuDsv4Factory.INSTANCE);
    }

    /** The CPU DeepSeek-V4 mixer factory. */
    public static List<CpuMixerFactory> cpuMixers() {
        return Collections.<CpuMixerFactory>singletonList(CpuDsv4Factory.INSTANCE);
    }

    /** No FFN blocks in this family. */
    public static List<GpuFfnFactory> gpuFfns() {
        return Collections.emptyList();
    }

    /** No FFN blocks in this family. */
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
     * Checks that a layer is a DeepSeek-V4 layer this family runs.
     *
     * @throws IllegalArgumentException      when the layer's mixer kind is not {@link MixerKind#DSV4} or it has no
     *                                       {@link Dsv4Spec}
     * @throws UnsupportedOperationException for a layer without the {@code q_lora} path or without RoPE
     */
    static void checkVariant(LayerSpec layer) {
        Objects.requireNonNull(layer, "layer");
        String where = "layer " + layer.stack() + "." + layer.il();
        if (layer.mixerKind() != MixerKind.DSV4 || layer.dsv4() == null) {
            throw new IllegalArgumentException(where + " has mixer kind " + layer.mixerKind() + (layer.dsv4() == null
                    ? " without a Dsv4Spec" : "") + ", not a " + MixerKind.DSV4 + " layer");
        }
        Dsv4Spec v4 = layer.dsv4();
        if (v4.qLoraRank() <= 0) {
            throw new UnsupportedOperationException(where + ": DeepSeek-V4 attention needs the q_lora path (wq_a, "
                    + "attn_q_a_norm, wq_b), whose normalized rank also feeds the indexer queries"
                    + " (deepseek4.cpp:918-922, 610)");
        }
        if (v4.ropeDim() <= 0 || layer.rope() == null) {
            throw new UnsupportedOperationException(where + ": DeepSeek-V4 attention rotates the last n_rot head dims"
                    + " (deepseek4.cpp:928-940); the layer has no RoPE");
        }
    }

    // ------------------------------------------------------------------ factories

    private enum GpuDsv4Factory implements GpuMixerFactory {
        INSTANCE;

        @Override
        public MixerKind kind() {
            return MixerKind.DSV4;
        }

        @Override
        public GpuMixer create(LayerSpec layer) {
            checkVariant(layer);
            return new GpuDsv4Mixer();
        }
    }

    private enum CpuDsv4Factory implements CpuMixerFactory {
        INSTANCE;

        @Override
        public MixerKind kind() {
            return MixerKind.DSV4;
        }

        @Override
        public CpuMixer create(LayerSpec layer) {
            checkVariant(layer);
            return new CpuDsv4Mixer();
        }
    }

    // ------------------------------------------------------------------ layer layout

    /** Shape lookup of the tensors of one layer: {@code {cols, rows}} of a tensor, or null when it is absent. */
    interface ShapeSource {
        long[] shape(String suffix);
    }

    /**
     * Shape and RoPE contract of one DeepSeek-V4 layer, shared by the GPU and CPU mixers and validated against the
     * tensors the layer holds (llama.cpp src/models/deepseek4.cpp:100-150 loads them, 886-1229 uses them). Immutable.
     *
     * <p>RoPE (deepseek4.cpp:909-916, 580-583, 613-615): the q / kv rotation and its inverse on the attention output
     * use the compress base with the context frequency scale, YaRN extrapolation, betas and original context on
     * compressed layers ({@code ratio != 0}), and the model base with scale 1 and no extrapolation on window-only
     * layers; the compressed rows and the indexer queries always use the compress-base parameters. Every rotation
     * takes {@code attn_factor = Dsv4Kernels.attnFactor(freq_scale, ext_factor)}, which cancels the YaRN magnitude
     * scale, and the model's rope type over the last {@code n_rot} dims.</p>
     */
    static final class Dsv4Layout {

        static final String WQ_A = TensorNames.weight(TensorNames.ATTN_Q_A);
        static final String Q_A_NORM = TensorNames.weight(TensorNames.ATTN_Q_A_NORM);
        static final String WQ_B = TensorNames.weight(TensorNames.ATTN_Q_B);
        static final String WKV = TensorNames.weight(TensorNames.ATTN_KV);
        static final String KV_NORM = TensorNames.weight(TensorNames.ATTN_KV_NORM);
        static final String WO_A = TensorNames.weight(TensorNames.ATTN_OUT_A);
        static final String WO_B = TensorNames.weight(TensorNames.ATTN_OUT_B);
        static final String SINKS = TensorNames.weight(TensorNames.ATTN_SINKS);
        static final String COMP_WKV = TensorNames.weight(TensorNames.ATTN_COMPRESSOR_WKV);
        static final String COMP_WGATE = TensorNames.weight(TensorNames.ATTN_COMPRESSOR_WGATE);
        static final String COMP_APE = TensorNames.weight(TensorNames.ATTN_COMPRESSOR_APE);
        static final String COMP_NORM = TensorNames.weight(TensorNames.ATTN_COMPRESSOR_NORM);
        static final String IDX_Q_B = TensorNames.weight(TensorNames.INDEXER_ATTN_Q_B);
        static final String IDX_PROJ = TensorNames.weight(TensorNames.INDEXER_PROJ);
        static final String IDX_COMP_WKV = TensorNames.weight(TensorNames.INDEXER_COMPRESSOR_WKV);
        static final String IDX_COMP_WGATE = TensorNames.weight(TensorNames.INDEXER_COMPRESSOR_WGATE);
        static final String IDX_COMP_APE = TensorNames.weight(TensorNames.INDEXER_COMPRESSOR_APE);
        static final String IDX_COMP_NORM = TensorNames.weight(TensorNames.INDEXER_COMPRESSOR_NORM);

        final String where;
        final int il;
        final int nEmbd;
        final int nHead;
        final int headDim;
        final int nRot;
        /** Unrotated head dims, {@code headDim - nRot}: the RoPE offset inside a head. */
        final int nope;
        final int qLora;
        final int nGroups;
        final int oLora;
        /** Input width of one {@code wo_a} group, {@code nHead / nGroups * headDim}. */
        final int oGroupDim;
        /** Compress ratio: 0 (window only), {@link Dsv4Spec#RATIO_CSA} or {@link Dsv4Spec#RATIO_HCA}. */
        final int ratio;
        final CompMode mode;
        /** Raw sliding window {@code n_swa}. */
        final int window;
        /** {@code f_norm_rms_eps} of every RMS norm of the layer. */
        final float rmsEps;
        final float kqScale;
        final boolean sinks;
        /** Compressor plane width of the CSA / HCA compressor ({@code 2 * headDim} or {@code headDim}); 0 without. */
        final int stateWidth;
        /** RoPE of q and kv (LLM_RoPE_Ext / CpuRope), offset {@link #nope}. */
        final RopeSpec attnRope;
        /** The same rotation as kernel scalars, undone on the attention output. */
        final RopeParams attnParams;
        /** RoPE of the compressed rows (CSA, HCA and indexer compressors). */
        final RopeParams compressParams;
        /** Indexer heads, head size, plane width and top-k; 0 on non-CSA layers. */
        final int idxHeads;
        final int idxDim;
        final int idxStateWidth;
        final int topK;
        /** Indexer configuration inputs (CSA layers only, else null / NaN). */
        private final float idxWeightScale;
        private final RopeSpec indexerRope;

        private Dsv4Layout(LayerSpec layer, ModelSpec spec, ShapeSource shapes) {
            checkVariant(layer);
            Dsv4Spec v4 = layer.dsv4();
            RopeSpec base = layer.rope();
            this.where = spec.arch().ggufName() + " layer " + layer.stack() + "." + layer.il();
            this.il = layer.il();
            this.nEmbd = spec.hparams().nEmbd();
            this.nHead = v4.nHead();
            this.headDim = v4.headDim();
            this.nRot = v4.ropeDim();
            this.nope = headDim - nRot;
            this.qLora = v4.qLoraRank();
            this.nGroups = v4.oGroupCount();
            this.oLora = v4.oLoraRank();
            this.oGroupDim = nHead / nGroups * headDim;
            this.ratio = v4.compressRatio();
            this.mode = CompMode.forRatio(ratio);
            this.window = v4.window();
            this.rmsEps = spec.hparams().normRmsEps();
            this.kqScale = v4.kqScale();
            this.sinks = v4.sinks();
            this.stateWidth = ratio == 0 ? 0 : Dsv4Kernels.stateWidth(ratio, headDim);
            Dsv4Kernels.checkRowWidth(headDim);
            RopeType type = base.type();
            Dsv4Kernels.ropeModeCode(type);
            if (base.nDims() != nRot) {
                throw new IllegalStateException(where + ": the layer RoPE rotates " + base.nDims() + " dims, the"
                        + " DeepSeek-V4 rope dim is " + nRot + " (hparams.n_rot(), deepseek4.cpp:898)");
            }

            RopeSpec compressed = ratio == 0 ? null : rope(type, nRot, nope, compressBase(v4), base.freqScale(),
                    base.extFactor(), base.betaFast(), base.betaSlow(), base.nCtxOrig());
            this.attnRope = compressed != null ? compressed
                    : rope(type, nRot, nope, base.freqBase(), 1.0f, 0.0f, 0.0f, 0.0f, 0);
            this.attnParams = params(attnRope);
            this.compressParams = compressed != null ? params(compressed) : null;

            matrix(shapes, WQ_A, nEmbd, qLora);
            vector(shapes, Q_A_NORM, qLora);
            matrix(shapes, WQ_B, qLora, (long) nHead * headDim);
            matrix(shapes, WKV, nEmbd, headDim);
            vector(shapes, KV_NORM, headDim);
            matrix(shapes, WO_A, oGroupDim, (long) oLora * nGroups);
            matrix(shapes, WO_B, (long) nGroups * oLora, nEmbd);
            if (sinks) {
                vector(shapes, SINKS, nHead);
            }
            if (ratio != 0) {
                matrix(shapes, COMP_WKV, nEmbd, stateWidth);
                matrix(shapes, COMP_WGATE, nEmbd, stateWidth);
                matrix(shapes, COMP_APE, stateWidth, ratio);
                vector(shapes, COMP_NORM, headDim);
            }
            if (mode == CompMode.SELECTED) {
                DsaSpec dsa = v4.indexer();
                this.idxHeads = dsa.indexerHeads();
                this.idxDim = dsa.indexerHeadDim();
                this.idxStateWidth = Dsv4Kernels.stateWidth(Dsv4Spec.RATIO_CSA, idxDim);
                this.topK = dsa.topK();
                this.idxWeightScale = dsa.weightScale();
                Dsv4Kernels.checkRowWidth(idxDim);
                if (!dsa.full() || dsa.kpool() != 0) {
                    throw new IllegalStateException(where + ": every DeepSeek-V4 CSA layer scores its own unpooled"
                            + " indexer rows (deepseek4.cpp:590-660), the spec says full=" + dsa.full() + ", kpool="
                            + dsa.kpool());
                }
                if (dsa.indexerRopeDim() != nRot || dsa.ropeType() != type || !dsa.ropeTail() || nRot > idxDim) {
                    throw new IllegalStateException(where + ": the DeepSeek-V4 indexer rotates the last n_rot = " + nRot
                            + " of its " + idxDim + " head dims with the model rope type " + type
                            + " (deepseek4.cpp:600-616), its spec says " + dsa.indexerRopeDim() + " dims of "
                            + dsa.ropeType() + (dsa.ropeTail() ? " at the tail" : " at the head"));
                }
                this.indexerRope = rope(type, nRot, idxDim - nRot, compressed.freqBase(), compressed.freqScale(),
                        compressed.extFactor(), compressed.betaFast(), compressed.betaSlow(), compressed.nCtxOrig());
                matrix(shapes, IDX_Q_B, qLora, (long) idxHeads * idxDim);
                matrix(shapes, IDX_PROJ, nEmbd, idxHeads);
                matrix(shapes, IDX_COMP_WKV, nEmbd, idxStateWidth);
                matrix(shapes, IDX_COMP_WGATE, nEmbd, idxStateWidth);
                matrix(shapes, IDX_COMP_APE, idxStateWidth, Dsv4Spec.RATIO_CSA);
                vector(shapes, IDX_COMP_NORM, idxDim);
            } else {
                this.idxHeads = 0;
                this.idxDim = 0;
                this.idxStateWidth = 0;
                this.topK = 0;
                this.idxWeightScale = Float.NaN;
                this.indexerRope = null;
            }
        }

        /**
         * The layout of a DeepSeek-V4 layer.
         *
         * @throws IllegalArgumentException      when the layer is not a DSV4 layer
         * @throws UnsupportedOperationException for a variant the family does not run
         * @throws IllegalStateException         when a tensor's shape contradicts the spec or a required tensor is
         *                                       absent
         */
        static Dsv4Layout of(LayerSpec layer, ModelSpec spec, ShapeSource shapes) {
            Objects.requireNonNull(spec, "spec");
            Objects.requireNonNull(shapes, "shapes");
            return new Dsv4Layout(layer, spec, shapes);
        }

        /** The kind of the layer's compressed attention cache, or null for a window-only layer. */
        CompressedKvPlan.Kind compressedKind() {
            switch (mode) {
                case SELECTED:
                    return CompressedKvPlan.Kind.CSA;
                case DENSE:
                    return CompressedKvPlan.Kind.HCA;
                default:
                    return null;
            }
        }

        /**
         * The lightning-indexer configuration of a CSA layer (deepseek4.cpp:590-660): the indexer heads, top-k and
         * {@code 1 / sqrt(idx_dim * idx_heads)} weight scale of its {@link DsaSpec}, the compress-base RoPE on the
         * last {@code n_rot} head dims, the indexer cache's Hadamard block, and full causal visibility, which the
         * per-row positions {@code kv_pos[r] = (r + 1) * ratio - 1} turn into the {@code n_visible} mask of the
         * compressed rows. The key-norm eps is unused (DeepSeek-V4 keys come from the indexer compressor).
         *
         * @throws IllegalStateException on a non-CSA layer
         */
        IndexerConfig indexerConfig(int lidRotBlock) {
            if (mode != CompMode.SELECTED) {
                throw new IllegalStateException(where + ": only CSA layers (ratio " + Dsv4Spec.RATIO_CSA
                        + ") have a lightning indexer");
            }
            return IndexerConfig.builder()
                    .heads(idxHeads, idxDim)
                    .topK(topK)
                    .weightScale(idxWeightScale)
                    .rope(indexerRope, idxDim - nRot)
                    .rotationBlock(lidRotBlock)
                    .keyNormEps(rmsEps)
                    .mask(KvWindow.FULL_CAUSAL)
                    .build();
        }

        /**
         * Position held by compressed row {@code row}: the last position of its block, {@code (row + 1) * ratio - 1}
         * (llama-kv-cache-dsv4.cpp:523-530), so a query at {@code p} sees exactly the rows {@code c < (p + 1) / ratio}.
         */
        static int compressedRowPosition(int row, int ratio) {
            return (row + 1) * ratio - 1;
        }

        /**
         * Hadamard block of the indexer cache, whatever its element type: llama.cpp always rotates the DeepSeek indexer
         * keys (llama-kv-cache.cpp:330-335), with {@code n_rot_k} starting at 64 and doubling while it divides the
         * head size (llama-kv-cache.cpp:343-350).
         */
        static int lidRotationBlock(int idxDim) {
            return RotationPlan.of(KvDType.F16, KvDType.F16, idxDim, 0, idxDim, 0, true, true).nRotK();
        }

        /**
         * Checks the layer's raw window cache: a single K-only row of {@code headDim} values per position with V
         * aliasing K, and the causal STANDARD window of {@code n_swa = window} that {@code LLM_DSV4_CompressedAttn}
         * applies (llama-hparams.h is_masked_swa, deepseek4.cpp:831-860).
         *
         * @throws IllegalStateException otherwise
         */
        void checkRawLayout(KvLayout kvl) {
            KvWindow w = kvl.window();
            if (!kvl.vAliasesK() || kvl.widthK() != headDim || w.type() != SwaType.STANDARD || !w.causal()
                    || w.nSwa() != window) {
                throw new IllegalStateException(where + ": the DeepSeek-V4 raw cache must hold one " + headDim
                        + "-value K row per position (V aliasing K) under a causal STANDARD window of " + window
                        + ", got " + kvl);
            }
        }

        /**
         * Checks a compressed cache plan against the layer: its kind's ratio, row width, plane shape and element type.
         * A CSA / HCA cache has {@code rawType}, so the raw window's Hadamard block also matches the compressed cache's
         * (both follow the same RotationPlan rule for the same type and head size, llama-kv-cache.cpp:323-350). The
         * indexer cache has the context's K type too (llama-kv-cache-dsv4.cpp:1308-1312), planned for its own row
         * width, so any type that stores whole rows of {@code idx_dim} values is accepted; its rotation is always the
         * forced one ({@link #lidRotationBlock}).
         *
         * @throws IllegalStateException otherwise
         */
        void checkCompressedPlan(CompressedKvPlan p, KvDType rawType) {
            boolean indexer = p.kind() == CompressedKvPlan.Kind.INDEXER;
            int r = indexer ? Dsv4Spec.RATIO_CSA : ratio;
            int width = indexer ? idxDim : headDim;
            boolean typeOk = indexer ? p.dtype().supportsWidth(idxDim) : p.dtype() == rawType;
            if (p.ratio() != r || p.width() != width || !typeOk
                    || p.stateRows() != Dsv4Kernels.stateRows(r) || p.stateWidth() != Dsv4Kernels.stateWidth(r, width)) {
                String type = indexer ? "values in whole blocks of their type" : rawType + " values";
                throw new IllegalStateException(where + ": the " + p.kind() + " cache must have ratio " + r + ", rows of "
                        + width + " " + type + " and " + Dsv4Kernels.stateRows(r) + " x "
                        + Dsv4Kernels.stateWidth(r, width) + " compressor planes, got " + p);
            }
        }

        /**
         * Checks that the step {@code [pos0, pos0 + n)} completes no compressed row beyond a cache of {@code rows}.
         *
         * @throws IllegalStateException naming the cache when the context exceeds it
         */
        void checkCapacity(int pos0, int n, int r, int rows, String what) {
            long needed = ((long) pos0 + n) / r;
            if (needed > rows) {
                throw new IllegalStateException(where + ": positions up to " + (pos0 + n - 1) + " need " + needed
                        + " compressed rows of ratio " + r + ", the " + what + " cache holds " + rows);
            }
        }

        private float compressBase(Dsv4Spec v4) {
            if (!(v4.compressRopeBase() > 0f)) {
                throw new IllegalStateException(where + ": DeepSeek-V4 compressed layers need compress_rope_base > 0,"
                        + " got " + v4.compressRopeBase());
            }
            return v4.compressRopeBase();
        }

        /**
         * A DeepSeek-V4 rotation of the last {@code nDims} dims at {@code nOffs}: {@code attn_factor} from
         * {@link Dsv4Kernels#attnFactor}, YaRN correction dims from {@link RopeMath#corrDims}
         * ({@code ggml_rope_yarn_corr_dims}); without an original context ({@code n_ctx_orig = 0}, window-only layers
         * with {@code ext_factor = 0}) the correction dims are never read and are 0.
         */
        static RopeSpec rope(RopeType type, int nDims, int nOffs, float freqBase, float freqScale, float extFactor,
                             float betaFast, float betaSlow, int nCtxOrig) {
            float[] corr = nCtxOrig > 0 ? RopeMath.corrDims(nDims, nCtxOrig, freqBase, betaFast, betaSlow)
                    : new float[]{0f, 0f};
            return RopeSpec.builder(type)
                    .nDims(nDims)
                    .nOffs(nOffs)
                    .freqBase(freqBase)
                    .freqScale(freqScale)
                    .yarn(extFactor, Dsv4Kernels.attnFactor(freqScale, extFactor), betaFast, betaSlow)
                    .corrDims(corr[0], corr[1])
                    .nCtxOrig(nCtxOrig)
                    .build();
        }

        /** The kernel scalars of a rotation. */
        static RopeParams params(RopeSpec r) {
            return RopeParams.of(r.nDims(), r.type(), r.freqBase(), r.freqScale(), r.extFactor(), r.attnFactor(),
                    r.corrLo(), r.corrHi());
        }

        private void matrix(ShapeSource shapes, String suffix, long cols, long rows) {
            long[] s = required(shapes, suffix);
            if (s[0] != cols || s[1] != rows) {
                throw new IllegalStateException(where + ": " + suffix + " has shape [" + s[0] + ", " + s[1]
                        + "], expected [" + cols + ", " + rows + "]");
            }
        }

        private void vector(ShapeSource shapes, String suffix, long n) {
            long[] s = required(shapes, suffix);
            if (s[0] * s[1] != n) {
                throw new IllegalStateException(where + ": " + suffix + " holds " + s[0] * s[1] + " values, expected "
                        + n);
            }
        }

        private long[] required(ShapeSource shapes, String suffix) {
            long[] s = shapes.shape(suffix);
            if (s == null) {
                throw new IllegalStateException(where + ": required tensor " + suffix + " is absent");
            }
            return s;
        }

        @Override
        public String toString() {
            return "Dsv4Layout{" + where + ", heads=" + nHead + "x" + headDim + ", rot=" + nRot + ", qLora=" + qLora
                    + ", groups=" + nGroups + "x" + oLora + ", ratio=" + ratio + ", window=" + window + ", sinks="
                    + sinks + (mode == CompMode.SELECTED ? ", indexer=" + idxHeads + "x" + idxDim + " top" + topK : "")
                    + '}';
        }
    }
}

package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.ops.mla.MlaBlocks;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.DsaSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Block factories of the DeepSeek sparse-attention family: the DSA sparse-MLA mixer ({@link MixerKind#DSA_MLA}) on
 * the GPU ({@link GpuDsaMlaMixer}) and the CPU ({@link CpuDsaMlaMixer}) for deepseek32, glm-dsa, dots3note and hy_v4.
 * The family has no FFN blocks and no layer extras.
 *
 * <p>{@link DsaLayout} is the shape contract both mixers share, validated against the tensors a layer holds;
 * {@link IndexerConfig} is the lightning-indexer configuration shared by {@link GpuDsaIndexer} and
 * {@link CpuDsaIndexerStage}. DSA layers with a pooled (QSA) indexer, {@code DsaSpec.kpool() > 0}, are not served
 * here: their factory rejects them.</p>
 */
public final class SparseBlocks {

    private SparseBlocks() {
    }

    /** The GPU DSA sparse-MLA mixer factory. */
    public static List<GpuMixerFactory> gpuMixers() {
        return Collections.<GpuMixerFactory>singletonList(GpuDsaFactory.INSTANCE);
    }

    /** The CPU DSA sparse-MLA mixer factory. */
    public static List<CpuMixerFactory> cpuMixers() {
        return Collections.<CpuMixerFactory>singletonList(CpuDsaFactory.INSTANCE);
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
     * Checks that a layer is a DSA sparse-MLA layer this family runs.
     *
     * @throws IllegalArgumentException      when the layer's mixer kind is not {@link MixerKind#DSA_MLA}
     * @throws UnsupportedOperationException for a pooled (QSA) indexer, a Q path without the {@code q_lora} rank, a
     *                                       non-absorbed or decompressed MLA cache, or MLA without RoPE
     */
    static void checkVariant(LayerSpec layer) {
        Objects.requireNonNull(layer, "layer");
        if (layer.mixerKind() != MixerKind.DSA_MLA) {
            throw new IllegalArgumentException("layer " + layer.stack() + "." + layer.il() + " has mixer kind "
                    + layer.mixerKind() + ", not " + MixerKind.DSA_MLA);
        }
        String where = "layer " + layer.stack() + "." + layer.il();
        DsaSpec dsa = layer.dsa();
        MlaSpec mla = layer.mla();
        if (dsa.kpool() > 0) {
            throw new UnsupportedOperationException(where + ": the pooled QSA indexer (kpool " + dsa.kpool()
                    + ") is not a DSA sparse-MLA layer");
        }
        if (mla.qLoraRank() <= 0) {
            throw new UnsupportedOperationException(where + ": DSA needs the q_lora path (wq_a, attn_q_a_norm, wq_b),"
                    + " whose normalized rank also feeds the indexer queries (deepseek32.cpp:207-218)");
        }
        if (!mla.absorbed() || !mla.latentCache()) {
            throw new UnsupportedOperationException(where + ": DSA attends over the absorbed latent cache only"
                    + " (llama-graph.cpp:3402-3478); neuralfree.mla.cache=decompressed is not supported for DSA");
        }
        if (!mla.ropeApplied() || layer.rope() == null) {
            throw new UnsupportedOperationException(where + ": DSA MLA layers apply RoPE to q_pe and k_pe");
        }
    }

    // ------------------------------------------------------------------ factories

    private enum GpuDsaFactory implements GpuMixerFactory {
        INSTANCE;

        @Override
        public MixerKind kind() {
            return MixerKind.DSA_MLA;
        }

        @Override
        public GpuMixer create(LayerSpec layer) {
            checkVariant(layer);
            return new GpuDsaMlaMixer();
        }
    }

    private enum CpuDsaFactory implements CpuMixerFactory {
        INSTANCE;

        @Override
        public MixerKind kind() {
            return MixerKind.DSA_MLA;
        }

        @Override
        public CpuMixer create(LayerSpec layer) {
            checkVariant(layer);
            return new CpuDsaMlaMixer();
        }
    }

    // ------------------------------------------------------------------ indexer configuration

    /**
     * Lightning-indexer configuration (llama.cpp src/models/deepseek32.cpp:207-316): head count and size, top-k width,
     * head-weight scale, the RoPE of the indexer queries and keys with its offset inside the head, the Hadamard
     * block of the q/k rotation (0 for none), the layer-norm eps of the key projection and the visibility mask of
     * the scores. Immutable.
     */
    public static final class IndexerConfig {

        private final int nHeads;
        private final int headDim;
        private final int topK;
        private final float weightScale;
        private final RopeSpec rope;
        private final int ropeOffset;
        private final int rotationBlock;
        private final float keyNormEps;
        private final KvWindow mask;

        private IndexerConfig(Builder b) {
            this.nHeads = b.nHeads;
            this.headDim = b.headDim;
            this.topK = b.topK;
            this.weightScale = b.weightScale;
            this.rope = b.rope;
            this.ropeOffset = b.ropeOffset;
            this.rotationBlock = b.rotationBlock;
            this.keyNormEps = b.keyNormEps;
            this.mask = b.mask;
        }

        public static Builder builder() {
            return new Builder();
        }

        /**
         * The configuration of a full DSA layer's indexer: shape, top-k and weight scale from its {@link DsaSpec},
         * the indexer RoPE derived from the layer's RoPE ({@link #indexerRope}), the q/k rotation of the indexer cache
         * plan and the visibility of the layer's KV window. Hadamard rotation follows the cache plan, which the memory
         * planner forces for every indexer with {@link DsaSpec#hadamard()} (llama-kv-cache.cpp:338-343).
         *
         * @param rotation   rotation plan of the layer's indexer cache
         * @param mask       KV window of the layer's cache group
         * @param keyNormEps eps of the indexer key layer norm ({@code f_norm_eps})
         * @throws IllegalStateException when the spec asks for the rotation and the cache plan does not rotate
         */
        public static IndexerConfig forLayer(LayerSpec layer, RotationPlan rotation, KvWindow mask, float keyNormEps) {
            Objects.requireNonNull(layer, "layer");
            Objects.requireNonNull(rotation, "rotation");
            DsaSpec dsa = Objects.requireNonNull(layer.dsa(), "layer DsaSpec");
            if (dsa.hadamard() && !rotation.rotateK()) {
                throw new IllegalStateException("layer " + layer.stack() + "." + layer.il() + ": the DSA indexer"
                        + " needs its Hadamard rotation but the indexer cache plan " + rotation + " does not rotate");
            }
            RopeSpec r = indexerRope(dsa, layer.rope());
            int offset = r != null && dsa.ropeTail() ? dsa.indexerHeadDim() - dsa.indexerRopeDim() : 0;
            return builder()
                    .heads(dsa.indexerHeads(), dsa.indexerHeadDim())
                    .topK(dsa.topK())
                    .weightScale(dsa.weightScale())
                    .rope(r, offset)
                    .rotationBlock(rotation.rotateK() ? rotation.nRotK() : 0)
                    .keyNormEps(keyNormEps)
                    .mask(mask)
                    .build();
        }

        /**
         * The RoPE of the indexer q/k: {@code dsa.indexerRopeDim()} dims of type {@code dsa.ropeType()} with the
         * layer's frequency base, scaling and YaRN parameters and no frequency factors (the graphs pass
         * {@code nullptr}: deepseek32.cpp:225-227 and 240-242, glm-dsa.cpp:261-263, dots3note.cpp:214-216 and
         * 227-229, hy-v4.cpp:361-374). The YaRN correction dims are recomputed when the dim count differs from the
         * layer's. Null when the indexer applies no RoPE.
         *
         * @throws IllegalStateException when the indexer needs RoPE and the layer has none
         */
        public static RopeSpec indexerRope(DsaSpec dsa, RopeSpec base) {
            Objects.requireNonNull(dsa, "dsa");
            if (dsa.ropeType() == RopeType.NONE || dsa.indexerRopeDim() == 0) {
                return null;
            }
            if (base == null) {
                throw new IllegalStateException("the DSA indexer RoPE needs the layer's RoPE parameters");
            }
            int nDims = dsa.indexerRopeDim();
            boolean sameDims = nDims == base.nDims();
            float lo = base.corrLo();
            float hi = base.corrHi();
            if (!sameDims && base.nCtxOrig() > 0) {
                float[] corr = RopeMath.corrDims(nDims, base.nCtxOrig(), base.freqBase(), base.betaFast(),
                        base.betaSlow());
                lo = corr[0];
                hi = corr[1];
            }
            RopeSpec.Builder b = RopeSpec.builder(dsa.ropeType())
                    .nDims(nDims)
                    .freqBase(base.freqBase())
                    .freqScale(base.freqScale())
                    .yarn(base.extFactor(), base.attnFactor(), base.betaFast(), base.betaSlow())
                    .corrDims(lo, hi)
                    .nCtxOrig(base.nCtxOrig())
                    .textPositions(base.textPositions());
            if (dsa.ropeType().usesSections()) {
                b.sections(base.sections());
            }
            if (sameDims) {
                b.thetaScale(base.thetaScale());
            }
            return b.build();
        }

        public int nHeads() { return nHeads; }
        public int headDim() { return headDim; }
        /** Query width of one token, {@code nHeads * headDim}. */
        public int queryWidth() { return nHeads * headDim; }
        public int topK() { return topK; }
        /** Factor of the head weights, {@code (headDim * nHeads)^-0.5} by default (deepseek32.cpp:269). */
        public float weightScale() { return weightScale; }
        /** RoPE of the indexer q/k; null for none. */
        public RopeSpec rope() { return rope; }
        /** First rotated dim within an indexer head. */
        public int ropeOffset() { return ropeOffset; }
        /** Hadamard block of the q/k rotation; 0 when the indexer is not rotated. */
        public int rotationBlock() { return rotationBlock; }
        /** The rotation transform, or null when {@link #rotationBlock()} is 0. */
        public HadamardSpec rotation() { return rotationBlock > 0 ? HadamardSpec.rotation(rotationBlock) : null; }
        public float keyNormEps() { return keyNormEps; }
        public KvWindow mask() { return mask; }

        @Override
        public String toString() {
            return "IndexerConfig{heads=" + nHeads + "x" + headDim + ", topK=" + topK + ", wScale=" + weightScale
                    + ", rope=" + rope + "@" + ropeOffset + ", rot=" + rotationBlock + ", mask=" + mask + '}';
        }

        public static final class Builder {
            private int nHeads;
            private int headDim;
            private int topK;
            private float weightScale = Float.NaN;
            private RopeSpec rope;
            private int ropeOffset;
            private int rotationBlock;
            private float keyNormEps;
            private KvWindow mask = KvWindow.FULL_CAUSAL;

            private Builder() {
            }

            public Builder heads(int n, int dim) { this.nHeads = n; this.headDim = dim; return this; }
            public Builder topK(int v) { this.topK = v; return this; }
            public Builder weightScale(float v) { this.weightScale = v; return this; }
            public Builder rope(RopeSpec spec, int offset) { this.rope = spec; this.ropeOffset = offset; return this; }
            public Builder rotationBlock(int v) { this.rotationBlock = v; return this; }
            public Builder keyNormEps(float v) { this.keyNormEps = v; return this; }
            public Builder mask(KvWindow v) { this.mask = Objects.requireNonNull(v, "mask"); return this; }

            /**
             * @throws IllegalArgumentException for a shape, top-k, mask or rotation the indexer kernels do not
             *                                  support, a non-positive weight scale, a negative eps, or RoPE dims
             *                                  outside the head
             */
            public IndexerConfig build() {
                SparseKernels.checkIndexer(nHeads, headDim);
                SparseKernels.checkTopK(topK);
                SparseKernels.checkIndexerMask(mask.type(), mask.nSwa());
                if (!(weightScale > 0f) || Float.isInfinite(weightScale)) {
                    throw new IllegalArgumentException("indexer weight scale must be positive and finite, got "
                            + weightScale);
                }
                if (!(keyNormEps >= 0f) || Float.isInfinite(keyNormEps)) {
                    throw new IllegalArgumentException("indexer key norm eps must be finite and >= 0, got "
                            + keyNormEps);
                }
                if (rope != null && (ropeOffset < 0
                        || (long) ropeOffset + RopeMath.rotatedWidth(rope.type(), rope.nDims()) > headDim)) {
                    throw new IllegalArgumentException("indexer RoPE of " + rope.nDims() + " dims at offset "
                            + ropeOffset + " exceeds the head size " + headDim);
                }
                if (rotationBlock < 0 || (rotationBlock > 0 && (Integer.bitCount(rotationBlock) != 1
                        || headDim % rotationBlock != 0 || rotationBlock > HadamardSpec.MAX_KERNEL_BLOCK))) {
                    throw new IllegalArgumentException("indexer rotation block " + rotationBlock + " must be 0 or a"
                            + " power of two dividing the head size " + headDim + " and at most "
                            + HadamardSpec.MAX_KERNEL_BLOCK);
                }
                return new IndexerConfig(this);
            }
        }
    }

    // ------------------------------------------------------------------ layer layout

    /** Shape lookup of the tensors of one layer: {@code {cols, rows}} of a tensor, or null when it is absent. */
    interface ShapeSource {
        long[] shape(String suffix);
    }

    /**
     * Shape contract of one DSA sparse-MLA layer, shared by the GPU and CPU mixers and validated against the
     * tensors the layer holds (llama.cpp src/models/deepseek32.cpp:200-396, glm-dsa.cpp:244-436, dots3note.cpp:179-402,
     * hy-v4.cpp:342-512). DSA runs on trunk layers only: llama.cpp builds the NextN/MTP blocks as dense MLA without
     * the indexer (llama-model.cpp:2830-2836, glm-dsa.cpp:519-526). The {@code attn_k_norm} RMS norm of {@code k_pe}
     * before RoPE (dots3note.cpp:331-332) follows the layer's attention facet through {@link MlaBlocks#kPeNorm}, the
     * source the plain MLA mixers share. The other optional stages are taken from the tensors exactly as the
     * llama.cpp loaders create them: the head-wise sigmoid gate of
     * {@code n_head} outputs broadcast over each head (dots3note.cpp:386-397) or the element-wise one of
     * {@code n_head * head_v} outputs (hy-v4.cpp:486-488), {@code attn_sinks} (hy-v4.cpp:483), the
     * {@code attn_output.scale} of {@code build_lora_mm} (llama-graph.cpp:1643-1645) and the indexer key-norm weight
     * and bias, which glm-dsa loads as optional (glm-dsa.cpp:134-135; {@code build_norm} skips an absent operand).
     *
     * <p>A shared layer reads the top-k its source layer left in the shared selection buffer, so the source must be
     * an earlier full DSA layer with the same top-k width and no full DSA layer may lie between them (it would
     * overwrite the selection); this is llama.cpp's {@code prev_top_k} of the nearest full layer
     * (glm-dsa.cpp:254, 350-355; hy-v4.cpp:439-443). Immutable.</p>
     */
    static final class DsaLayout {

        /** Output gate form. */
        enum Gate { NONE, PER_HEAD, ELEMENTWISE }

        static final String WQ_A = TensorNames.weight(TensorNames.ATTN_Q_A);
        static final String Q_A_NORM = TensorNames.weight(TensorNames.ATTN_Q_A_NORM);
        static final String WQ_B = TensorNames.weight(TensorNames.ATTN_Q_B);
        static final String WKV_A = TensorNames.weight(TensorNames.ATTN_KV_A_MQA);
        static final String KV_A_NORM = TensorNames.weight(TensorNames.ATTN_KV_A_NORM);
        static final String K_PE_NORM = TensorNames.weight(TensorNames.ATTN_K_NORM);
        static final String WK_B = TensorNames.weight(TensorNames.ATTN_K_B);
        static final String WV_B = TensorNames.weight(TensorNames.ATTN_V_B);
        static final String WO = TensorNames.weight(TensorNames.ATTN_OUT);
        static final String WO_SCALE = TensorNames.scale(TensorNames.ATTN_OUT);
        static final String GATE = TensorNames.weight(TensorNames.ATTN_GATE);
        static final String SINKS = TensorNames.weight(TensorNames.ATTN_SINKS);
        static final String IDX_Q_B = TensorNames.weight(TensorNames.INDEXER_ATTN_Q_B);
        static final String IDX_PROJ = TensorNames.weight(TensorNames.INDEXER_PROJ);
        static final String IDX_K = TensorNames.weight(TensorNames.INDEXER_ATTN_K);
        static final String IDX_K_NORM = TensorNames.weight(TensorNames.INDEXER_K_NORM);
        static final String IDX_K_NORM_B = TensorNames.bias(TensorNames.INDEXER_K_NORM);

        final String where;
        final int nEmbd;
        final int nHead;
        final int qLora;
        final int kvLora;
        final int nope;
        final int rope;
        final int headK;
        final int headV;
        /** Latent row and absorbed query head width, {@code kvLora + rope}. */
        final int dK;
        final float kqScale;
        final float softcap;
        /** {@code f_norm_rms_eps} of {@code attn_q_a_norm}, {@code attn_kv_a_norm} and {@code attn_k_norm}. */
        final float rmsEps;
        /** {@code f_norm_eps} of the indexer key layer norm. */
        final float normEps;
        final RopeSpec ropeSpec;
        final boolean kPeNorm;
        final Gate gate;
        final boolean sinks;
        final boolean woScale;
        final boolean full;
        final int sourceLayer;
        final int topK;
        final int idxHeads;
        final int idxDim;
        final boolean idxKeyNorm;
        final boolean idxKeyBias;

        private DsaLayout(LayerSpec layer, ModelSpec spec, ShapeSource shapes) {
            checkVariant(layer);
            MlaSpec mla = layer.mla();
            DsaSpec dsa = layer.dsa();
            this.where = spec.arch().ggufName() + " layer " + layer.stack() + "." + layer.il();
            this.nEmbd = spec.hparams().nEmbd();
            this.nHead = mla.nHead();
            this.qLora = mla.qLoraRank();
            this.kvLora = mla.kvLoraRank();
            this.nope = mla.headKNope();
            this.rope = mla.headKRope();
            this.headK = mla.headK();
            this.headV = mla.headV();
            this.dK = kvLora + rope;
            this.kqScale = mla.kqScale();
            this.softcap = layer.attention() != null ? layer.attention().softcap() : 0f;
            this.rmsEps = spec.hparams().normRmsEps();
            this.normEps = spec.hparams().normEps();
            this.ropeSpec = layer.rope();
            this.full = dsa.full();
            this.sourceLayer = dsa.sourceLayer();
            this.topK = dsa.topK();
            this.idxHeads = dsa.indexerHeads();
            this.idxDim = dsa.indexerHeadDim();
            if (layer.il() >= spec.nLayer()) {
                throw new IllegalStateException(where + " is a NextN block; llama.cpp runs NextN/MTP blocks as dense"
                        + " MLA without the DSA indexer (llama-model.cpp:2830-2836, glm-dsa.cpp:519-526)");
            }
            if (RopeMath.rotatedWidth(ropeSpec.type(), ropeSpec.nDims()) != rope) {
                throw new IllegalStateException(where + ": the layer RoPE rotates " + ropeSpec.nDims()
                        + " dims of type " + ropeSpec.type() + ", the MLA rope part q_pe / k_pe has " + rope
                        + " (both are hparams.n_rot(), deepseek32.cpp:166, 350-355)");
            }

            matrix(shapes, WQ_A, nEmbd, qLora);
            vector(shapes, Q_A_NORM, qLora);
            matrix(shapes, WQ_B, qLora, (long) nHead * headK);
            matrix(shapes, WKV_A, nEmbd, dK);
            vector(shapes, KV_A_NORM, kvLora);
            matrix(shapes, WK_B, nope, (long) kvLora * nHead);
            matrix(shapes, WV_B, kvLora, (long) headV * nHead);
            matrix(shapes, WO, (long) nHead * headV, nEmbd);

            this.kPeNorm = MlaBlocks.kPeNorm(layer);
            if (kPeNorm) {
                vector(shapes, K_PE_NORM, rope);
            }
            long[] g = shapes.shape(GATE);
            if (g == null) {
                this.gate = Gate.NONE;
            } else if (g[0] == nEmbd && g[1] == nHead) {
                this.gate = Gate.PER_HEAD;
            } else if (g[0] == nEmbd && g[1] == (long) nHead * headV) {
                this.gate = Gate.ELEMENTWISE;
            } else {
                throw new IllegalStateException(where + ": " + GATE + " has shape [" + g[0] + ", " + g[1]
                        + "], expected [" + nEmbd + ", " + nHead + "] (per head) or [" + nEmbd + ", "
                        + (long) nHead * headV + "] (element-wise)");
            }
            this.sinks = shapes.shape(SINKS) != null;
            if (sinks) {
                vector(shapes, SINKS, nHead);
            }
            this.woScale = shapes.shape(WO_SCALE) != null;
            if (woScale) {
                vector(shapes, WO_SCALE, 1);
            }
            if (full) {
                matrix(shapes, IDX_Q_B, qLora, (long) idxHeads * idxDim);
                matrix(shapes, IDX_PROJ, nEmbd, idxHeads);
                matrix(shapes, IDX_K, nEmbd, idxDim);
                this.idxKeyNorm = shapes.shape(IDX_K_NORM) != null;
                if (idxKeyNorm) {
                    vector(shapes, IDX_K_NORM, idxDim);
                }
                this.idxKeyBias = shapes.shape(IDX_K_NORM_B) != null;
                if (idxKeyBias) {
                    vector(shapes, IDX_K_NORM_B, idxDim);
                }
            } else {
                this.idxKeyNorm = false;
                this.idxKeyBias = false;
                checkSharedSource(spec, layer);
            }
            SparseKernels.checkTopK(topK);
            SparseKernels.checkSparseMla(nHead, kvLora, rope, topK);
        }

        /**
         * Checks that the source of a shared layer is an earlier full DSA trunk layer with the same top-k width and
         * that no full DSA layer lies between them.
         */
        private void checkSharedSource(ModelSpec spec, LayerSpec layer) {
            if (!TensorNames.STACK_BLK.equals(layer.stack()) || sourceLayer < 0 || sourceLayer >= layer.il()
                    || layer.il() >= spec.nLayer()) {
                throw new IllegalStateException(where + ": a shared DSA layer must be a trunk layer reusing the top-k"
                        + " of an earlier full layer, got source layer " + sourceLayer);
            }
            LayerSpec src = spec.layer(sourceLayer);
            if (!isFullDsa(src) || src.dsa().topK() != topK) {
                throw new IllegalStateException(where + ": source layer " + sourceLayer + " is not a full DSA layer"
                        + " selecting " + topK + " cells");
            }
            for (int il = sourceLayer + 1; il < layer.il(); il++) {
                if (isFullDsa(spec.layer(il))) {
                    throw new IllegalStateException(where + ": full DSA layer " + il + " lies between the shared"
                            + " layer and its source layer " + sourceLayer + " and would overwrite the selection");
                }
            }
        }

        private static boolean isFullDsa(LayerSpec l) {
            return l.mixerKind() == MixerKind.DSA_MLA && l.dsa() != null && l.dsa().full();
        }

        /**
         * The layout of a DSA sparse-MLA layer.
         *
         * @throws IllegalStateException when a tensor's shape contradicts the spec
         * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException or IllegalStateException when a required
         *         tensor is absent (the shape source decides)
         */
        static DsaLayout of(LayerSpec layer, ModelSpec spec, ShapeSource shapes) {
            Objects.requireNonNull(spec, "spec");
            Objects.requireNonNull(shapes, "shapes");
            return new DsaLayout(layer, spec, shapes);
        }

        /** Width of the gate projection output per token (0 without a gate). */
        int gateWidth() {
            switch (gate) {
                case PER_HEAD:
                    return nHead;
                case ELEMENTWISE:
                    return nHead * headV;
                default:
                    return 0;
            }
        }

        /** Floats per token of the q buffer, which later holds the decompressed attention ({@code nHead * headV}). */
        int qBufferWidth() {
            return nHead * Math.max(headK, headV);
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
            return "DsaLayout{" + where + ", heads=" + nHead + ", qLora=" + qLora + ", kvLora=" + kvLora + ", nope="
                    + nope + ", rope=" + rope + ", headV=" + headV + ", gate=" + gate + ", sinks=" + sinks
                    + ", kPeNorm=" + kPeNorm + ", full=" + full + ", source=" + sourceLayer + '}';
        }
    }
}

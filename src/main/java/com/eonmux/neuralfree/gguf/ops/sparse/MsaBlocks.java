package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.IndexerPlan;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DsaSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Block factories of the MiniMax / QSA sparse-attention family, exposed to the block catalogs, and the helpers its
 * mixers share.
 *
 * <p><b>Routing.</b> The family serves {@link MixerKind#MSA} with one GPU and one CPU factory, which pick the mixer
 * from the layer's sub-specs ({@link #variant}):
 * <ul>
 *   <li>{@link LayerSpec#msa()} with {@code sparse()}: MiniMax-M3 block-sparse attention ({@link GpuMsaMixer},
 *       {@link CpuMsaMixer}; models/minimax-m3.cpp:339-535);</li>
 *   <li>{@link LayerSpec#msa()} without {@code sparse()}: a dense lead layer, which llama.cpp runs through the
 *       standard {@code build_attn} (models/minimax-m3.cpp:334-338). It is served by the {@link MixerKind#ATTENTION}
 *       factory passed to {@link #gpuMixers(GpuMixerFactory)} / {@link #cpuMixers(CpuMixerFactory)}, bound to the
 *       attention-only view of the layer ({@link #denseAttentionLayer});</li>
 *   <li>{@link LayerSpec#dsa()} with a k-pool ({@code dsa().kpool() > 0}): the QSA pooled-indexer mixer of the
 *       HYBRID_IDX architectures ({@link GpuQsaMixer}, {@link CpuQsaMixer}): nope MLA over the pooled selection
 *       (glm5-next, models/glm5-next.cpp:892-1108) or gated GQA over it (qwen4exp, models/qwen4exp.cpp:741-1026).</li>
 * </ul>
 * The family has no FFN blocks and no layer extras.
 *
 * <p><b>Shared selection.</b> A glm5-next layer whose indexer is not full reuses the pool selection of the nearest
 * full layer before it in the same step (models/glm5-next.cpp:1077-1083). Every QSA layer requests the selection
 * temporary {@link #QSA_SELECTION} with the same model-wide size ({@link #selectionInts}), so the scratch serves
 * all of them one buffer: the full layer writes it and the following shared layers read it. No other block uses
 * that name.
 */
public final class MsaBlocks {

    /** Scratch name of the per-token pool selection shared by the QSA layers of one step. */
    static final String QSA_SELECTION = "qsa.selection";

    /** {@code LLM_Batch_BiasAdd(i dim, i seq_len, R bias, RW data)}: {@code data[s*dim + d] += bias[d]}. */
    static final String BATCH_BIAS_ADD = GpuOps.BATCH_BIAS_ADD;
    private static final int ARG_BIAS_SEQ_LEN = 1;
    /** Argument index of {@code n_tokens} in {@code LLM_KV_Store_*} ({@code (i n_tokens, i row_elems, ...)}). */
    private static final int ARG_STORE_N_TOKENS = 0;

    private MsaBlocks() {
    }

    // ================================================================== factory exposure

    /**
     * The GPU mixer factory of {@link MixerKind#MSA} without a dense-attention delegate: dense MiniMax-M3 lead layers
     * then fail at {@code create} and need {@link #gpuMixers(GpuMixerFactory)}.
     */
    public static List<GpuMixerFactory> gpuMixers() {
        return gpuMixers(null);
    }

    /**
     * The GPU mixer factory of {@link MixerKind#MSA}, with dense MiniMax-M3 lead layers served by
     * {@code denseAttention}.
     *
     * @param denseAttention the {@link MixerKind#ATTENTION} factory, or null when no attention family is registered
     * @throws IllegalArgumentException when {@code denseAttention} serves another kind
     */
    public static List<GpuMixerFactory> gpuMixers(GpuMixerFactory denseAttention) {
        if (denseAttention != null && denseAttention.kind() != MixerKind.ATTENTION) {
            throw new IllegalArgumentException("the dense-attention factory must serve ATTENTION, not "
                    + denseAttention.kind());
        }
        return Collections.singletonList(new GpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return MixerKind.MSA;
            }

            @Override
            public GpuMixer create(LayerSpec layer) {
                switch (variant(layer)) {
                    case DENSE:
                        if (denseAttention == null) {
                            throw noAttentionFactory(layer, "GPU", "MsaBlocks.gpuMixers(GpuMixerFactory)");
                        }
                        return new GpuDense(denseAttention.create(denseAttentionLayer(layer)));
                    case MSA:
                        return new GpuMsaMixer();
                    default:
                        return new GpuQsaMixer();
                }
            }

            @Override
            public String toString() {
                return "MsaBlocks.gpu(MSA)";
            }
        });
    }

    /**
     * The CPU mixer factory of {@link MixerKind#MSA} without a dense-attention delegate (see {@link #gpuMixers()}).
     */
    public static List<CpuMixerFactory> cpuMixers() {
        return cpuMixers(null);
    }

    /**
     * The CPU mixer factory of {@link MixerKind#MSA}, with dense MiniMax-M3 lead layers served by
     * {@code denseAttention}.
     *
     * @param denseAttention the {@link MixerKind#ATTENTION} factory, or null when no attention family is registered
     * @throws IllegalArgumentException when {@code denseAttention} serves another kind
     */
    public static List<CpuMixerFactory> cpuMixers(CpuMixerFactory denseAttention) {
        if (denseAttention != null && denseAttention.kind() != MixerKind.ATTENTION) {
            throw new IllegalArgumentException("the dense-attention factory must serve ATTENTION, not "
                    + denseAttention.kind());
        }
        return Collections.singletonList(new CpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return MixerKind.MSA;
            }

            @Override
            public CpuMixer create(LayerSpec layer) {
                switch (variant(layer)) {
                    case DENSE:
                        if (denseAttention == null) {
                            throw noAttentionFactory(layer, "CPU", "MsaBlocks.cpuMixers(CpuMixerFactory)");
                        }
                        return new CpuDense(denseAttention.create(denseAttentionLayer(layer)));
                    case MSA:
                        return new CpuMsaMixer();
                    default:
                        return new CpuQsaMixer();
                }
            }

            @Override
            public String toString() {
                return "MsaBlocks.cpu(MSA)";
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

    private static IllegalStateException noAttentionFactory(LayerSpec layer, String side, String entry) {
        return new IllegalStateException("dense MSA lead layer " + describe(layer) + " runs standard attention, but no "
                + side + " ATTENTION factory was passed to " + entry);
    }

    // ================================================================== dense lead layers

    /**
     * The attention-only view of a dense MiniMax-M3 lead layer: the same layer index, weights, norms, residual,
     * attention, RoPE and FFN, with {@link MixerKind#ATTENTION} as its mixer and no layer extras (the executor binds
     * those around the whole layer).
     */
    static LayerSpec denseAttentionLayer(LayerSpec layer) {
        return LayerSpec.builder(layer.il(), MixerKind.ATTENTION, layer.ffnKind())
                .weightLayer(layer.weightLayer())
                .stack(layer.stack())
                .norm(layer.norm())
                .residual(layer.residual())
                .attention(layer.attention())
                .rope(layer.rope())
                .ffn(layer.ffn())
                .moe(layer.moe())
                .build();
    }

    /** A dense lead layer: the standard attention mixer bound to the attention-only view. */
    private static final class GpuDense implements GpuMixer {
        private final GpuMixer attention;

        GpuDense(GpuMixer attention) {
            this.attention = Objects.requireNonNull(attention, "attention");
        }

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            checkVariant(layer, Variant.DENSE);
            attention.bind(ctx, denseAttentionLayer(layer));
        }

        @Override
        public void decode(int pos, ScratchSlot in, ScratchSlot out) {
            attention.decode(pos, in, out);
        }

        @Override
        public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
            attention.prefill(pos0, nTokens, in, out);
        }

        @Override
        public void shiftContext(int nKeep, int nDiscard) {
            attention.shiftContext(nKeep, nDiscard);
        }

        @Override
        public String shiftRefusal() {
            return attention.shiftRefusal();
        }

        @Override
        public void close() {
            attention.close();
        }

        @Override
        public String toString() {
            return "MsaBlocks.dense{" + attention + '}';
        }
    }

    /** CPU twin of {@link GpuDense}. */
    private static final class CpuDense implements CpuMixer {
        private final CpuMixer attention;

        CpuDense(CpuMixer attention) {
            this.attention = Objects.requireNonNull(attention, "attention");
        }

        @Override
        public void bind(CpuForwardContext ctx, LayerSpec layer) {
            checkVariant(layer, Variant.DENSE);
            attention.bind(ctx, denseAttentionLayer(layer));
        }

        @Override
        public void decode(int pos, float[] in, float[] out) {
            attention.decode(pos, in, out);
        }

        @Override
        public void prefill(int pos0, int n, float[] in, float[] out) {
            attention.prefill(pos0, n, in, out);
        }

        @Override
        public void shiftContext(int nKeep, int nDiscard) {
            attention.shiftContext(nKeep, nDiscard);
        }

        @Override
        public String shiftRefusal() {
            return attention.shiftRefusal();
        }

        @Override
        public void close() {
            attention.close();
        }

        @Override
        public String toString() {
            return "MsaBlocks.dense{" + attention + '}';
        }
    }

    // ================================================================== layer checks

    /** The mixers of the family. */
    enum Variant {
        /** Dense MiniMax-M3 lead layer (standard attention). */
        DENSE,
        /** MiniMax-M3 block-sparse attention. */
        MSA,
        /** QSA k-pool selection. */
        QSA
    }

    /**
     * The variant of an {@link MixerKind#MSA} layer.
     *
     * @throws IllegalArgumentException for another kind, or a layer with neither an MSA spec nor a k-pool indexer
     */
    static Variant variant(LayerSpec layer) {
        Objects.requireNonNull(layer, "layer");
        if (layer.mixerKind() != MixerKind.MSA) {
            throw new IllegalArgumentException("the MSA family cannot run layer " + describe(layer) + " ("
                    + layer.mixerKind() + ")");
        }
        if (layer.msa() != null) {
            return layer.msa().sparse() ? Variant.MSA : Variant.DENSE;
        }
        if (layer.dsa() != null && layer.dsa().kpool() > 0) {
            return Variant.QSA;
        }
        throw new IllegalArgumentException("MSA layer " + describe(layer) + " carries neither an MSA spec nor a k-pool"
                + " indexer spec");
    }

    /**
     * Checks that a mixer of {@code expected} may bind {@code layer}.
     *
     * @throws IllegalArgumentException otherwise
     */
    static void checkVariant(LayerSpec layer, Variant expected) {
        Variant v = variant(layer);
        if (v != expected) {
            throw new IllegalArgumentException("layer " + describe(layer) + " is a " + v + " layer, not " + expected);
        }
    }

    /** Layer name used in messages. */
    static String describe(LayerSpec layer) {
        return layer.stack() + "." + layer.il();
    }

    /**
     * Values per token of the mixer input and output: the residual stream width divided by the hyper-connection
     * count ({@code n_embd}; glm5-next and qwen4exp run their mixers on one {@code n_embd} row mixed out of the
     * {@code hcMult * n_embd} stream, as the layer executors pass it).
     *
     * @throws IllegalStateException when the stream width is not a multiple of the hyper-connection count
     */
    static int mixerWidth(LayerSpec layer) {
        ResidualSpec r = layer.residual();
        if (r.hcMult() < 1 || r.streamWidth() % r.hcMult() != 0) {
            throw new IllegalStateException("layer " + describe(layer) + ": stream width " + r.streamWidth()
                    + " is not a multiple of the hyper-connection count " + r.hcMult());
        }
        return r.streamWidth() / r.hcMult();
    }

    /**
     * Checks the attention features the sparse mixers implement: a cached causal layer that owns its KV, without
     * sinks, position bias, softcap or output scale, temperature, value norm or scale, value experts, KQV clamp,
     * sliding window or linear replacement, and with output gate {@code allowedGate}.
     *
     * @throws IllegalArgumentException naming the first unsupported feature
     */
    static void checkAttention(LayerSpec layer, AttentionSpec a, AttnGateKind allowedGate) {
        String what = null;
        if (!a.usesMemory() || !a.causal()) {
            what = "a non-causal or cache-less attention";
        } else if (a.sharesKv(layer.il())) {
            what = "a KV cache shared from layer " + a.kvOwnerLayer();
        } else if (a.sinks()) {
            what = "attention sinks";
        } else if (a.biasMode() != AttentionSpec.BiasMode.NONE || a.alibiMaxBias() != 0f) {
            what = "position bias " + a.biasMode();
        } else if (a.softcap() != 0f || a.outScale() != 0f) {
            what = "a logit softcap or output scale";
        } else if (a.temperature().enabled()) {
            what = "attention temperature";
        } else if (a.vNorm() || a.valueScale() != 0f) {
            what = "a value norm or scale";
        } else if (a.valueExperts() != null || a.linearReplace()) {
            what = "value experts or a linear replacement";
        } else if (a.clampKqv() > 0f) {
            what = "a KQV clamp";
        } else if (a.effectiveSwaType() != SwaType.NONE) {
            what = "a sliding window";
        } else if (a.gate() != allowedGate) {
            what = "output gate " + a.gate();
        }
        if (what != null) {
            throw new IllegalArgumentException("layer " + describe(layer) + ": the sparse-attention mixer does not "
                    + "implement " + what);
        }
    }

    /**
     * The layer's rotary spec, checked to fit heads of {@code headDim} values and, when {@code declaredDims > 0}, to
     * rotate that many dims (a sub-spec's own rope width must agree with the layer's {@code n_rot}).
     *
     * @throws IllegalArgumentException when the layer has no rotary spec or the widths disagree
     */
    static RopeSpec rope(LayerSpec layer, int headDim, int declaredDims, String what) {
        RopeSpec r = layer.rope();
        if (r == null) {
            throw new IllegalArgumentException("layer " + describe(layer) + ": " + what + " needs a rotary spec");
        }
        if (declaredDims > 0 && r.nDims() != declaredDims) {
            throw new IllegalArgumentException("layer " + describe(layer) + ": " + what + " rotates " + declaredDims
                    + " dims but the layer's rotary spec rotates " + r.nDims());
        }
        if (r.nOffs() + r.nDims() > headDim) {
            throw new IllegalArgumentException("layer " + describe(layer) + ": " + what + " rotary dims ["
                    + r.nOffs() + ", " + (r.nOffs() + r.nDims()) + ") exceed the head of " + headDim);
        }
        return r;
    }

    /**
     * Checks that a cache addresses positions without a ring, so the positions {@code [0, pos0 + T)} of a step never
     * exceed its rows and the score rows stay bounded by the cache.
     *
     * @throws IllegalStateException for a ring layout
     */
    static void checkNoRing(LayerSpec layer, KvLayout layout, String what) {
        if (layout.ring()) {
            throw new IllegalStateException("layer " + describe(layer) + ": the " + what + " is a ring of "
                    + layout.rows() + " rows; sparse selection needs every position cached");
        }
    }

    // ================================================================== QSA geometry

    /** Pools selected per token of a k-pool indexer: {@code floor(top_k / kpool)}. */
    static int selectionWidth(DsaSpec dsa) {
        return MsaKernels.selectedPools(dsa.topK(), dsa.kpool());
    }

    /**
     * Ints of the shared selection buffer: {@code ubatch} rows of the widest selection of any k-pool layer of the
     * model (at least one value), identical for every QSA layer.
     */
    static int selectionInts(ModelSpec spec, int ubatch) {
        int width = 1;
        List<LayerSpec> all = new ArrayList<>(spec.layers());
        all.addAll(spec.nextnLayers());
        for (LayerSpec l : all) {
            DsaSpec d = l.dsa();
            if (l.mixerKind() == MixerKind.MSA && d != null && d.kpool() > 0) {
                width = Math.max(width, selectionWidth(d));
            }
        }
        return Math.multiplyExact(ubatch, width);
    }

    /**
     * Most pools a step of at most {@code ubatch} tokens completes: {@code (pos0 + T) / kpool - pos0 / kpool} is at
     * most {@code (T - 1) / kpool + 1}, never more than {@code T}.
     */
    static int maxNewPools(int ubatch, int kpool) {
        return Math.min(ubatch, (ubatch - 1) / kpool + 1);
    }

    /**
     * Checks that a shared-indexer QSA layer can read the selection {@link #QSA_SELECTION} written by its source:
     * the source is an earlier full k-pool layer of the same pool size and selection width, and no other full
     * k-pool layer lies between them (it would overwrite the selection). A full layer passes unchecked.
     *
     * @throws IllegalStateException otherwise
     */
    static void checkQsaSource(ModelSpec spec, LayerSpec layer) {
        DsaSpec dsa = layer.dsa();
        if (dsa.full()) {
            return;
        }
        int source = dsa.sourceLayer();
        List<LayerSpec> all = new ArrayList<>(spec.layers());
        all.addAll(spec.nextnLayers());
        LayerSpec src = null;
        for (LayerSpec l : all) {
            if (l.il() == source) {
                src = l;
            } else if (l.il() > source && l.il() < layer.il() && isFullKpool(l)) {
                throw new IllegalStateException("layer " + describe(layer) + " reuses the pool selection of layer "
                        + source + ", but full k-pool layer " + describe(l) + " lies between them");
            }
        }
        if (src == null || source >= layer.il() || !isFullKpool(src) || src.dsa().kpool() != dsa.kpool()
                || selectionWidth(src.dsa()) != selectionWidth(dsa)) {
            throw new IllegalStateException("layer " + describe(layer) + " reuses the pool selection of layer "
                    + source + ", which is not an earlier full k-pool layer with " + dsa.kpool() + "-row pools and "
                    + selectionWidth(dsa) + " selected pools");
        }
    }

    private static boolean isFullKpool(LayerSpec l) {
        return l.mixerKind() == MixerKind.MSA && l.dsa() != null && l.dsa().kpool() > 0 && l.dsa().full();
    }

    /**
     * Checks the glm5-next latent attention the QSA mixers implement: absorbed nope-only MLA with a q LoRA over a
     * latent cache whose K row is the {@code kv_lora_rank} latent and serves as V (models/glm5-next.cpp:1056-1100). V
     * is the K row, so the cache must not rotate V separately.
     *
     * @throws IllegalArgumentException for another MLA form
     * @throws IllegalStateException    for a cache of another layout
     */
    static void checkQsaMla(LayerSpec layer, MlaSpec m, KvLayout kv, int rotV) {
        if (m.headKRope() != 0 || !m.absorbed() || !m.latentCache() || m.qLoraRank() <= 0) {
            throw new IllegalArgumentException("layer " + describe(layer) + ": QSA MLA is absorbed nope-only"
                    + " attention with a q LoRA over the latent cache (glm5-next.cpp:1059), the spec is " + m);
        }
        if (!kv.vAliasesK() || kv.widthK() != m.kvLoraRank() || rotV != 0) {
            throw new IllegalStateException("layer " + describe(layer) + ": QSA MLA needs a latent cache of "
                    + m.kvLoraRank() + "-value rows read as K and V, the layout is " + kv);
        }
    }

    /**
     * Checks a k-pool indexer cache: unrotated (llama.cpp never rotates the k-pool keys, glm5-next.cpp:904-946,
     * qwen4exp.cpp:762-812), {@code idxDim * kpoolRow} values wide, and holding a row for every position of the
     * layer's KV cache.
     *
     * @throws IllegalArgumentException when the spec asks for a Hadamard indexer
     * @throws IllegalStateException    for a cache of another layout
     */
    static void checkQsaIndexer(LayerSpec layer, DsaSpec dsa, IndexerPlan plan, KvLayout kv) {
        if (dsa.hadamard()) {
            throw new IllegalArgumentException("layer " + describe(layer) + ": the k-pool indexer is not rotated");
        }
        RotationPlan rot = plan.rotation();
        int width = dsa.indexerHeadDim() * dsa.kpoolRow();
        if (rot.rotateK() || rot.rotateV() || plan.width() != width || plan.rows() < kv.rows()) {
            throw new IllegalStateException("layer " + describe(layer) + ": the k-pool indexer cache needs "
                    + kv.rows() + " unrotated rows of " + width + " values, the plan is " + plan);
        }
    }

    // ================================================================== GPU tensors

    /**
     * Required matrix {@code <base>.weight} with {@code cols} inputs and {@code rows} outputs.
     *
     * @throws MissingTensorException when it is absent
     * @throws IllegalStateException  when its shape differs
     */
    static GpuWeight gpuMatrix(GpuWeights ws, LayerSpec layer, String base, long cols, long rows) {
        GpuWeight w = ws.layer(layer, TensorNames.weight(base));
        if (w.cols() != cols || w.rows() != rows) {
            throw new IllegalStateException("layer " + describe(layer) + ": " + w.name() + " is " + w.cols() + "x"
                    + w.rows() + ", the sparse-attention mixer needs " + cols + "x" + rows);
        }
        return w;
    }

    /** Optional matrix, as {@link #gpuMatrix}, or null when absent. */
    static GpuWeight gpuMatrixOrNull(GpuWeights ws, LayerSpec layer, String base, long cols, long rows) {
        return ws.layerOrNull(layer, TensorNames.weight(base)) == null ? null : gpuMatrix(ws, layer, base, cols, rows);
    }

    /**
     * A per-head 3-D weight ({@code ne = {cols, rows, nHead}}) read by the id-matvec kernels, which address it as
     * one binding without transforms.
     *
     * @throws IllegalStateException when it is sharded, rotated, scaled or of another shape
     */
    static GpuWeight gpuHeadStack(GpuWeights ws, LayerSpec layer, String base, long cols, long rows, int nHead) {
        GpuWeight w = gpuMatrix(ws, layer, base, cols, rows * nHead);
        if (!w.isSingleShard() || w.hadamard() != null || w.hasScale()) {
            throw new IllegalStateException("layer " + describe(layer) + ": " + w.name() + " must be one unscaled "
                    + "buffer without an input rotation for the per-head id-matvec");
        }
        return w;
    }

    /**
     * Device F32 values of a {@code floats}-element tensor: the tensor's own buffer when it is a single-buffer F32
     * tensor, otherwise a converted copy made once at bind and added to {@code owned}.
     *
     * @throws IllegalStateException when the tensor holds another element count
     */
    static GpuBuffer gpuF32(GpuForwardContext ctx, GpuWeight w, long floats, List<GpuBuffer> owned) {
        long n = w.rows() * w.cols();
        if (n != floats) {
            throw new IllegalStateException("tensor " + w.name() + " holds " + n + " values, " + floats + " expected");
        }
        if (w.type() == GGMLType.F32 && w.isSingleShard()) {
            return w.shard(0);
        }
        int nRows = Math.toIntExact(w.rows());
        int[] rows = new int[nRows];
        for (int r = 0; r < nRows; r++) {
            rows[r] = r;
        }
        GpuBuffer b = ctx.backend().allocateBuffer(Math.multiplyExact(floats, (long) Float.BYTES));
        owned.add(b);
        ctx.ops().embed(w, rows, b, 0);
        return b;
    }

    /**
     * A norm weight, norm bias or projection bias of {@code width} values as a single-buffer F32 {@link GpuWeight}
     * (the form {@link GpuOps#norm} and {@code LLM_Batch_BiasAdd} read), or null when it is absent and optional.
     *
     * @param name full layer suffix ({@code attn_q_norm.weight})
     * @throws MissingTensorException when a required tensor is absent
     * @throws IllegalStateException  when it does not hold exactly {@code width} values
     */
    static GpuWeight gpuVector(GpuForwardContext ctx, GpuWeights ws, LayerSpec layer, String name, int width,
                               boolean required, List<GpuBuffer> owned) {
        GpuWeight w = ws.layerOrNull(layer, name);
        if (w == null) {
            if (required) {
                throw new MissingTensorException(ws.archName(), layer.il(), layer.tensorName(name));
            }
            return null;
        }
        if (w.type() == GGMLType.F32 && w.isSingleShard() && w.cols() == width && w.rows() == 1) {
            return w;
        }
        return GpuWeight.of(w.name(), GGMLType.F32, new long[]{width}, gpuF32(ctx, w, width, owned));
    }

    /**
     * Device buffer of the layer's RoPE frequency factors, or null when the spec has none.
     *
     * @throws MissingTensorException when the spec names a tensor that is not resident
     */
    static GpuBuffer gpuFreqFactors(GpuForwardContext ctx, GpuWeights ws, RopeSpec rope, List<GpuBuffer> owned) {
        if (rope.freqFactors() == null) {
            return null;
        }
        GpuWeight w = ws.get(rope.freqFactors().name());
        return gpuF32(ctx, w, w.rows() * w.cols(), owned);
    }

    /**
     * Checks that a scratch temporary starts at element 0 of its buffer, as the dispatches prepared over it assume.
     *
     * @throws IllegalStateException otherwise
     */
    static ScratchSlot atZero(ScratchSlot s) {
        if (s.offset() != 0) {
            throw new IllegalStateException("scratch temporary " + s.name() + " starts at element " + s.offset()
                    + ", the sparse-attention dispatches need element 0");
        }
        return s;
    }

    // ================================================================== CPU tensors

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
                    + w.cols() + "x" + w.rows() + ", the sparse-attention mixer needs " + cols + "x" + rows);
        }
        return w;
    }

    /** Optional matrix, as {@link #cpuMatrix}, or null when absent. */
    static CpuTensorRef cpuMatrixOrNull(CpuWeights ws, LayerSpec layer, String base, long cols, long rows) {
        return ws.layerOrNull(layer, TensorNames.weight(base)) == null ? null : cpuMatrix(ws, layer, base, cols, rows);
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
                    + v.length + " values, the sparse-attention mixer needs " + width);
        }
        return v;
    }

    /** The F32 frequency factors of the layer's RoPE, or null when the spec has none. */
    static float[] cpuFreqFactors(CpuWeights ws, RopeSpec rope) {
        return rope.freqFactors() == null ? null : ws.floats(rope.freqFactors());
    }

    // ================================================================== CPU positions

    /**
     * Writes the stored position of every row a step over positions {@code [0, nPos)} can address
     * ({@code layout.rowOf(p)} for {@code p} in the last {@code min(rows, nPos)} positions) into
     * {@code kvPos[row]}: the CPU form of the group's {@code kv_pos} buffer read by {@link CpuMsa} and {@link CpuQsa}.
     */
    static void fillKvPos(CpuMemory memory, int group, KvLayout layout, int nPos, int[] kvPos) {
        int n = Math.min(layout.rows(), nPos);
        for (int p = nPos - n; p < nPos; p++) {
            int row = layout.rowOf(p);
            kvPos[row] = memory.positionOfRow(group, row);
        }
    }

    /** Writes {@code qPos[t] = pos0 + t} for {@code t < nTokens}. */
    static void fillQueryPositions(int pos0, int nTokens, int[] qPos) {
        for (int t = 0; t < nTokens; t++) {
            qPos[t] = pos0 + t;
        }
    }

    /**
     * In-place block-diagonal Hadamard rotation of {@code nRows} rows of {@code width} values from index 0, the CPU
     * form of {@link #rotate}; nothing when {@code block} is 0.
     */
    static void cpuRotate(int block, float[] x, int nRows, int width) {
        if (block > 0) {
            CpuHadamard.applyInPlace(HadamardSpec.rotation(block), x, 0, nRows, width);
        }
    }

    // ================================================================== GPU dispatches shared by the mixers

    /**
     * {@code LLM_RoPE_Ext} over {@code nHeads} heads of each row of {@code x} (element 0) at positions {@code pos},
     * prepared for one token; {@link #runRope} sets the token count and the section stride per step. Validated for
     * {@code ubatch} tokens.
     *
     * @param ff the layer's frequency factors, or null
     */
    static PreparedDispatch prepareRope(GpuForwardContext ctx, RopeSpec rope, GpuBuffer ff, int nHeads,
                                        int headStride, int rowStride, GpuBuffer pos, GpuBuffer x) {
        int ubatch = ctx.scratch().ubatch();
        RopeKernels.validateRopeExt(rope.type(), rope.nDims(), rope.nOffs(), ubatch, nHeads, headStride, rowStride,
                ubatch, 0);
        Object[] args = RopeKernels.ropeExtArgs(rope, 1, nHeads, headStride, rowStride, rope.nOffs(), 1, 0,
                ff != null, pos, ff != null ? ff : ctx.placeholders().readOnlyFloats(), x);
        int local = ctx.ops().local();
        long items = RopeKernels.ropeExtWorkItems(rope.type(), 1, nHeads, rope.nDims());
        return ctx.ops().prepare(RopeKernels.ROPE_EXT, new long[]{RopeKernels.roundUp(items, local)},
                new long[]{local}, args);
    }

    /** Runs a {@link #prepareRope} dispatch over {@code nTokens} rows whose position sections are {@code posStride} apart. */
    static void runRope(PreparedDispatch d, RopeSpec rope, int nTokens, int nHeads, int posStride, int local) {
        d.setInt(RopeKernels.ARG_N_TOKENS, nTokens);
        d.setInt(RopeKernels.ARG_POS_STRIDE, posStride);
        d.setGlobal(RopeKernels.roundUp(RopeKernels.ropeExtWorkItems(rope.type(), nTokens, nHeads, rope.nDims()),
                local));
        d.dispatch();
    }

    /**
     * {@code LLM_KV_Store_<type>} of {@code rowElems} values per token from {@code src} (rows {@code srcRowStride}
     * apart from element 0) into cache rows {@code cellIdx[t]} of {@code dst}, prepared for one token; validated for
     * {@code ubatch} tokens.
     */
    static PreparedDispatch prepareStore(GpuForwardContext ctx, KvDType type, int rowElems, int srcRowStride,
                                         int dstRowStride, GpuBuffer src, GpuBuffer cellIdx, GpuBuffer dst) {
        RopeKernels.validateKvStore(type, ctx.scratch().ubatch(), rowElems, srcRowStride, 0, dstRowStride);
        Object[] args = RopeKernels.kvStoreArgs(type, 1, rowElems, srcRowStride, 0, dstRowStride, src, cellIdx, dst);
        int local = ctx.ops().local();
        return ctx.ops().prepare(RopeKernels.kvStore(type),
                new long[]{RopeKernels.roundUp(RopeKernels.kvStoreWorkItems(type, 1, rowElems), local)},
                new long[]{local}, args);
    }

    /** Runs a {@link #prepareStore} dispatch for {@code nTokens} tokens. */
    static void runStore(PreparedDispatch d, KvDType type, int nTokens, int rowElems, int local) {
        d.setInt(ARG_STORE_N_TOKENS, nTokens);
        d.setGlobal(RopeKernels.roundUp(RopeKernels.kvStoreWorkItems(type, nTokens, rowElems), local));
        d.dispatch();
    }

    /**
     * {@code LLM_Batch_BiasAdd} of {@code bias} onto rows of {@code width} values of {@code data} (element 0),
     * prepared for one row; {@link #runBiasAdd} sets the row count. Null when {@code bias} is null.
     */
    static PreparedDispatch prepareBiasAdd(GpuForwardContext ctx, GpuWeight bias, int width, GpuBuffer data) {
        if (bias == null) {
            return null;
        }
        int local = ctx.ops().local();
        return ctx.ops().prepare(BATCH_BIAS_ADD, new long[]{RopeKernels.roundUp(width, local)}, new long[]{local},
                width, 1, bias.shard(0), data);
    }

    /** Runs a {@link #prepareBiasAdd} dispatch over {@code nRows} rows; a null dispatch does nothing. */
    static void runBiasAdd(PreparedDispatch d, int nRows) {
        if (d != null) {
            d.setInt(ARG_BIAS_SEQ_LEN, nRows);
            d.dispatch();
        }
    }

    /**
     * {@code LLM_TopK_Rows} selecting {@code k} columns per row of {@code scores} into {@code out}, prepared for one
     * row of one column; {@link #runTopK} sets the row and column counts per step.
     *
     * @throws IllegalArgumentException for a width outside {@code [1, TOPK_MAX]}
     */
    static PreparedDispatch prepareTopK(GpuForwardContext ctx, int k, GpuBuffer scores, GpuBuffer out) {
        SparseKernels.checkTopK(k);
        SparseKernels.checkTopKLocal(SparseKernels.TOPK_LOCAL);
        return ctx.ops().prepare(SparseKernels.TOPK_ROWS, SparseKernels.topKGlobal(1, SparseKernels.TOPK_LOCAL),
                SparseKernels.local(SparseKernels.TOPK_LOCAL), 1, 1, k, scores, out);
    }

    /** Runs a {@link #prepareTopK} dispatch over {@code nRows} rows of {@code nCols} scores. */
    static void runTopK(PreparedDispatch d, int nRows, int nCols) {
        d.setInt(SparseKernels.ARG_TOPK_N_ROWS, nRows);
        d.setInt(SparseKernels.ARG_TOPK_N_COLS, nCols);
        d.setGlobal(SparseKernels.topKGlobal(nRows, SparseKernels.TOPK_LOCAL));
        d.dispatch();
    }

    /**
     * {@code LLM_MSA_GatherAttn} of {@code nHead} query heads of {@code hd} values over the selected blocks of the
     * layer's cache, prepared for one token; {@link #runGather} sets the token count per step. {@code q} rows are
     * {@code qStride} apart and output rows {@code outStride} apart, both from element 0. An MLA latent cache
     * ({@code kv.vAliasesK()}) reads V from the first {@code hd} values of the K row.
     *
     * @param selHeads {@code nKvHeads} for one selection per KV head (MSA), 1 for one per token (QSA)
     * @param kvPos    the cache group's {@code kv_pos}
     * @param blocks   {@code blocks[(t*selHeads + sh)*k + s]}
     * @throws IllegalArgumentException for a shape the kernel cannot run
     */
    static PreparedDispatch prepareGather(GpuForwardContext ctx, int nHead, int nKvHeads, int hd, int blk, int k,
                                          float kqScale, int selHeads, boolean tail, KvLayout kv, GpuBuffer q,
                                          int qStride, GpuBuffer kCache, GpuBuffer vCache, GpuBuffer qPos,
                                          GpuBuffer kvPos, GpuBuffer blocks, GpuBuffer out, int outStride) {
        int local = MsaKernels.GATHER_LOCAL;
        KvDType vType = kv.vAliasesK() ? kv.dtypeK() : kv.dtypeV();
        int vStride = kv.vAliasesK() ? kv.widthK() : kv.widthV();
        MsaKernels.checkGatherAttn(nHead, nKvHeads, hd, blk, k, selHeads, kv.rows(), kv.widthK(), vStride,
                kv.dtypeK(), vType, local);
        Object[] args = {
                1, nHead, nKvHeads, hd, blk, k, kqScale, selHeads, tail ? 1 : 0, kv.rows(), kv.widthK(), vStride,
                MsaKernels.kvTypeCode(kv.dtypeK()), MsaKernels.kvTypeCode(vType), qStride, 0, outStride, 0,
                q, kCache, vCache, qPos, kvPos, blocks, out};
        return ctx.ops().prepare(MsaKernels.MSA_GATHER_ATTN, MsaKernels.gatherGlobal(1, nHead, local),
                MsaKernels.local(local), args);
    }

    /** Runs a {@link #prepareGather} dispatch for {@code nTokens} tokens. */
    static void runGather(PreparedDispatch d, int nTokens, int nHead) {
        d.setInt(MsaKernels.ARG_GA_N_TOKENS, nTokens);
        d.setGlobal(MsaKernels.gatherGlobal(nTokens, nHead, MsaKernels.GATHER_LOCAL));
        d.dispatch();
    }

    /**
     * In-place block-diagonal Hadamard rotation of {@code nRows} rows of {@code width} values at element 0
     * ({@code llama_mul_mat_hadamard}, src/llama-impl.h:57-75); nothing when {@code block} is 0.
     */
    static void rotate(GpuOps ops, int block, GpuBuffer x, int nRows, int width) {
        if (block > 0) {
            ops.hadamard(HadamardSpec.rotation(block), x, 0, x, 0, nRows, width);
        }
    }

    /**
     * Checks at bind that rows of {@code width} values can be rotated in blocks of {@code block} (0: no rotation).
     *
     * @throws IllegalArgumentException when the width is not a multiple of the block or the block exceeds the kernel
     */
    static void checkRotation(int block, int width) {
        if (block > 0) {
            HadamardSpec h = HadamardSpec.rotation(block);
            h.checkWidth(width);
            if (!h.kernelSupported()) {
                throw new IllegalArgumentException("KV rotation block " + block + " exceeds the kernel limit of "
                        + HadamardSpec.MAX_KERNEL_BLOCK);
            }
        }
    }

    // ================================================================== close helpers

    /** Closes every dispatch and owned buffer, rethrowing the first failure with the others suppressed. */
    static void release(List<PreparedDispatch> prepared, List<GpuBuffer> owned) {
        RuntimeException first = Closeables.closeAll(prepared, null);
        prepared.clear();
        first = Closeables.closeAll(owned, first);
        owned.clear();
        if (first != null) {
            throw first;
        }
    }

}

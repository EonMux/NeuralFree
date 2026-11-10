package com.eonmux.neuralfree.gguf.ops.mla;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuMatmulTap;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceKvFormats;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.attn.AttnExtKernels;
import com.eonmux.neuralfree.gguf.ops.attn.AttnExtMath;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.attn.FlashAttnTiling;
import com.eonmux.neuralfree.gguf.ops.attn.FlashBindGuard;
import com.eonmux.neuralfree.gguf.ops.moe.GpuGroupedRows;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.utils.Closeables;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * GPU MLA mixer (models/deepseek2.cpp:493-620, models/minicpm3.cpp:95-200, models/kimi-k3.cpp:483-558,
 * models/bailingmoe3.cpp:300-353). One step of {@code n} tokens:
 * <ol>
 *   <li>Q: {@code attn_q}, or {@code attn_q_a -> LLM_RowNorm(attn_q_a_norm) -> attn_q_b}, into {@code [t][h][head_k]};
 *       {@code attn_kv_a_mqa} into {@code [t][kv_lora | n_rot]}.</li>
 *   <li>Optional {@code LLM_RowNorm(attn_k_norm)} over the strided shared key (dots3note.cpp:330-332);
 *       {@code LLM_RoPE_Ext} on the query rope dims ({@code n_offs = n_nope}) and on the shared key
 *       ({@code n_offs = kv_lora}); {@code LLM_RowNorm(attn_kv_a_norm)} over the strided latent prefix.</li>
 *   <li>Latent cache: {@code {T}_MatVecId(attn_k_b)} with slot {@code s = t*n_head + h -> h} absorbs {@code q_nope}
 *       into {@code kv_lora} values per head; a step of several tokens sorts the slots by head once
 *       ({@link GpuGroupedRows}, the heads as experts) and runs the absorption and the expansion below as grouped
 *       GEMMs over the aligned rows ({@code {T}_GroupedGemm}, {@code {T}_GroupedGemmMma} on CUDA or Vulkan tensor
 *       cores) when the weight types and the device's grid allow; two {@code LLM_Copy2D_F32} build
 *       {@code [q_abs | q_pe]}; the K row is the {@code attn_kv_a_mqa} row itself. Decompressed cache:
 *       {@code attn_kv_b} gives {@code [k_nope | v]} per head; {@code LLM_MLA_KV_Assemble} (one token) or copies
 *       (several) build K and V. Decompressed cache over absorbed weights (deepseek2 converted without
 *       {@code attn_kv_b}): one {@code {T}_MatVecIdT} per token reads {@code attn_k_b} transposed into
 *       {@code k_nope} and one {@code {T}_MatVecId} per token expands {@code attn_v_b} into V, with the same
 *       copies, rotations and stores as the {@code attn_kv_b} cache.</li>
 *   <li>Optional {@code LLM_AttnTempScale}, the Hadamard rotations the memory plan assigns the cache,
 *       {@code LLM_KV_Store_*}, then flash attention: decode, tiled prefill or per-row decode, with
 *       {@code LLM_FlashAttn_Combine} after a split, and the per-head {@code attn_sinks} logits in the softmax when
 *       the layer has them (hy-v4.cpp:325-327). The latent cache binds one buffer to K and V
 *       ({@code d_k = kv_lora + n_rot}, {@code d_v = kv_lora}).</li>
 *   <li>Latent cache: un-rotation, then {@code {T}_MatVecId(attn_v_b)} per head into {@code [t][h][head_v]}.
 *       Optional {@code LLM_AttnOutGate} from {@code attn_gate(x)}; then {@code attn_output}.</li>
 * </ol>
 * Each of {@code attn_q}, {@code attn_q_a}, {@code attn_q_b}, {@code attn_kv_a_mqa}, {@code attn_kv_b} (decompressed
 * cache), {@code attn_gate} and {@code attn_output} adds the {@link GpuProjectionDeltas} registered on it (LoRA)
 * right after its product ({@code build_lora_mm}, llama-graph.cpp:1629-1667). The latent cache never runs
 * {@code attn_kv_b} (it uses the absorbed {@code attn_k_b} / {@code attn_v_b}), so it refuses deltas on it.
 * Every buffer comes from the scratch temporaries, the memory and the weights, except the constant head-slot index
 * table this mixer owns. Dispatches are prepared at bind; a step re-sets only token counts, key ranges, positions,
 * splits and the row offsets of per-head products that exceed the device's grid in one call. With
 * {@link com.eonmux.neuralfree.gguf.BatchInvariance} on, the step keeps the single-row kernel choices at every row
 * count: the per-head absorption and expansion never take their grouped GEMMs, and the attention runs one
 * flash-decode launch per query row on the unpaged and on the paged cache, so a row's key split comes from its own
 * key count (the paged launch reads the row's segment through the focus table).
 */
public final class GpuMlaMixer implements GpuMixer {

    static final String TEMP_Q_A = "mla.q_a";
    static final String TEMP_Q = "mla.q";
    static final String TEMP_KV_A = "mla.kv_a";
    static final String TEMP_LATENT = "mla.latent";
    static final String TEMP_Q_LATENT = "mla.q_latent";
    static final String TEMP_KV_C = "mla.kv_c";
    static final String TEMP_KV_B = "mla.kv_b";
    static final String TEMP_K = "mla.k";
    static final String TEMP_V = "mla.v";
    static final String TEMP_ATTN = "mla.attn";
    static final String TEMP_GATE = "mla.gate";
    /** {@code LLM_MLA_KV_Assemble} (kernels/llm/mla.cl): 6 scalars then 4 buffers, all at element 0. */
    static final String KV_ASSEMBLE = "LLM_MLA_KV_Assemble";
    /** Scratch prefix of the head sort of the grouped per-head products ({@link GpuGroupedRows}). */
    static final String TEMP_HEAD_ROWS = "mla.head_rows";

    private final List<PreparedDispatch> prepared = new ArrayList<>();
    private final List<GpuBuffer> owned = new ArrayList<>();
    private boolean bound;
    private boolean closed;

    private GpuOps ops;
    private GpuMemory memory;
    /** The identity KV-scale placeholder the flash kernels read on an unscaled latent cache. */
    private GpuBuffer kvScalesPlaceholder;
    private com.eonmux.neuralfree.acceleration.backend.ComputeBackend backend;
    private GpuBuffer focus;
    /** The one-sequence slot count and query position of the batch-invariant per-row paged launches. */
    private GpuBuffer focusKvLen;
    private GpuBuffer focusQPos;
    private int il;
    private int local;
    private int idLocal;
    /** Tokens per {@code {T}_MatVecId} call: {@code n_sel = tokens * n_head} stays within the grid's second dim. */
    private int idChunkTokens;
    private MlaBlocks.Geometry g;
    private KvLayout layout;
    private GpuKvLayer kv;
    private GpuStepInputs steps;
    private boolean owns;
    private int nEmbd;
    private int nOut;
    private RopeSpec rope;
    private GpuBuffer ff;
    private boolean hasFf;
    private String shiftBlocker;
    private HadamardSpec rotKSpec;
    private HadamardSpec rotVSpec;

    /**
     * Why the absorbed MLA paths (the latent cache and the decompressed cache over {@code attn_k_b} /
     * {@code attn_v_b}) refuse deltas on {@code attn_kv_b}, which they never multiply.
     */
    static final String ABSORBED_KV_B_REFUSAL = "the absorbed attn_k_b / attn_v_b weights replace attn_kv_b in this"
            + " MLA layer, which therefore never multiplies attn_kv_b; a LoRA delta on attn_kv_b needs a conversion"
            + " with the fused attn_kv_b instead of the per-head pair";

    private GpuWeight wq;
    private GpuWeight wqA;
    private GpuWeight wqB;
    private GpuWeight wkvA;
    private GpuWeight wkvB;
    /** The absorbed per-head {@code attn_k_b} and {@code attn_v_b}, bound whenever the layer has them. */
    private GpuWeight wkB;
    private GpuWeight wvB;
    private GpuWeight wo;
    private GpuWeight wGate;
    private GpuProjectionDeltas.Site qDeltas;
    private GpuProjectionDeltas.Site qaDeltas;
    private GpuProjectionDeltas.Site qbDeltas;
    private GpuProjectionDeltas.Site kvaDeltas;
    private GpuProjectionDeltas.Site kvbDeltas;
    private GpuProjectionDeltas.Site outDeltas;
    private GpuProjectionDeltas.Site gateDeltas;

    private ScratchSlot qa;
    private ScratchSlot q;
    private ScratchSlot kva;
    private ScratchSlot lat;
    private ScratchSlot qlat;
    private ScratchSlot kvc;
    private ScratchSlot kvb;
    private ScratchSlot k;
    private ScratchSlot v;
    private ScratchSlot attn;
    private ScratchSlot gate;

    private PreparedDispatch normQ;
    private PreparedDispatch normKPe;
    private PreparedDispatch normKv;
    private PreparedDispatch ropeQ;
    private PreparedDispatch ropeK;
    private PreparedDispatch absorb;
    private PreparedDispatch expand;
    /** The per-head products of a decompressed cache over absorbed weights: the transposed
     * {@code attn_k_b} read into k_nope, and the {@code attn_v_b} expansion into V. */
    private PreparedDispatch kDecomp;
    private PreparedDispatch vExpand;
    /** Head-sorted slot rows of a several-token latent step; null when the per-head products stay per row. */
    private GpuGroupedRows headRows;
    private PreparedDispatch absorbGrouped;
    private PreparedDispatch expandGrouped;
    private PreparedDispatch assemble;
    private PreparedDispatch tempScale;
    private PreparedDispatch storeK;
    private PreparedDispatch storeV;
    private PreparedDispatch fd;
    private PreparedDispatch fdRow;
    private PreparedDispatch fa;
    private PreparedDispatch fdPaged;
    private PreparedDispatch faPaged;
    private PreparedDispatch fc;
    /** The causal and sink bits of the flash {@code flags} argument; each launch adds its direct bit. */
    private int flashFlags;
    /** The flash-attention argument builders of this layer's bound geometry. */
    private MlaFlashArgs flashArgs;
    private PreparedDispatch outGate;

    private FlashAttnTiling decodeTiling;
    private FlashAttnTiling prefillTiling;
    private FlashAttnTiling pagedTiling;
    private long partialCapacity;
    private int pageShift = -1;
    private int maxPages;
    private int maxLaunchSeq;

    GpuMlaMixer() {
    }

    /**
     * Why the GPU cannot run this MLA layer, or null: the shared limits of {@link MlaBlocks#unsupportedCommon}.
     * A decompressed cache over absorbed {@code attn_k_b}/{@code attn_v_b} runs like any other decompressed cache,
     * through {@code {T}_MatVecIdT} ({@code attn_k_b}) and {@code {T}_MatVecId} ({@code attn_v_b}).
     */
    static String unsupported(LayerSpec layer) {
        return MlaBlocks.unsupportedCommon(layer);
    }

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException      when the layer is not an MLA layer
     * @throws UnsupportedOperationException for a variant {@link #unsupported} names, or a weight type without
     *                                       id-matvec kernels
     * @throws IllegalStateException         when the model has no memory, a tensor or the cache has the wrong shape,
     *                                       or the mixer is already bound or closed
     * @throws MissingTensorException        when a required tensor is absent
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        MlaBlocks.checkKind(layer);
        if (bound || closed) {
            throw new IllegalStateException("MLA mixer is already bound or closed");
        }
        String why = unsupported(layer);
        if (why != null) {
            throw new UnsupportedOperationException(why);
        }
        try {
            bindChecked(ctx, layer);
            bound = true;
        } catch (RuntimeException e) {
            try {
                release();
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
    }

    private void bindChecked(GpuForwardContext ctx, LayerSpec layer) {
        if (!ctx.hasMemory()) {
            throw new IllegalStateException("layer " + layer.il() + ": the MLA mixer needs the model's KV memory");
        }
        il = layer.il();
        ops = ctx.ops();
        backend = ctx.backend();
        local = ops.local();
        NormKernels.checkReduceLocal(local);
        memory = ctx.memory();
        kv = memory.kv(il);
        if (kv == null) {
            throw new IllegalStateException("layer " + il + ": the memory holds no KV cache for this MLA layer");
        }
        layout = kv.layout();
        owns = memory.plan().layer(il).ownsKv();
        steps = memory.stepInputs();
        g = MlaBlocks.Geometry.of(layer, layout, memory.plan().layer(il).rotation());
        rotKSpec = g.rotK > 0 ? HadamardSpec.rotation(g.rotK) : null;
        rotVSpec = g.rotV > 0 ? HadamardSpec.rotation(g.rotV) : null;
        rope = MlaBlocks.rope(layer);
        shiftBlocker = g.shiftBlocker(rope, layout);

        GpuWeights ws = ctx.weightSource().weightsFor(layer.weightLayer());
        bindWeights(ws, layer);
        bindDeltas(ctx.projectionDeltas(), layer);
        GpuBuffer ph = ctx.placeholders().readOnlyFloats();
        kvScalesPlaceholder = ctx.placeholders().kvScalesOnes();
        bindRopeFactors(ws, layer, ph);
        bindTemps(ctx.scratch(), layer);
        prepareProjections(ctx, layer, ws, ph);
        prepareAttention(ctx, layer, ws, ph);
        if (owns && rope != null && shiftRefusalChecked() == null) {
            ops.prewarm(RopeKernels.kvRopeShift(layout.dtypeK()), new long[]{shiftLocal()});
        }
    }

    private void bindWeights(GpuWeights ws, LayerSpec layer) {
        wkvA = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_KV_A_MQA));
        nEmbd = MlaBlocks.toInt(wkvA.cols(), "MLA input width");
        MlaBlocks.checkShape(wkvA.name(), wkvA.cols(), wkvA.rows(), nEmbd, g.kvaWidth);
        if (g.qLora > 0) {
            wqA = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_Q_A));
            wqB = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_Q_B));
            MlaBlocks.checkShape(wqA.name(), wqA.cols(), wqA.rows(), nEmbd, g.qLora);
            MlaBlocks.checkShape(wqB.name(), wqB.cols(), wqB.rows(), g.qLora, g.qWidth);
        } else {
            wq = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_Q));
            MlaBlocks.checkShape(wq.name(), wq.cols(), wq.rows(), nEmbd, g.qWidth);
        }
        if (!g.latent && !g.absorbed) {
            wkvB = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_KV_B));
            MlaBlocks.checkShape(wkvB.name(), wkvB.cols(), wkvB.rows(), g.kvLora, (long) g.nHead * (g.nope + g.headV));
        }
        wo = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_OUT));
        MlaBlocks.checkShape(wo.name(), wo.cols(), wo.rows(), (long) g.nHead * g.headV, wo.rows());
        nOut = MlaBlocks.toInt(wo.rows(), "MLA output width");
        if (MlaBlocks.gateKind(layer) != AttnGateKind.NONE) {
            wGate = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_GATE));
            MlaBlocks.checkShape(wGate.name(), wGate.cols(), wGate.rows(), nEmbd, MlaBlocks.gateWidth(layer));
        }
    }

    /** The delta sites of the projections this layer runs. */
    private void bindDeltas(GpuProjectionDeltas deltas, LayerSpec layer) {
        kvaDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_KV_A_MQA), wkvA);
        if (g.qLora > 0) {
            qaDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_Q_A), wqA);
            qbDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_Q_B), wqB);
        } else {
            qDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_Q), wq);
        }
        if (wkvB != null) {
            kvbDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_KV_B), wkvB);
        } else {
            deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_KV_B), ABSORBED_KV_B_REFUSAL);
        }
        outDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_OUT), wo);
        if (wGate != null) {
            gateDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_GATE), wGate);
        }
    }

    private void bindRopeFactors(GpuWeights ws, LayerSpec layer, GpuBuffer ph) {
        ff = ph;
        hasFf = false;
        if (rope == null || rope.freqFactors() == null) {
            return;
        }
        TensorRef ref = rope.freqFactors();
        GpuWeight w = ws.getOrNull(ref.name());
        if (w == null) {
            throw new MissingTensorException(ws.archName(), layer.il(), ref.name());
        }
        int pairs = RopeMath.pairCount(rope.type(), rope.nDims());
        if (w.type() != GGMLType.F32 || !w.isSingleShard() || w.rows() * w.cols() < pairs) {
            throw new IllegalStateException(w.name() + " must be one F32 buffer of at least " + pairs
                    + " frequency factors, got " + w);
        }
        ff = w.shard(0);
        hasFf = true;
    }

    private void bindTemps(GpuScratch scratch, LayerSpec layer) {
        long u = scratch.ubatch();
        long rows = u * g.nHead;
        if (g.qLora > 0) {
            qa = scratch.temp(TEMP_Q_A, u * g.qLora);
        }
        q = scratch.temp(TEMP_Q, u * g.qWidth);
        kva = scratch.temp(TEMP_KV_A, u * g.kvaWidth);
        attn = scratch.temp(TEMP_ATTN, rows * g.headV);
        if (g.latent) {
            lat = scratch.temp(TEMP_LATENT, rows * g.kvLora);
            qlat = scratch.temp(TEMP_Q_LATENT, rows * g.kvaWidth);
        } else if (g.absorbed) {
            kvb = scratch.temp(TEMP_KV_B, rows * g.nope);
            k = scratch.temp(TEMP_K, rows * g.headK);
            v = scratch.temp(TEMP_V, rows * g.headV);
        } else {
            kvc = scratch.temp(TEMP_KV_C, u * g.kvLora);
            kvb = scratch.temp(TEMP_KV_B, rows * (g.nope + g.headV));
            k = scratch.temp(TEMP_K, rows * g.headK);
            v = scratch.temp(TEMP_V, rows * g.headV);
        }
        if (wGate != null) {
            gate = scratch.temp(TEMP_GATE, u * MlaBlocks.gateWidth(layer));
        }
    }

    // ------------------------------------------------------------------ prepared dispatches

    private void prepareProjections(GpuForwardContext ctx, LayerSpec layer, GpuWeights ws, GpuBuffer ph) {
        int u = ctx.scratch().ubatch();
        float eps = ctx.spec().hparams().normRmsEps();
        if (g.qLora > 0) {
            GpuWeight n = normWeight(ws, layer, TensorNames.ATTN_Q_A_NORM, g.qLora);
            normQ = keep(ops.prepare(NormKernels.ROW_NORM, rowNormGlobal(u), new long[]{local},
                    NormKernels.rowNormArgs(NormKind.RMS, eps, u, g.qLora, g.qLora, qa.buffer(), qa.offset(),
                            qa.buffer(), qa.offset(), n.shard(0), null, 1, ph)));
        }
        if (MlaBlocks.kPeNorm(layer)) {
            GpuWeight kPe = normWeight(ws, layer, TensorNames.ATTN_K_NORM, g.rope);
            normKPe = keep(ops.prepare(NormKernels.ROW_NORM, rowNormGlobal(u), new long[]{local},
                    NormKernels.rowNormArgs(NormKind.RMS, eps, u, g.rope, g.kvaWidth, kva.buffer(),
                            kva.offset() + g.kvLora, kva.buffer(), kva.offset() + g.kvLora, kPe.shard(0), null, 1,
                            ph)));
        }
        GpuWeight kvNorm = normWeight(ws, layer, TensorNames.ATTN_KV_A_NORM, g.kvLora);
        normKv = keep(ops.prepare(NormKernels.ROW_NORM, rowNormGlobal(u), new long[]{local},
                NormKernels.rowNormArgs(NormKind.RMS, eps, u, g.kvLora, g.kvaWidth, kva.buffer(), kva.offset(),
                        kva.buffer(), kva.offset(), kvNorm.shard(0), null, 1, ph)));
        if (rope != null) {
            ropeQ = keep(ops.prepare(RopeKernels.ROPE_EXT, ropeGlobal(u, g.nHead), new long[]{local},
                    RopeKernels.ropeExtArgs(rope, u, g.nHead, g.headK, g.qWidth, g.nope, u, q.offset(), hasFf,
                            steps.ropePos(), ff, q.buffer())));
            ropeK = keep(ops.prepare(RopeKernels.ROPE_EXT, ropeGlobal(u, 1), new long[]{local},
                    RopeKernels.ropeExtArgs(rope, u, 1, g.kvaWidth, g.kvaWidth, g.kvLora, u, kva.offset(), hasFf,
                            steps.ropePos(), ff, kva.buffer())));
        }
        int rows = u * g.nHead;
        if (g.absorbed) {
            wkB = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_K_B));
            wvB = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_V_B));
            MlaBlocks.checkShape(wkB.name(), wkB.cols(), wkB.rows(), g.nope, (long) g.kvLora * g.nHead);
            MlaBlocks.checkShape(wvB.name(), wvB.cols(), wvB.rows(), g.kvLora, (long) g.headV * g.nHead);
            requirePlainHeads(wkB);
            requirePlainHeads(wvB);
            idLocal = Math.min(MoeKernels.MAX_LOCAL, local);
            MoeKernels.checkIdLocal(wkB.type(), idLocal);
            MoeKernels.checkIdLocal(wvB.type(), idLocal);
            GpuBuffer slots = headSlots(ctx, rows);
            if (g.latent) {
                idChunkTokens = idChunkTokens(ctx.kernels().caps().getMaxWorkGroupCount()[1]);
                absorb = keep(ops.prepare(MoeKernels.matVecId(wkB.type()),
                        MoeKernels.idMatVecGlobal(g.kvLora, rows, idLocal), new long[]{idLocal, 1},
                        MoeKernels.matVecIdArgs(g.kvLora, g.nope, rows, g.headK, q.offset(), lat.offset(), false, 0,
                                g.nHead, slots, wkB.shard(0), q.buffer(), ph, lat.buffer())));
                expand = keep(ops.prepare(MoeKernels.matVecId(wvB.type()),
                        MoeKernels.idMatVecGlobal(g.headV, rows, idLocal), new long[]{idLocal, 1},
                        MoeKernels.matVecIdArgs(g.headV, g.kvLora, rows, g.kvLora, lat.offset(), attn.offset(),
                                false, 0, g.nHead, slots, wvB.shard(0), lat.buffer(), ph, attn.buffer())));
                prepareGroupedHeads(ctx, wkB, wvB, slots, ph);
            } else {
                // The absorbed attn_k_b stores each head's block [kv_lora x n_nope] row-major, so the
                // k_nope product reads it transposed ({T}_MatVecIdT); the launches run one token at a
                // time (x_stride = 0 broadcasts the token's latent row to its heads), like the CPU twin,
                // and the head count of one such launch must still fit the grid.
                idChunkTokens(ctx.kernels().caps().getMaxWorkGroupCount()[1]);
                kDecomp = keep(ops.prepare(MoeKernels.matVecIdT(wkB.type()),
                        MoeKernels.idMatVecGlobal(g.nope, rows, idLocal), new long[]{idLocal, 1},
                        MoeKernels.matVecIdArgs(g.nope, g.kvLora, rows, 0, kva.offset(), kvb.offset(), false, 0,
                                g.nHead, slots, wkB.shard(0), kva.buffer(), ph, kvb.buffer())));
                vExpand = keep(ops.prepare(MoeKernels.matVecId(wvB.type()),
                        MoeKernels.idMatVecGlobal(g.headV, rows, idLocal), new long[]{idLocal, 1},
                        MoeKernels.matVecIdArgs(g.headV, g.kvLora, rows, 0, kva.offset(), v.offset(), false, 0,
                                g.nHead, slots, wvB.shard(0), kva.buffer(), ph, v.buffer())));
            }
        } else if (kva.offset() == 0 && kvb.offset() == 0 && k.offset() == 0 && v.offset() == 0) {
            assemble = keep(ops.prepare(KV_ASSEMBLE,
                    new long[]{RopeKernels.roundUp((long) g.nHead * (g.headK + g.headV), local)}, new long[]{local},
                    g.nHead, g.nope, g.rope, g.headK, g.headV, g.kvLora,
                    kva.buffer(), kvb.buffer(), k.buffer(), v.buffer()));
        }
        AttentionSpec.Temperature t = MlaBlocks.temperature(layer);
        if (t != null) {
            ScratchSlot qs = g.latent ? qlat : q;
            tempScale = keep(ops.prepare(AttnExtKernels.TEMP_SCALE, AttnExtKernels.global(u, g.nHead, g.dQ, local),
                    new long[]{local}, AttnExtKernels.tempScaleArgs(t, u, g.nHead, g.dQ, qs.buffer(), qs.offset(),
                            g.nHead * g.dQ, steps.qPos())));
        }
        if (owns) {
            ScratchSlot ks = g.latent ? kva : k;
            storeK = keep(prepareStore(layout.dtypeK(), true, u, layout.widthK(), ks, kv.k()));
            if (!g.latent) {
                storeV = keep(prepareStore(layout.dtypeV(), false, u, layout.widthV(), v, kv.v()));
            }
        }
        if (wGate != null) {
            AttnGateKind kind = MlaBlocks.gateKind(layer);
            GpuWeight bias = ws.layerOrNull(layer, TensorNames.bias(TensorNames.ATTN_GATE));
            if (bias != null) {
                MlaBlocks.checkF32Vector(bias.name(), bias.type(), bias.rows() * bias.cols(), g.nHead);
                requireOneBinding(bias);
            } else if (AttnExtMath.gateRequiresBias(kind)) {
                throw new MissingTensorException(ws.archName(), il, TensorNames.bias(TensorNames.ATTN_GATE));
            }
            int gw = MlaBlocks.gateWidth(layer);
            outGate = keep(ops.prepare(AttnExtKernels.OUT_GATE, AttnExtKernels.global(u, g.nHead, g.headV, local),
                    new long[]{local}, AttnExtKernels.outGateArgs(kind, u, g.nHead, g.headV, attn.buffer(),
                            attn.offset(), g.nHead * g.headV, gate.buffer(), gate.offset(), gw,
                            bias != null ? bias.shard(0) : ph, bias != null)));
        }
    }

    /**
     * The store of one side of the layer's cache: {@code kSide} picks the K or V rows, their packed side planes and
     * (for a TurboQuant key stream) the centroid table of a vendor latent cache.
     */
    private PreparedDispatch prepareStore(KvDType type, boolean kSide, int u, int rowElems, ScratchSlot src,
                                          GpuBuffer dst) {
        KvLayout layout = kv.layout();
        GpuBuffer side = kv.vendorSide();
        long items = type.perTokenHead() || type.vendorFormat()
                ? RopeKernels.kvStoreWorkItemsPerTokenHead(type, 1, rowElems, rowElems)
                : RopeKernels.kvStoreWorkItems(type, 1, rowElems);
        return ops.prepare(RopeKernels.kvStore(type),
                new long[]{RopeKernels.roundUp(u * items, local)}, new long[]{local},
                RopeKernels.kvStoreArgs(type, u, rowElems, rowElems, src.offset(), rowElems, rowElems,
                        src.buffer(), steps.cellIdx(kv.group()), ops.placeholders().kvScalesOnes(), dst, null,
                        side != null ? side : ops.placeholders().scratchInts(),
                        side != null ? DeviceKvFormats.sideRowBytes(layout) : 0,
                        DeviceKvFormats.sidePlaneOffset(layout, !kSide), kSide ? 0 : 1));
    }

    private void prepareAttention(GpuForwardContext ctx, LayerSpec layer, GpuWeights ws, GpuBuffer ph) {
        FlashBindGuard.requireServable(ctx.kernels(), "MLA layer " + il, g.nHead, g.nHeadKv, g.dQ, g.dV);
        int smem = ctx.kernels().faSmemFloats();
        int cus = FlashAttnTiling.planningUnits(ctx.kernels().caps());
        boolean fast = FlashAttnTiling.decodeFastPath(ctx.kernels().decodeWarpPath(),
                layout.dtypeK(), layout.dtypeV(), g.dQ, g.dV);
        decodeTiling = FlashAttnTiling.decode(g.nHead, g.nHeadKv, g.dQ, g.dV, layout.rows(), smem, cus, fast);
        prefillTiling = FlashAttnTiling.prefill(g.nHead, g.nHeadKv, g.dQ, g.dV, layout.rows(), smem, cus);
        partialCapacity = decodeTiling.decodePartialFloats();
        if (prefillTiling.plan() == FlashAttnTiling.Plan.TILED) {
            partialCapacity = Math.max(partialCapacity,
                    MlaFlashArgs.prefillPartialFloats(prefillTiling, ctx.scratch().ubatch(), layout.rows()));
        } else {
            partialCapacity = Math.max(partialCapacity, prefillTiling.decodePartialFloats());
        }
        if (steps.pageShift() >= 0) {
            pageShift = steps.pageShift();
            maxPages = steps.maxPages();
            maxLaunchSeq = Math.min(steps.pagedMaxSequences(), ctx.scratch().ubatch());
            pagedTiling = FlashAttnTiling.decodePaged(maxLaunchSeq, g.nHead, g.nHeadKv, g.dQ, g.dV,
                    maxPages << pageShift, smem, cus, fast);
            partialCapacity = Math.max(partialCapacity, pagedTiling.decodePagedPartialFloats(maxLaunchSeq));
        }
        GpuBuffer partials = ctx.scratch().partials(MlaBlocks.toInt(partialCapacity, "MLA flash partial floats"));
        ScratchSlot qs = g.latent ? qlat : q;
        ScratchSlot os = g.latent ? lat : attn;
        float softcap = MlaBlocks.softcap(layer);
        GpuBuffer sinks = sinksBuffer(ws, layer);
        GpuBuffer sinkArg = sinks != null ? sinks : ph;
        int hasSinks = sinks != null ? 1 : 0;
        /* Every step rebuilds the flag word from this field: a latent cache's V alias bit is part of it. */
        flashFlags = AttnKernels.attnFlags(layout.window().causal(), sinks != null, false)
                | (layout.vAliasesK() ? AttnKernels.FLAG_V_ALIAS : 0);
        flashArgs = new MlaFlashArgs(g, layout, kv, steps, kvScalesPlaceholder, prefillTiling, pagedTiling,
                flashFlags, pageShift, maxPages);
        String kernel = AttnKernels.flashDecode(layout.dtypeK());
        fd = keep(ops.prepare(kernel, decodeTiling.decodeGlobal(1), FlashAttnTiling.decodeLocal(),
                flashArgs.decodeArgs(decodeTiling, qs, os, softcap, sinkArg, ph, partials)));
        if (prefillTiling.plan() == FlashAttnTiling.Plan.TILED) {
            fa = keep(ops.prepare(AttnKernels.flashAttn(layout.dtypeK()), prefillTiling.prefillGlobal(1, 1),
                    FlashAttnTiling.prefillLocal(), flashArgs.prefillArgs(qs, os, softcap, sinkArg, ph, partials)));
        } else {
            fdRow = keep(ops.prepare(kernel, prefillTiling.decodeGlobal(1), FlashAttnTiling.decodeLocal(),
                    flashArgs.decodeArgs(prefillTiling, qs, os, softcap, sinkArg, ph, partials)));
        }
        if (pagedTiling != null) {
            fdPaged = keep(ops.prepare(AttnKernels.flashDecodePaged(layout.dtypeK(), layout.dtypeV()),
                    pagedTiling.decodePagedGlobal(1, 1), FlashAttnTiling.decodeLocal(),
                    flashArgs.pagedDecodeArgs(qs, os, softcap, sinkArg, ph, partials)));
            if (prefillTiling.plan() == FlashAttnTiling.Plan.TILED) {
                faPaged = keep(ops.prepare(AttnKernels.flashAttnPaged(layout.dtypeK(), layout.dtypeV()),
                        prefillTiling.prefillGlobal(1, 1), FlashAttnTiling.prefillLocal(),
                        flashArgs.pagedPrefillArgs(qs, os, softcap, sinkArg, ph, partials)));
            }
        }
        Object[] c = new Object[AttnKernels.FC_ARG_COUNT];
        c[AttnKernels.ARG_FC_N_ROWS] = g.nHead;
        c[AttnKernels.ARG_FC_D_V] = g.dV;
        c[AttnKernels.ARG_FC_N_SPLIT] = 1;
        c[AttnKernels.ARG_FC_N_HEAD] = g.nHead;
        c[AttnKernels.ARG_FC_HAS_SINKS] = hasSinks;
        c[AttnKernels.ARG_FC_OUT_OFF] = os.offset();
        c[AttnKernels.ARG_FC_BUF_PARTIAL] = partials;
        c[AttnKernels.ARG_FC_BUF_SINKS] = sinkArg;
        c[AttnKernels.ARG_FC_BUF_OUT] = os.buffer();
        fc = keep(ops.prepare(AttnKernels.FLASH_COMBINE, FlashAttnTiling.combineGlobal(g.nHead),
                FlashAttnTiling.combineLocal(), c));
    }

    /**
     * The layer's {@code attn_sinks} as one F32 buffer of {@code n_head} logits, or null when the layer has no sinks.
     */
    private GpuBuffer sinksBuffer(GpuWeights ws, LayerSpec layer) {
        if (!MlaBlocks.sinks(layer)) {
            return null;
        }
        GpuWeight w = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_SINKS));
        MlaBlocks.checkF32Vector(w.name(), w.type(), w.rows() * w.cols(), g.nHead);
        requireOneBinding(w);
        return w.shard(0);
    }

    /** RMS-norm weight of {@code len} values; {@code LLM_RowNorm} reads it as one F32 vector. */
    private static GpuWeight normWeight(GpuWeights ws, LayerSpec layer, String base, int len) {
        GpuWeight w = ws.layer(layer, TensorNames.weight(base));
        MlaBlocks.checkF32Vector(w.name(), w.type(), w.rows() * w.cols(), len);
        requireOneBinding(w);
        return w;
    }

    /**
     * The grouped form of the per-head products of a several-token step, when the device can run it: the constant
     * slot table is the expert-id list of {@code n} tokens of {@code n_head} pairs, so {@link GpuGroupedRows} sorts
     * the step's rows by head and the absorption ({@code x_stride = head_k}) and expansion
     * ({@code x_stride = kv_lora}) multiply each head's rows as one tiled GEMM, writing the same rows as the
     * per-head {@code {T}_MatVecId}.
     */
    private void prepareGroupedHeads(GpuForwardContext ctx, GpuWeight wkB, GpuWeight wvB, GpuBuffer slots,
                                     GpuBuffer ph) {
        int u = ctx.scratch().ubatch();
        LlmKernels kernels = ctx.kernels();
        if (u < 2 || !GpuGroupedRows.alignUsable(kernels, u * g.nHead, g.nHead, Math.max(g.kvLora, g.headV))
                || !GpuGroupedRows.groupedUsable(kernels, wkB.type(), g.nope)
                || !GpuGroupedRows.groupedUsable(kernels, wvB.type(), g.kvLora)) {
            return;
        }
        headRows = new GpuGroupedRows(ops, ctx.scratch(), kernels, TEMP_HEAD_ROWS, u, g.nHead, g.nHead, slots, true);
        int tiles = headRows.nTiles();
        absorbGrouped = keep(ops.prepare(GpuGroupedRows.groupedKernel(ops, wkB.type(), false),
                MoeKernels.groupedGemmGlobal(g.kvLora, tiles), MoeKernels.groupedGemmLocal(),
                MoeKernels.groupedGemmArgs(g.kvLora, g.nope, tiles, 1, g.headK, q.offset(), lat.offset(), false, 0,
                        g.nHead, headRows.gemmToken(), headRows.gemmSlot(), wkB.shard(0), q.buffer(), ph,
                        lat.buffer())));
        expandGrouped = keep(ops.prepare(GpuGroupedRows.groupedKernel(ops, wvB.type(), false),
                MoeKernels.groupedGemmGlobal(g.headV, tiles), MoeKernels.groupedGemmLocal(),
                MoeKernels.groupedGemmArgs(g.headV, g.kvLora, tiles, 1, g.kvLora, lat.offset(), attn.offset(), false,
                        0, g.nHead, headRows.gemmToken(), headRows.gemmSlot(), wvB.shard(0), lat.buffer(), ph,
                        attn.buffer())));
    }

    /**
     * {@code attn_k_b} / {@code attn_v_b} enter plain {@code ggml_mul_mat} products (deepseek2.cpp:120,
     * llama-graph.cpp build_attn {@code v_mla}): one binding, no input Hadamard or input transform and no tensor
     * scale, which the per-head {@code {T}_MatVecId} would not apply.
     */
    private static void requirePlainHeads(GpuWeight w) {
        requireOneBinding(w);
        if (w.hadamard() != null || w.hasScale() || (w.inputTransform() != null && !w.inputTransform().isIdentity())) {
            throw new UnsupportedOperationException(w.name() + " carries an input transform (Hadamard) or a tensor"
                    + " scale; the per-head MLA products take plain weights");
        }
    }

    private static void requireOneBinding(GpuWeight w) {
        if (!w.isSingleShard()) {
            throw new IllegalStateException(w.name() + " is split into " + w.shardCount() + " buffers; the MLA "
                    + "kernels bind the whole tensor once");
        }
    }

    /**
     * The constant slot table of the per-head {@code {T}_MatVecId} calls: entry {@code s = t*n_head + h} is head
     * {@code h}, so slot {@code s} reads head {@code h}'s matrix of {@code attn_k_b} / {@code attn_v_b}. Written
     * once at bind; this mixer owns and releases it.
     */
    private GpuBuffer headSlots(GpuForwardContext ctx, int rows) {
        long bytes = (long) rows * Integer.BYTES;
        GpuBuffer b = ctx.backend().allocateBuffer(bytes, true);
        owned.add(b);
        ByteBuffer bb = ByteBuffer.allocate(MlaBlocks.toInt(bytes, "MLA head-slot bytes")).order(ByteOrder.LITTLE_ENDIAN);
        for (int s = 0; s < rows; s++) {
            bb.putInt(s % g.nHead);
        }
        GpuEvent e = ctx.queue().enqueueWriteBuffer(b, 0, bytes, bb.array());
        if (e != null) {
            e.close();
        }
        return b;
    }

    /**
     * Whole tokens per {@code {T}_MatVecId} call whose {@code n_head} slots each fit a grid of at most
     * {@code maxGroupsY} work-groups in its second dimension (CUDA and Vulkan: 65535).
     *
     * @throws IllegalStateException when even one token's heads exceed the limit
     */
    private int idChunkTokens(long maxGroupsY) {
        long tokens = Math.min(Integer.MAX_VALUE, maxGroupsY) / g.nHead;
        if (tokens < 1) {
            throw new IllegalStateException("layer " + il + ": " + g.nHead + " MLA heads exceed the device's "
                    + maxGroupsY + " work-groups in the second grid dimension");
        }
        return (int) tokens;
    }

    private long[] rowNormGlobal(int nRows) {
        return NormKernels.reduceGlobal(NormKernels.rowNormGroups(nRows), local);
    }

    private long[] ropeGlobal(int nTokens, int nHeads) {
        return new long[]{RopeKernels.roundUp(RopeKernels.ropeExtWorkItems(rope.type(), nTokens, nHeads, rope.nDims()),
                local)};
    }

    private PreparedDispatch keep(PreparedDispatch d) {
        prepared.add(d);
        return d;
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        run(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        run(pos0, nTokens, in, out);
    }

    /**
     * The cache writes of a step: the latent KV projection with its norms and RoPE, then the latent store (a latent
     * cache) or the per-head K/V decompression, the K/V rotations and both stores. The query projection, the
     * absorption, the attention, the output expansion and the output projection do not run.
     */
    @Override
    public void prefillKV(int pos0, int n, ScratchSlot in) {
        checkBound();
        Objects.requireNonNull(in, "in");
        if (n < 1 || n > ops.scratch().ubatch()) {
            throw new IllegalArgumentException("MLA step of " + n + " tokens is outside [1, " + ops.scratch().ubatch()
                    + "]");
        }
        if (pos0 < 0) {
            throw new IllegalArgumentException("layer " + il + ": negative position " + pos0);
        }
        if (steps.pos0() != pos0 || steps.nTokens() != n) {
            throw new IllegalStateException("layer " + il + ": step inputs hold positions [" + steps.pos0() + ", "
                    + (steps.pos0() + steps.nTokens()) + "), the MLA step runs [" + pos0 + ", " + (pos0 + n) + ")");
        }
        in.require((long) n * nEmbd);
        if (!owns) {
            return;
        }
        projectKV(n, in);
        if (g.latent) {
            latentIntoKV(n);
        } else {
            decompressedIntoKV(n);
        }
    }

    @Override
    public String kvRefusal() {
        checkBound();
        return null;
    }

    /**
     * The latent row alone: the projection, the optional RMS norm of the shared rotary key, RoPE of the rotary
     * dims and the RMS norm of the latent prefix. The query's products, its norm and its RoPE do not run.
     */
    private void projectKV(int n, ScratchSlot in) {
        ops.matmul(wkvA, in.buffer(), in.offset(), kva.buffer(), kva.offset(), n);
        kvaDeltas.applyRows(n, in.buffer(), in.offset(), kva.buffer(), kva.offset());
        if (normKPe != null) {
            normKPe.setInt(NormKernels.ARG_ROW_NORM_N_ROWS, n);
            normKPe.setGlobal(rowNormGlobal(n));
            normKPe.dispatch();
        }
        if (rope != null) {
            runRope(ropeK, n, 1);
        }
        normKv.setInt(NormKernels.ARG_ROW_NORM_N_ROWS, n);
        normKv.setGlobal(rowNormGlobal(n));
        normKv.dispatch();
    }

    /** The latent step up to the store: the K rotation of the latent row and the store. */
    private void latentIntoKV(int n) {
        if (rotKSpec != null) {
            ops.hadamard(rotKSpec, kva.buffer(), kva.offset(), kva.buffer(), kva.offset(), n, g.kvaWidth);
        }
        runStore(storeK, layout.dtypeK(), n, layout.widthK());
    }

    /**
     * The decompressed step up to the stores: decompression, per-head K/V, the K and V rotations and both stores.
     * The temperature and the Q rotation (query-only) do not run.
     */
    private void decompressedIntoKV(int n) {
        decompressHeads(n);
        int rows = g.headRows(n);
        if (rotKSpec != null) {
            ops.hadamard(rotKSpec, k.buffer(), k.offset(), k.buffer(), k.offset(), rows, g.headK);
        }
        if (rotVSpec != null) {
            ops.hadamard(rotVSpec, v.buffer(), v.offset(), v.buffer(), v.offset(), rows, g.headV);
        }
        runStore(storeK, layout.dtypeK(), n, layout.widthK());
        runStore(storeV, layout.dtypeV(), n, layout.widthV());
    }

    /**
     * The decompression of a decompressed-cache step into its per-head K and V: the {@code attn_kv_b} product
     * and its assembly, or the absorbed {@code attn_k_b} / {@code attn_v_b} products when the cache runs over
     * them.
     */
    private void decompressHeads(int n) {
        if (g.absorbed) {
            absorbedIntoHeads(n);
        } else {
            kvbIntoHeads(n);
        }
    }

    /** The {@code attn_kv_b} decompression: the product (over the copied latent rows of a several-token
     * step), then {@code LLM_MLA_KV_Assemble} or the copies that split {@code [k_nope | v]} per head. */
    private void kvbIntoHeads(int n) {
        int rows = g.headRows(n);
        int kvRow = g.nope + g.headV;
        if (n == 1) {
            ops.matmul(wkvB, kva.buffer(), kva.offset(), kvb.buffer(), kvb.offset(), 1);
            kvbDeltas.applyRows(1, kva.buffer(), kva.offset(), kvb.buffer(), kvb.offset());
        } else {
            ops.copy2d(kva.buffer(), kva.offset(), g.kvaWidth, kvc.buffer(), kvc.offset(), g.kvLora, n, g.kvLora);
            ops.matmul(wkvB, kvc.buffer(), kvc.offset(), kvb.buffer(), kvb.offset(), n);
            kvbDeltas.applyRows(n, kvc.buffer(), kvc.offset(), kvb.buffer(), kvb.offset());
        }
        if (n == 1 && assemble != null) {
            assemble.dispatch();
        } else {
            ops.copy2d(kvb.buffer(), kvb.offset(), kvRow, k.buffer(), k.offset(), g.headK, rows, g.nope);
            ops.copy2d(kvb.buffer(), kvb.offset() + g.nope, kvRow, v.buffer(), v.offset(), g.headV, rows, g.headV);
            ropeIntoK(n);
        }
    }

    /** The absorbed decompression: the per-token transposed {@code attn_k_b} reads into k_nope and the
     * {@code attn_v_b} expansions straight into V, then the k_nope copy and the rope broadcast. */
    private void absorbedIntoHeads(int n) {
        int rows = g.headRows(n);
        runPerToken(kDecomp, n, g.nope, kva.offset(), kvb.offset());
        runPerToken(vExpand, n, g.headV, kva.offset(), v.offset());
        ops.copy2d(kvb.buffer(), kvb.offset(), g.nope, k.buffer(), k.offset(), g.headK, rows, g.nope);
        ropeIntoK(n);
    }

    /** The shared rotary key broadcast into the k_rope dims of every head. */
    private void ropeIntoK(int n) {
        if (g.rope > 0) {
            for (int h = 0; h < g.nHead; h++) {
                ops.copy2d(kva.buffer(), kva.offset() + g.kvLora, g.kvaWidth, k.buffer(),
                        k.offset() + h * g.headK + g.nope, g.qWidth, n, g.rope);
            }
        }
    }

    private void run(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        checkBound();
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ops.scratch().ubatch()) {
            throw new IllegalArgumentException("MLA step of " + n + " tokens is outside [1, " + ops.scratch().ubatch()
                    + "]");
        }
        if (steps.pos0() != pos0 || steps.nTokens() != n) {
            throw new IllegalStateException("layer " + il + ": step inputs hold positions [" + steps.pos0() + ", "
                    + (steps.pos0() + steps.nTokens()) + "), the MLA step runs [" + pos0 + ", " + (pos0 + n) + ")");
        }
        in.require((long) n * nEmbd);
        out.require((long) n * nOut);
        // The query, latent and gate projections of a single-token step all read this row, so its Q8_1
        // blocks are quantized once.
        GpuOps.SharedQ8 share = n == 1 && GpuOps.shareableQ8Activations(in.offset(), nEmbd)
                ? ops.shareQ8Activations(in.buffer(), in.offset(), nEmbd) : null;
        try {
            project(n, in);
            if (g.latent) {
                latentStep(pos0, n);
            } else {
                decompressedStep(pos0, n);
            }
            if (outGate != null) {
                ops.matmul(wGate, in.buffer(), in.offset(), gate.buffer(), gate.offset(), n);
                gateDeltas.applyRows(n, in.buffer(), in.offset(), gate.buffer(), gate.offset());
                outGate.setInt(AttnExtKernels.ARG_GATE_N_TOKENS, n);
                outGate.setGlobal(AttnExtKernels.global(n, g.nHead, g.headV, local));
                outGate.dispatch();
            }
            ops.matmul(wo, attn.buffer(), attn.offset(), out.buffer(), out.offset(), n);
            outDeltas.applyRows(n, attn.buffer(), attn.offset(), out.buffer(), out.offset());
        } finally {
            if (share != null) {
                share.close();
            }
        }
    }

    /**
     * Q and the latent row: projections, the optional RMS norm of the shared rotary key, RoPE of the rotary dims, RMS
     * norm of the latent prefix (dots3note.cpp:330-344).
     */
    private void project(int n, ScratchSlot in) {
        if (g.qLora > 0) {
            ops.matmul(wqA, in.buffer(), in.offset(), qa.buffer(), qa.offset(), n);
            qaDeltas.applyRows(n, in.buffer(), in.offset(), qa.buffer(), qa.offset());
            normQ.setInt(NormKernels.ARG_ROW_NORM_N_ROWS, n);
            normQ.setGlobal(rowNormGlobal(n));
            normQ.dispatch();
            ops.matmul(wqB, qa.buffer(), qa.offset(), q.buffer(), q.offset(), n);
            qbDeltas.applyRows(n, qa.buffer(), qa.offset(), q.buffer(), q.offset());
        } else {
            ops.matmul(wq, in.buffer(), in.offset(), q.buffer(), q.offset(), n);
            qDeltas.applyRows(n, in.buffer(), in.offset(), q.buffer(), q.offset());
        }
        ops.matmul(wkvA, in.buffer(), in.offset(), kva.buffer(), kva.offset(), n);
        kvaDeltas.applyRows(n, in.buffer(), in.offset(), kva.buffer(), kva.offset());
        if (normKPe != null) {
            normKPe.setInt(NormKernels.ARG_ROW_NORM_N_ROWS, n);
            normKPe.setGlobal(rowNormGlobal(n));
            normKPe.dispatch();
        }
        if (rope != null) {
            runRope(ropeQ, n, g.nHead);
            runRope(ropeK, n, 1);
        }
        normKv.setInt(NormKernels.ARG_ROW_NORM_N_ROWS, n);
        normKv.setGlobal(rowNormGlobal(n));
        normKv.dispatch();
    }

    private void runRope(PreparedDispatch d, int n, int nHeads) {
        d.setInt(RopeKernels.ARG_N_TOKENS, n);
        d.setInt(RopeKernels.ARG_POS_STRIDE, n);
        d.setGlobal(ropeGlobal(n, nHeads));
        d.dispatch();
    }

    /** Absorbed Q, latent K row, attention over the latent cache, per-head expansion by {@code attn_v_b}. */
    private void latentStep(int pos0, int n) {
        latentInto(n);
        attend(pos0, n, qlat, lat);
        latentFrom(n);
    }

    /** The batched latent step: the same pipeline, its attention through the paged step. */
    private void latentBatched(PagedStep step, StepLayout s, int n) {
        latentInto(n);
        attendBatched(step, s, qlat, lat);
        latentFrom(n);
    }

    /** The latent pipeline up to the attention: absorbed Q, the latent K row, its rotation and the store. */
    private void latentInto(int n) {
        int rows = g.headRows(n);
        GpuMatmulTap.heads(ops.queue(), wkB, g.nHead, q.buffer(), q.offset(), g.headK, g.nope, n);
        /* Batch invariance keeps the per-head {T}_MatVecId at every row count: the grouped GEMM tiles the rows
         * differently from the per-head product, so a row's value would depend on the step's token count. */
        if (n > 1 && headRows != null && !BatchInvariance.enabled()) {
            headRows.update(n);
            headRows.dispatch();
            absorbGrouped.dispatch();
        } else {
            runPerHead(absorb, n, g.kvLora, g.headK, q.offset(), lat.offset());
        }
        ops.copy2d(lat.buffer(), lat.offset(), g.kvLora, qlat.buffer(), qlat.offset(), g.dQ, rows, g.kvLora);
        if (g.rope > 0) {
            ops.copy2d(q.buffer(), q.offset() + g.nope, g.headK, qlat.buffer(), qlat.offset() + g.kvLora, g.dQ, rows,
                    g.rope);
        }
        runTemperature(n);
        if (rotKSpec != null) {
            ops.hadamard(rotKSpec, qlat.buffer(), qlat.offset(), qlat.buffer(), qlat.offset(), rows, g.dQ);
            ops.hadamard(rotKSpec, kva.buffer(), kva.offset(), kva.buffer(), kva.offset(), n, g.kvaWidth);
        }
        runStore(storeK, layout.dtypeK(), n, layout.widthK());
    }

    /** The latent pipeline after the attention: the inverse rotation and the per-head {@code attn_v_b} expansion. */
    private void latentFrom(int n) {
        int rows = g.headRows(n);
        if (rotKSpec != null) {
            ops.hadamard(rotKSpec, lat.buffer(), lat.offset(), lat.buffer(), lat.offset(), rows, g.kvLora);
        }
        GpuMatmulTap.heads(ops.queue(), wvB, g.nHead, lat.buffer(), lat.offset(), g.kvLora, g.kvLora, n);
        if (n > 1 && headRows != null && !BatchInvariance.enabled()) {
            expandGrouped.dispatch();
        } else {
            runPerHead(expand, n, g.headV, g.kvLora, lat.offset(), attn.offset());
        }
    }

    /**
     * A per-head {@code {T}_MatVecId} over the {@code n * n_head} rows of the step, in calls of at most
     * {@link #idChunkTokens} tokens. Every call starts on a token boundary, so slot {@code s} of the call is head
     * {@code s % n_head} of the constant slot table.
     */
    private void runPerHead(PreparedDispatch d, int n, int outDim, int xStride, int inOff, int outOff) {
        for (int t0 = 0; t0 < n; t0 += idChunkTokens) {
            int sel = Math.min(idChunkTokens, n - t0) * g.nHead;
            int row0 = t0 * g.nHead;
            d.setInt(MoeKernels.ARG_ID_N_SEL, sel);
            d.setInt(MoeKernels.ARG_ID_IN_OFF, inOff + row0 * xStride);
            d.setInt(MoeKernels.ARG_ID_OUT_OFF, outOff + row0 * outDim);
            d.setGlobal(MoeKernels.idMatVecGlobal(outDim, sel, idLocal));
            d.dispatch();
        }
    }

    /**
     * A per-head product of the absorbed decompressed cache, one launch per token: {@code x_stride = 0}
     * broadcasts the token's latent row to its {@code n_head} head slots, which write the token's
     * {@code n_head * out_dim} outputs.
     */
    private void runPerToken(PreparedDispatch d, int n, int outDim, int inOff, int outOff) {
        for (int t = 0; t < n; t++) {
            d.setInt(MoeKernels.ARG_ID_N_SEL, g.nHead);
            d.setInt(MoeKernels.ARG_ID_IN_OFF, inOff + t * g.kvaWidth);
            d.setInt(MoeKernels.ARG_ID_OUT_OFF, outOff + t * g.nHead * outDim);
            d.setGlobal(MoeKernels.idMatVecGlobal(outDim, g.nHead, idLocal));
            d.dispatch();
        }
    }

    /** Per-head K/V decompression ({@code attn_kv_b} or the absorbed pair), attention over the per-head cache. */
    private void decompressedStep(int pos0, int n) {
        decompressedInto(n);
        attend(pos0, n, q, attn);
        decompressedFrom(n);
    }

    /** The batched decompressed step: the same pipeline, its attention through the paged step. */
    private void decompressedBatched(PagedStep step, StepLayout s, int n) {
        decompressedInto(n);
        attendBatched(step, s, q, attn);
        decompressedFrom(n);
    }

    /** The decompressed pipeline up to the attention: decompression, per-head K/V and the stores. */
    private void decompressedInto(int n) {
        decompressHeads(n);
        int rows = g.headRows(n);
        runTemperature(n);
        if (rotKSpec != null) {
            ops.hadamard(rotKSpec, q.buffer(), q.offset(), q.buffer(), q.offset(), rows, g.headK);
            ops.hadamard(rotKSpec, k.buffer(), k.offset(), k.buffer(), k.offset(), rows, g.headK);
        }
        if (rotVSpec != null) {
            ops.hadamard(rotVSpec, v.buffer(), v.offset(), v.buffer(), v.offset(), rows, g.headV);
        }
        runStore(storeK, layout.dtypeK(), n, layout.widthK());
        runStore(storeV, layout.dtypeV(), n, layout.widthV());
    }

    /** The decompressed pipeline after the attention: the inverse V rotation. */
    private void decompressedFrom(int n) {
        int rows = g.headRows(n);
        if (rotVSpec != null) {
            ops.hadamard(rotVSpec, attn.buffer(), attn.offset(), attn.buffer(), attn.offset(), rows, g.headV);
        }
    }

    private void runTemperature(int n) {
        if (tempScale == null) {
            return;
        }
        tempScale.setInt(AttnExtKernels.ARG_TEMP_N_TOKENS, n);
        tempScale.setGlobal(AttnExtKernels.global(n, g.nHead, g.dQ, local));
        tempScale.dispatch();
    }

    private void runStore(PreparedDispatch d, KvDType type, int n, int rowElems) {
        if (d == null) {
            return;
        }
        d.setInt(RopeKernels.ARG_N_TOKENS, n);
        d.setGlobal(RopeKernels.roundUp(RopeKernels.kvStoreWorkItems(type, n, rowElems), local));
        d.dispatch();
    }

    // ------------------------------------------------------------------ attention

    /**
     * One step over the rows of {@code s}: the contiguous step at {@code s.pos0(0)} when no paged step is current,
     * the paged batched step through the memory's current {@code PagedStep} otherwise.
     *
     * @throws UnsupportedOperationException when the step is not contiguous and the memory holds no paged step
     */
    @Override
    public void batch(StepLayout s, ScratchSlot in, ScratchSlot out) {
        checkBound();
        Objects.requireNonNull(s, "s");
        PagedStep step = steps.currentPagedStep();
        if (step == null) {
            if (!s.contiguousSingle()) {
                throw new UnsupportedOperationException(batchRefusal());
            }
            run(s.pos0(0), s.nTokens(), in, out);
            return;
        }
        int n = s.nTokens();
        if (n < 1 || n > ops.scratch().ubatch()) {
            throw new IllegalArgumentException("MLA step of " + n + " tokens is outside [1, "
                    + ops.scratch().ubatch() + "]");
        }
        in.require((long) n * nEmbd);
        out.require((long) n * nOut);
        project(n, in);
        if (g.latent) {
            latentBatched(step, s, n);
        } else {
            decompressedBatched(step, s, n);
        }
        if (outGate != null) {
            ops.matmul(wGate, in.buffer(), in.offset(), gate.buffer(), gate.offset(), n);
            gateDeltas.applyRows(n, in.buffer(), in.offset(), gate.buffer(), gate.offset());
            outGate.setInt(AttnExtKernels.ARG_GATE_N_TOKENS, n);
            outGate.setGlobal(AttnExtKernels.global(n, g.nHead, g.headV, local));
            outGate.dispatch();
        }
        ops.matmul(wo, attn.buffer(), attn.offset(), out.buffer(), out.offset(), n);
        outDeltas.applyRows(n, attn.buffer(), attn.offset(), out.buffer(), out.offset());
    }

    /**
     * The batched attention: one paged decode launch when every segment of {@code s} carries one row, else one
     * paged prefill per segment ({@code LLM_FlashAttnPaged_*}; a PER_ROW plan falls back to the paged decode once
     * per row over the segment's page-table block). Batch invariance runs every segment per row through the focus
     * table with that segment's own slot count, the launches of the segments' single-sequence steps, so the batch's
     * longest sequence cannot size another sequence's key split. A tree verify step's mask rides the paged prefill
     * like the standard attention's: every segment's rows are its launch's fresh rows, whose relativized mask rows
     * and depth positions the launch uploads, and whose count rides the flags (a masked step always holds a
     * multi-row segment, so it never takes the decode path).
     *
     * @throws UnsupportedOperationException when the memory holds no paged step for this mixer, or the step carries
     *         a tree mask this layer's launch cannot take
     */
    private void attendBatched(PagedStep step, StepLayout s, ScratchSlot qs, ScratchSlot os) {
        if (fdPaged == null) {
            throw new UnsupportedOperationException("layer " + il + " runs a batched step on an unpaged cache");
        }
        int nSeg = step.nSeg();
        if (steps.treeMask() != null) {
            if (BatchInvariance.enabled()) {
                throw new UnsupportedOperationException("layer " + il + " runs a tree verify step on a batch-invariant"
                        + " plan, whose every segment runs through the per-row fallback");
            }
            if (faPaged == null) {
                throw new UnsupportedOperationException("layer " + il + " runs a tree verify step on a PER_ROW plan,"
                        + " whose segments run through the per-row fallback");
            }
        }
        if (BatchInvariance.enabled()) {
            int[] slots = step.seqSlots(kv.group());
            for (int seg = 0; seg < nSeg; seg++) {
                rowsFallback(step, seg, slots[step.segSeq(seg)], qs, os);
            }
            return;
        }
        boolean allSingle = true;
        for (int i = 0; i < nSeg; i++) {
            allSingle &= step.segLen(i) == 1;
        }
        if (allSingle) {
            decodeBatched(step, nSeg, qs, os);
            return;
        }
        int qStride = g.nHead * g.dQ;
        int outStride = g.nHead * g.dV;
        boolean treeStep = steps.treeMask() != null;
        GpuBuffer tree = treeStep ? steps.treeMask() : null;
        for (int seg = 0; seg < nSeg; seg++) {
            int len = step.segLen(seg);
            int keys = step.seqSlots(kv.group())[step.segSeq(seg)];
            int tokStart = step.segStart(seg);
            if (treeStep) {
                // the segment's rows rope at their tree depth positions and see their own ancestor bits,
                // both launch-local ([0, len) of the segment positions and mask buffers)
                steps.uploadSegmentTreePositions(tokStart, len);
                steps.uploadTreeMaskSegment(tokStart, len);
            } else {
                steps.uploadSegmentPositions(step.segPos0(seg), len);
            }
            if (faPaged != null) {
                FlashAttnTiling.Split sp = prefillTiling.prefillSplit(len, keys, partialCapacity);
                if (treeStep) {
                    faPaged.setBuffer(AttnKernels.ARG_FAP_BUF_TREE_MASK, tree);
                }
                faPaged.setInt(AttnKernels.ARG_FA_N_TOKENS, len);
                faPaged.setInt(AttnKernels.ARG_FA_N_SPLIT, sp.count());
                faPaged.setInt(AttnKernels.ARG_FA_SPLIT_LEN, sp.length());
                faPaged.setInt(AttnKernels.ARG_FA_KV_END, keys);
                // a tree step's fresh rows are the segment's rows, packed beside the tree bit
                faPaged.setInt(AttnKernels.ARG_FA_FLAGS, treeStep
                        ? AttnKernels.treeFlags(pagedFlags(sp.direct()), len) : pagedFlags(sp.direct()));
                faPaged.setInt(AttnKernels.ARG_FA_Q_OFF, (int) (qs.offset() + (long) tokStart * qStride));
                faPaged.setInt(AttnKernels.ARG_FA_OUT_OFF, (int) (os.offset() + (long) tokStart * outStride));
                faPaged.setInt(AttnKernels.ARG_FAP_PT_OFF, seg * maxPages);
                faPaged.setGlobal(prefillTiling.prefillGlobal(len, sp.count()));
                faPaged.dispatch();
                if (!sp.direct()) {
                    combine(g.headRows(len), sp.count(), (int) (os.offset() + (long) tokStart * outStride));
                }
            } else {
                rowsFallback(step, seg, keys, qs, os);
            }
        }
    }

    /**
     * Segment {@code seg} row by row through the paged decode kernel, each row over the segment's own {@code keys}
     * slots and query position: the segment's page-table row, slot count and the rows' positions go to the mixer's
     * one-sequence buffers, which the launch's one sequence reads ({@code LLM_FlashDecodePaged_*} takes its
     * sequence's slot count and query position from the bound buffers, not from the launch scalars), and the
     * launch's rows point back at the step's buffers afterwards.
     */
    private void rowsFallback(PagedStep step, int seg, int keys, ScratchSlot qs, ScratchSlot os) {
        int qStride = g.nHead * g.dQ;
        int outStride = g.nHead * g.dV;
        int tokStart = step.segStart(seg);
        ops.queue().enqueueCopyBuffer(steps.pageTable(kv.group()),
                (long) seg * maxPages * Integer.BYTES, focusTable(), 0, (long) maxPages * Integer.BYTES);
        writeLaunchInt(focusKvLen(), keys);
        fdPaged.setBuffer(AttnKernels.ARG_FDP_BUF_PAGE_TABLE, focusTable());
        fdPaged.setBuffer(AttnKernels.ARG_FDP_BUF_SEQ_KV_LEN, focusKvLen());
        fdPaged.setBuffer(AttnKernels.ARG_FDP_BUF_Q_POS_SEQ, focusQPos());
        for (int t = 0; t < step.segLen(seg); t++) {
            writeLaunchInt(focusQPos(), step.segPos0(seg) + t);
            decodePagedRow(qs.offset() + (tokStart + t) * qStride, os.offset() + (tokStart + t) * outStride, keys);
        }
        restorePagedRows();
    }

    /** Points the paged decode launch's rows back at the step's page table, slot counts and positions. */
    private void restorePagedRows() {
        fdPaged.setBuffer(AttnKernels.ARG_FDP_BUF_PAGE_TABLE, steps.pageTable(kv.group()));
        fdPaged.setBuffer(AttnKernels.ARG_FDP_BUF_SEQ_KV_LEN, steps.seqKvLen(kv.group()));
        fdPaged.setBuffer(AttnKernels.ARG_FDP_BUF_Q_POS_SEQ, steps.qPosSeq());
        fdPaged.setInt(AttnKernels.ARG_FDP_Q_SEQ_STRIDE, g.nHead * g.dQ);
        fdPaged.setInt(AttnKernels.ARG_FDP_OUT_SEQ_STRIDE, g.nHead * g.dV);
    }

    /**
     * The flags of a paged launch: the layer's flash flags without the causal bit when the step is non-causal
     * ({@link GpuStepInputs#stepNonCausal()}, every row then sees its whole sequence), plus the direct bit.
     */
    private int pagedFlags(boolean direct) {
        int f = steps.stepNonCausal() ? flashFlags & ~AttnKernels.FLAG_CAUSAL : flashFlags;
        return direct ? f | AttnKernels.FLAG_DIRECT : f;
    }

    /** One paged decode launch over the step's {@code nSeq} one-row segments (launch order = segment order). */
    private void decodeBatched(PagedStep step, int nSeq, ScratchSlot qs, ScratchSlot os) {
        int[] slots = step.seqSlots(kv.group());
        int maxKeys = 0;
        for (int i = 0; i < nSeq; i++) {
            maxKeys = Math.max(maxKeys, slots[step.segSeq(i)]);
        }
        FlashAttnTiling.Split sp = pagedTiling.decodeSplit(maxKeys);
        fdPaged.setInt(AttnKernels.ARG_FD_N_SPLIT, sp.count());
        fdPaged.setInt(AttnKernels.ARG_FD_SPLIT_LEN, sp.length());
        fdPaged.setInt(AttnKernels.ARG_FD_FLAGS, pagedFlags(sp.direct()));
        fdPaged.setInt(AttnKernels.ARG_FDP_N_SEQ, nSeq);
        fdPaged.setInt(AttnKernels.ARG_FD_Q_OFF, qs.offset());
        fdPaged.setInt(AttnKernels.ARG_FD_OUT_OFF, os.offset());
        fdPaged.setGlobal(pagedTiling.decodePagedGlobal(nSeq, sp.count()));
        fdPaged.dispatch();
        if (!sp.direct()) {
            combine(nSeq * g.nHead, sp.count(), os.offset());
        }
    }

    /** One query row through the paged decode kernel, whose one sequence sits at page-table row 0. */
    private void decodePagedRow(long qOff, long outOff, int keys) {
        FlashAttnTiling.Split sp = pagedTiling.decodeSplit(keys);
        fdPaged.setInt(AttnKernels.ARG_FD_N_SPLIT, sp.count());
        fdPaged.setInt(AttnKernels.ARG_FD_SPLIT_LEN, sp.length());
        fdPaged.setInt(AttnKernels.ARG_FD_FLAGS, pagedFlags(sp.direct()));
        fdPaged.setInt(AttnKernels.ARG_FDP_N_SEQ, 1);
        fdPaged.setInt(AttnKernels.ARG_FDP_Q_SEQ_STRIDE, 0);
        fdPaged.setInt(AttnKernels.ARG_FDP_OUT_SEQ_STRIDE, 0);
        fdPaged.setInt(AttnKernels.ARG_FD_Q_OFF, (int) qOff);
        fdPaged.setInt(AttnKernels.ARG_FD_OUT_OFF, (int) outOff);
        fdPaged.setGlobal(pagedTiling.decodePagedGlobal(1, sp.count()));
        fdPaged.dispatch();
        if (!sp.direct()) {
            combine(g.nHead, sp.count(), (int) outOff);
        }
    }

    /**
     * The one-sequence page table of the per-row fallback launches, allocated at first use and owned by this
     * mixer ({@code LLM_FlashDecodePaged_*} reads its launch's one sequence at page-table row 0).
     */
    private GpuBuffer focusTable() {
        if (focus == null) {
            focus = backend.allocateBuffer((long) maxPages * Integer.BYTES, true);
            owned.add(focus);
        }
        return focus;
    }

    /** The one-int slot count of the batch-invariant per-row paged launches, allocated at first use. */
    private GpuBuffer focusKvLen() {
        if (focusKvLen == null) {
            focusKvLen = backend.allocateBuffer(Integer.BYTES, true);
            owned.add(focusKvLen);
        }
        return focusKvLen;
    }

    /** The one-int query position of the batch-invariant per-row paged launches, allocated at first use. */
    private GpuBuffer focusQPos() {
        if (focusQPos == null) {
            focusQPos = backend.allocateBuffer(Integer.BYTES, true);
            owned.add(focusQPos);
        }
        return focusQPos;
    }

    /** Writes one little-endian int to element 0 of {@code dst}, ordered on the queue before later launches. */
    private void writeLaunchInt(GpuBuffer dst, int value) {
        byte[] b = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array();
        GpuEvent e = ops.queue().enqueueWriteBuffer(dst, 0, Integer.BYTES, b);
        if (e != null) {
            e.close();
        }
    }

    /**
     * The attention of one sequence's {@code n} rows: one decode launch for one row, else the tiled prefill, or the
     * per-row decode when the plan has no prefill kernel. Batch invariance always runs one decode launch per row
     * over the decode tiling, the exact launches of {@code n} single-row steps, so a row's key split comes from its
     * own key count instead of the tiled call's token tile.
     */
    private void attend(int pos0, int n, ScratchSlot qs, ScratchSlot os) {
        if (n == 1) {
            decodeRow(fd, decodeTiling, pos0, qs.offset(), os.offset());
        } else if (BatchInvariance.enabled()) {
            for (int t = 0; t < n; t++) {
                decodeRow(fd, decodeTiling, pos0 + t, qs.offset() + t * g.nHead * g.dQ,
                        os.offset() + t * g.nHead * g.dV);
            }
        } else if (fa != null) {
            tiled(pos0, n, os);
        } else {
            for (int t = 0; t < n; t++) {
                decodeRow(fdRow, prefillTiling, pos0 + t, qs.offset() + t * g.nHead * g.dQ,
                        os.offset() + t * g.nHead * g.dV);
            }
        }
    }

    /** One query row: {@code LLM_FlashDecode_*} over its key range, then the combine when the range split. */
    private void decodeRow(PreparedDispatch d, FlashAttnTiling t, int pos, int qOff, int outOff) {
        int start = MlaBlocks.kvStart(layout, pos);
        int end = MlaBlocks.kvEnd(layout, pos);
        FlashAttnTiling.Split s = t.decodeSplit(end - start);
        d.setInt(AttnKernels.ARG_FD_N_SPLIT, s.count());
        d.setInt(AttnKernels.ARG_FD_SPLIT_LEN, s.length());
        d.setInt(AttnKernels.ARG_FD_KV_START, start);
        d.setInt(AttnKernels.ARG_FD_KV_END, end);
        d.setInt(AttnKernels.ARG_FD_Q_POS, pos);
        d.setInt(AttnKernels.ARG_FD_FLAGS, flashFlags | (s.direct() ? AttnKernels.FLAG_DIRECT : 0));
        d.setInt(AttnKernels.ARG_FD_Q_OFF, qOff);
        d.setInt(AttnKernels.ARG_FD_OUT_OFF, outOff);
        d.setGlobal(t.decodeGlobal(s.count()));
        d.dispatch();
        if (!s.direct()) {
            combine(g.nHead, s.count(), outOff);
        }
    }

    /** All query rows in one {@code LLM_FlashAttn_*} call, split over keys when the call is small. */
    private void tiled(int pos0, int n, ScratchSlot os) {
        int start = MlaBlocks.kvStart(layout, pos0);
        int end = MlaBlocks.kvEnd(layout, pos0 + n - 1);
        FlashAttnTiling.Split s = prefillTiling.prefillSplit(n, end - start, partialCapacity);
        fa.setInt(AttnKernels.ARG_FA_N_TOKENS, n);
        fa.setInt(AttnKernels.ARG_FA_N_SPLIT, s.count());
        fa.setInt(AttnKernels.ARG_FA_SPLIT_LEN, s.length());
        fa.setInt(AttnKernels.ARG_FA_KV_START, start);
        fa.setInt(AttnKernels.ARG_FA_KV_END, end);
        fa.setInt(AttnKernels.ARG_FA_FLAGS, flashFlags | (s.direct() ? AttnKernels.FLAG_DIRECT : 0));
        fa.setGlobal(prefillTiling.prefillGlobal(n, s.count()));
        fa.dispatch();
        if (!s.direct()) {
            combine(g.headRows(n), s.count(), os.offset());
        }
    }

    private void combine(int nRows, int nSplit, int outOff) {
        fc.setInt(AttnKernels.ARG_FC_N_ROWS, nRows);
        fc.setInt(AttnKernels.ARG_FC_N_SPLIT, nSplit);
        fc.setInt(AttnKernels.ARG_FC_OUT_OFF, outOff);
        fc.setGlobal(FlashAttnTiling.combineGlobal(nRows));
        fc.dispatch();
    }

    // ================================================================== context shift

    /**
     * Applies the per-row RoPE deltas of the memory's last {@code shiftPositions(nKeep, nDiscard)} (which already
     * moved the rows) to this layer's K rows: {@code LLM_KV_RopeShift_*} over the rotary dims of every head, undoing
     * and re-applying the planned Hadamard block when the rows are rotated (llama-kv-cache.cpp:1945-2068). Layers that
     * read another layer's cache, and layers without RoPE, have nothing to shift.
     *
     * @throws UnsupportedOperationException for multi-position RoPE, a cache geometry the shift kernels reject, or a
     *                                       memory that does not shift positions on the device
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        String refusal = shiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("layer " + il + ": context shift unsupported: " + refusal);
        }
        if (!owns || rope == null) {
            return;
        }
        GpuBuffer delta = ((DeviceModelMemory) memory).shiftDelta(kv.group());
        KvDType type = layout.dtypeK();
        int shiftLocal = shiftLocal();
        Object[] args = RopeKernels.kvRopeShiftArgs(type, rope, layout.rows(), g.shiftHeads, g.shiftHeadStride,
                g.shiftRowStride, g.shiftOffs, g.shiftRotK(), hasFf, delta, ff, kv.k());
        long items = RopeKernels.kvRopeShiftWorkItems(type, layout.rows(), g.shiftHeads, rope.nDims(), shiftLocal);
        long global = type.quantized() ? items : RopeKernels.roundUp(items, shiftLocal);
        ops.enqueue(RopeKernels.kvRopeShift(type), new long[]{global}, new long[]{shiftLocal}, args);
    }

    /** Work-group size of the {@code LLM_KV_RopeShift_*} launch: {@code L} capped at {@code LLM_DIMX}. */
    private int shiftLocal() {
        return Math.min(LlmKernelConstants.LLM_DIMX, Integer.highestOneBit(local));
    }

    /** Null when {@link #shiftContext} can run: no own RoPE cache, or a shiftable geometry on device memory. */
    @Override
    public String shiftRefusal() {
        checkBound();
        return shiftRefusalChecked();
    }

    private String shiftRefusalChecked() {
        if (!owns || rope == null) {
            return null;
        }
        if (shiftBlocker != null) {
            return shiftBlocker;
        }
        if (!(memory instanceof DeviceModelMemory)) {
            return "context shift needs DeviceModelMemory, got " + memory.getClass().getSimpleName();
        }
        return null;
    }

    // ================================================================== close

    private void checkBound() {
        if (!bound || closed) {
            throw new IllegalStateException("MLA mixer is not bound or already closed");
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        release();
    }

    private void release() {
        RuntimeException first = Closeables.closeAll(prepared, null);
        prepared.clear();
        first = Closeables.close(headRows, first);
        headRows = null;
        first = Closeables.closeAll(owned, first);
        owned.clear();
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuMlaMixer{layer=" + il + ", " + g + '}';
    }
}

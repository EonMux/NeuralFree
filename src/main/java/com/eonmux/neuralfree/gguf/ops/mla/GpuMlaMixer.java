package com.eonmux.neuralfree.gguf.ops.mla;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.attn.AttnExtKernels;
import com.eonmux.neuralfree.gguf.ops.attn.AttnExtMath;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.attn.FlashAttnTiling;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
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
 *       into {@code kv_lora} values per head; two {@code LLM_Copy2D_F32} build {@code [q_abs | q_pe]}; the K row is
 *       the {@code attn_kv_a_mqa} row itself. Decompressed cache: {@code attn_kv_b} gives {@code [k_nope | v]} per
 *       head; {@code LLM_MLA_KV_Assemble} (one token) or copies (several) build K and V.</li>
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
 * splits and the row offsets of per-head products that exceed the device's grid in one call.
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

    private final List<PreparedDispatch> prepared = new ArrayList<>();
    private final List<GpuBuffer> owned = new ArrayList<>();
    private boolean bound;
    private boolean closed;

    private GpuOps ops;
    private GpuMemory memory;
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

    /** Why the latent cache refuses deltas on {@code attn_kv_b}, which it never multiplies. */
    static final String LATENT_KV_B_REFUSAL = "the latent MLA cache runs the absorbed attn_k_b / attn_v_b instead of"
            + " attn_kv_b; set " + MemoryConfig.PROP_MLA_CACHE
            + "=decompressed to apply a LoRA delta on attn_kv_b";

    private GpuWeight wq;
    private GpuWeight wqA;
    private GpuWeight wqB;
    private GpuWeight wkvA;
    private GpuWeight wkvB;
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
    private PreparedDispatch assemble;
    private PreparedDispatch tempScale;
    private PreparedDispatch storeK;
    private PreparedDispatch storeV;
    private PreparedDispatch fd;
    private PreparedDispatch fdRow;
    private PreparedDispatch fa;
    private PreparedDispatch fc;
    private PreparedDispatch outGate;

    private FlashAttnTiling decodeTiling;
    private FlashAttnTiling prefillTiling;
    private long partialCapacity;

    GpuMlaMixer() {
    }

    /**
     * Why the GPU cannot run this MLA layer, or null: the shared limits of {@link MlaBlocks#unsupportedCommon}, and
     * a decompressed cache over absorbed {@code attn_k_b}/{@code attn_v_b}, whose key decompression multiplies by
     * the transposed {@code attn_k_b} (no typed GPU kernel; {@code LLM_MLA_AbsorbedKVDecomp} reads F32 weights of one
     * token only).
     */
    static String unsupported(LayerSpec layer) {
        String why = MlaBlocks.unsupportedCommon(layer);
        if (why != null) {
            return why;
        }
        MlaSpec m = layer.mla();
        if (m.absorbed() && !m.latentCache()) {
            return "layer " + layer.il() + ": a decompressed MLA cache over absorbed attn_k_b/attn_v_b weights has no "
                    + "GPU kernel; use neuralfree.mla.cache=latent (or auto)";
        }
        return null;
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
        if (!g.latent) {
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
        if (g.latent) {
            deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_KV_B), LATENT_KV_B_REFUSAL);
        } else {
            kvbDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_KV_B), wkvB);
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
        if (g.latent) {
            GpuWeight wkB = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_K_B));
            GpuWeight wvB = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_V_B));
            MlaBlocks.checkShape(wkB.name(), wkB.cols(), wkB.rows(), g.nope, (long) g.kvLora * g.nHead);
            MlaBlocks.checkShape(wvB.name(), wvB.cols(), wvB.rows(), g.kvLora, (long) g.headV * g.nHead);
            requirePlainHeads(wkB);
            requirePlainHeads(wvB);
            idLocal = Math.min(MoeKernels.MAX_LOCAL, local);
            MoeKernels.checkIdLocal(wkB.type(), idLocal);
            MoeKernels.checkIdLocal(wvB.type(), idLocal);
            idChunkTokens = idChunkTokens(ctx.kernels().caps().getMaxWorkGroupCount()[1]);
            GpuBuffer slots = headSlots(ctx, rows);
            absorb = keep(ops.prepare(MoeKernels.matVecId(wkB.type()), MoeKernels.idMatVecGlobal(g.kvLora, rows, idLocal),
                    new long[]{idLocal, 1}, MoeKernels.matVecIdArgs(g.kvLora, g.nope, rows, g.headK, q.offset(),
                            lat.offset(), false, 0, g.nHead, slots, wkB.shard(0), q.buffer(), ph, lat.buffer())));
            expand = keep(ops.prepare(MoeKernels.matVecId(wvB.type()), MoeKernels.idMatVecGlobal(g.headV, rows, idLocal),
                    new long[]{idLocal, 1}, MoeKernels.matVecIdArgs(g.headV, g.kvLora, rows, g.kvLora, lat.offset(),
                            attn.offset(), false, 0, g.nHead, slots, wvB.shard(0), lat.buffer(), ph,
                            attn.buffer())));
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
            storeK = keep(prepareStore(layout.dtypeK(), u, layout.widthK(), ks, kv.k()));
            if (!g.latent) {
                storeV = keep(prepareStore(layout.dtypeV(), u, layout.widthV(), v, kv.v()));
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

    private PreparedDispatch prepareStore(KvDType type, int u, int rowElems, ScratchSlot src, GpuBuffer dst) {
        return ops.prepare(RopeKernels.kvStore(type),
                new long[]{RopeKernels.roundUp(RopeKernels.kvStoreWorkItems(type, u, rowElems), local)},
                new long[]{local}, RopeKernels.kvStoreArgs(type, u, rowElems, rowElems, src.offset(), rowElems, src.buffer(),
                        steps.cellIdx(kv.group()), dst));
    }

    private void prepareAttention(GpuForwardContext ctx, LayerSpec layer, GpuWeights ws, GpuBuffer ph) {
        int smem = ctx.kernels().faSmemFloats();
        int cus = Math.max(1, ctx.kernels().caps().getComputeUnits());
        decodeTiling = FlashAttnTiling.decode(g.nHead, g.nHeadKv, g.dQ, g.dV, layout.rows(), smem, cus);
        prefillTiling = FlashAttnTiling.prefill(g.nHead, g.nHeadKv, g.dQ, g.dV, layout.rows(), smem, cus);
        partialCapacity = decodeTiling.decodePartialFloats();
        if (prefillTiling.plan() == FlashAttnTiling.Plan.TILED) {
            partialCapacity = Math.max(partialCapacity,
                    prefillPartialFloats(prefillTiling, ctx.scratch().ubatch(), layout.rows()));
        } else {
            partialCapacity = Math.max(partialCapacity, prefillTiling.decodePartialFloats());
        }
        GpuBuffer partials = ctx.scratch().partials(MlaBlocks.toInt(partialCapacity, "MLA flash partial floats"));
        ScratchSlot qs = g.latent ? qlat : q;
        ScratchSlot os = g.latent ? lat : attn;
        float softcap = MlaBlocks.softcap(layer);
        GpuBuffer sinks = sinksBuffer(ws, layer);
        GpuBuffer sinkArg = sinks != null ? sinks : ph;
        int hasSinks = sinks != null ? 1 : 0;
        String kernel = AttnKernels.flashDecode(layout.dtypeK());
        fd = keep(ops.prepare(kernel, decodeTiling.decodeGlobal(1), FlashAttnTiling.decodeLocal(),
                decodeArgs(decodeTiling, qs, os, softcap, hasSinks, sinkArg, ph, partials)));
        if (prefillTiling.plan() == FlashAttnTiling.Plan.TILED) {
            fa = keep(ops.prepare(AttnKernels.flashAttn(layout.dtypeK()), prefillTiling.prefillGlobal(1, 1),
                    FlashAttnTiling.prefillLocal(), prefillArgs(qs, os, softcap, hasSinks, sinkArg, ph, partials)));
        } else {
            fdRow = keep(ops.prepare(kernel, prefillTiling.decodeGlobal(1), FlashAttnTiling.decodeLocal(),
                    decodeArgs(prefillTiling, qs, os, softcap, hasSinks, sinkArg, ph, partials)));
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
     * Partial floats the largest split of a tiled prefill needs: the most {@link FlashAttnTiling#prefillSplit} chooses
     * for any call of up to {@code ubatch} tokens over the whole cache, unconstrained by capacity. A smaller capacity
     * only lowers the split count, because {@code prefillSplit} fits its splits to the capacity it is given.
     */
    private static long prefillPartialFloats(FlashAttnTiling p, int ubatch, int nKvMax) {
        long cap = 0;
        for (int n = 1; n <= ubatch; n++) {
            FlashAttnTiling.Split s = p.prefillSplit(n, nKvMax, Long.MAX_VALUE);
            if (!s.direct()) {
                cap = Math.max(cap, FlashAttnTiling.partialFloats((long) n * p.nHead(), s.count(), p.dV()));
            }
        }
        return cap;
    }

    /**
     * {@code LLM_FlashDecode_*} arguments in kernel signature order; the per-step values start as one direct split at
     * row 0.
     */
    private Object[] decodeArgs(FlashAttnTiling t, ScratchSlot qs, ScratchSlot os, float softcap, int hasSinks,
                                GpuBuffer sinks, GpuBuffer ph, GpuBuffer partials) {
        KvWindow w = layout.window();
        Object[] a = new Object[AttnKernels.FD_ARG_COUNT];
        a[AttnKernels.ARG_FD_N_HEAD] = g.nHead;
        a[AttnKernels.ARG_FD_N_HEAD_KV] = g.nHeadKv;
        a[AttnKernels.ARG_FD_D_K] = g.dQ;
        a[AttnKernels.ARG_FD_D_V] = g.dV;
        a[AttnKernels.ARG_FD_G_TILE] = t.gTile();
        a[AttnKernels.ARG_FD_N_SPLIT] = 1;
        a[AttnKernels.ARG_FD_SPLIT_LEN] = t.maxSplitLen();
        a[AttnKernels.ARG_FD_KV_START] = 0;
        a[AttnKernels.ARG_FD_KV_END] = 1;
        a[AttnKernels.ARG_FD_Q_POS] = 0;
        a[AttnKernels.ARG_FD_K_ROW_STRIDE] = g.kRowStride;
        a[AttnKernels.ARG_FD_K_HEAD_STRIDE] = g.kHeadStride;
        a[AttnKernels.ARG_FD_V_ROW_STRIDE] = g.vRowStride;
        a[AttnKernels.ARG_FD_V_HEAD_STRIDE] = g.vHeadStride;
        a[AttnKernels.ARG_FD_SCALE] = g.kqScale;
        a[AttnKernels.ARG_FD_SOFTCAP] = softcap;
        a[AttnKernels.ARG_FD_MAX_BIAS] = 0.0f;
        a[AttnKernels.ARG_FD_CAUSAL] = w.causal() ? 1 : 0;
        a[AttnKernels.ARG_FD_SWA_TYPE] = w.type().code();
        a[AttnKernels.ARG_FD_N_SWA] = w.nSwa();
        a[AttnKernels.ARG_FD_BIAS_MODE] = AttentionSpec.BiasMode.NONE.code();
        a[AttnKernels.ARG_FD_N_BUCKETS] = 0;
        a[AttnKernels.ARG_FD_MAX_DISTANCE] = 0;
        a[AttnKernels.ARG_FD_HAS_SINKS] = hasSinks;
        a[AttnKernels.ARG_FD_DIRECT_OUT] = 1;
        a[AttnKernels.ARG_FD_Q_OFF] = qs.offset();
        a[AttnKernels.ARG_FD_OUT_OFF] = os.offset();
        a[AttnKernels.ARG_FD_BUF_Q] = qs.buffer();
        a[AttnKernels.ARG_FD_BUF_K] = kv.k();
        a[AttnKernels.ARG_FD_BUF_V] = kv.v();
        a[AttnKernels.ARG_FD_BUF_KV_POS] = steps.kvPos(kv.group());
        a[AttnKernels.ARG_FD_BUF_SINKS] = sinks;
        a[AttnKernels.ARG_FD_BUF_REL_BIAS] = ph;
        a[AttnKernels.ARG_FD_BUF_OUT] = os.buffer();
        a[AttnKernels.ARG_FD_BUF_PARTIAL] = partials;
        return a;
    }

    /** {@code LLM_FlashAttn_*} arguments in kernel signature order; the per-call values start as one token, one split. */
    private Object[] prefillArgs(ScratchSlot qs, ScratchSlot os, float softcap, int hasSinks, GpuBuffer sinks,
                                 GpuBuffer ph, GpuBuffer partials) {
        KvWindow w = layout.window();
        Object[] a = new Object[AttnKernels.FA_ARG_COUNT];
        a[AttnKernels.ARG_FA_N_TOKENS] = 1;
        a[AttnKernels.ARG_FA_N_HEAD] = g.nHead;
        a[AttnKernels.ARG_FA_N_HEAD_KV] = g.nHeadKv;
        a[AttnKernels.ARG_FA_D_K] = g.dQ;
        a[AttnKernels.ARG_FA_D_V] = g.dV;
        a[AttnKernels.ARG_FA_BQ] = prefillTiling.bq();
        a[AttnKernels.ARG_FA_BK] = prefillTiling.bk();
        a[AttnKernels.ARG_FA_G_TILE] = prefillTiling.gTile();
        a[AttnKernels.ARG_FA_N_SPLIT] = 1;
        a[AttnKernels.ARG_FA_SPLIT_LEN] = prefillTiling.bk();
        a[AttnKernels.ARG_FA_KV_START] = 0;
        a[AttnKernels.ARG_FA_KV_END] = 1;
        a[AttnKernels.ARG_FA_Q_ROW_STRIDE] = g.nHead * g.dQ;
        a[AttnKernels.ARG_FA_OUT_ROW_STRIDE] = g.nHead * g.dV;
        a[AttnKernels.ARG_FA_K_ROW_STRIDE] = g.kRowStride;
        a[AttnKernels.ARG_FA_K_HEAD_STRIDE] = g.kHeadStride;
        a[AttnKernels.ARG_FA_V_ROW_STRIDE] = g.vRowStride;
        a[AttnKernels.ARG_FA_V_HEAD_STRIDE] = g.vHeadStride;
        a[AttnKernels.ARG_FA_SCALE] = g.kqScale;
        a[AttnKernels.ARG_FA_SOFTCAP] = softcap;
        a[AttnKernels.ARG_FA_MAX_BIAS] = 0.0f;
        a[AttnKernels.ARG_FA_CAUSAL] = w.causal() ? 1 : 0;
        a[AttnKernels.ARG_FA_SWA_TYPE] = w.type().code();
        a[AttnKernels.ARG_FA_N_SWA] = w.nSwa();
        a[AttnKernels.ARG_FA_BIAS_MODE] = AttentionSpec.BiasMode.NONE.code();
        a[AttnKernels.ARG_FA_N_BUCKETS] = 0;
        a[AttnKernels.ARG_FA_MAX_DISTANCE] = 0;
        a[AttnKernels.ARG_FA_HAS_SINKS] = hasSinks;
        a[AttnKernels.ARG_FA_DIRECT_OUT] = 1;
        a[AttnKernels.ARG_FA_Q_OFF] = qs.offset();
        a[AttnKernels.ARG_FA_OUT_OFF] = os.offset();
        a[AttnKernels.ARG_FA_BUF_Q] = qs.buffer();
        a[AttnKernels.ARG_FA_BUF_K] = kv.k();
        a[AttnKernels.ARG_FA_BUF_V] = kv.v();
        a[AttnKernels.ARG_FA_BUF_Q_POS] = steps.qPos();
        a[AttnKernels.ARG_FA_BUF_KV_POS] = steps.kvPos(kv.group());
        a[AttnKernels.ARG_FA_BUF_SINKS] = sinks;
        a[AttnKernels.ARG_FA_BUF_REL_BIAS] = ph;
        a[AttnKernels.ARG_FA_BUF_OUT] = os.buffer();
        a[AttnKernels.ARG_FA_BUF_PARTIAL] = partials;
        return a;
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
     * {@code attn_k_b} / {@code attn_v_b} enter plain {@code ggml_mul_mat} products (deepseek2.cpp:120,
     * llama-graph.cpp build_attn {@code v_mla}): one binding, no input Hadamard and no tensor scale, which the per-head
     * {@code {T}_MatVecId} would not apply.
     */
    private static void requirePlainHeads(GpuWeight w) {
        requireOneBinding(w);
        if (w.hadamard() != null || w.hasScale()) {
            throw new UnsupportedOperationException(w.name() + " carries an input Hadamard or a tensor scale; the "
                    + "per-head MLA products take plain weights");
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
        int rows = g.headRows(n);
        runPerHead(absorb, n, g.kvLora, g.headK, q.offset(), lat.offset());
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
        attend(pos0, n, qlat, lat);
        if (rotKSpec != null) {
            ops.hadamard(rotKSpec, lat.buffer(), lat.offset(), lat.buffer(), lat.offset(), rows, g.kvLora);
        }
        runPerHead(expand, n, g.headV, g.kvLora, lat.offset(), attn.offset());
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

    /** {@code attn_kv_b} decompression, per-head K/V, attention over the per-head cache. */
    private void decompressedStep(int pos0, int n) {
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
            if (g.rope > 0) {
                for (int h = 0; h < g.nHead; h++) {
                    ops.copy2d(kva.buffer(), kva.offset() + g.kvLora, g.kvaWidth, k.buffer(),
                            k.offset() + h * g.headK + g.nope, g.qWidth, n, g.rope);
                }
            }
        }
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
        attend(pos0, n, q, attn);
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

    private void attend(int pos0, int n, ScratchSlot qs, ScratchSlot os) {
        if (n == 1) {
            decodeRow(fd, decodeTiling, pos0, qs.offset(), os.offset());
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
        d.setInt(AttnKernels.ARG_FD_DIRECT_OUT, s.direct() ? 1 : 0);
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
        fa.setInt(AttnKernels.ARG_FA_DIRECT_OUT, s.direct() ? 1 : 0);
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

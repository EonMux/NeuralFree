package com.eonmux.neuralfree.gguf.ops.encdec;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.attn.FlashAttnTiling;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.ArrayList;
import java.util.List;

/**
 * T5 decoder cross-attention of a {@code CROSS_ATTENTION_T5} layer on the GPU (llama.cpp src/models/t5.cpp:164-182; the
 * block contract is in {@link EncDecBlocks}): {@code out = wo_cross @ attn(wq_cross @ in, cross K/V)} with
 * {@code in = cross_attn_norm(h)} supplied by the executor, attending bidirectionally and without position bias over
 * the cross cache rows {@code [0, nEnc)} that {@link #encode} filled from the encoder output.
 *
 * <p>Per step, in order: {@code wq_cross} into the {@code Q} slot; the K rotation of Q when the cross cache is
 * quantized and rotated (llama-graph.cpp:3225-3231); flash attention ({@code causal = 0},
 * {@code bias_mode = 0}, {@code kv_pos} the identity rows, {@code kv_end = nEnc}) into {@code ATTN_OUT}, merged by
 * {@code LLM_FlashAttn_Combine} when split; the inverse V rotation (llama-graph.cpp:3270); {@code wo_cross} into
 * {@code out}. Every kernel argument except the per-step scalars is bound at {@link #bind}; a step allocates no device
 * memory and never waits on the host.</p>
 */
public final class GpuCrossAttentionMixer implements GpuMixer {

    private LayerSpec layer;
    private GpuOps ops;
    private int local;
    private int ubatch;
    private AttentionSpec cross;

    private GpuWeight wqCross;
    private GpuWeight wkCross;
    private GpuWeight wvCross;
    private GpuWeight woCross;
    /** LoRA deltas of the cross-attention projections ({@code build_lora_mm}, t5.cpp:164-182). */
    private GpuProjectionDeltas.Site qDeltas;
    private GpuProjectionDeltas.Site kDeltas;
    private GpuProjectionDeltas.Site vDeltas;
    private GpuProjectionDeltas.Site oDeltas;

    private ScratchSlot q;
    private ScratchSlot k;
    private ScratchSlot v;
    private ScratchSlot attnOut;

    private GpuKvLayer crossKv;
    private int identityLength;
    private HadamardSpec rotK;
    private HadamardSpec rotV;

    private PreparedDispatch storeK;
    private PreparedDispatch storeV;
    private PreparedDispatch decode;
    private PreparedDispatch tiled;
    private PreparedDispatch combine;
    private FlashAttnTiling decodePlan;
    private FlashAttnTiling prefillPlan;
    private long partialCapacity;
    private final List<PreparedDispatch> prepared = new ArrayList<>();

    private int nEnc;
    private boolean closed;

    /** Unbound mixer; {@link #bind} attaches it to its layer. */
    public GpuCrossAttentionMixer() {
    }

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        if (this.layer != null) {
            throw new IllegalStateException("T5 cross-attention mixer is already bound to layer "
                    + EncDecBlocks.describe(this.layer));
        }
        EncDecBlocks.checkLayer(layer);
        this.layer = layer;
        this.ops = ctx.ops();
        this.local = ops.local();
        this.cross = layer.crossAttention();
        try {
            bindWeights(ctx.weightSource().weightsFor(layer.weightLayer()), layer.residual().streamWidth());
            bindDeltas(ctx.projectionDeltas());
            bindScratch(ctx.scratch());
            GpuStepInputs steps = bindMemory(ctx.memory());
            bindDispatches(ctx, (DeviceStepInputs) steps);
        } catch (RuntimeException e) {
            close();
            throw e;
        }
    }

    private void bindWeights(GpuWeights ws, int width) {
        wqCross = matrix(ws, EncDecBlocks.W_CROSS_Q, width, cross.qDim());
        wkCross = matrix(ws, EncDecBlocks.W_CROSS_K, width, cross.kDim());
        wvCross = matrix(ws, EncDecBlocks.W_CROSS_V, width, cross.vDim());
        woCross = matrix(ws, EncDecBlocks.W_CROSS_O, cross.outDim(), width);
    }

    private void bindDeltas(GpuProjectionDeltas deltas) {
        qDeltas = deltas.serve(layer, EncDecBlocks.W_CROSS_Q, wqCross);
        kDeltas = deltas.serve(layer, EncDecBlocks.W_CROSS_K, wkCross);
        vDeltas = deltas.serve(layer, EncDecBlocks.W_CROSS_V, wvCross);
        oDeltas = deltas.serve(layer, EncDecBlocks.W_CROSS_O, woCross);
    }

    private GpuWeight matrix(GpuWeights ws, String suffix, long cols, long rows) {
        GpuWeight w = ws.layer(layer, suffix);
        EncDecBlocks.checkMatrix(layer, w.name(), w.cols(), w.rows(), cols, rows);
        return w;
    }

    private void bindScratch(GpuScratch scratch) {
        ubatch = scratch.ubatch();
        long ub = ubatch;
        q = scratch.slot(ScratchSlots.Q).require(ub * cross.qDim());
        k = scratch.slot(ScratchSlots.K).require(ub * cross.kDim());
        v = scratch.slot(ScratchSlots.V).require(ub * cross.vDim());
        attnOut = scratch.slot(ScratchSlots.ATTN_OUT).require(ub * cross.outDim());
    }

    private GpuStepInputs bindMemory(GpuMemory memory) {
        crossKv = memory.crossKv(layer.il());
        if (crossKv == null) {
            throw new IllegalStateException("T5 decoder layer " + EncDecBlocks.describe(layer)
                    + " needs a cross-attention KV cache, the memory has none");
        }
        KvLayout l = crossKv.layout();
        if (l.widthK() != cross.kDim() || l.widthV() != cross.vDim() || l.vAliasesK()) {
            throw new IllegalStateException("T5 decoder layer " + EncDecBlocks.describe(layer) + ": cross cache rows "
                    + l.widthK() + "/" + l.widthV() + " do not hold K/V of " + cross.kDim() + "/" + cross.vDim());
        }
        GpuStepInputs steps = memory.stepInputs();
        if (!(steps instanceof DeviceStepInputs)) {
            throw new IllegalStateException("T5 cross-attention addresses its cache with DeviceStepInputs.identityRows();"
                    + " the memory's step inputs are " + steps.getClass().getName());
        }
        identityLength = ((DeviceStepInputs) steps).identityLength();
        rotK = rotation(crossKv.rotation().nRotK());
        rotV = rotation(crossKv.rotation().nRotV());
        return steps;
    }

    private static HadamardSpec rotation(int block) {
        return block > 0 ? HadamardSpec.rotation(block) : null;
    }

    private void bindDispatches(GpuForwardContext ctx, DeviceStepInputs steps) {
        int smem = ctx.kernels().faSmemFloats();
        int cus = Math.max(1, ctx.kernels().caps().getComputeUnits());
        KvLayout l = crossKv.layout();
        KvDType typeK = l.dtypeK();
        KvDType typeV = l.dtypeV();
        decodePlan = FlashAttnTiling.decode(cross.nHead(), cross.nHeadKv(), cross.headDimK(), cross.headDimV(),
                l.rows(), smem, cus);
        prefillPlan = FlashAttnTiling.prefill(cross.nHead(), cross.nHeadKv(), cross.headDimK(), cross.headDimV(),
                l.rows(), smem, cus);
        partialCapacity = decodePlan.decodePartialFloats();
        GpuBuffer partials = ctx.scratch().partials(EncDecBlocks.toInt(partialCapacity, "flash partial floats"));
        GpuBuffer ro = ctx.placeholders().readOnlyFloats();
        GpuBuffer identity = steps.identityRows();

        storeK = prepareStore(typeK, cross.kDim(), k, identity, crossKv.k(), l.widthK());
        storeV = prepareStore(typeV, cross.vDim(), v, identity, crossKv.v(), l.widthV());

        int swa = SwaType.NONE.code();
        int noBias = AttentionSpec.BiasMode.NONE.code();
        float scale = cross.scoreScale();
        decode = track(ops.prepare(AttnKernels.flashDecode(typeK, typeV), decodePlan.decodeGlobal(1),
                FlashAttnTiling.decodeLocal(), AttnKernels.withKvTypes(new Object[]{
                cross.nHead(), cross.nHeadKv(), cross.headDimK(), cross.headDimV(), decodePlan.gTile(),
                1, decodePlan.maxSplitLen(), 0, 1, 0,
                l.widthK(), cross.headDimK(), l.widthV(), cross.headDimV(),
                scale, 0f, 0f,
                0, swa, 0,
                noBias, 0, 0, 0, 1, q.offset(), attnOut.offset(),
                q.buffer(), crossKv.k(), crossKv.v(), identity, ro, ro, attnOut.buffer(), partials}, typeK, typeV)));
        if (prefillPlan.plan() == FlashAttnTiling.Plan.TILED) {
            tiled = track(ops.prepare(AttnKernels.flashAttn(typeK, typeV), prefillPlan.prefillGlobal(1, 1),
                    FlashAttnTiling.prefillLocal(), AttnKernels.withKvTypes(new Object[]{
                    1, cross.nHead(), cross.nHeadKv(), cross.headDimK(), cross.headDimV(),
                    prefillPlan.bq(), prefillPlan.bk(), prefillPlan.gTile(),
                    1, prefillPlan.bk(), 0, 1,
                    cross.qDim(), cross.outDim(),
                    l.widthK(), cross.headDimK(), l.widthV(), cross.headDimV(),
                    scale, 0f, 0f,
                    0, swa, 0,
                    noBias, 0, 0, 0, 1, q.offset(), attnOut.offset(),
                    q.buffer(), crossKv.k(), crossKv.v(), steps.qPos(), identity, ro, ro, attnOut.buffer(),
                    partials}, typeK, typeV)));
        }
        combine = track(ops.prepare(AttnKernels.FLASH_COMBINE, FlashAttnTiling.combineGlobal(cross.nHead()),
                FlashAttnTiling.combineLocal(),
                cross.nHead(), cross.headDimV(), 1, cross.nHead(), 0, attnOut.offset(),
                partials, ro, attnOut.buffer()));
    }

    private PreparedDispatch prepareStore(KvDType type, int rowElems, ScratchSlot src, GpuBuffer cellIdx,
                                          GpuBuffer dst, int dstRowStride) {
        Object[] args = RopeKernels.kvStoreArgs(type, 1, rowElems, rowElems, src.offset(), dstRowStride,
                src.buffer(), cellIdx, dst);
        return track(ops.prepare(RopeKernels.kvStore(type), storeGlobal(type, 1, rowElems), new long[]{local},
                args));
    }

    private long[] storeGlobal(KvDType type, int nTokens, int rowElems) {
        return new long[]{RopeKernels.roundUp(RopeKernels.kvStoreWorkItems(type, nTokens, rowElems), local)};
    }

    private PreparedDispatch track(PreparedDispatch d) {
        prepared.add(d);
        return d;
    }

    // ================================================================== encoder output

    /**
     * Computes this layer's cross K/V from the encoder output and stores them in rows {@code [0, nEnc)} of the cross
     * cache (llama.cpp t5.cpp:170-174: {@code wk_cross @ embd_enc}, {@code wv_cross @ embd_enc}). Called once per
     * prompt, after the encoder ran and before the first decoder step; a later call replaces the previous encoder
     * output.
     *
     * @param enc    encoder output rows {@code enc[encOff + r*width ..]} (after {@code enc.output_norm})
     * @param nEnc   encoder tokens; llama.cpp encodes in one ubatch (llama-context.cpp:1628), so at most
     *               {@code ubatch} and the cross cache rows
     * @throws IllegalStateException    before {@link #bind} or after {@link #close()}
     * @throws IllegalArgumentException when {@code nEnc} is outside {@code [1, ubatch]}, exceeds the cross cache or
     *                                  the identity rows, or {@code enc} is the K or V scratch slot
     */
    public void encode(GpuBuffer enc, int encOff, int nEnc) {
        ensureBound();
        int rows = crossKv.layout().rows();
        if (nEnc < 1 || nEnc > ubatch || nEnc > rows || nEnc > identityLength) {
            throw new IllegalArgumentException("T5 decoder layer " + EncDecBlocks.describe(layer) + ": encoder output of "
                    + nEnc + " tokens must be in [1, " + Math.min(Math.min(ubatch, rows), identityLength) + "] (ubatch "
                    + ubatch + ", cross cache rows " + rows + ", identity rows " + identityLength + ")");
        }
        if (enc == k.buffer() || enc == v.buffer()) {
            throw new IllegalArgumentException("encoder output must not live in the K or V scratch slot");
        }
        ops.matmul(wkCross, enc, encOff, k.buffer(), k.offset(), nEnc);
        kDeltas.applyRows(nEnc, enc, encOff, k.buffer(), k.offset());
        ops.matmul(wvCross, enc, encOff, v.buffer(), v.offset(), nEnc);
        vDeltas.applyRows(nEnc, enc, encOff, v.buffer(), v.offset());
        rotate(rotK, k, nEnc, cross.kDim());
        rotate(rotV, v, nEnc, cross.vDim());
        KvLayout l = crossKv.layout();
        store(storeK, l.dtypeK(), cross.kDim(), nEnc);
        store(storeV, l.dtypeV(), cross.vDim(), nEnc);
        this.nEnc = nEnc;
    }

    /** Encoder tokens of the current cross cache; 0 before the first {@link #encode}. */
    public int encodedTokens() {
        return nEnc;
    }

    private void store(PreparedDispatch d, KvDType type, int rowElems, int n) {
        d.setInt(RopeKernels.ARG_N_TOKENS, n);
        d.setGlobal(storeGlobal(type, n, rowElems));
        d.dispatch();
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        step(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        step(pos0, nTokens, in, out);
    }

    /** {@code out = wo_cross @ attn(wq_cross @ in, cross cache)} (t5.cpp:164-182). */
    private void step(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        ensureBound();
        if (nEnc == 0) {
            throw new IllegalStateException("T5 decoder layer " + EncDecBlocks.describe(layer)
                    + " runs before encode() stored the encoder output");
        }
        if (pos0 < 0 || n < 1 || n > ubatch) {
            throw new IllegalArgumentException("T5 decoder step needs pos0 >= 0 and nTokens in [1, " + ubatch + "], got "
                    + pos0 + "/" + n);
        }
        long rows = (long) n * layer.residual().streamWidth();
        in.require(rows);
        out.require(rows);
        checkViews(in, out);
        project(wqCross, qDeltas, in, q, n);
        rotate(rotK, q, n, cross.qDim());
        attend(pos0, n);
        rotate(rotV, attnOut, n, cross.outDim());
        project(woCross, oDeltas, attnOut, out, n);
    }

    private void checkViews(ScratchSlot in, ScratchSlot out) {
        GpuBuffer i = in.buffer();
        GpuBuffer o = out.buffer();
        if (i == o || i == q.buffer() || o == q.buffer() || i == attnOut.buffer() || o == attnOut.buffer()) {
            throw new IllegalArgumentException("T5 cross-attention input " + in.name() + " and output " + out.name()
                    + " must be distinct from each other and from the Q and ATTN_OUT slots");
        }
    }

    private void project(GpuWeight w, GpuProjectionDeltas.Site deltas, ScratchSlot src, ScratchSlot dst, int n) {
        if (n == 1) {
            ops.matvec(w, src.buffer(), src.offset(), dst.buffer(), dst.offset(), 1);
        } else {
            ops.matmul(w, src.buffer(), src.offset(), dst.buffer(), dst.offset(), n);
        }
        deltas.applyRows(n, src.buffer(), src.offset(), dst.buffer(), dst.offset());
    }

    /** In-place block rotation of {@code n} rows of {@code rowWidth}; nothing when the cache is not rotated. */
    private void rotate(HadamardSpec h, ScratchSlot s, int n, int rowWidth) {
        if (h != null) {
            ops.hadamard(h, s.buffer(), s.offset(), s.buffer(), s.offset(), n, rowWidth);
        }
    }

    /**
     * Flash attention of the {@code n} query rows over the cross rows {@code [0, nEnc)}: the tiled kernel for several
     * rows when a tiling fits, else the decode kernel once per row; split ranges merged by the combine.
     */
    private void attend(int pos0, int n) {
        if (n == 1 || tiled == null) {
            for (int t = 0; t < n; t++) {
                decodeRow(t, pos0 + t);
            }
            return;
        }
        FlashAttnTiling.Split s = prefillPlan.prefillSplit(n, nEnc, partialCapacity);
        tiled.setInt(AttnKernels.ARG_FA_N_TOKENS, n);
        tiled.setInt(AttnKernels.ARG_FA_N_SPLIT, s.count());
        tiled.setInt(AttnKernels.ARG_FA_SPLIT_LEN, s.length());
        tiled.setInt(AttnKernels.ARG_FA_KV_END, nEnc);
        tiled.setInt(AttnKernels.ARG_FA_DIRECT_OUT, s.direct() ? 1 : 0);
        tiled.setGlobal(prefillPlan.prefillGlobal(n, s.count()));
        tiled.dispatch();
        if (!s.direct()) {
            combine(n * cross.nHead(), s.count(), attnOut.offset());
        }
    }

    /** Query row {@code t} of the step (position {@code qPos}, unused by the unmasked kernel) through decode. */
    private void decodeRow(int t, int qPos) {
        FlashAttnTiling.Split s = decodePlan.decodeSplit(nEnc);
        int outOff = attnOut.offset() + t * cross.outDim();
        decode.setInt(AttnKernels.ARG_FD_N_SPLIT, s.count());
        decode.setInt(AttnKernels.ARG_FD_SPLIT_LEN, s.length());
        decode.setInt(AttnKernels.ARG_FD_KV_END, nEnc);
        decode.setInt(AttnKernels.ARG_FD_Q_POS, qPos);
        decode.setInt(AttnKernels.ARG_FD_DIRECT_OUT, s.direct() ? 1 : 0);
        decode.setInt(AttnKernels.ARG_FD_Q_OFF, q.offset() + t * cross.qDim());
        decode.setInt(AttnKernels.ARG_FD_OUT_OFF, outOff);
        decode.setGlobal(decodePlan.decodeGlobal(s.count()));
        decode.dispatch();
        if (!s.direct()) {
            combine(cross.nHead(), s.count(), outOff);
        }
    }

    private void combine(int rows, int nSplit, int outOff) {
        combine.setInt(AttnKernels.ARG_FC_N_ROWS, rows);
        combine.setInt(AttnKernels.ARG_FC_N_SPLIT, nSplit);
        combine.setInt(AttnKernels.ARG_FC_OUT_OFF, outOff);
        combine.setGlobal(FlashAttnTiling.combineGlobal(rows));
        combine.dispatch();
    }

    /**
     * Context shift: the cross cache holds the encoder output, which has no decoder positions, so nothing moves (the
     * decoder self-attention cache is shifted by its own mixer and the memory).
     *
     * @throws IllegalArgumentException for a negative range
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        ensureBound();
        if (nKeep < 0 || nDiscard < 0) {
            throw new IllegalArgumentException("context shift needs nKeep >= 0 and nDiscard >= 0, got " + nKeep + "/"
                    + nDiscard);
        }
    }

    /** The cross cache holds no decoder positions, so the shift never refuses here. */
    @Override
    public String shiftRefusal() {
        ensureBound();
        return null;
    }

    // ================================================================== lifecycle

    private void ensureBound() {
        if (closed) {
            throw new IllegalStateException("T5 cross-attention mixer is closed");
        }
        if (layer == null) {
            throw new IllegalStateException("T5 cross-attention mixer is not bound");
        }
    }

    /** Waits for the queue, then releases the prepared dispatches. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (ops != null) {
            ops.queue().finish();
        }
        for (PreparedDispatch d : prepared) {
            d.close();
        }
        prepared.clear();
    }

    @Override
    public String toString() {
        return "GpuCrossAttentionMixer{" + (layer == null ? "unbound" : EncDecBlocks.describe(layer))
                + ", nEnc=" + nEnc + (closed ? ", closed" : "") + '}';
    }
}

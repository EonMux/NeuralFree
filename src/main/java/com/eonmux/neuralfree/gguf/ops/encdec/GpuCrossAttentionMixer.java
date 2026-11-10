package com.eonmux.neuralfree.gguf.ops.encdec;

import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
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
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.attn.FlashAttnTiling;
import com.eonmux.neuralfree.gguf.ops.attn.FlashBindGuard;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.ArrayList;
import java.util.Arrays;
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
 * {@code out}. Every kernel argument except the per-step scalars and the cross cache of the step's sequence is bound
 * at {@link #bind}; a step allocates no device memory and never waits on the host.</p>
 *
 * <p>The memory keeps one cross cache per sequence ({@code LayerMemoryPlan.crossStreams()}: the paged memory's
 * sequence count, {@code GpuMemory.crossKv(il, seq)}): {@link #encode(int, GpuBuffer, int, int)} fills sequence
 * {@code seq}'s cache from its own source, and a paged batch ({@link #batch}) attends every segment to its own
 * sequence's encoder rows only, as llama.cpp's cross mask keeps each decoder row on the encoder rows of its sequence
 * (llama-graph.cpp:1093-1105), the twin of {@code CpuCrossAttentionMixer.batch}. The contiguous calls run sequence
 * 0.</p>
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

    /** Sequence {@code s}'s cross cache at index {@code s}. */
    private GpuKvLayer[] crossKv;
    private GpuStepInputs steps;
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

    /** Encoder tokens of each sequence's cross cache; 0 before its first {@link #encode}. */
    private int[] nEnc;
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
        int il = layer.il();
        if (memory.crossKv(il) == null) {
            throw new IllegalStateException("T5 decoder layer " + EncDecBlocks.describe(layer)
                    + " needs a cross-attention KV cache, the memory has none");
        }
        int streams = Math.max(1, memory.plan().layer(il).crossStreams());
        crossKv = new GpuKvLayer[streams];
        for (int seq = 0; seq < streams; seq++) {
            crossKv[seq] = checkCrossCache(memory.crossKv(il, seq), crossKv[0]);
        }
        nEnc = new int[streams];
        GpuStepInputs steps = memory.stepInputs();
        this.steps = steps;
        if (!(steps instanceof DeviceStepInputs)) {
            throw new IllegalStateException("T5 cross-attention addresses its cache with DeviceStepInputs.identityRows();"
                    + " the memory's step inputs are " + steps.getClass().getName());
        }
        identityLength = ((DeviceStepInputs) steps).identityLength();
        rotK = rotation(crossKv[0].rotation().nRotK());
        rotV = rotation(crossKv[0].rotation().nRotV());
        return steps;
    }

    /** A cross cache holding this layer's K/V rows in the layout and rotation of sequence 0's ({@code first}). */
    private GpuKvLayer checkCrossCache(GpuKvLayer c, GpuKvLayer first) {
        KvLayout l = c.layout();
        if (l.widthK() != cross.kDim() || l.widthV() != cross.vDim() || l.vAliasesK()) {
            throw new IllegalStateException("T5 decoder layer " + EncDecBlocks.describe(layer) + ": cross cache rows "
                    + l.widthK() + "/" + l.widthV() + " do not hold K/V of " + cross.kDim() + "/" + cross.vDim());
        }
        if (first != null && (l.rows() != first.layout().rows() || l.dtypeK() != first.layout().dtypeK()
                || l.dtypeV() != first.layout().dtypeV()
                || c.rotation().nRotK() != first.rotation().nRotK()
                || c.rotation().nRotV() != first.rotation().nRotV())) {
            throw new IllegalStateException("T5 decoder layer " + EncDecBlocks.describe(layer) + ": the cross caches"
                    + " of the sequences differ in layout or rotation (" + l + " vs " + first.layout() + ")");
        }
        return c;
    }

    private static HadamardSpec rotation(int block) {
        return block > 0 ? HadamardSpec.rotation(block) : null;
    }

    private void bindDispatches(GpuForwardContext ctx, DeviceStepInputs steps) {
        FlashBindGuard.requireServable(ctx.kernels(), "T5 decoder layer " + EncDecBlocks.describe(layer)
                        + " (cross attention)", cross.nHead(), cross.nHeadKv(), cross.headDimK(), cross.headDimV());
        int smem = ctx.kernels().faSmemFloats();
        int cus = FlashAttnTiling.planningUnits(ctx.kernels().caps());
        KvLayout l = crossKv[0].layout();
        KvDType typeK = l.dtypeK();
        KvDType typeV = l.dtypeV();
        decodePlan = FlashAttnTiling.decode(cross.nHead(), cross.nHeadKv(), cross.headDimK(), cross.headDimV(),
                l.rows(), smem, cus,
                FlashAttnTiling.decodeFastPath(ctx.kernels().caps().hasFeature(DeviceFeature.SUBGROUP_SHUFFLE),
                        typeK, typeV, cross.headDimK(), cross.headDimV()));
        prefillPlan = FlashAttnTiling.prefill(cross.nHead(), cross.nHeadKv(), cross.headDimK(), cross.headDimV(),
                l.rows(), smem, cus);
        partialCapacity = decodePlan.decodePartialFloats();
        GpuBuffer partials = ctx.scratch().partials(EncDecBlocks.toInt(partialCapacity, "flash partial floats"));
        GpuBuffer ro = ctx.placeholders().readOnlyFloats();
        GpuBuffer identity = steps.identityRows();

        storeK = prepareStore(typeK, cross.kDim(), k, identity, crossKv[0].k(), l.widthK());
        storeV = prepareStore(typeV, cross.vDim(), v, identity, crossKv[0].v(), l.widthV());

        int swa = SwaType.NONE.code();
        int noBias = AttentionSpec.BiasMode.NONE.code();
        float scale = cross.scoreScale();
        int directFlags = AttnKernels.attnFlags(false, false, true);
        decode = track(ops.prepare(AttnKernels.flashDecode(typeK, typeV), decodePlan.decodeGlobal(1),
                FlashAttnTiling.decodeLocal(), AttnKernels.withKvTypes(new Object[]{
                cross.nHead(), cross.nHeadKv(), cross.headDimK(), cross.headDimV(), decodePlan.gTile(),
                1, decodePlan.maxSplitLen(), 0, 1, 0,
                l.widthK(), cross.headDimK(), l.widthV(), cross.headDimV(),
                scale, 0f, 0f,
                directFlags, swa, 0,
                noBias, 0, 0, q.offset(), attnOut.offset(),
                q.buffer(), crossKv[0].k(), crossKv[0].v(), identity, ro, ro, attnOut.buffer(), partials,
                ctx.placeholders().kvScalesOnes(), ro}, typeK, typeV)));
        if (prefillPlan.plan() == FlashAttnTiling.Plan.TILED) {
            tiled = track(ops.prepare(AttnKernels.flashAttn(typeK, typeV), prefillPlan.prefillGlobal(1, 1),
                    FlashAttnTiling.prefillLocal(), AttnKernels.withKvTypes(new Object[]{
                    1, cross.nHead(), cross.nHeadKv(), cross.headDimK(), cross.headDimV(),
                    prefillPlan.bq(), prefillPlan.bk(), prefillPlan.gTile(),
                    1, prefillPlan.bk(), 0, 1,
                    cross.qDim(), cross.outDim(),
                    l.widthK(), cross.headDimK(), l.widthV(), cross.headDimV(),
                    scale, 0f, 0f,
                    directFlags, swa, 0,
                    noBias, 0, 0, q.offset(), attnOut.offset(),
                    q.buffer(), crossKv[0].k(), crossKv[0].v(), steps.qPos(), identity, ro, ro, attnOut.buffer(),
                    partials, ctx.placeholders().kvScalesOnes(), ro}, typeK, typeV)));
        }
        combine = track(ops.prepare(AttnKernels.FLASH_COMBINE, FlashAttnTiling.combineGlobal(cross.nHead()),
                FlashAttnTiling.combineLocal(),
                cross.nHead(), cross.headDimV(), 1, cross.nHead(), 0, attnOut.offset(),
                partials, ro, attnOut.buffer()));
    }

    private PreparedDispatch prepareStore(KvDType type, int rowElems, ScratchSlot src, GpuBuffer cellIdx,
                                          GpuBuffer dst, int dstRowStride) {
        Object[] args = RopeKernels.kvStoreArgs(type, 1, rowElems, rowElems, src.offset(), dstRowStride, rowElems,
                src.buffer(), cellIdx, ops.placeholders().kvScalesOnes(), dst);
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

    /** {@link #encode(int, GpuBuffer, int, int)} into sequence 0's cross cache, the one the contiguous calls read. */
    public void encode(GpuBuffer enc, int encOff, int nEnc) {
        encode(0, enc, encOff, nEnc);
    }

    /**
     * Computes this layer's cross K/V from the encoder output of sequence {@code seq}'s source and stores them in rows
     * {@code [0, nEnc)} of that sequence's cross cache (llama.cpp t5.cpp:170-174: {@code wk_cross @ embd_enc},
     * {@code wv_cross @ embd_enc}). Called once per source, after the encoder ran and before the sequence's first
     * decoder step; a later call replaces the sequence's previous encoder output and leaves the other sequences'
     * caches alone.
     *
     * @param seq    the sequence whose cross cache takes the source, in {@code [0, crossStreams)}
     * @param enc    encoder output rows {@code enc[encOff + r*width ..]} (after {@code enc.output_norm})
     * @param nEnc   encoder tokens; llama.cpp encodes in one ubatch (llama-context.cpp:1628), so at most
     *               {@code ubatch} and the cross cache rows
     * @throws IllegalStateException     before {@link #bind} or after {@link #close()}
     * @throws IndexOutOfBoundsException when the memory keeps no cross cache for {@code seq}
     * @throws IllegalArgumentException  when {@code nEnc} is outside {@code [1, ubatch]}, exceeds the cross cache or
     *                                   the identity rows, or {@code enc} is the K or V scratch slot
     */
    public void encode(int seq, GpuBuffer enc, int encOff, int nEnc) {
        ensureBound();
        GpuKvLayer cache = cache(seq);
        int rows = cache.layout().rows();
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
        KvLayout l = cache.layout();
        storeK.setBuffer(RopeKernels.kvStoreDstArg(l.dtypeK()), cache.k());
        storeV.setBuffer(RopeKernels.kvStoreDstArg(l.dtypeV()), cache.v());
        store(storeK, l.dtypeK(), cross.kDim(), nEnc);
        store(storeV, l.dtypeV(), cross.vDim(), nEnc);
        this.nEnc[seq] = nEnc;
    }

    /** Encoder tokens of sequence 0's cross cache; 0 before its first {@link #encode}. */
    public int encodedTokens() {
        return encodedTokens(0);
    }

    /**
     * Encoder tokens of sequence {@code seq}'s cross cache; 0 before its first {@link #encode}.
     *
     * @throws IndexOutOfBoundsException when the memory keeps no cross cache for {@code seq}
     */
    public int encodedTokens(int seq) {
        ensureBound();
        cache(seq);
        return nEnc[seq];
    }

    private GpuKvLayer cache(int seq) {
        if (seq < 0 || seq >= crossKv.length) {
            throw new IndexOutOfBoundsException("T5 decoder layer " + EncDecBlocks.describe(layer) + " keeps "
                    + crossKv.length + " cross caches, not one for sequence " + seq);
        }
        return crossKv[seq];
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

    /**
     * One step over the rows of {@code s}: the contiguous step when the memory runs no paged step, else the paged
     * step: the projections run over the whole step and every segment attends to its own sequence's cross cache, so
     * each row computes what it computes run one sequence at a time. The cross attention is bidirectional over the
     * encoder rows, so a non-causal step changes nothing here.
     *
     * @throws IllegalStateException when a segment's sequence has no encoded source
     */
    @Override
    public void batch(StepLayout s, ScratchSlot in, ScratchSlot out) {
        ensureBound();
        if (steps.currentPagedStep() == null) {
            GpuMixer.super.batch(s, in, out);
            return;
        }
        int n = s.nTokens();
        for (int seg = 0; seg < s.nSeg(); seg++) {
            cache(s.seq(seg));
            requireEncoded(s.seq(seg));
        }
        checkStep(n, in, out);
        project(wqCross, qDeltas, in, q, n);
        rotate(rotK, q, n, cross.qDim());
        for (int seg = 0; seg < s.nSeg(); seg++) {
            attend(s.seq(seg), s.tokStart(seg), s.len(seg), true, 0);
        }
        rotate(rotV, attnOut, n, cross.outDim());
        project(woCross, oDeltas, attnOut, out, n);
    }

    @Override
    public String batchRefusal() {
        return "the T5 cross-attention mixer mixes several sequences per step only over paged memory (a"
                + " currentPagedStep)";
    }

    /** {@code out = wo_cross @ attn(wq_cross @ in, cross cache)} over sequence 0's cache (t5.cpp:164-182). */
    private void step(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        ensureBound();
        requireEncoded(0);
        if (pos0 < 0) {
            throw new IllegalArgumentException("T5 decoder step needs pos0 >= 0, got " + pos0);
        }
        checkStep(n, in, out);
        project(wqCross, qDeltas, in, q, n);
        rotate(rotK, q, n, cross.qDim());
        attend(0, 0, n, false, pos0);
        rotate(rotV, attnOut, n, cross.outDim());
        project(woCross, oDeltas, attnOut, out, n);
    }

    private void requireEncoded(int seq) {
        if (nEnc[seq] == 0) {
            throw new IllegalStateException("T5 decoder layer " + EncDecBlocks.describe(layer) + " runs sequence "
                    + seq + " before encode() stored its encoder output");
        }
    }

    private void checkStep(int n, ScratchSlot in, ScratchSlot out) {
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("T5 decoder step needs nTokens in [1, " + ubatch + "], got " + n);
        }
        long rows = (long) n * layer.residual().streamWidth();
        in.require(rows);
        out.require(rows);
        checkViews(in, out);
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
     * Flash attention of the query rows {@code [t0, t0 + n)} over sequence {@code seq}'s cross rows
     * {@code [0, nEnc)}: the tiled kernel for several rows when a tiling fits, else the decode kernel once per row;
     * split ranges merged by the combine. The kernels' visibility is off (no causal mask, no window, no bias), so
     * the query positions they take ({@code pos0 + t} on the contiguous step, the step's row positions on a
     * {@code paged} one) are never read.
     */
    private void attend(int seq, int t0, int n, boolean paged, int pos0) {
        GpuKvLayer cache = crossKv[seq];
        int kvEnd = nEnc[seq];
        if (n == 1 || tiled == null) {
            decode.setBuffer(AttnKernels.ARG_FD_BUF_K, cache.k());
            decode.setBuffer(AttnKernels.ARG_FD_BUF_V, cache.v());
            for (int t = t0; t < t0 + n; t++) {
                decodeRow(t, kvEnd, paged ? steps.rowPosition(t) : pos0 + t);
            }
            return;
        }
        int outOff = attnOut.offset() + t0 * cross.outDim();
        FlashAttnTiling.Split s = prefillPlan.prefillSplit(n, kvEnd, partialCapacity);
        tiled.setBuffer(AttnKernels.ARG_FA_BUF_K, cache.k());
        tiled.setBuffer(AttnKernels.ARG_FA_BUF_V, cache.v());
        tiled.setInt(AttnKernels.ARG_FA_Q_OFF, q.offset() + t0 * cross.qDim());
        tiled.setInt(AttnKernels.ARG_FA_OUT_OFF, outOff);
        tiled.setInt(AttnKernels.ARG_FA_N_TOKENS, n);
        tiled.setInt(AttnKernels.ARG_FA_N_SPLIT, s.count());
        tiled.setInt(AttnKernels.ARG_FA_SPLIT_LEN, s.length());
        tiled.setInt(AttnKernels.ARG_FA_KV_END, kvEnd);
        tiled.setInt(AttnKernels.ARG_FA_FLAGS, s.direct() ? AttnKernels.FLAG_DIRECT : 0);
        tiled.setGlobal(prefillPlan.prefillGlobal(n, s.count()));
        tiled.dispatch();
        if (!s.direct()) {
            combine(n * cross.nHead(), s.count(), outOff);
        }
    }

    /**
     * Query row {@code t} of the step (position {@code qPos}, unused by the unmasked kernel) over the bound cross
     * cache's rows {@code [0, kvEnd)} through decode.
     */
    private void decodeRow(int t, int kvEnd, int qPos) {
        FlashAttnTiling.Split s = decodePlan.decodeSplit(kvEnd);
        int outOff = attnOut.offset() + t * cross.outDim();
        decode.setInt(AttnKernels.ARG_FD_N_SPLIT, s.count());
        decode.setInt(AttnKernels.ARG_FD_SPLIT_LEN, s.length());
        decode.setInt(AttnKernels.ARG_FD_KV_END, kvEnd);
        decode.setInt(AttnKernels.ARG_FD_Q_POS, qPos);
        decode.setInt(AttnKernels.ARG_FD_FLAGS, s.direct() ? AttnKernels.FLAG_DIRECT : 0);
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
                + ", nEnc=" + (nEnc == null ? "[]" : Arrays.toString(nEnc)) + (closed ? ", closed" : "") + '}';
    }
}

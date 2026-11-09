package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.CompressedStateCopy;
import com.eonmux.neuralfree.gguf.memory.CompressorRollback;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.RecurrentStateCopy;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.StateTransfer;
import com.eonmux.neuralfree.gguf.memory.gpu.GroupPositions;
import com.eonmux.neuralfree.gguf.memory.gpu.KvShift;
import com.eonmux.neuralfree.gguf.memory.gpu.SequenceRules;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.memory.paged.StatefulPagedSequences;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.IntPredicate;

/**
 * Heap memory of one model, allocated from a {@link MemoryPlan}: the CPU twin of {@code DeviceModelMemory} with the
 * same plan, layouts, row addressing and sequence rules ({@link SequenceRules}). The position stored in each row of
 * each KV group is kept in a {@link GroupPositions} record; {@link #beginStep(int, int)} marks a step's rows before
 * its K/V are written. A rollback of memory with DSV4 compressed caches stays within their rollback snapshots
 * ({@link CompressorRollback}) and copies that snapshot plane into the live compressor plane of every compressed
 * cache.
 *
 * <p>A paged plan ({@code neuralfree.parallel > 1} or {@code neuralfree.kv.paged}) replaces the single contiguous
 * sequence by {@link PagedSequences} over the plan's page pools: {@link #sequences()} returns it,
 * {@link #beginPagedStep} applies a step's row writes to the {@link GroupPositions} mirrors, and the causal K/V
 * writes go through the physical rows the step's {@code cellIdx} names. The single-sequence calls
 * ({@link #sequence}, {@link #canRollback}, {@link #rollback}, {@link #snapshot}) keep ruling sequence 0's
 * {@link SequenceState}; the batched paths ({@code CpuForwardBackend.forwardBatch},
 * {@code CpuEncDecBackend.forwardBatch}) drive the other sequences. The T5 cross-attention caches keep one stream per
 * sequence ({@link LayerMemoryPlan#crossStreams()}, {@link #crossKv(int, int)}), each holding its sequence's encoder
 * output; the sequence operations leave them alone, since a sequence's encoding is replaced only by encoding a new
 * source.
 * The DSV4 compressed caches keep one stream per sequence ({@link LayerMemoryPlan.CompressedKvPlan#streams()},
 * {@link #compressed(int, LayerMemoryPlan.CompressedKvPlan.Kind, int)}) with its own {@link CompressorRollback}, and
 * the sequence operations keep them in step as llama.cpp's DSV4 cache does (llama-kv-cache-dsv4.cpp:1595-1690): a
 * removal of a whole sequence zeroes its streams, a tail removal restores the rollback snapshot plane it drops to
 * (refused beyond the snapshots, as is any removal inside the sequence), a copy takes whole sequences only and copies
 * the streams, and position shifts are refused.
 *
 * <p>Head shapes of the KV caches come from the model spec: latent caches have one head of
 * {@code (kv_lora + n_rot, kv_lora)}; MLA decompressed caches {@code n_head} heads of {@code (head_k, head_v)};
 * DeepSeek-V4 raw caches one head of the row width; cross-attention caches the decoder layer's cross-attention
 * heads; every other cache the layer's {@code n_head_kv} heads of {@code (head_dim_k, head_dim_v)}.
 *
 * <p>Not thread-safe for its sequence operations; reads of the caches may run in parallel.
 */
public final class HeapModelMemory implements CpuMemory {

    private static final Logger LOGGER = LogManager.getLogger(HeapModelMemory.class);
    private static final long MIB = 1L << 20;

    private final MemoryPlan plan;
    private final SequenceState sequence;
    private final GroupPositions[] positions;
    private final KvAccessor[] kv;
    /** Per layer, the cross-attention K/V of every stream (one per paged sequence); null without one. */
    private final KvAccessor[][] crossKv;
    private final CpuRecurrentState[] recurrent;
    private final KvAccessor[] indexer;
    private final KvAccessor[] msa;
    /** Per layer and kind, the compressed cache of every stream (one per paged sequence). */
    private final List<Map<LayerMemoryPlan.CompressedKvPlan.Kind, CpuCompressedKv[]>> compressed;
    private final List<CpuKvLayer> ownedKv = new ArrayList<>();
    private final List<CpuRecurrentState> ownedStates = new ArrayList<>();
    private final List<CpuCompressedKv> ownedCompressed = new ArrayList<>();
    /** The cell-addressed indexer and MSA key caches this memory owns, which page copies move with the K/V. */
    private final List<KvAccessor> ownedSideCaches = new ArrayList<>();
    /** The compressor rollback bookkeeping of each sequence; entry 0 also rules the single-sequence calls. */
    private final CompressorRollback[] rollbacks;
    private final PagedSequences paged;
    /** The paged sequences with the recurrent-state slots and compressed streams kept in step ({@link #sequences()}). */
    private final StatefulPagedSequences pagedWithStates;
    private int[][] shiftDeltas;
    /** Whether a paged shift queued key re-rotations the layers have not applied yet. */
    private boolean pagedShiftPending;
    private PagedStep pagedStep;
    private int snapshotPast = -1;
    private boolean closed;

    private HeapModelMemory(MemoryPlan plan) {
        this.plan = plan;
        this.sequence = new SequenceState(plan.nCtx());
        this.paged = plan.groups().isEmpty() || !plan.group(0).paged() ? null
                : new PagedSequences(plan, plan.config().parallel());
        this.rollbacks = new CompressorRollback[paged != null ? paged.maxSequences() : 1];
        for (int seq = 0; seq < rollbacks.length; seq++) {
            rollbacks[seq] = new CompressorRollback(plan.rollbackSnapshots());
        }
        this.pagedWithStates = paged != null
                ? new StatefulPagedSequences(paged, new SequenceStates(), plan.hasCompressed(), rollbacks) : null;
        this.positions = new GroupPositions[plan.groups().size()];
        for (int g = 0; g < positions.length; g++) {
            positions[g] = new GroupPositions(plan.group(g));
        }
        int n = plan.layers().size();
        this.kv = new KvAccessor[n];
        this.crossKv = new KvAccessor[n][];
        this.recurrent = new CpuRecurrentState[n];
        this.indexer = new KvAccessor[n];
        this.msa = new KvAccessor[n];
        this.compressed = new ArrayList<>(n);
        for (int il = 0; il < n; il++) {
            compressed.add(new EnumMap<>(LayerMemoryPlan.CompressedKvPlan.Kind.class));
        }
    }

    /**
     * Head shape of the KV caches of a virtual layer: {@code {nHeadKv, headDimK, headDimV}}, which must fill the
     * planned row widths.
     */
    public interface KvHeads {

        /**
         * @param cross true for the T5 decoder cross-attention cache
         * @throws IllegalArgumentException when the layer has no heads for that cache
         */
        int[] of(int layer, KvLayout layout, boolean cross);

        /** Head shapes from the model spec, per the rules of {@link HeapModelMemory}. */
        static KvHeads fromSpec(ModelSpec spec) {
            Objects.requireNonNull(spec, "spec");
            return (il, layout, cross) -> heads(spec, il, layout, cross);
        }
    }

    /**
     * Allocates every cache the plan describes, with head shapes from the model spec.
     *
     * @param spec   the model the plan was built for
     * @param target memory of the target model whose caches {@code TargetLayer} plans read; null when the plan has
     *               none
     * @throws IllegalArgumentException when a KV layer has no attention spec, a head shape does not fill the planned
     *                                  rows, or a shared layer's source has no cache
     */
    public static HeapModelMemory allocate(MemoryPlan plan, ModelSpec spec, CpuMemory target) {
        return allocate(plan, KvHeads.fromSpec(spec), target);
    }

    /**
     * Allocates the caches of the virtual layers {@code layerGate} names, with head shapes from the model spec; the
     * other layers of the plan stay unallocated ({@code kv(il)} and friends read null for them). A hybrid load
     * allocates the layers of its side over the shared plan this way.
     *
     * @throws IllegalArgumentException as {@link #allocate(MemoryPlan, ModelSpec, CpuMemory)}
     */
    public static HeapModelMemory allocate(MemoryPlan plan, ModelSpec spec, CpuMemory target,
                                           IntPredicate layerGate) {
        return allocate(plan, KvHeads.fromSpec(spec), target, layerGate);
    }

    /**
     * Allocates every cache the plan describes, with the given head shapes.
     *
     * @throws IllegalArgumentException as {@link #allocate(MemoryPlan, ModelSpec, CpuMemory)}
     */
    public static HeapModelMemory allocate(MemoryPlan plan, KvHeads heads, CpuMemory target) {
        return allocate(plan, heads, target, il -> true);
    }

    /**
     * Allocates the caches of the virtual layers {@code layerGate} names, with the given head shapes.
     *
     * @throws IllegalArgumentException as {@link #allocate(MemoryPlan, ModelSpec, CpuMemory)}
     */
    public static HeapModelMemory allocate(MemoryPlan plan, KvHeads heads, CpuMemory target, IntPredicate layerGate) {
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(heads, "heads");
        Objects.requireNonNull(layerGate, "layerGate");
        HeapModelMemory m = new HeapModelMemory(plan);
        m.allocateLayers(heads, layerGate);
        m.bindShared(target);
        LOGGER.info("CPU memory: KV {} MiB, recurrent state {} MiB (+{} MiB snapshots), side caches {} MiB over {} "
                        + "layers in {} KV groups (ctx {}, ubatch {})", plan.kvBytes() / MIB, plan.stateBytes() / MIB,
                plan.snapshotBytes() / MIB, plan.sideCacheBytes() / MIB, plan.layers().size(), plan.groups().size(),
                plan.nCtx(), plan.ubatch());
        return m;
    }

    private void allocateLayers(KvHeads heads, IntPredicate layerGate) {
        for (LayerMemoryPlan lp : plan.layers()) {
            int il = lp.layer();
            if (!layerGate.test(il)) {
                continue;
            }
            if (lp.ownsKv()) {
                int[] h = shape(heads, il, lp.kv(), false);
                CpuKvLayer c = CpuKvLayer.allocate(il, lp.kv(), lp.kvGroup(), h[0], h[1], h[2]);
                ownedKv.add(c);
                kv[il] = c;
            }
            if (lp.hasState()) {
                CpuRecurrentState s = CpuRecurrentState.allocate(il, lp.states(), plan.snapshotsReserved());
                ownedStates.add(s);
                recurrent[il] = s;
            }
            if (lp.indexer() != null) {
                indexer[il] = CpuIndexerCache.allocate(il, lp.indexer(), plan.group(lp.kvGroup()));
                ownedSideCaches.add(indexer[il]);
            }
            if (lp.msa() != null) {
                msa[il] = CpuMsaCache.allocate(il, lp.msa(), plan.group(lp.kvGroup()));
                ownedSideCaches.add(msa[il]);
            }
            for (LayerMemoryPlan.CompressedKvPlan cp : lp.compressed()) {
                if (compressed.get(il).containsKey(cp.kind())) {
                    throw new IllegalArgumentException("layer " + il + " plans two " + cp.kind() + " compressed caches");
                }
                CpuCompressedKv[] streams = new CpuCompressedKv[cp.streams()];
                for (int seq = 0; seq < streams.length; seq++) {
                    streams[seq] = CpuCompressedKv.allocate(il, cp);
                    ownedCompressed.add(streams[seq]);
                }
                compressed.get(il).put(cp.kind(), streams);
            }
            if (lp.crossKv() != null) {
                int[] h = shape(heads, il, lp.crossKv(), true);
                KvAccessor[] streams = new KvAccessor[lp.crossStreams()];
                for (int s = 0; s < streams.length; s++) {
                    streams[s] = CpuKvLayer.allocate(il, lp.crossKv(), -1, h[0], h[1], h[2]);
                }
                crossKv[il] = streams;
            }
        }
    }

    private static int[] shape(KvHeads heads, int il, KvLayout layout, boolean cross) {
        int[] h = heads.of(il, layout, cross);
        if (h == null || h.length != 3) {
            throw new IllegalArgumentException("head shape of layer " + il + " must be {nHeadKv, headDimK, headDimV}");
        }
        return h;
    }

    /** {@code {nHeadKv, headDimK, headDimV}} of a cache of virtual layer {@code il}. */
    private static int[] heads(ModelSpec spec, int il, KvLayout layout, boolean cross) {
        if (layout.vAliasesK()) {
            return new int[]{1, layout.widthK(), layout.widthV()};
        }
        LayerSpec ls = layerSpec(spec, il);
        if (cross) {
            AttentionSpec a = ls.crossAttention();
            if (a == null) {
                throw new IllegalArgumentException("layer " + il + " plans cross-attention K/V but has no cross-attention");
            }
            return new int[]{a.nHeadKv(), a.headDimK(), a.headDimV()};
        }
        if (ls.dsv4() != null) {
            return new int[]{1, layout.widthK(), layout.widthV()};
        }
        MlaSpec m = ls.mla();
        if (m != null) {
            return new int[]{m.nHead(), m.headK(), m.headV()};
        }
        AttentionSpec a = ls.attention();
        if (a == null) {
            throw new IllegalArgumentException("layer " + il + " plans a KV cache but has no attention");
        }
        return new int[]{a.nHeadKv(), a.headDimK(), a.headDimV()};
    }

    private static LayerSpec layerSpec(ModelSpec spec, int il) {
        if (il < spec.nLayer()) {
            return spec.layer(il);
        }
        int nextn = il - spec.nLayer();
        if (nextn >= spec.nextnLayers().size()) {
            throw new IllegalArgumentException("memory plan layer " + il + " is beyond the " + spec.nLayer()
                    + " trunk and " + spec.nextnLayers().size() + " NextN layers of " + spec.arch());
        }
        return spec.nextnLayers().get(nextn);
    }

    private void bindShared(CpuMemory target) {
        for (LayerMemoryPlan lp : plan.layers()) {
            LayerMemoryPlan.SharedKv src = lp.sharedFrom();
            if (lp.kind() != LayerMemoryPlan.Kind.SHARED_KV || src == null) {
                continue;
            }
            int il = lp.layer();
            if (src.target()) {
                if (target == null) {
                    throw new IllegalArgumentException("layer " + il + " reads target layer " + src.layer()
                            + " but no target memory was given");
                }
                kv[il] = target.kv(src.layer());
                msa[il] = target.msa(src.layer());
            } else {
                kv[il] = kv[src.layer()];
                msa[il] = msa[src.layer()];
            }
            // a KV-sharing layer with its own indexer plan (mistral-large4) allocated it in allocateLayers
            if (lp.indexer() == null) {
                indexer[il] = src.target() ? target.indexer(src.layer()) : indexer[src.layer()];
            }
            if (kv[il] == null) {
                throw new IllegalArgumentException("layer " + il + " shares the KV of " + src + ", which has no cache");
            }
        }
    }

    // ------------------------------------------------------------------ CpuMemory

    @Override
    public MemoryPlan plan() {
        return plan;
    }

    @Override
    public SequenceState sequence() {
        return sequence;
    }

    /** {@link PagedSequences} over the plan's page pools on a paged plan, else the contiguous adapter. */
    @Override
    public com.eonmux.neuralfree.gguf.memory.SequenceMemory sequences() {
        checkOpen();
        return paged != null ? pagedWithStates : CpuMemory.super.sequences();
    }

    @Override
    public KvAccessor kv(int il) {
        return kv[checkLayer(il)];
    }

    @Override
    public KvAccessor crossKv(int il) {
        KvAccessor[] streams = crossKv[checkLayer(il)];
        return streams == null ? null : streams[0];
    }

    @Override
    public KvAccessor crossKv(int il, int seq) {
        KvAccessor[] streams = crossKv[checkLayer(il)];
        if (streams == null) {
            return null;
        }
        if (seq < 0 || seq >= streams.length) {
            throw new IndexOutOfBoundsException("layer " + il + " holds " + streams.length
                    + " cross-attention streams, not sequence " + seq);
        }
        return streams[seq];
    }

    @Override
    public RecurrentStateAccessor recurrent(int il) {
        return recurrent[checkLayer(il)];
    }

    @Override
    public KvAccessor indexer(int il) {
        return indexer[checkLayer(il)];
    }

    @Override
    public KvAccessor msa(int il) {
        return msa[checkLayer(il)];
    }

    @Override
    public KvAccessor compressed(int il, LayerMemoryPlan.CompressedKvPlan.Kind kind) {
        return compressed(il, kind, 0);
    }

    @Override
    public RecurrentStateAccessor compressorState(int il, LayerMemoryPlan.CompressedKvPlan.Kind kind) {
        return compressorState(il, kind, 0);
    }

    @Override
    public KvAccessor compressed(int il, LayerMemoryPlan.CompressedKvPlan.Kind kind, int seq) {
        return stream(il, kind, seq);
    }

    @Override
    public RecurrentStateAccessor compressorState(int il, LayerMemoryPlan.CompressedKvPlan.Kind kind, int seq) {
        CpuCompressedKv c = stream(il, kind, seq);
        return c != null ? c.state() : null;
    }

    /** Stream {@code seq} of a layer's compressed cache, or null when the layer keeps none of {@code kind}. */
    private CpuCompressedKv stream(int il, LayerMemoryPlan.CompressedKvPlan.Kind kind, int seq) {
        CpuCompressedKv[] streams = compressed.get(checkLayer(il)).get(Objects.requireNonNull(kind, "kind"));
        if (streams == null) {
            return null;
        }
        if (seq < 0 || seq >= streams.length) {
            throw new IndexOutOfBoundsException("layer " + il + " keeps " + streams.length + " " + kind
                    + " compressed streams, not " + seq);
        }
        return streams[seq];
    }

    @Override
    public int positionOfRow(int group, int row) {
        checkOpen();
        return positions[checkGroup(group)].position(row);
    }

    /**
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]} or a position has no row in
     *                                  a group; nothing changes then
     */
    @Override
    public void beginStep(int pos0, int nTokens) {
        checkOpen();
        if (nTokens < 1 || nTokens > plan.ubatch()) {
            throw new IllegalArgumentException("step of " + nTokens + " tokens outside [1, " + plan.ubatch() + "]");
        }
        if (pos0 < 0 || (long) pos0 + nTokens > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("step positions [" + pos0 + ", " + ((long) pos0 + nTokens)
                    + ") are out of range");
        }
        for (GroupPositions p : positions) {
            p.checkStep(pos0, nTokens);
        }
        boolean overwrote = false;
        for (GroupPositions p : positions) {
            overwrote |= p.store(pos0, nTokens);
        }
        if (overwrote) {
            sequence.markWrapped();
        }
        rollbacks[0].onStep(nTokens);
    }

    @Override
    public PagedSequences pagedSequences() {
        checkOpen();
        return paged;
    }

    @Override
    public String pagedStepRefusal(PagedStep step) {
        return paged == null ? getClass().getSimpleName() + " holds no paged sequences" : null;
    }

    @Override
    public void beginPagedStep(PagedStep step, int[] positions, int[][] ropeSections) {
        checkOpen();
        String refusal = pagedStepRefusal(step);
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        if (step.maxSequences() != paged.maxSequences() || step.groups() != this.positions.length
                || step.pageSize() != plan.group(0).pageSize() || step.nPages() != paged.pageCount()) {
            throw new IllegalArgumentException("the step does not belong to this paged memory");
        }
        // mirror changes the page bookkeeping made besides the step's own writes (removals, shifts, partial copies),
        // then the step's writes, so attention sees every row's true position
        drainRowReports();
        for (int g = 0; g < this.positions.length; g++) {
            int[] rows = step.kvPosRows(g);
            int[] values = step.kvPosValues(g);
            for (int i = 0; i < rows.length; i++) {
                this.positions[g].set(rows[i], values[i]);
            }
        }
        for (int seg = 0; seg < step.nSeg(); seg++) {
            rollbacks[step.segSeq(seg)].onStep(step.segLen(seg));
        }
        pagedStep = step;
    }

    /**
     * Opens one paged step over {@code layout}, the order a host backend runs it in: reserves the step's rows
     * ({@code PagedSequences.prepare}), runs the page copies queued by earlier sequence operations and then the
     * step's own copy-on-write copies ({@link #copyPage}), commits the reservation, and mirrors the step's rows
     * ({@link #beginPagedStep}). The caller runs the layers and closes the step with {@link #endPagedStep}.
     *
     * @param positions the position of every row of the step
     * @return the step the layers read through {@link #currentPagedStep}
     * @throws UnsupportedOperationException when the memory is not paged
     * @throws com.eonmux.neuralfree.gguf.memory.MemoryFullException when the rows cannot be reserved; nothing
     *                                                               changes then
     */
    public PagedStep openPagedStep(com.eonmux.neuralfree.gguf.memory.StepLayout layout, int[] positions) {
        checkOpen();
        if (paged == null) {
            throw new UnsupportedOperationException(pagedStepRefusal(null));
        }
        PagedStep step = paged.prepare(layout, positions);
        for (com.eonmux.neuralfree.gguf.memory.paged.PageCopy copy
                : paged.pendingCopies().toArray(new com.eonmux.neuralfree.gguf.memory.paged.PageCopy[0])) {
            copyPage(copy);
            paged.releaseCopies(copy);
        }
        moveRows(paged.pendingRowMoves().toArray(new com.eonmux.neuralfree.gguf.memory.paged.RowMove[0]));
        paged.clearRowMoves();
        for (com.eonmux.neuralfree.gguf.memory.paged.PageCopy copy : step.pageCopies()) {
            copyPage(copy);
        }
        paged.commit(step);
        paged.releaseCopies(step.pageCopies().toArray(new com.eonmux.neuralfree.gguf.memory.paged.PageCopy[0]));
        beginPagedStep(step, positions, null);
        return step;
    }

    /** Applies the page bookkeeping's queued mirror changes to the {@code GroupPositions} and consumes them. */
    private void drainRowReports() {
        boolean shifted = false;
        for (com.eonmux.neuralfree.gguf.memory.paged.RowReport r : paged.pendingReports()) {
            int[] rows = r.rows();
            int[] values = r.values();
            for (int i = 0; i < rows.length; i++) {
                positions[r.group()].set(rows[i], values[i]);
            }
            if (r.absolute()) {
                continue;
            }
            // a shifted row's key needs the RoPE re-rotation its delta names: record it for the layers
            if (shiftDeltas == null) {
                shiftDeltas = new int[positions.length][];
                for (int g = 0; g < shiftDeltas.length; g++) {
                    shiftDeltas[g] = new int[positions[g].rows()];
                }
            }
            for (int i = 0; i < rows.length; i++) {
                shiftDeltas[r.group()][rows[i]] += r.rowDelta(i);
            }
            shifted = true;
        }
        paged.clearReports();
        pagedShiftPending |= shifted;
    }

    /**
     * The paged context shift of one sequence (llama.cpp server's slot shift): removes its positions
     * {@code [nKeep, nKeep + nDiscard)} and moves every later position down. The row mirrors and the per-row
     * deltas apply when the backend asks for them ({@link #applyPagedShiftDeltas()}); the refusals of the
     * sequence view stand (a DeepSeek-V4 compressor beyond its rollback snapshots refuses the interior removal by
     * name).
     *
     * @throws UnsupportedOperationException when the sequence view refuses the removal or the move
     */
    public void shiftSequence(int seq, int nKeep, int nDiscard) {
        checkOpen();
        shiftDeltas = null; // the layers re-rotate by this move's deltas alone
        sequences().remove(seq, nKeep, nKeep + nDiscard);
        sequences().add(seq, nKeep + nDiscard, -1, -nDiscard);
    }

    /**
     * The {@code seq_add} of one sequence (llama.cpp {@code llama_memory_seq_add}): moves its stored positions
     * {@code [p0, p1)} by {@code delta}. The row mirrors, the compaction's row moves and the per-row RoPE deltas
     * apply when the backend asks for them ({@link #applyPagedShiftDeltas()}), as {@link #shiftSequence}'s do.
     *
     * @throws UnsupportedOperationException when the sequence view refuses the move
     */
    public void addSequence(int seq, int p0, int p1, int delta) {
        checkOpen();
        shiftDeltas = null; // the layers re-rotate by this edit's deltas alone
        sequences().add(seq, p0, p1, delta);
    }

    /**
     * The {@code seq_div} of one sequence (llama.cpp {@code llama_memory_seq_div}): divides its stored positions
     * {@code [p0, p1)} by {@code d}, reporting each row's own RoPE shift for the backend's re-rotation.
     *
     * @throws IllegalArgumentException      when {@code p0} is negative or {@code d} is below 2
     * @throws UnsupportedOperationException when the sequence view refuses the division
     */
    public void divSequence(int seq, int p0, int p1, int d) {
        checkOpen();
        shiftDeltas = null; // the layers re-rotate by this edit's deltas alone
        sequences().div(seq, p0, p1, d);
    }

    /**
     * The KV chunk reuse of one sequence (llama.cpp server's cached-prompt re-adoption): re-appends the held rows
     * of cached pages at its append cursor, so its prompt skips their prefill — the run of {@code rows} rows starts
     * at source row {@code srcRow0} of run page 0 of {@code srcPages} (one page per KV group per run page, the
     * cache blocks') and lands at the sequence's own slots at the positions {@code [pos0, pos0 + rows)} its prompt
     * rows take. The row mirrors and the copied rows' RoPE deltas apply when the backend asks for them
     * ({@link #applyPagedShiftDeltas()}), as {@link #addSequence}'s do; the re-append's page copies run first
     * ({@link #runPagedEditCopies()}), before the copied keys re-rotate.
     *
     * @throws IllegalArgumentException      for a run outside the pages, a token count or position mismatch, or
     *                                       source rows the cache does not hold
     * @throws com.eonmux.neuralfree.gguf.memory.MemoryFullException when the re-append needs pages the pools lack
     * @throws UnsupportedOperationException when the memory holds no paged sequences, or a group trims a window
     */
    public void reuseSequence(int seq, int[][] srcPages, int srcRow0, int pos0, int rows, int delta,
                              int[] tokens, int embeddingToken) {
        checkOpen();
        if (paged == null) {
            throw new UnsupportedOperationException("the memory holds no paged sequences, so no cached rows can be"
                    + " re-appended");
        }
        shiftDeltas = null; // the layers re-rotate by this edit's deltas alone
        paged.reusePages(seq, srcPages, srcRow0, pos0, rows, delta, tokens, embeddingToken);
    }

    /**
     * Runs the page copies a sequence edit queued (the privatization of every shared page it writes), before the
     * layers re-rotate the moved keys: a copy deferred to the next step would overwrite the rotated rows with the
     * source's unrotated ones. The row moves of a compaction stay queued for the next step, which runs them after
     * the rotation.
     */
    public void runPagedEditCopies() {
        checkOpen();
        if (paged == null) {
            return;
        }
        for (com.eonmux.neuralfree.gguf.memory.paged.PageCopy copy
                : paged.pendingCopies().toArray(new com.eonmux.neuralfree.gguf.memory.paged.PageCopy[0])) {
            copyPage(copy);
            paged.releaseCopies(copy);
        }
    }

    /**
     * Applies the queued row-move mirrors and records the per-row shift deltas the layers re-rotate
     * ({@code #pagedShiftPending}): the backend runs the re-rotation at once, while the moved rows still sit at
     * the cells the shift reports name — the compaction's row moves then carry the rotated keys to their new
     * cells.
     */
    public boolean applyPagedShiftDeltas() {
        checkOpen();
        drainRowReports();
        return pagedShiftPending;
    }

    /** Whether a paged shift queued key re-rotations the layers have not applied yet. */
    public boolean pagedShiftPending() {
        checkOpen();
        return pagedShiftPending;
    }

    /** Clears the pending paged-shift flag once the layers re-rotated the moved keys. */
    public void clearPagedShift() {
        checkOpen();
        pagedShiftPending = false;
    }

    /**
     * Runs one page copy on the host: every cache layer of the copy's group moves its rows
     * {@code [srcPage * pageSize + srcRow0, .. + rows)} to the destination page's rows
     * {@code [dstPage * pageSize + dstRow0, .. + rows)} (a copy-on-write copy lands at the same rows, a KV chunk
     * reuse's copy at the destination's own slots), and so do the cell-addressed side caches of the group (the DSA
     * indexer keys and the MSA index keys, {@code RowParts}). A group no layer declares (the pure-recurrent plan's
     * bookkeeping group) holds no rows and the copy is a no-op.
     *
     * @throws IllegalArgumentException when cache layers declare the group but none belongs to the copy's group
     */
    @Override
    public void copyPage(com.eonmux.neuralfree.gguf.memory.paged.PageCopy copy) {
        int ps = plan.group(0).pageSize();
        int src0 = copy.srcPage() * ps + copy.srcRow0();
        int dst0 = copy.dstPage() * ps + copy.dstRow0();
        int moved = 0;
        for (int il = 0; il < kv.length; il++) {
            KvAccessor k = kv[il];
            if (k != null && k.group() == copy.group()) {
                k.copyRows(src0, dst0, copy.rows());
                moved++;
            }
        }
        for (KvAccessor side : ownedSideCaches) {
            if (side.group() == copy.group()) {
                side.copyRows(src0, dst0, copy.rows());
            }
        }
        if (moved == 0 && declaresGroup(copy.group())) {
            throw new IllegalArgumentException("no cache layer of group " + copy.group() + " runs the page copy "
                    + copy);
        }
    }

    /** Whether any cache layer of {@code group} reads the KV group's cells. */
    private boolean declaresGroup(int group) {
        for (LayerMemoryPlan l : plan.layers()) {
            if (l.kvGroup() == group) {
                return true;
            }
        }
        return false;
    }

    /**
     * Carries the queued row moves of a shifted sequence on the host: every cache layer of the move's group moves
     * the run's keys and values, and so do the cell-addressed side caches, as a page copy does. The destinations
     * privatized before the moves were queued, so every write lands on the sequence's own rows.
     */
    public void moveRows(com.eonmux.neuralfree.gguf.memory.paged.RowMove[] moves) {
        checkOpen();
        for (com.eonmux.neuralfree.gguf.memory.paged.RowMove move : moves) {
            int moved = 0;
            for (int il = 0; il < kv.length; il++) {
                KvAccessor k = kv[il];
                if (k != null && k.group() == move.group()) {
                    k.copyRows(move.srcRow(), move.dstRow(), move.rows());
                    moved++;
                }
            }
            for (KvAccessor side : ownedSideCaches) {
                if (side.group() == move.group()) {
                    side.copyRows(move.srcRow(), move.dstRow(), move.rows());
                }
            }
            if (moved == 0 && declaresGroup(move.group())) {
                throw new IllegalArgumentException("no cache layer of group " + move.group() + " runs the row move "
                        + move);
            }
        }
    }

    @Override
    public String movePositionRowsRefusal(int[] srcPos, int[] dstPos) {
        String structural = checkMovePositionRows(srcPos, dstPos);
        if (structural != null) {
            return structural;
        }
        checkOpen();
        for (int i = 0; i < srcPos.length; i++) {
            for (int g = 0; g < positions.length; g++) {
                MemoryPlan.KvGroup group = positions[g].group();
                if (group.holds(srcPos[i]) && positions[g].position(group.rowOf(srcPos[i])) != srcPos[i]) {
                    return "the row of position " + srcPos[i] + " holds " + positions[g]
                            .position(group.rowOf(srcPos[i])) + "; the tree compaction needs its verify rows in"
                            + " their slots";
                }
            }
        }
        return null;
    }

    @Override
    public void movePositionRows(int[] srcPos, int[] dstPos) {
        String refusal = movePositionRowsRefusal(srcPos, dstPos);
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        for (int i = 0; i < srcPos.length; i++) {
            for (int g = 0; g < positions.length; g++) {
                MemoryPlan.KvGroup group = positions[g].group();
                if (!group.holds(srcPos[i])) {
                    continue; // a context shard holds other positions
                }
                int src = group.rowOf(srcPos[i]);
                int dst = group.holds(dstPos[i]) ? group.rowOf(dstPos[i]) : -1;
                if (dst < 0) {
                    continue; // the destination sits outside this group's rows (a context shard's other block)
                }
                int moved = 0;
                for (int il = 0; il < kv.length; il++) {
                    KvAccessor k = kv[il];
                    if (k != null && k.group() == g) {
                        k.copyRows(src, dst, 1);
                        moved++;
                    }
                }
                for (KvAccessor side : ownedSideCaches) {
                    if (side.group() == g) {
                        side.copyRows(src, dst, 1);
                    }
                }
                if (moved == 0 && declaresGroup(g)) {
                    throw new IllegalArgumentException("no cache layer of group " + g + " runs the row move of"
                            + " position " + srcPos[i]);
                }
                positions[g].set(dst, dstPos[i]);
                positions[g].set(src, -1);
            }
        }
    }

    /** The ordering contract of a compaction: equal lengths, sources above their destinations, both ascending. */
    private static String checkMovePositionRows(int[] srcPos, int[] dstPos) {
        if (srcPos == null || dstPos == null) {
            throw new IllegalArgumentException("a row compaction names sources and destinations");
        }
        if (srcPos.length != dstPos.length) {
            return "the compaction names " + srcPos.length + " sources and " + dstPos.length + " destinations";
        }
        for (int i = 0; i < srcPos.length; i++) {
            if (srcPos[i] <= dstPos[i]) {
                return "the compaction moves position " + srcPos[i] + " down onto " + dstPos[i] + ", which is not"
                        + " below its source";
            }
            if (i > 0 && (srcPos[i] <= srcPos[i - 1] || dstPos[i] <= dstPos[i - 1])) {
                return "the compaction's positions must ascend strictly, got " + srcPos[i] + "/" + dstPos[i]
                        + " after " + srcPos[i - 1] + "/" + dstPos[i - 1];
            }
        }
        return null;
    }

    /** Runs the queued compaction now: the pending page copies, then the queued row moves. */
    @Override
    public void carryRowMoves() {
        checkOpen();
        if (paged == null) {
            return;
        }
        for (com.eonmux.neuralfree.gguf.memory.paged.PageCopy copy
                : paged.pendingCopies().toArray(new com.eonmux.neuralfree.gguf.memory.paged.PageCopy[0])) {
            copyPage(copy);
            paged.releaseCopies(copy);
        }
        moveRows(paged.pendingRowMoves().toArray(new com.eonmux.neuralfree.gguf.memory.paged.RowMove[0]));
        paged.clearRowMoves();
    }

    @Override
    public void endPagedStep() {
        checkOpen();
        pagedStep = null;
    }

    @Override
    public PagedStep currentPagedStep() {
        checkOpen();
        return pagedStep;
    }

    // ------------------------------------------------------------------ sequence operations

    @Override
    public void reset() {
        checkOpen();
        for (CpuRecurrentState s : ownedStates) {
            s.reset();
        }
        for (CpuCompressedKv c : ownedCompressed) {
            c.reset();
        }
        for (GroupPositions p : positions) {
            p.clear();
        }
        if (paged != null) {
            paged.clear();
        }
        sequence.clear();
        for (CompressorRollback r : rollbacks) {
            r.reset();
        }
        snapshotPast = -1;
    }

    @Override
    public boolean canRollback(int nTokens) {
        checkOpen();
        if (nTokens < 0) {
            throw new IllegalArgumentException("rollback count " + nTokens + " is negative");
        }
        return nTokens == 0
                || SequenceRules.rollbackBlocker(plan, positions, sequence, rollbacks[0], nTokens) == null;
    }

    @Override
    public boolean ringsKeepWindow(int nWrite, int nDrop) {
        checkOpen();
        return SequenceRules.ringBlocker(positions, sequence, nWrite, nDrop) == null;
    }

    @Override
    public void rollback(int nTokens) {
        checkOpen();
        if (nTokens < 0) {
            throw new IllegalArgumentException("rollback count " + nTokens + " is negative");
        }
        if (nTokens == 0) {
            return;
        }
        String blocker = SequenceRules.rollbackBlocker(plan, positions, sequence, rollbacks[0], nTokens);
        if (blocker != null) {
            throw new IllegalStateException("cannot roll back " + nTokens + " positions: " + blocker);
        }
        if (plan.hasCompressed()) {
            restoreStreams(0, rollbacks[0].onRollback(nTokens));
        }
        drop(sequence.nPast() - nTokens, nTokens);
        sequence.truncate(nTokens);
    }

    @Override
    public void dropRecomputed(int nTokens) {
        checkOpen();
        if (nTokens < 0) {
            throw new IllegalArgumentException("drop count " + nTokens + " is negative");
        }
        if (nTokens == 0) {
            return;
        }
        if (!plan.hasOnlyPositionalState()) {
            throw new UnsupportedOperationException("the memory keeps recurrent state the dropped positions do not"
                    + " re-evaluate; it can only roll back through snapshot() and restore()");
        }
        if (plan.hasCompressed()) {
            throw new UnsupportedOperationException("the memory keeps DSV4 compressor state, whose planes the dropped"
                    + " positions do not re-evaluate");
        }
        if (nTokens > sequence.nPast()) {
            throw new IllegalStateException("only " + sequence.nPast() + " positions are stored");
        }
        drop(sequence.nPast() - nTokens, nTokens);
        sequence.truncate(nTokens);
    }

    @Override
    public void dropKeepRecurrent(int nTokens) {
        checkOpen();
        if (nTokens < 0) {
            throw new IllegalArgumentException("drop count " + nTokens + " is negative");
        }
        if (nTokens == 0) {
            return;
        }
        if (plan.hasCompressed()) {
            throw new UnsupportedOperationException("the memory keeps DSV4 compressor state, whose planes the"
                    + " dropped positions do not re-evaluate");
        }
        if (plan.hasPositionalState()) {
            throw new UnsupportedOperationException("the memory keeps positional state the dropped positions"
                    + " re-evaluate: it can only drop through dropRecomputed, which refuses the recurrent layers");
        }
        if (nTokens > sequence.nPast()) {
            throw new IllegalStateException("only " + sequence.nPast() + " positions are stored");
        }
        String ring = SequenceRules.ringBlocker(positions, sequence, 0, nTokens);
        if (ring != null) {
            throw new IllegalStateException("cannot drop " + nTokens + " positions: " + ring);
        }
        drop(sequence.nPast() - nTokens, nTokens);
        sequence.truncate(nTokens);
    }

    @Override
    public void snapshot() {
        checkOpen();
        String blocker = SequenceRules.snapshotBlocker(plan);
        if (blocker != null) {
            throw new UnsupportedOperationException(blocker);
        }
        if (paged != null) {
            throw new UnsupportedOperationException("a paged memory keeps one recurrent-state row per sequence; the"
                    + " whole-memory snapshot would rewind every sequence at once, so it is refused");
        }
        for (CpuRecurrentState s : ownedStates) {
            s.snapshot();
        }
        snapshotPast = sequence.nPast();
    }

    @Override
    public void restore() {
        checkOpen();
        if (snapshotPast < 0) {
            throw new IllegalStateException("no recurrent-state snapshot since the last reset");
        }
        int n = sequence.nPast() - snapshotPast;
        if (n < 0) {
            throw new IllegalStateException("memory holds " + sequence.nPast() + " positions, fewer than the "
                    + snapshotPast + " of the snapshot");
        }
        String blocker = n > 0 ? SequenceRules.ringBlocker(positions, sequence, 0, n) : null;
        if (blocker != null) {
            throw new IllegalStateException("cannot restore the snapshot: " + blocker);
        }
        for (CpuRecurrentState s : ownedStates) {
            s.restore();
        }
        if (n > 0) {
            drop(snapshotPast, n);
            sequence.truncate(n);
        }
    }

    private void drop(int pos0, int n) {
        for (GroupPositions p : positions) {
            p.drop(pos0, n);
        }
    }

    /**
     * Context shift (llama.cpp {@code seq_rm} + {@code seq_add}), the heap twin of
     * {@code DeviceModelMemory.shiftPositions}: drops positions {@code [nKeep, nKeep + nDiscard)}, moves every later
     * position down by {@code nDiscard} (rows move so each position stays in {@code rowOf(pos)}), and returns the
     * per-row RoPE deltas each attention mixer applies to its K rows.
     *
     * @return the per-row deltas of each KV group, indexed by group then row (copies)
     * @throws IllegalArgumentException      for an invalid range
     * @throws UnsupportedOperationException for recurrent state, sparse side caches, target-model caches, or a ring
     *                                       that no longer holds kept positions the next query attends; nothing
     *                                       changes then
     */
    public int[][] shiftPositions(int nKeep, int nDiscard) {
        checkOpen();
        SequenceRules.checkShift(sequence, nKeep, nDiscard);
        String why = SequenceRules.shiftBlocker(plan);
        if (why != null) {
            throw new UnsupportedOperationException("context shift unsupported: " + why);
        }
        for (GroupPositions p : positions) {
            String ring = SequenceRules.ringShiftBlocker(p, sequence, nKeep, nDiscard);
            if (ring != null) {
                throw new UnsupportedOperationException("context shift unsupported: " + ring);
            }
        }
        KvShift[] shifts = new KvShift[positions.length];
        for (int g = 0; g < positions.length; g++) {
            shifts[g] = positions[g].planShift(nKeep, nDiscard);
        }
        for (CpuKvLayer c : ownedKv) {
            c.moveRows(shifts[c.group()]);
        }
        int[][] deltas = new int[positions.length][];
        for (int g = 0; g < positions.length; g++) {
            positions[g].apply(shifts[g]);
            deltas[g] = new int[positions[g].rows()];
            for (int r = 0; r < deltas[g].length; r++) {
                deltas[g][r] = shifts[g].delta(r);
            }
        }
        shiftDeltas = deltas;
        SequenceRules.shiftSequence(sequence, nKeep, nDiscard);
        return shiftDelta();
    }

    /**
     * Per-row deltas of the last {@link #shiftPositions(int, int)}, indexed by group then row (copies).
     *
     * @throws IllegalStateException when no shift has happened
     */
    public int[][] shiftDelta() {
        checkOpen();
        if (shiftDeltas == null) {
            throw new IllegalStateException("the context has not been shifted");
        }
        int[][] copy = new int[shiftDeltas.length][];
        for (int g = 0; g < copy.length; g++) {
            copy[g] = shiftDeltas[g].clone();
        }
        return copy;
    }

    private int checkLayer(int il) {
        checkOpen();
        if (il < 0 || il >= kv.length) {
            throw new IndexOutOfBoundsException("memory has no layer " + il + " (" + kv.length + " layers)");
        }
        return il;
    }

    // ------------------------------------------------------------------ state transfer

    /**
     * The host transfer of this memory's state: the stored row bytes of every owned cache layer, the live
     * recurrent state tensors and the per-group position mirrors ({@link StateTransfer}).
     */
    @Override
    public StateTransfer stateTransfer() {
        checkOpen();
        List<StateTransfer.Cache> caches = new ArrayList<>(ownedKv.size());
        for (CpuKvLayer c : ownedKv) {
            caches.add(new HostCache(c));
        }
        List<StateTransfer.State> states = new ArrayList<>(ownedStates.size());
        for (CpuRecurrentState s : ownedStates) {
            states.add(new HostState(s));
        }
        return new HostTransfer(List.copyOf(caches), List.copyOf(states));
    }

    /** The heap transfer: rows and states straight from the arrays, positions from the group mirrors. */
    private final class HostTransfer implements StateTransfer {
        private final List<Cache> caches;
        private final List<State> states;

        HostTransfer(List<Cache> caches, List<State> states) {
            this.caches = caches;
            this.states = states;
        }

        @Override
        public List<Cache> caches() {
            return caches;
        }

        @Override
        public List<State> states() {
            return states;
        }

        @Override
        public int[] positions(int group) {
            checkOpen();
            return positions[checkGroup(group)].snapshot();
        }

        @Override
        public void setPositions(int group, int[] mirror) {
            checkOpen();
            GroupPositions p = positions[checkGroup(group)];
            if (mirror.length != p.rows()) {
                throw new IllegalArgumentException("a mirror of " + mirror.length + " rows for KV group " + group
                        + " of " + p.rows() + " rows");
            }
            for (int r = 0; r < mirror.length; r++) {
                p.set(r, mirror[r]);
            }
        }
    }

    /** One owned heap cache layer under the transfer. */
    private static final class HostCache implements StateTransfer.Cache {
        private final CpuKvLayer layer;

        HostCache(CpuKvLayer layer) {
            this.layer = layer;
        }

        @Override
        public int layer() {
            return layer.layer();
        }

        @Override
        public int group() {
            return layer.group();
        }

        @Override
        public KvLayout layout() {
            return layer.layout();
        }

        @Override
        public void read(boolean v, int row0, int rows, byte[] dst) {
            layer.readRowBytes(v, row0, rows, dst);
        }

        @Override
        public void write(boolean v, int row0, int rows, byte[] src) {
            layer.writeRowBytes(v, row0, rows, src);
        }

        @Override
        public java.util.List<com.eonmux.neuralfree.gguf.memory.KvDType.SidePlane> sidePlanes(boolean v) {
            int n = layer.sidePlaneCount(v);
            java.util.List<com.eonmux.neuralfree.gguf.memory.KvDType.SidePlane> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                out.add((v ? layer.layout().dtypeV() : layer.layout().dtypeK()).sidePlanes().get(i));
            }
            return out;
        }

        @Override
        public void readSide(boolean v, int plane, int row0, int rows, byte[] dst) {
            layer.readSideBytes(v, plane, row0, rows, dst);
        }

        @Override
        public void writeSide(boolean v, int plane, int row0, int rows, byte[] src) {
            layer.writeSideBytes(v, plane, row0, rows, src);
        }
    }

    /** One owned heap recurrent state under the transfer. */
    private static final class HostState implements StateTransfer.State {
        private final CpuRecurrentState state;

        HostState(CpuRecurrentState state) {
            this.state = state;
        }

        @Override
        public int layer() {
            return state.layer();
        }

        @Override
        public List<LayerMemoryPlan.StateTensor> tensors() {
            return state.tensors();
        }

        @Override
        public float[] read(String name) {
            return state.tensor(name).clone();
        }

        @Override
        public void write(String name, float[] values) {
            float[] live = state.tensor(name);
            if (values.length != live.length) {
                throw new IllegalArgumentException("state tensor " + name + " of layer " + state.layer() + " holds "
                        + live.length + " floats, got " + values.length);
            }
            System.arraycopy(values, 0, live, 0, live.length);
        }
    }

    private int checkGroup(int group) {
        if (group < 0 || group >= positions.length) {
            throw new IndexOutOfBoundsException("KV group " + group + " outside [0, " + positions.length + ")");
        }
        return group;
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("CPU memory is closed");
        }
    }

    /** The recurrent-state slots and compressed streams {@link StatefulPagedSequences} keeps in step. */
    private final class SequenceStates implements StatefulPagedSequences.States {

        @Override
        public void clear(int seq) {
            clearStateSlot(seq);
            for (Map<LayerMemoryPlan.CompressedKvPlan.Kind, CpuCompressedKv[]> layer : compressed) {
                for (CpuCompressedKv[] streams : layer.values()) {
                    streams[seq].reset();
                }
            }
        }

        @Override
        public void restore(int seq, int plane) {
            restoreStreams(seq, plane);
        }

        @Override
        public void copy(int src, int dst) {
            for (Map<LayerMemoryPlan.CompressedKvPlan.Kind, CpuCompressedKv[]> layer : compressed) {
                for (CpuCompressedKv[] streams : layer.values()) {
                    streams[dst].copyFrom(streams[src]);
                }
            }
        }
    }

    /** Copies rollback snapshot plane {@code plane} into the live plane of every compressed stream of {@code seq}. */
    private void restoreStreams(int seq, int plane) {
        for (Map<LayerMemoryPlan.CompressedKvPlan.Kind, CpuCompressedKv[]> layer : compressed) {
            for (CpuCompressedKv[] streams : layer.values()) {
                streams[seq].restorePlane(plane);
            }
        }
    }

    /**
     * Null when the memory has no recurrent state, or holds it in per-sequence slots of a paged plan; otherwise
     * the reason by name. A paged memory with DSV4 compressed caches can: a context checkpoint copies the streams
     * beside the recurrent rows ({@link #saveCompressedState}).
     */
    @Override
    public String sequenceStateRefusal() {
        if (!plan.hasRecurrentState()) {
            return null;
        }
        if (paged == null) {
            return "the contiguous CPU memory keeps one recurrent state, not one per sequence"
                    + " (neuralfree.parallel > 1 or neuralfree.kv.paged=true)";
        }
        return null;
    }

    @Override
    public RecurrentStateCopy saveSequenceState(int seq) {
        checkOpen();
        checkSequenceState(seq);
        List<float[]> rows = new ArrayList<>();
        for (CpuRecurrentState s : ownedStates) {
            rows.addAll(Arrays.asList(s.copySlot(seq)));
        }
        return RecurrentStateCopy.of(rows.toArray(new float[0][]));
    }

    @Override
    public void restoreSequenceState(int seq, RecurrentStateCopy copy) {
        checkOpen();
        Objects.requireNonNull(copy, "copy");
        checkSequenceState(seq);
        int total = 0;
        for (CpuRecurrentState s : ownedStates) {
            total += s.tensors().size();
        }
        if (copy.rows() != total) {
            throw new IllegalArgumentException("state copy " + copy + " does not hold the " + total
                    + " state rows of this memory");
        }
        int first = 0;
        for (CpuRecurrentState s : ownedStates) {
            s.restoreSlot(seq, copy, first);
            first += s.tensors().size();
        }
    }

    @Override
    public void clearSequenceState(int seq) {
        checkOpen();
        checkSequenceState(seq);
        clearStateSlot(seq);
    }

    /**
     * Null when the memory keeps no DSV4 compressed caches, or holds them in per-sequence streams of a paged
     * plan; otherwise the reason by name.
     */
    @Override
    public String compressedStateRefusal() {
        if (!plan.hasCompressed()) {
            return null;
        }
        if (paged == null) {
            return "the contiguous CPU memory keeps one DSV4 compressed stream, not one per sequence"
                    + " (neuralfree.parallel > 1 or neuralfree.kv.paged=true)";
        }
        return null;
    }

    @Override
    public CompressedStateCopy saveCompressedState(int seq) {
        checkOpen();
        checkCompressedState(seq);
        List<CompressedStateCopy.Stream> streams = new ArrayList<>();
        for (Map<LayerMemoryPlan.CompressedKvPlan.Kind, CpuCompressedKv[]> layer : compressed) {
            for (CpuCompressedKv[] kindStreams : layer.values()) {
                streams.add(copyCompressed(kindStreams[seq]));
            }
        }
        CompressorRollback r = rollbacks[seq];
        return CompressedStateCopy.of(streams, r.lastStepTokens(), r.pending());
    }

    @Override
    public void restoreCompressedState(int seq, CompressedStateCopy copy) {
        checkOpen();
        Objects.requireNonNull(copy, "copy");
        checkCompressedState(seq);
        int perSequence = 0;
        for (Map<LayerMemoryPlan.CompressedKvPlan.Kind, CpuCompressedKv[]> layer : compressed) {
            perSequence += layer.size();
        }
        if (copy.streams() != perSequence) {
            throw new IllegalArgumentException("compressed state copy " + copy + " does not hold the " + perSequence
                    + " streams of one sequence of this memory");
        }
        int i = 0;
        for (Map<LayerMemoryPlan.CompressedKvPlan.Kind, CpuCompressedKv[]> layer : compressed) {
            for (CpuCompressedKv[] kindStreams : layer.values()) {
                applyCompressed(kindStreams[seq], copy.stream(i++));
            }
        }
        rollbacks[seq].restoreFrom(copy.lastStepTokens(), copy.pending());
    }

    /** Refuses the compressed-state operations by {@link #compressedStateRefusal()} and a sequence outside. */
    private void checkCompressedState(int seq) {
        String refusal = compressedStateRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        if (paged != null && (seq < 0 || seq >= paged.maxSequences())) {
            throw new IndexOutOfBoundsException("sequence " + seq + " outside [0, " + paged.maxSequences() + ")");
        }
    }

    /** The copied stream of one compressed cache: its compressed rows and both compressor tensors. */
    private static CompressedStateCopy.Stream copyCompressed(CpuCompressedKv c) {
        int rows = c.plan().rows();
        int width = c.plan().width();
        float[][] out = new float[rows][width];
        for (int r = 0; r < rows; r++) {
            c.readK(r, 0, out[r], 0);
        }
        return CompressedStateCopy.Stream.of(out, c.state().tensor(CpuMemory.COMPRESSOR_KV),
                c.state().tensor(CpuMemory.COMPRESSOR_SCORE));
    }

    /** Writes one copied stream back into its compressed cache: the rows and every compressor plane. */
    private static void applyCompressed(CpuCompressedKv c, CompressedStateCopy.Stream copy) {
        int rows = c.plan().rows();
        int width = c.plan().width();
        if (copy.rows() != rows) {
            throw new IllegalArgumentException("compressed state copy holds " + copy.rows() + " rows, the cache of"
                    + " layer " + c.layer() + " keeps " + rows);
        }
        for (int r = 0; r < rows; r++) {
            float[] row = copy.row(r);
            if (row.length != width) {
                throw new IllegalArgumentException("compressed state copy row " + r + " holds " + row.length
                        + " floats, the cache of layer " + c.layer() + " keeps " + width);
            }
            c.writeK(r, row, 0);
        }
        float[] kv = copy.compressorKv();
        float[] score = copy.compressorScore();
        float[] liveKv = c.state().tensor(CpuMemory.COMPRESSOR_KV);
        float[] liveScore = c.state().tensor(CpuMemory.COMPRESSOR_SCORE);
        if (kv.length != liveKv.length || score.length != liveScore.length) {
            throw new IllegalArgumentException("compressed state copy holds " + kv.length + "/" + score.length
                    + " compressor floats, the cache of layer " + c.layer() + " keeps " + liveKv.length + "/"
                    + liveScore.length);
        }
        System.arraycopy(kv, 0, liveKv, 0, kv.length);
        System.arraycopy(score, 0, liveScore, 0, score.length);
    }

    /** Refuses the per-sequence state operations by {@link #sequenceStateRefusal()} and a sequence outside. */
    private void checkSequenceState(int seq) {
        String refusal = sequenceStateRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        if (paged != null && (seq < 0 || seq >= paged.maxSequences())) {
            throw new IndexOutOfBoundsException("sequence " + seq + " outside [0, " + paged.maxSequences() + ")");
        }
    }

    /** Zeroes the live state slot {@code seq} of every recurrent-state layer this memory owns. */
    private void clearStateSlot(int seq) {
        for (CpuRecurrentState s : ownedStates) {
            if (seq < s.slots()) {
                s.clearSlot(seq);
            }
        }
    }

    /** Marks the memory closed; its arrays are released with the object. Idempotent. */
    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "HeapModelMemory{" + plan + ", " + (paged != null ? paged : sequence) + '}';
    }
}

package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.CompressorRollback;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs.GroupPositions;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs.KvShift;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs.SequenceRules;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Heap memory of one model, allocated from a {@link MemoryPlan}: the CPU twin of {@code DeviceModelMemory} with the
 * same plan, layouts, row addressing and sequence rules ({@link SequenceRules}). The position stored in each row of
 * each KV group is kept in a {@link GroupPositions} record; {@link #beginStep(int, int)} marks a step's rows before
 * its K/V are written. A rollback of memory with DSV4 compressed caches stays within their rollback snapshots
 * ({@link CompressorRollback}) and copies that snapshot plane into the live compressor plane of every compressed
 * cache.
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
    private final KvAccessor[] crossKv;
    private final CpuRecurrentState[] recurrent;
    private final KvAccessor[] indexer;
    private final KvAccessor[] msa;
    private final List<Map<LayerMemoryPlan.CompressedKvPlan.Kind, CpuCompressedKv>> compressed;
    private final List<CpuKvLayer> ownedKv = new ArrayList<>();
    private final List<CpuRecurrentState> ownedStates = new ArrayList<>();
    private final List<CpuCompressedKv> ownedCompressed = new ArrayList<>();
    private final CompressorRollback compressorRollback;
    private int[][] shiftDeltas;
    private int snapshotPast = -1;
    private boolean closed;

    private HeapModelMemory(MemoryPlan plan) {
        this.plan = plan;
        this.sequence = new SequenceState(plan.nCtx());
        this.compressorRollback = new CompressorRollback(plan.rollbackSnapshots());
        this.positions = new GroupPositions[plan.groups().size()];
        for (int g = 0; g < positions.length; g++) {
            positions[g] = new GroupPositions(plan.group(g));
        }
        int n = plan.layers().size();
        this.kv = new KvAccessor[n];
        this.crossKv = new KvAccessor[n];
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
     * Allocates every cache the plan describes, with the given head shapes.
     *
     * @throws IllegalArgumentException as {@link #allocate(MemoryPlan, ModelSpec, CpuMemory)}
     */
    public static HeapModelMemory allocate(MemoryPlan plan, KvHeads heads, CpuMemory target) {
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(heads, "heads");
        HeapModelMemory m = new HeapModelMemory(plan);
        m.allocateLayers(heads);
        m.bindShared(target);
        LOGGER.info("CPU memory: KV {} MiB, recurrent state {} MiB (+{} MiB snapshots), side caches {} MiB over {} "
                        + "layers in {} KV groups (ctx {}, ubatch {})", plan.kvBytes() / MIB, plan.stateBytes() / MIB,
                plan.snapshotBytes() / MIB, plan.sideCacheBytes() / MIB, plan.layers().size(), plan.groups().size(),
                plan.nCtx(), plan.ubatch());
        return m;
    }

    private void allocateLayers(KvHeads heads) {
        for (LayerMemoryPlan lp : plan.layers()) {
            int il = lp.layer();
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
            }
            if (lp.msa() != null) {
                msa[il] = CpuMsaCache.allocate(il, lp.msa(), plan.group(lp.kvGroup()));
            }
            for (LayerMemoryPlan.CompressedKvPlan cp : lp.compressed()) {
                if (compressed.get(il).containsKey(cp.kind())) {
                    throw new IllegalArgumentException("layer " + il + " plans two " + cp.kind() + " compressed caches");
                }
                CpuCompressedKv c = CpuCompressedKv.allocate(il, cp);
                ownedCompressed.add(c);
                compressed.get(il).put(cp.kind(), c);
            }
            if (lp.crossKv() != null) {
                int[] h = shape(heads, il, lp.crossKv(), true);
                crossKv[il] = CpuKvLayer.allocate(il, lp.crossKv(), -1, h[0], h[1], h[2]);
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
                indexer[il] = target.indexer(src.layer());
                msa[il] = target.msa(src.layer());
            } else {
                kv[il] = kv[src.layer()];
                indexer[il] = indexer[src.layer()];
                msa[il] = msa[src.layer()];
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

    @Override
    public KvAccessor kv(int il) {
        return kv[checkLayer(il)];
    }

    @Override
    public KvAccessor crossKv(int il) {
        return crossKv[checkLayer(il)];
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
        return compressed.get(checkLayer(il)).get(Objects.requireNonNull(kind, "kind"));
    }

    @Override
    public RecurrentStateAccessor compressorState(int il, LayerMemoryPlan.CompressedKvPlan.Kind kind) {
        CpuCompressedKv c = compressed.get(checkLayer(il)).get(Objects.requireNonNull(kind, "kind"));
        return c != null ? c.state() : null;
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
        compressorRollback.onStep(nTokens);
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
        sequence.clear();
        compressorRollback.reset();
        snapshotPast = -1;
    }

    @Override
    public boolean canRollback(int nTokens) {
        checkOpen();
        if (nTokens < 0) {
            throw new IllegalArgumentException("rollback count " + nTokens + " is negative");
        }
        return nTokens == 0
                || SequenceRules.rollbackBlocker(plan, positions, sequence, compressorRollback, nTokens) == null;
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
        String blocker = SequenceRules.rollbackBlocker(plan, positions, sequence, compressorRollback, nTokens);
        if (blocker != null) {
            throw new IllegalStateException("cannot roll back " + nTokens + " positions: " + blocker);
        }
        if (plan.hasCompressed()) {
            int plane = compressorRollback.onRollback(nTokens);
            for (CpuCompressedKv c : ownedCompressed) {
                c.restorePlane(plane);
            }
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

    /** Marks the memory closed; its arrays are released with the object. Idempotent. */
    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "HeapModelMemory{" + plan + ", " + sequence + '}';
    }
}

package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.CompressorRollback;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs.SequenceRules;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Device memory of one model, allocated from a {@link MemoryPlan}: KV caches of the owning layers, recurrent
 * states, DSA / MSA / DSV4 side caches, T5 cross-attention K/V and the per-step inputs. Every buffer is
 * zero-filled at allocation; all commands go to the queue given at allocation, in order with the forward pass.
 *
 * <p>Sequence operations (the {@link com.eonmux.neuralfree.gguf.memory.ModelMemory} contract):
 * <ul>
 *   <li>{@link #reset()}: {@code kv_pos = -1} in every group, recurrent states and DSV4 compressed caches zeroed,
 *       sequence and snapshot cleared. KV rows are not cleared: attention only reads rows {@code kv_pos} marks.</li>
 *   <li>{@link #canRollback(int)} / {@link #rollback(int)}: memory without recurrent state, when every
 *       sliding-window ring still holds the window of the next query
 *       ({@link SequenceRules#ringBlocker(DeviceStepInputs.GroupPositions[], SequenceState, int, int)}); with DSV4
 *       compressed caches also within the rollback snapshots ({@link CompressorRollback}), whose plane is then
 *       copied into the live compressor plane of every compressed cache.</li>
 *   <li>{@link #snapshot()} / {@link #restore()}: device copies of every recurrent state into the shadow buffers
 *       allocated with the memory when the plan reserves them ({@link MemoryPlan#snapshotsReserved()}); the restore
 *       also drops the KV positions written after the snapshot (same ring rule).</li>
 *   <li>{@link #shiftPositions(int, int)}: the context shift of KV-only memory.</li>
 * </ul>
 * Not thread-safe; used by the thread that drives the queue.
 */
public final class DeviceModelMemory implements GpuMemory {

    private static final Logger LOGGER = LogManager.getLogger(DeviceModelMemory.class);
    private static final long MIB = 1L << 20;

    private final ComputeBackend backend;
    private final ComputeQueue queue;
    private final MemoryPlan plan;
    private final SequenceState sequence;
    private final GpuKvLayer[] kv;
    private final GpuKvLayer[] crossKv;
    private final DeviceRecurrentState[] recurrent;
    private final GpuIndexerCache[] indexer;
    private final GpuMsaCache[] msa;
    private final List<List<GpuCompressedKv>> compressed;
    private final List<DeviceKvLayer> ownedKv = new ArrayList<>();
    private final List<DeviceRecurrentState> ownedStates = new ArrayList<>();
    private final List<DeviceCompressedKv> ownedCompressed = new ArrayList<>();
    private final List<AutoCloseable> owned = new ArrayList<>();
    private final CompressorRollback compressorRollback;
    private DeviceStepInputs stepInputs;
    private int snapshotPast = -1;
    private boolean closed;

    private DeviceModelMemory(ComputeBackend backend, ComputeQueue queue, MemoryPlan plan) {
        this.backend = backend;
        this.queue = queue;
        this.plan = plan;
        this.sequence = new SequenceState(plan.nCtx());
        this.compressorRollback = new CompressorRollback(plan.rollbackSnapshots());
        int n = plan.layers().size();
        this.kv = new GpuKvLayer[n];
        this.crossKv = new GpuKvLayer[n];
        this.recurrent = new DeviceRecurrentState[n];
        this.indexer = new GpuIndexerCache[n];
        this.msa = new GpuMsaCache[n];
        List<List<GpuCompressedKv>> comp = new ArrayList<>(n);
        for (int il = 0; il < n; il++) {
            comp.add(new ArrayList<>());
        }
        this.compressed = comp;
    }

    /**
     * Allocates every buffer the plan describes.
     *
     * @param textPositions section pattern of the RoPE position buffer
     * @param target        memory of the target model whose caches {@code TargetLayer} plans read; null when the
     *                      plan has none
     * @throws IllegalArgumentException when a shared layer's source has no cache, or a target layer is requested
     *                                  without a target memory
     * @throws GpuOutOfMemoryException  when a buffer exceeds the device's maximum allocation; nothing stays
     *                                  allocated then
     */
    public static DeviceModelMemory allocate(ComputeBackend backend, ComputeQueue queue, MemoryPlan plan,
                                             MropeTextPositions textPositions, GpuMemory target) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(queue, "queue");
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(textPositions, "textPositions");
        DeviceModelMemory m = new DeviceModelMemory(backend, queue, plan);
        try {
            m.allocateLayers();
            m.bindShared(target);
            m.stepInputs = DeviceStepInputs.allocate(backend, queue, plan.groups(), plan.ubatch(),
                    m.identityLength(), textPositions, m.sequence);
            m.stepInputs.trackCompressorSteps(m.compressorRollback);
        } catch (RuntimeException e) {
            m.close();
            throw e;
        }
        LOGGER.info("GPU memory: KV {} MiB, recurrent state {} MiB (+{} MiB snapshots), side caches {} MiB, positions "
                        + "{} KiB over {} layers in {} KV groups (ctx {}, ubatch {})",
                plan.kvBytes() / MIB, plan.stateBytes() / MIB, plan.snapshotBytes() / MIB, plan.sideCacheBytes() / MIB,
                plan.positionBytes() / 1024, plan.layers().size(), plan.groups().size(), plan.nCtx(), plan.ubatch());
        return m;
    }

    private void allocateLayers() {
        for (LayerMemoryPlan lp : plan.layers()) {
            int il = lp.layer();
            if (lp.ownsKv()) {
                DeviceKvLayer c = own(DeviceKvLayer.allocate(backend, queue, il, lp.kv(), lp.kvGroup(), lp.rotation(),
                        "KV cache of layer " + il));
                ownedKv.add(c);
                kv[il] = c;
            }
            if (lp.hasState()) {
                DeviceRecurrentState s = own(DeviceRecurrentState.allocate(backend, queue, il, lp.states(),
                        plan.snapshotsReserved()));
                ownedStates.add(s);
                recurrent[il] = s;
            }
            if (lp.indexer() != null) {
                indexer[il] = own(DeviceIndexerCache.allocate(backend, queue, il, lp.indexer(), lp.kvGroup()));
            }
            if (lp.msa() != null) {
                msa[il] = own(DeviceMsaCache.allocate(backend, queue, il, lp.msa(), lp.kvGroup()));
            }
            for (LayerMemoryPlan.CompressedKvPlan cp : lp.compressed()) {
                DeviceCompressedKv c = own(DeviceCompressedKv.allocate(backend, queue, il, cp));
                ownedCompressed.add(c);
                compressed.get(il).add(c);
            }
            if (lp.crossKv() != null) {
                crossKv[il] = own(DeviceKvLayer.allocate(backend, queue, il, lp.crossKv(), -1, lp.rotation(),
                        "cross-attention KV of layer " + il));
            }
        }
        for (int il = 0; il < compressed.size(); il++) {
            compressed.set(il, Collections.unmodifiableList(compressed.get(il)));
        }
    }

    private void bindShared(GpuMemory target) {
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

    private int identityLength() {
        int n = plan.ubatch();
        for (LayerMemoryPlan lp : plan.layers()) {
            if (lp.crossKv() != null) {
                n = Math.max(n, lp.crossKv().rows());
            }
        }
        return n;
    }

    private <T extends AutoCloseable> T own(T resource) {
        owned.add(resource);
        return resource;
    }

    // ------------------------------------------------------------------ GpuMemory

    @Override
    public MemoryPlan plan() {
        return plan;
    }

    @Override
    public SequenceState sequence() {
        return sequence;
    }

    @Override
    public GpuKvLayer kv(int il) {
        return kv[checkLayer(il)];
    }

    @Override
    public GpuKvLayer crossKv(int il) {
        return crossKv[checkLayer(il)];
    }

    @Override
    public GpuRecurrentState recurrent(int il) {
        return recurrent[checkLayer(il)];
    }

    @Override
    public GpuIndexerCache indexer(int il) {
        return indexer[checkLayer(il)];
    }

    @Override
    public GpuMsaCache msa(int il) {
        return msa[checkLayer(il)];
    }

    @Override
    public List<GpuCompressedKv> compressed(int il) {
        return compressed.get(checkLayer(il));
    }

    @Override
    public DeviceStepInputs stepInputs() {
        checkOpen();
        return stepInputs;
    }

    // ------------------------------------------------------------------ sequence operations

    @Override
    public void reset() {
        checkOpen();
        for (DeviceRecurrentState s : ownedStates) {
            s.reset(queue);
        }
        for (DeviceCompressedKv c : ownedCompressed) {
            c.reset(queue);
        }
        stepInputs.clear();
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
        if (nTokens == 0) {
            return true;
        }
        return SequenceRules.rollbackBlocker(plan, stepInputs.allPositions(), sequence, compressorRollback, nTokens)
                == null;
    }

    @Override
    public boolean ringsKeepWindow(int nWrite, int nDrop) {
        checkOpen();
        return SequenceRules.ringBlocker(stepInputs.allPositions(), sequence, nWrite, nDrop) == null;
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
        String blocker = SequenceRules.rollbackBlocker(plan, stepInputs.allPositions(), sequence, compressorRollback,
                nTokens);
        if (blocker != null) {
            throw new IllegalStateException("cannot roll back " + nTokens + " positions: " + blocker);
        }
        if (plan.hasCompressed()) {
            int plane = compressorRollback.onRollback(nTokens);
            for (DeviceCompressedKv c : ownedCompressed) {
                c.restorePlane(queue, plane);
            }
        }
        stepInputs.drop(sequence.nPast() - nTokens, nTokens);
        sequence.truncate(nTokens);
    }

    @Override
    public void snapshot() {
        checkOpen();
        String blocker = SequenceRules.snapshotBlocker(plan);
        if (blocker != null) {
            throw new UnsupportedOperationException(blocker);
        }
        for (DeviceRecurrentState s : ownedStates) {
            s.snapshot(queue);
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
        String blocker = n > 0 ? SequenceRules.ringBlocker(stepInputs.allPositions(), sequence, 0, n) : null;
        if (blocker != null) {
            throw new IllegalStateException("cannot restore the snapshot: " + blocker);
        }
        for (DeviceRecurrentState s : ownedStates) {
            s.restore(queue);
        }
        if (n > 0) {
            stepInputs.drop(snapshotPast, n);
            sequence.truncate(n);
        }
    }

    // ------------------------------------------------------------------ context shift

    /**
     * Context shift (llama.cpp {@code seq_rm} + {@code seq_add}): drops positions {@code [nKeep, nKeep + nDiscard)}
     * and moves every later position down by {@code nDiscard}. Rows move so that each position stays in
     * {@code rowOf(pos)} (in-place copies for full caches, through a staging buffer for ring rotations), every
     * group's {@code kv_pos} is rewritten, and the per-row deltas are written for {@code LLM_KV_RopeShift_*}, which
     * each attention mixer then applies to its K rows. Callers check that every mixer supports the shift first.
     *
     * @return the {@code shift_delta} buffer of each KV group, indexed by group
     * @throws IllegalArgumentException      for {@code nKeep < 0}, {@code nDiscard <= 0} or a range beyond the
     *                                       stored positions
     * @throws UnsupportedOperationException for recurrent state, sparse side caches, caches read from a target
     *                                       model, or a ring that no longer holds kept positions the next query
     *                                       attends ({@link SequenceRules#ringShiftBlocker}); nothing changes then
     */
    public List<GpuBuffer> shiftPositions(int nKeep, int nDiscard) {
        checkOpen();
        SequenceRules.checkShift(sequence, nKeep, nDiscard);
        String why = SequenceRules.shiftBlocker(plan);
        if (why != null) {
            throw new UnsupportedOperationException("context shift unsupported: " + why);
        }
        int groups = stepInputs.groups();
        for (int g = 0; g < groups; g++) {
            String ring = SequenceRules.ringShiftBlocker(stepInputs.positions(g), sequence, nKeep, nDiscard);
            if (ring != null) {
                throw new UnsupportedOperationException("context shift unsupported: " + ring);
            }
        }
        DeviceStepInputs.KvShift[] shifts = new DeviceStepInputs.KvShift[groups];
        for (int g = 0; g < groups; g++) {
            shifts[g] = stepInputs.positions(g).planShift(nKeep, nDiscard);
        }
        stepInputs.reserveShiftDeltas();
        long stagingBytes = 0;
        for (DeviceKvLayer c : ownedKv) {
            DeviceStepInputs.KvShift s = shifts[c.group()];
            if (s.needsStaging()) {
                stagingBytes = Math.max(stagingBytes, s.movedRows() * c.maxRowBytes());
            }
        }
        GpuBuffer staging = stagingBytes > 0 ? allocate(backend, stagingBytes, "context-shift staging") : null;
        try {
            for (DeviceKvLayer c : ownedKv) {
                c.moveRows(queue, shifts[c.group()], staging);
            }
        } finally {
            if (staging != null) {
                queue.finish();
                staging.close();
            }
        }
        List<GpuBuffer> deltas = new ArrayList<>(groups);
        for (int g = 0; g < groups; g++) {
            stepInputs.applyShift(g, shifts[g]);
            deltas.add(stepInputs.shiftDelta(g));
        }
        SequenceRules.shiftSequence(sequence, nKeep, nDiscard);
        return Collections.unmodifiableList(deltas);
    }

    /** {@code shift_delta} of a KV group after the last {@link #shiftPositions(int, int)}. */
    public GpuBuffer shiftDelta(int group) {
        checkOpen();
        return stepInputs.shiftDelta(group);
    }

    // ------------------------------------------------------------------ allocation helpers

    /**
     * Allocates a device buffer of {@code bytes} rounded up to a multiple of 4 and zero-fills it on {@code queue}.
     *
     * @param what description for error messages
     * @throws GpuOutOfMemoryException when the size exceeds the device's maximum allocation
     */
    static GpuBuffer allocateZeroed(ComputeBackend backend, ComputeQueue queue, long bytes, String what) {
        GpuBuffer b = allocate(backend, bytes, what);
        try {
            queue.enqueueFillBuffer(b, 0, roundUp4(bytes), 0);
        } catch (RuntimeException e) {
            b.close();
            throw e;
        }
        return b;
    }

    private static GpuBuffer allocate(ComputeBackend backend, long bytes, String what) {
        if (bytes <= 0) {
            throw new IllegalArgumentException(what + " has a non-positive size of " + bytes + " bytes");
        }
        long size = roundUp4(bytes);
        long max = backend.capabilities().getMaxAllocSizeBytes();
        if (max > 0 && size > max) {
            throw new GpuOutOfMemoryException(what + " needs " + size + " bytes, above the device's maximum allocation of "
                    + max + " bytes; lower neuralfree.ctx or choose a smaller neuralfree.kv.type", size,
                    backend.backendType().name());
        }
        return backend.allocateBuffer(size, false);
    }

    private static long roundUp4(long bytes) {
        return (bytes + 3) & ~3L;
    }

    private int checkLayer(int il) {
        checkOpen();
        if (il < 0 || il >= kv.length) {
            throw new IndexOutOfBoundsException("memory has no layer " + il + " (" + kv.length + " layers)");
        }
        return il;
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("GPU memory is closed");
        }
    }

    /** Releases every owned buffer; caches read from a target model stay open. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        if (stepInputs != null) {
            try {
                stepInputs.close();
            } catch (RuntimeException e) {
                failure = e;
            }
        }
        for (int i = owned.size() - 1; i >= 0; i--) {
            try {
                owned.get(i).close();
            } catch (Exception e) {
                if (failure == null) {
                    failure = e instanceof RuntimeException ? (RuntimeException) e
                            : new IllegalStateException("failed to release GPU memory", e);
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        owned.clear();
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "DeviceModelMemory{" + plan + ", " + sequence + '}';
    }
}

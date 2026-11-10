package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.CompressorRollback;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.RecurrentStateCopy;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.StateTransfer;
import com.eonmux.neuralfree.gguf.memory.paged.PageCopy;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.memory.paged.RowMove;
import com.eonmux.neuralfree.gguf.memory.paged.StatefulPagedSequences;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.IntPredicate;

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
 *       ({@link SequenceRules#ringBlocker(GroupPositions[], SequenceState, int, int)}); with DSV4
 *       compressed caches also within the rollback snapshots ({@link CompressorRollback}), whose plane is then
 *       copied into the live compressor plane of every compressed cache.</li>
 *   <li>{@link #snapshot()} / {@link #restore()}: device copies of every recurrent state into the shadow buffers
 *       allocated with the memory when the plan reserves them ({@link MemoryPlan#snapshotsReserved()}); the restore
 *       also drops the KV positions written after the snapshot (same ring rule).</li>
 *   <li>{@link #shiftPositions(int, int)}: the context shift of KV-only contiguous memory;
 *       {@link #shiftSequence(int, int, int)}: the per-sequence shift of a paged memory, whose moved keys the
 *       attention mixers re-rotate before {@link #moveShiftedRows} carries them to their new cells.</li>
 *   <li>{@link #stateTransfer()}: host reads and writes of the K/V rows, recurrent states and position mirrors,
 *       under the session file ({@link #saveSession}) and the slot files of a paged memory.</li>
 *   <li>{@link #pagedSequences()}: on a paged plan ({@code neuralfree.parallel > 1} or
 *       {@code neuralfree.kv.paged}) the {@link PagedSequences} over the plan's page pools, which
 *       {@code GpuForwardBackend.forwardBatch} drives: {@code beginPagedStep} uploads a step's page tables,
 *       {@code cell_idx} and {@code kv_pos} to the device, {@link #copyPage(PageCopy)} enqueues a page copy on the
 *       cache layers and cell-addressed side caches, and the paged attention kernels address the rows through
 *       them.</li>
 *   <li>Per-sequence streams: a paged plan keeps one DSV4 compressed stream of each kind and one T5 cross-attention
 *       stream per sequence ({@link #compressed(int, LayerMemoryPlan.CompressedKvPlan.Kind, int)},
 *       {@link #crossKv(int, int)}); the single-sequence accessors serve stream 0. {@link #sequences()} keeps the
 *       compressed streams in step with the sequence operations ({@link StatefulPagedSequences}) and every paged
 *       segment advances its sequence's rollback record.</li>
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
    /** Per layer, the cross-attention K/V of every stream (one per paged sequence); null without one. */
    private final GpuKvLayer[][] crossKv;
    private final DeviceRecurrentState[] recurrent;
    private final GpuIndexerCache[] indexer;
    private final GpuMsaCache[] msa;
    private final List<List<GpuCompressedKv>> compressed;
    private final List<DeviceKvLayer> ownedKv = new ArrayList<>();
    private final List<DeviceRecurrentState> ownedStates = new ArrayList<>();
    /**
     * Per layer and kind, every compressed stream the plan describes ({@code CompressedKvPlan.streams()}, one per
     * paged sequence), indexed by sequence.
     */
    private final List<Map<LayerMemoryPlan.CompressedKvPlan.Kind, DeviceCompressedKv[]>> streams;
    /** The cell-addressed indexer and MSA key buffers of a group this memory owns, which page copies move. */
    private final List<SideRows> ownedSideRows = new ArrayList<>();
    private final List<AutoCloseable> owned = new ArrayList<>();
    /** The compressor rollback bookkeeping of each sequence; entry 0 also rules the single-sequence calls. */
    private final CompressorRollback[] rollbacks;
    private final PagedSequences paged;
    /** The paged sequences with the compressed streams kept in step ({@link #sequences()}). */
    private final StatefulPagedSequences pagedWithStates;
    private DeviceStepInputs stepInputs;
    private int snapshotPast = -1;
    /** The page copies the running {@link #shiftSequence} enqueued, released once the queue executed them. */
    private final List<PageCopy> shiftCopies = new ArrayList<>();
    /** Per-row RoPE deltas of the last {@link #shiftSequence}, group order ({@link #pagedShiftDeltas()}). */
    private int[][] lastShiftDeltas;
    private boolean closed;

    private DeviceModelMemory(ComputeBackend backend, ComputeQueue queue, MemoryPlan plan) {
        this.backend = backend;
        this.queue = queue;
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
        int n = plan.layers().size();
        this.kv = new GpuKvLayer[n];
        this.crossKv = new GpuKvLayer[n][];
        this.recurrent = new DeviceRecurrentState[n];
        this.indexer = new GpuIndexerCache[n];
        this.msa = new GpuMsaCache[n];
        List<List<GpuCompressedKv>> comp = new ArrayList<>(n);
        this.streams = new ArrayList<>(n);
        for (int il = 0; il < n; il++) {
            comp.add(new ArrayList<>());
            streams.add(new EnumMap<>(LayerMemoryPlan.CompressedKvPlan.Kind.class));
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
        return allocate(backend, queue, plan, textPositions, target, il -> true);
    }

    /**
     * Allocates the buffers of the virtual layers {@code layerGate} names; the other layers of the plan stay
     * unallocated ({@code kv(il)} and friends read null for them). The per-group position and step buffers are
     * allocated for every group of the plan, as both sides of a hybrid load address the same groups. A hybrid load
     * allocates the layers of its side over the shared plan this way.
     *
     * @throws IllegalArgumentException as {@link #allocate(ComputeBackend, ComputeQueue, MemoryPlan,
     *                                      MropeTextPositions, GpuMemory)}
     * @throws GpuOutOfMemoryException  when a buffer exceeds the device's maximum allocation; nothing stays
     *                                  allocated then
     */
    public static DeviceModelMemory allocate(ComputeBackend backend, ComputeQueue queue, MemoryPlan plan,
                                             MropeTextPositions textPositions, GpuMemory target,
                                             IntPredicate layerGate) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(queue, "queue");
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(textPositions, "textPositions");
        Objects.requireNonNull(layerGate, "layerGate");
        DeviceModelMemory m = new DeviceModelMemory(backend, queue, plan);
        try {
            m.allocateLayers(layerGate);
            m.bindShared(target);
            m.stepInputs = DeviceStepInputs.allocate(backend, queue, plan.groups(), plan.ubatch(),
                    plan.identityLength(), textPositions, m.sequence, m.paged);
            m.stepInputs.trackCompressorSteps(m.rollbacks);
            if (plan.contextShiftReserved()) {
                m.stepInputs.reserveShiftDeltas();
            }
        } catch (RuntimeException e) {
            m.close();
            throw e;
        }
        LOGGER.info("GPU memory: KV {} MiB, recurrent state {} MiB (+{} MiB snapshots), side caches {} MiB, positions "
                        + "{} KiB over {} layers in {} KV groups (ctx {}, ubatch{}), {}",
                plan.kvBytes() / MIB, plan.stateBytes() / MIB, plan.snapshotBytes() / MIB, plan.sideCacheBytes() / MIB,
                plan.positionBytes() / 1024, plan.layers().size(), plan.groups().size(), plan.nCtx(), plan.ubatch(),
                m.paged != null ? "paged " + m.paged : "single sequence");
        return m;
    }

    private void allocateLayers(IntPredicate layerGate) {
        for (LayerMemoryPlan lp : plan.layers()) {
            int il = lp.layer();
            if (!layerGate.test(il)) {
                continue;
            }
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
                ownedSideRows.add(new SideRows(lp.kvGroup(), indexer[il].keys(),
                        lp.indexer().dtype().rowBytes(lp.indexer().width())));
            }
            if (lp.msa() != null) {
                msa[il] = own(DeviceMsaCache.allocate(backend, queue, il, lp.msa(), lp.kvGroup()));
                ownedSideRows.add(new SideRows(lp.kvGroup(), msa[il].indexKeys(),
                        lp.msa().dtype().rowBytes(lp.msa().width())));
            }
            for (LayerMemoryPlan.CompressedKvPlan cp : lp.compressed()) {
                if (streams.get(il).containsKey(cp.kind())) {
                    throw new IllegalArgumentException("layer " + il + " plans two " + cp.kind() + " compressed caches");
                }
                if (cp.streams() != rollbacks.length) {
                    throw new IllegalArgumentException("layer " + il + " plans " + cp.streams() + " " + cp.kind()
                            + " streams for " + rollbacks.length + " sequences");
                }
                DeviceCompressedKv[] kindStreams = new DeviceCompressedKv[cp.streams()];
                for (int seq = 0; seq < kindStreams.length; seq++) {
                    kindStreams[seq] = own(DeviceCompressedKv.allocate(backend, queue, il, cp));
                }
                streams.get(il).put(cp.kind(), kindStreams);
                compressed.get(il).add(kindStreams[0]);
            }
            if (lp.crossStreams() > 0) {
                crossKv[il] = new GpuKvLayer[lp.crossStreams()];
            }
            for (int seq = 0; seq < lp.crossStreams(); seq++) {
                crossKv[il][seq] = own(DeviceKvLayer.allocate(backend, queue, il, lp.crossKv(), -1, lp.rotation(),
                        "cross-attention KV of layer " + il + (seq > 0 ? " stream " + seq : "")));
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
                msa[il] = target.msa(src.layer());
            } else {
                kv[il] = kv[src.layer()];
                msa[il] = msa[src.layer()];
            }
            // a KV-sharing layer with its own indexer plan (mistral-large4) allocated it above
            if (lp.indexer() == null) {
                indexer[il] = src.target() ? target.indexer(src.layer()) : indexer[src.layer()];
            }
            if (kv[il] == null) {
                throw new IllegalArgumentException("layer " + il + " shares the KV of " + src + ", which has no cache");
            }
        }
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
        GpuKvLayer[] layer = crossKv[checkLayer(il)];
        return layer == null ? null : layer[0];
    }

    @Override
    public GpuKvLayer crossKv(int il, int seq) {
        GpuKvLayer[] layer = crossKv[checkLayer(il)];
        if (layer == null) {
            return null;
        }
        if (seq < 0 || seq >= layer.length) {
            throw new IndexOutOfBoundsException("layer " + il + " holds " + layer.length
                    + " cross-attention streams, not sequence " + seq);
        }
        return layer[seq];
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
    public GpuCompressedKv compressed(int il, LayerMemoryPlan.CompressedKvPlan.Kind kind, int seq) {
        DeviceCompressedKv[] kindStreams = streams.get(checkLayer(il)).get(Objects.requireNonNull(kind, "kind"));
        if (kindStreams == null) {
            return null;
        }
        if (seq < 0 || seq >= kindStreams.length) {
            throw new IndexOutOfBoundsException("layer " + il + " holds " + kindStreams.length + " " + kind
                    + " compressed streams, not sequence " + seq);
        }
        return kindStreams[seq];
    }

    @Override
    public DeviceStepInputs stepInputs() {
        checkOpen();
        return stepInputs;
    }

    // ------------------------------------------------------------------ paged sequences

    /** The paged bookkeeping over the plan's page pools on a paged plan, else null. */
    @Override
    public PagedSequences pagedSequences() {
        checkOpen();
        return paged;
    }

    /**
     * The {@link PagedSequences} over the plan's page pools on a paged plan, run through
     * {@link StatefulPagedSequences} so the sequence operations keep each sequence's DSV4 compressed streams in step;
     * else the contiguous adapter.
     */
    @Override
    public SequenceMemory sequences() {
        checkOpen();
        return pagedWithStates != null ? pagedWithStates : GpuMemory.super.sequences();
    }

    @Override
    public String pagedStepRefusal(PagedStep step) {
        return paged == null ? getClass().getSimpleName() + " holds no paged sequences" : null;
    }

    /**
     * Uploads the paged step to the device: the page bookkeeping's queued {@code kv_pos} changes, the step's row
     * writes, {@code cell_idx}, the per-group page tables of the step's segments in launch order and the
     * per-sequence addressing the paged attention kernels read.
     *
     * @throws UnsupportedOperationException when this memory is not paged
     * @throws IllegalArgumentException      when the step belongs to another memory
     */
    @Override
    public void beginPagedStep(PagedStep step, int[] positions, int[][] ropeSections) {
        checkOpen();
        String refusal = pagedStepRefusal(step);
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        stepInputs.beginPagedStep(step, positions, ropeSections);
    }

    /**
     * {@link #beginPagedStep(PagedStep, int[], int[][])} with the step's mask rule: a {@code nonCausal} step's rows
     * attend each other without the causal mask (a media chunk's batch) until {@link #endPagedStep}.
     *
     * @throws UnsupportedOperationException when this memory is not paged
     */
    public void beginPagedStep(PagedStep step, int[] positions, int[][] ropeSections, boolean nonCausal) {
        checkOpen();
        String refusal = pagedStepRefusal(step);
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        stepInputs.beginPagedStep(step, positions, ropeSections, nonCausal);
    }

    @Override
    public void endPagedStep() {
        checkOpen();
        stepInputs.endPagedStep();
    }

    @Override
    public PagedStep currentPagedStep() {
        checkOpen();
        return stepInputs.currentPagedStep();
    }

    /**
     * Enqueues one page copy of the copy-on-write protocol on the device: every owned KV cache layer of the copy's
     * group moves rows {@code [srcPage * pageSize + srcRow0, .. + rows)} to the destination page's rows
     * {@code [dstPage * pageSize + dstRow0, .. + rows)} (a copy-on-write copy lands at the same rows, a KV chunk
     * reuse's copy at the destination's own slots), and so does every owned cell-addressed side cache of the group
     * (DSA indexer keys, MSA index keys), whose rows follow the KV cells. A group no owned cache reads (a
     * recurrent-only model's bookkeeping group) moves nothing.
     */
    @Override
    public void copyPage(PageCopy copy) {
        checkOpen();
        int ps = plan.group(0).pageSize();
        long src = (long) copy.srcPage() * ps + copy.srcRow0();
        long dst = (long) copy.dstPage() * ps + copy.dstRow0();
        for (DeviceKvLayer c : ownedKv) {
            if (c.group() != copy.group()) {
                continue;
            }
            KvLayout l = c.layout();
            long kBytes = (long) copy.rows() * l.rowBytesK();
            queue.enqueueCopyBuffer(c.k(), src * l.rowBytesK(), c.k(), dst * l.rowBytesK(), kBytes);
            if (!l.vAliasesK()) {
                queue.enqueueCopyBuffer(c.v(), src * l.rowBytesV(), c.v(), dst * l.rowBytesV(),
                        (long) copy.rows() * l.rowBytesV());
            }
            copyScales(c, l, src, dst, copy.rows());
        }
        for (SideRows side : ownedSideRows) {
            if (side.group == copy.group()) {
                queue.enqueueCopyBuffer(side.keys, src * side.rowBytes, side.keys, dst * side.rowBytes,
                        copy.rows() * side.rowBytes);
            }
        }
        // a group no owned cache reads (a recurrent-only model's bookkeeping group, whose pages only track the
        // sequences' slots) has nothing to move: the pool accounting is the copy
    }

    /**
     * Copy of the paged host {@code kv_pos} mirror of every KV group (group order), or null for the contiguous form.
     * The mirror holds the positions the paged attention kernels read alongside the page tables: a copy, a removal
     * and a sliding-window trim change it between steps, so a tensor-parallel rank follows it from the memory that
     * prepared the step ({@link #installPagedMirror}).
     */
    public int[][] pagedMirrorSnapshot() {
        checkOpen();
        if (paged == null) {
            return null;
        }
        int groups = stepInputs.groups();
        int[][] out = new int[groups][];
        for (int g = 0; g < groups; g++) {
            out[g] = stepInputs.positions(g).snapshot();
        }
        return out;
    }

    /**
     * Installs a paged host {@code kv_pos} mirror ({@link #pagedMirrorSnapshot()}) on this memory: the next
     * {@link #beginPagedStep} uploads it whole. A rank cache takes the mirror of the memory whose page bookkeeping
     * planned the step, so copies, removals and trims the rank never performed itself reach its kernels.
     *
     * @throws UnsupportedOperationException for the contiguous form
     * @throws IllegalArgumentException      when the snapshot holds another group count or row count
     */
    public void installPagedMirror(int[][] mirror) {
        checkOpen();
        Objects.requireNonNull(mirror, "mirror");
        if (paged == null) {
            throw new UnsupportedOperationException("the contiguous memory holds no paged kv_pos mirror");
        }
        if (mirror.length != stepInputs.groups()) {
            throw new IllegalArgumentException("a paged mirror of " + mirror.length + " KV group(s) cannot install on "
                    + stepInputs.groups());
        }
        for (int g = 0; g < mirror.length; g++) {
            GroupPositions positions = stepInputs.positions(g);
            if (mirror[g].length != positions.rows()) {
                throw new IllegalArgumentException("group " + g + " mirror holds " + mirror[g].length + " row(s) to "
                        + positions.rows());
            }
            for (int row = 0; row < mirror[g].length; row++) {
                positions.set(row, mirror[g][row]);
            }
        }
    }

    // ------------------------------------------------------------------ sequence operations

    @Override
    public void reset() {
        checkOpen();
        for (DeviceRecurrentState s : ownedStates) {
            s.reset(queue);
        }
        for (Map<LayerMemoryPlan.CompressedKvPlan.Kind, DeviceCompressedKv[]> layer : streams) {
            for (DeviceCompressedKv[] kindStreams : layer.values()) {
                for (DeviceCompressedKv c : kindStreams) {
                    c.reset(queue);
                }
            }
        }
        stepInputs.clear();
        if (paged != null) {
            paged.clear();
            paged.clearReports();
        }
        sequence.clear();
        for (CompressorRollback r : rollbacks) {
            r.reset();
        }
        shiftCopies.clear();
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
        return SequenceRules.rollbackBlocker(plan, stepInputs.allPositions(), sequence, rollbacks[0], nTokens)
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
        String blocker = SequenceRules.rollbackBlocker(plan, stepInputs.allPositions(), sequence, rollbacks[0],
                nTokens);
        if (blocker != null) {
            throw new IllegalStateException("cannot roll back " + nTokens + " positions: " + blocker);
        }
        if (plan.hasCompressed()) {
            restoreStreams(0, rollbacks[0].onRollback(nTokens));
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

    // ------------------------------------------------------------------ per-sequence recurrent state

    /**
     * Null when the memory has no recurrent state, or holds it in per-sequence rows of a paged plan; otherwise
     * the reason by name.
     */
    @Override
    public String sequenceStateRefusal() {
        if (!plan.hasRecurrentState()) {
            return null;
        }
        return paged != null ? null
                : "the contiguous GPU memory keeps one recurrent state, not one per sequence"
                        + " (neuralfree.parallel > 1 or neuralfree.kv.paged=true)";
    }

    /**
     * Copies the state row of paged sequence {@code seq} of every state tensor into a host copy: the recurrent
     * half of a context checkpoint over the device memory. The queue finishes before the copy returns.
     *
     * @throws UnsupportedOperationException carrying {@link #sequenceStateRefusal()} when it is not null
     */
    @Override
    public RecurrentStateCopy saveSequenceState(int seq) {
        checkOpen();
        checkSequenceRow(seq);
        List<float[]> rows = new ArrayList<>();
        for (DeviceRecurrentState s : ownedStates) {
            s.readSlot(queue, seq, rows);
        }
        queue.finish();
        return RecurrentStateCopy.of(rows.toArray(new float[0][]));
    }

    /**
     * Writes {@code copy} (taken by {@link #saveSequenceState} on this memory) into the state row of paged
     * sequence {@code seq}; the queue finishes before the write returns.
     *
     * @throws UnsupportedOperationException carrying {@link #sequenceStateRefusal()} when it is not null
     * @throws IllegalArgumentException      when the copy's rows do not match this memory's state tensors
     */
    @Override
    public void restoreSequenceState(int seq, RecurrentStateCopy copy) {
        checkOpen();
        Objects.requireNonNull(copy, "copy");
        checkSequenceRow(seq);
        int total = 0;
        for (DeviceRecurrentState s : ownedStates) {
            total += s.tensors().size();
        }
        if (copy.rows() != total) {
            throw new IllegalArgumentException("state copy " + copy + " does not hold the " + total
                    + " state rows of this memory");
        }
        int first = 0;
        for (DeviceRecurrentState s : ownedStates) {
            s.writeSlot(queue, seq, copy, first);
            first += s.tensors().size();
        }
        queue.finish();
    }

    /**
     * Zeroes the state row of paged sequence {@code seq}, so the next sequence in the slot starts from a fresh
     * state.
     *
     * @throws UnsupportedOperationException carrying {@link #sequenceStateRefusal()} when it is not null
     */
    @Override
    public void clearSequenceState(int seq) {
        checkOpen();
        checkSequenceRow(seq);
        for (DeviceRecurrentState s : ownedStates) {
            s.clearSlot(queue, seq);
        }
    }

    /** Refuses the per-sequence state operations by {@link #sequenceStateRefusal()} and a sequence outside. */
    private void checkSequenceRow(int seq) {
        String refusal = sequenceStateRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        if (paged != null && (seq < 0 || seq >= paged.maxSequences())) {
            throw new IndexOutOfBoundsException("sequence " + seq + " outside [0, " + paged.maxSequences() + ")");
        }
    }

    /** Copies rollback snapshot plane {@code plane} into the live plane of every compressed stream of {@code seq}. */
    private void restoreStreams(int seq, int plane) {
        for (Map<LayerMemoryPlan.CompressedKvPlan.Kind, DeviceCompressedKv[]> layer : streams) {
            for (DeviceCompressedKv[] kindStreams : layer.values()) {
                kindStreams[seq].restorePlane(queue, plane);
            }
        }
    }

    /** The per-sequence state {@link StatefulPagedSequences} keeps in step: the DSV4 compressed streams. */
    private final class SequenceStates implements StatefulPagedSequences.States {

        @Override
        public void clear(int seq) {
            for (Map<LayerMemoryPlan.CompressedKvPlan.Kind, DeviceCompressedKv[]> layer : streams) {
                for (DeviceCompressedKv[] kindStreams : layer.values()) {
                    kindStreams[seq].reset(queue);
                }
            }
        }

        @Override
        public void restore(int seq, int plane) {
            restoreStreams(seq, plane);
        }

        @Override
        public void copy(int src, int dst) {
            for (Map<LayerMemoryPlan.CompressedKvPlan.Kind, DeviceCompressedKv[]> layer : streams) {
                for (DeviceCompressedKv[] kindStreams : layer.values()) {
                    kindStreams[dst].copyFrom(queue, kindStreams[src]);
                }
            }
        }
    }

    /** One owned cell-addressed side cache buffer of a KV group: its rows follow the group's cells. */
    private static final class SideRows {
        final int group;
        final GpuBuffer keys;
        final long rowBytes;

        SideRows(int group, GpuBuffer keys, long rowBytes) {
            this.group = group;
            this.keys = keys;
            this.rowBytes = rowBytes;
        }
    }

    // ------------------------------------------------------------------ state transfer

    /**
     * The host transfer of this memory's state ({@link StateTransfer}): K/V rows of every owned cache layer and the
     * live recurrent state tensors move through the memory's queue, and the contiguous form's position mirrors install
     * on the host and the device {@code kv_pos} alike. Session and slot files and the host-RAM prompt cache read and
     * write the device state through it.
     */
    @Override
    public StateTransfer stateTransfer() {
        checkOpen();
        return new DeviceStateTransfer(queue, stepInputs, ownedKv, ownedStates);
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
        if (paged != null) {
            throw new UnsupportedOperationException("context shift unsupported: a paged memory maps positions"
                    + " through its page tables; shift through " + SequenceMemory.class.getSimpleName()
                    + ".add (PagedSequences.add)");
        }
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
        KvShift[] shifts = new KvShift[groups];
        for (int g = 0; g < groups; g++) {
            shifts[g] = stepInputs.positions(g).planShift(nKeep, nDiscard);
        }
        stepInputs.reserveShiftDeltas();
        long stagingBytes = 0;
        for (DeviceKvLayer c : ownedKv) {
            KvShift s = shifts[c.group()];
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

    /**
     * The paged context shift of one sequence (llama.cpp server's slot shift, server-context.cpp:3800-3808):
     * removes its positions {@code [nKeep, nKeep + nDiscard)} and moves every later position down by
     * {@code nDiscard} ({@code seq_rm} + {@code seq_add}). The move runs on the sequence view
     * ({@link #sequences()}, whose refusals stand — a DeepSeek-V4 compressor refuses the interior removal by
     * name), the privatizations it queued copy on the device at once, the move's reports apply to the position
     * mirrors at once, and the move's per-row RoPE deltas install whole in every group's {@code shift_delta}
     * buffer. The caller then re-rotates the keys ({@code LLM_KV_RopeShift_*} over these deltas, as llama.cpp's
     * K-shift graph ropes every cell by its per-cell shift), and finishes with {@link #moveShiftedRows()}: the
     * rows still sit at the cells the deltas name until then, and the device {@code kv_pos} follows at the next
     * {@link #beginPagedStep}.
     *
     * @throws IllegalArgumentException      for {@code nKeep < 0}, {@code nDiscard <= 0} or a sequence outside the
     *                                       memory
     * @throws UnsupportedOperationException for the contiguous form ({@link #shiftPositions} shifts it), and any
     *                                       refusal of the sequence view
     */
    public void shiftSequence(int seq, int nKeep, int nDiscard) {
        checkOpen();
        if (paged == null) {
            throw new UnsupportedOperationException("context shift unsupported: the contiguous memory shifts"
                    + " through " + DeviceModelMemory.class.getSimpleName() + ".shiftPositions");
        }
        if (nKeep < 0 || nDiscard <= 0) {
            throw new IllegalArgumentException("context shift needs nKeep >= 0 and nDiscard > 0, got " + nKeep + "/"
                    + nDiscard);
        }
        if (seq < 0 || seq >= paged.maxSequences()) {
            throw new IndexOutOfBoundsException("sequence " + seq + " outside [0, " + paged.maxSequences() + ")");
        }
        try {
            sequences().remove(seq, nKeep, nKeep + nDiscard);
            sequences().add(seq, nKeep + nDiscard, -1, -nDiscard);
            runShiftCopies();
            installShiftDeltas();
        } catch (RuntimeException e) {
            queue.finish();
            releaseShiftCopies();
            throw e;
        }
    }

    /**
     * The {@code seq_add} of one sequence (llama.cpp {@code llama_memory_seq_add}): moves its stored positions
     * {@code [p0, p1)} by {@code delta}, running the privatizations it queued on the device and installing the
     * move's whole per-row RoPE deltas. The caller re-rotates the keys and finishes with
     * {@link #moveShiftedRows()}, as after {@link #shiftSequence(int, int, int)}.
     *
     * @throws UnsupportedOperationException for the contiguous form, and any refusal of the sequence view
     */
    public void addSequence(int seq, int p0, int p1, int delta) {
        sequenceEdit(seq, () -> sequences().add(seq, p0, p1, delta));
    }

    /**
     * The {@code seq_div} of one sequence (llama.cpp {@code llama_memory_seq_div}): divides its stored positions
     * {@code [p0, p1)} by {@code d}, installing each row's own RoPE shift for the caller's re-rotation. The rows
     * keep their cells, so no row move follows.
     *
     * @throws IllegalArgumentException      when {@code p0} is negative or {@code d} is below 2
     * @throws UnsupportedOperationException for the contiguous form, and any refusal of the sequence view
     */
    public void divSequence(int seq, int p0, int p1, int d) {
        sequenceEdit(seq, () -> sequences().div(seq, p0, p1, d));
    }

    /**
     * The KV chunk reuse of one sequence (llama.cpp server's cached-prompt re-adoption): re-appends the held rows
     * of cached pages at its append cursor, so its prompt skips their prefill — the run of {@code rows} rows starts
     * at source row {@code srcRow0} of run page 0 of {@code srcPages} (one page per KV group per run page, the
     * cache blocks') and lands at the sequence's own slots at the positions {@code [pos0, pos0 + rows)} its prompt
     * rows take. The edit's page copies run on the device ({@code runShiftCopies}) before the copied keys
     * re-rotate by the installed whole per-row deltas, as after {@link #addSequence}; the re-append queues no row
     * moves, so nothing carries after the rotation.
     *
     * @throws IllegalArgumentException      for a run outside the pages, a token count or position mismatch, or
     *                                       source rows the cache does not hold
     * @throws com.eonmux.neuralfree.gguf.memory.MemoryFullException when the re-append needs pages the pools lack
     * @throws UnsupportedOperationException for the contiguous form, and any refusal of the sequence view
     */
    public void reuseSequence(int seq, int[][] srcPages, int srcRow0, int pos0, int rows, int delta,
                              int[] tokens, int embeddingToken) {
        sequenceEdit(seq, () -> paged.reusePages(seq, srcPages, srcRow0, pos0, rows, delta, tokens,
                embeddingToken));
    }

    /** Runs one sequence position edit on the device, failing without leaving half-applied state. */
    private void sequenceEdit(int seq, Runnable edit) {
        checkOpen();
        if (paged == null) {
            throw new UnsupportedOperationException("position edits unsupported: the contiguous memory edits through"
                    + " " + DeviceModelMemory.class.getSimpleName() + ".shiftPositions");
        }
        if (seq < 0 || seq >= paged.maxSequences()) {
            throw new IndexOutOfBoundsException("sequence " + seq + " outside [0, " + paged.maxSequences() + ")");
        }
        try {
            edit.run();
            runShiftCopies();
            installShiftDeltas();
        } catch (RuntimeException e) {
            queue.finish();
            releaseShiftCopies();
            throw e;
        }
    }

    /**
     * Runs the row moves a {@link #shiftSequence} queued on the device and releases the shift's page copies: every
     * cache layer of a move's group carries the run's keys and values to their new cells, and so do the group's
     * cell-addressed side caches. Call it after the mixers re-rotated the keys, while they still sit at the cells
     * the shift's deltas name; it waits for the queue, so the copies' sources return to their pools only once the
     * copies executed.
     *
     * @throws UnsupportedOperationException when the memory holds no paged sequences
     */
    public void moveShiftedRows() {
        checkOpen();
        if (paged == null) {
            throw new UnsupportedOperationException("the memory holds no paged sequences");
        }
        try {
            List<RowMove> moves = paged.pendingRowMoves();
            if (!moves.isEmpty()) {
                enqueueRowMoves(moves);
                paged.clearRowMoves();
            }
        } finally {
            queue.finish();
            releaseShiftCopies();
        }
    }

    /** Enqueues the page copies the move's privatizations queued: the re-rotation addresses their cells. */
    private void runShiftCopies() {
        for (PageCopy copy : paged.pendingCopies()) {
            copyPage(copy);
            shiftCopies.add(copy);
        }
    }

    /** Returns the shift's executed page copies' sources to their pools. */
    private void releaseShiftCopies() {
        if (shiftCopies.isEmpty()) {
            return;
        }
        paged.releaseCopies(shiftCopies.toArray(new PageCopy[0]));
        shiftCopies.clear();
    }

    /**
     * Drains the move's reports (their mirror changes apply here, not at the next step, so a second shift cannot
     * collect the same deltas twice) and installs every group's whole delta array: the reported rows take their
     * deltas, every other row 0.
     */
    private void installShiftDeltas() {
        int[][] deltas = stepInputs.drainPagedShiftDeltas();
        stepInputs.reserveShiftDeltas();
        for (int g = 0; g < deltas.length; g++) {
            if (deltas[g] == null) {
                deltas[g] = new int[stepInputs.positions(g).rows()];
            }
            stepInputs.installShiftDeltas(g, deltas[g]);
        }
        lastShiftDeltas = deltas;
    }

    /**
     * The per-row RoPE deltas of the last {@link #shiftSequence} (group order, a new array over copies): each row's
     * stored position change, 0 for a row the shift kept in place. A tensor-parallel rank installs the same deltas
     * with {@link #installPagedShiftDeltas} so its split mixers re-rotate the rank's keys identically.
     *
     * @return the deltas, or null before the first shift
     */
    public int[][] pagedShiftDeltas() {
        checkOpen();
        if (lastShiftDeltas == null) {
            return null;
        }
        int[][] out = new int[lastShiftDeltas.length][];
        for (int g = 0; g < lastShiftDeltas.length; g++) {
            out[g] = lastShiftDeltas[g].clone();
        }
        return out;
    }

    /**
     * Installs the per-row RoPE deltas of another memory's shift ({@link #pagedShiftDeltas()}) as this memory's
     * shift deltas, so its mixers re-rotate the keys of the same rows.
     *
     * @throws UnsupportedOperationException for the contiguous form
     * @throws IllegalArgumentException      when the deltas hold another group count or row count
     */
    public void installPagedShiftDeltas(int[][] deltas) {
        checkOpen();
        Objects.requireNonNull(deltas, "deltas");
        if (paged == null) {
            throw new UnsupportedOperationException("the contiguous memory holds no paged shift deltas");
        }
        if (deltas.length != stepInputs.groups()) {
            throw new IllegalArgumentException("shift deltas of " + deltas.length + " KV group(s) cannot install on "
                    + stepInputs.groups());
        }
        stepInputs.reserveShiftDeltas();
        for (int g = 0; g < deltas.length; g++) {
            stepInputs.installShiftDeltas(g, deltas[g]);
        }
    }

    /**
     * Carries the row moves another memory's shift planned ({@link PagedSequences#pendingRowMoves()}) on this
     * memory's caches: the keys and values of the moved cells follow the same rows. Call it after the mixers
     * re-rotated the keys, as {@link #moveShiftedRows()} does for a shift this memory planned.
     *
     * @throws UnsupportedOperationException when the memory holds no paged sequences
     */
    public void moveShiftedRows(List<RowMove> moves) {
        checkOpen();
        Objects.requireNonNull(moves, "moves");
        if (paged == null) {
            throw new UnsupportedOperationException("the memory holds no paged sequences");
        }
        try {
            enqueueRowMoves(moves);
        } finally {
            queue.finish();
        }
    }

    /**
     * Runs what a sequence-level compaction queued on the paged bookkeeping: the pending copy-on-write page copies
     * on the device caches ({@code PagedSequences#pendingCopies()}), then the pending row moves
     * ({code pendingRowMoves()}), and returns the copies' sources once the queue finished, so the compaction is
     * complete when the call returns. The tree verify's commit calls it after its {@code moveRows}; a plain shift
     * queues and carries the same way through {@link #moveShiftedRows()}.
     */
    public void carryRowMoves() {
        checkOpen();
        if (paged == null) {
            return;
        }
        List<PageCopy> copies = paged.pendingCopies();
        try {
            for (PageCopy copy : copies) {
                copyPage(copy);
            }
            moveShiftedRows();
        } finally {
            if (!copies.isEmpty()) {
                paged.releaseCopies(copies.toArray(new PageCopy[0]));
            }
        }
    }

    /**
     * Enqueues one device copy per row move: every cache layer of the move's group carries the run's rows to their
     * new cells, and so do the group's cell-addressed side caches. A group no layer declares (the pure-recurrent
     * plan's bookkeeping group) holds no rows and the move is a no-op.
     *
     * @throws IllegalArgumentException when cache layers declare the group but none belongs to the move's group
     */
    private void enqueueRowMoves(List<RowMove> moves) {
        for (RowMove move : moves) {
            int moved = 0;
            for (DeviceKvLayer c : ownedKv) {
                if (c.group() != move.group()) {
                    continue;
                }
                KvLayout l = c.layout();
                queue.enqueueCopyBuffer(c.k(), (long) move.srcRow() * l.rowBytesK(), c.k(),
                        (long) move.dstRow() * l.rowBytesK(), (long) move.rows() * l.rowBytesK());
                if (!l.vAliasesK()) {
                    queue.enqueueCopyBuffer(c.v(), (long) move.srcRow() * l.rowBytesV(), c.v(),
                            (long) move.dstRow() * l.rowBytesV(), (long) move.rows() * l.rowBytesV());
                }
                copyScales(c, l, move.srcRow(), move.dstRow(), move.rows());
                moved++;
            }
            for (SideRows side : ownedSideRows) {
                if (side.group == move.group()) {
                    queue.enqueueCopyBuffer(side.keys, (long) move.srcRow() * side.rowBytes, side.keys,
                            (long) move.dstRow() * side.rowBytes, (long) move.rows() * side.rowBytes);
                }
            }
            if (moved == 0 && declaresGroup(move.group())) {
                throw new IllegalArgumentException("no cache layer of group " + move.group() + " runs the row move "
                        + move);
            }
        }
    }

    /**
     * Carries the side rows of a cache with its code rows: the interleaved per-(row, head) scales of a
     * per-token-head cache and the packed plane rows of a vendor cache's side buffer.
     */
    private void copyScales(DeviceKvLayer c, KvLayout l, long srcRow, long dstRow, int rows) {
        long rowBytes = 2L * l.nHeadKv() * Float.BYTES;
        if (c.rowScales() != null) {
            queue.enqueueCopyBuffer(c.rowScales(), srcRow * rowBytes, c.rowScales(), dstRow * rowBytes,
                    (long) rows * rowBytes);
        }
        if (c.vendorSide() != null && c.sideRowBytes() > 0) {
            long base = DeviceKvFormats.sidePlaneBase(l.nHeadKv());
            long stride = c.sideRowBytes();
            queue.enqueueCopyBuffer(c.vendorSide(), base + srcRow * stride, c.vendorSide(), base + dstRow * stride,
                    (long) rows * stride);
        }
    }

    /** Whether any layer of the plan reads the KV cells of {@code group}. */
    private boolean declaresGroup(int group) {
        for (LayerMemoryPlan lp : plan.layers()) {
            if (lp.kvGroup() == group) {
                return true;
            }
        }
        return false;
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

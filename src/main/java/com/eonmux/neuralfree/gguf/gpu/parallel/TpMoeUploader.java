package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.ops.moe.MoeExpertRemap;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Uploads the per-rank shards of a tensor-parallel expert split ({@link TpMoePlan}): every routed expert's
 * intermediate-width rows of its gate and up projections, the matching columns of its down projection, and the same
 * slices of the shared expert. Expert tensors are stored {@code [in, rows, n_expert]}, so the uploader copies each
 * expert's row run into the rank's buffer; a down projection keeps whole device blocks of the stored type. Under
 * expert parallel it instead copies the rank's whole experts — its contiguous block, or the runs an
 * {@link EplbPlacement} packs — as whole-expert shards at their global expert ids ({@link GpuWeight#expertRuns}) so
 * the kernels address the experts by their global id. Router tensors, per-expert scales and the shared expert's
 * output-width vectors stay whole (the down bias only on rank 0).
 *
 * <p>Every {@link #uploadRank} allocates on the rank's backend, enqueues on its queue and finishes it before
 * returning. The uploader owns every buffer it allocates; {@link #close()} releases them.</p>
 */
public final class TpMoeUploader implements AutoCloseable {

    private final TpMoePlan plan;
    private final GpuWeightUploader weights;
    private final WeightStore store;
    private final List<GpuBuffer> owned = new ArrayList<>();
    /** The physical bank buffers of every rank and expert tensor, for a runtime rearrangement. */
    private final Map<String, Bank> banks = new LinkedHashMap<>();
    /** The physical per-expert scale buffers of every rank and scale tensor. */
    private final Map<String, Bank> scales = new LinkedHashMap<>();
    /** The logical-to-physical mapping buffers of every rank and weight layer. */
    private final Map<Integer, Map<Integer, RemapBuffers>> remaps = new LinkedHashMap<>();
    /** The physical layout a rearrangement last migrated to, per weight layer; absent keeps the plan's. */
    private final Map<Integer, int[]> applied = new LinkedHashMap<>();

    public TpMoeUploader(TpMoePlan plan, GpuWeightUploader weights, WeightStore store) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.store = Objects.requireNonNull(store, "store");
    }

    /**
     * Uploads rank {@code rank}'s shard of every split expert tensor.
     *
     * @return the rank's shard weights, keyed by their full tensor names
     * @throws IllegalArgumentException when the model has no data for a split tensor
     */
    public Map<String, GpuWeight> uploadRank(int rank, ComputeBackend backend, ComputeQueue queue) {
        return uploadRank(rank, backend, queue, true);
    }

    /**
     * {@link #uploadRank(int, ComputeBackend, ComputeQueue)} that can leave the whole support tensors to the model
     * uploader: rank 0 reads them from the model device's resident weights.
     */
    public Map<String, GpuWeight> uploadRank(int rank, ComputeBackend backend, ComputeQueue queue,
                                             boolean support) {
        Map<String, GpuWeight> out = new LinkedHashMap<>();
        for (TpMoePlan.Layer layer : plan.layers()) {
            for (Map.Entry<String, TpMoePlan.Shard> entry : layer.tensors().entrySet()) {
                String name = layer.tensor(entry.getKey());
                GpuWeightUploader.TensorPlan p = weights.plan(name);
                if (p == null) {
                    throw new IllegalArgumentException("the model has no tensor " + name);
                }
                GpuWeight w = upload(layer, entry.getValue(), p, rank, backend, queue);
                if (w != null) {
                    out.put(name, w);
                }
            }
            if (support) {
                support(layer, rank, backend, queue, out);
            }
        }
        queue.finish();
        weights.syncCopier(queue);
        return out;
    }

    /**
     * Uploads rank {@code rank}'s logical-to-physical replica mappings: for every layer whose experts are remapped,
     * the {@code [logical * maxReplicas]} table of physical slots and the {@code [logical]} replica counts the
     * forward path indexes by the router's logical ids.
     *
     * @return the layer's mapping buffers, keyed by weight layer
     */
    public Map<Integer, MoeExpertRemap.Layer> uploadRemap(int rank, ComputeBackend backend, ComputeQueue queue) {
        Map<Integer, MoeExpertRemap.Layer> out = new LinkedHashMap<>();
        for (TpMoePlan.Layer layer : plan.layers()) {
            if (!layer.remaps()) {
                continue;
            }
            int[] flat = layer.logicalToPhysicalFlat();
            int[] counts = layer.replicaCount();
            GpuBuffer map = intBuffer(flat, backend, queue);
            GpuBuffer replica = intBuffer(counts, backend, queue);
            out.put(layer.weightLayer(), new MoeExpertRemap.Layer(map, replica, layer.nExpert(),
                    layer.physicalExperts(), layer.maxReplicas()));
            remaps.computeIfAbsent(rank, r -> new LinkedHashMap<>())
                    .put(layer.weightLayer(), new RemapBuffers(map, replica, layer.nExpert(),
                            layer.physicalExperts(), layer.maxReplicas()));
        }
        return out;
    }

    /** An I32 buffer holding {@code values}, uploaded on {@code queue}. */
    private GpuBuffer intBuffer(int[] values, ComputeBackend backend, ComputeQueue queue) {
        GpuBuffer dst = allocate((long) values.length * Integer.BYTES, backend);
        ByteBuffer staged = ByteBuffer.allocateDirect(values.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        staged.asIntBuffer().put(values);
        queue.enqueueWriteBuffer(dst, 0L, (long) values.length * Integer.BYTES, staged);
        return dst;
    }

    /**
     * Migrates the expert banks in place to new physical layouts (vLLM's rearrangement weight migration): for every
     * rank, each changed slot's rows are re-copied from the model's source tensors into the rank's bank, the physical
     * per-expert scale vectors are rewritten and the logical-to-physical map and replica counts are updated; the
     * rank's queue is finished before the next rank starts. Slots that keep their logical expert are not rewritten.
     *
     * @param layouts the new physical layout per weight layer: the logical expert each slot reproduces
     * @param group   the ranked group whose devices hold the banks
     * @throws IllegalArgumentException for a layout of an unknown or non-remapped layer, a wrong layout length, a
     *                                  slot outside the logical experts, a logical expert with no slot, or a larger
     *                                  maximum replica count than the uploaded map holds
     * @throws IllegalStateException   when no remap buffers were uploaded (the plan was not remapped)
     */
    public void rearrange(Map<Integer, int[]> layouts, TensorParallelGroup group) {
        Objects.requireNonNull(layouts, "layouts");
        Objects.requireNonNull(group, "group");
        if (group.rankCount() != plan.ranks()) {
            throw new IllegalArgumentException("the banks are uploaded over " + plan.ranks() + " rank(s), the group"
                    + " holds " + group.rankCount());
        }
        if (remaps.isEmpty()) {
            throw new IllegalStateException("no remap buffers were uploaded: a rearrangement needs an expert-parallel"
                    + " plan whose layers are remapped (" + EngineConfig.PROP_EXPERT_PARALLEL + "=true and a"
                    + " rebalancing placement)");
        }
        Map<Integer, int[]> currents = new LinkedHashMap<>();
        for (TpMoePlan.Layer layer : plan.layers()) {
            int[] newLayout = layouts.get(layer.weightLayer());
            if (newLayout == null) {
                continue;
            }
            int[] current = appliedLayout(layer);
            if (Arrays.equals(current, newLayout)) {
                continue;
            }
            checkLayout(layer, newLayout, current);
            currents.put(layer.weightLayer(), current);
        }
        for (int rank = 0; rank < group.rankCount(); rank++) {
            ComputeQueue queue = group.rankQueue(rank);
            for (TpMoePlan.Layer layer : plan.layers()) {
                int[] current = currents.get(layer.weightLayer());
                if (current == null) {
                    continue;
                }
                migrateLayer(layer, rank, current, layouts.get(layer.weightLayer()), queue);
            }
            queue.finish();
            weights.syncCopier(queue);
        }
        for (Map.Entry<Integer, int[]> entry : currents.entrySet()) {
            applied.put(entry.getKey(), layouts.get(entry.getKey()).clone());
        }
    }

    /** The physical layout the banks currently reproduce: the last migration's, or the plan's own. */
    private int[] appliedLayout(TpMoePlan.Layer layer) {
        int[] appliedLayout = applied.get(layer.weightLayer());
        return appliedLayout != null ? appliedLayout.clone() : layer.phyToLog();
    }

    /**
     * Checks a new layout against the layer and the uploaded replica map.
     *
     * @throws IllegalArgumentException naming the first problem: a non-remapped layer, a wrong length, a slot
     *                                  outside the logical experts, an expert with no slot, or a larger maximum
     *                                  replica count than the uploaded map holds
     */
    private void checkLayout(TpMoePlan.Layer layer, int[] newLayout, int[] current) {
        if (!layer.remaps()) {
            throw new IllegalArgumentException("layer " + layer.il() + " does not remap its routed ids, so its"
                    + " experts cannot be rearranged in place");
        }
        if (newLayout.length != layer.physicalExperts()) {
            throw new IllegalArgumentException("the new layout of layer " + layer.il() + " holds "
                    + newLayout.length + " slot(s), the layer's physical bank has " + layer.physicalExperts());
        }
        int logical = layer.nExpert();
        int[] replicas = new int[logical];
        for (int slot = 0; slot < newLayout.length; slot++) {
            int expert = newLayout[slot];
            if (expert < 0 || expert >= logical) {
                throw new IllegalArgumentException("the new layout of layer " + layer.il() + " puts expert " + expert
                        + " at slot " + slot + ", outside [0, " + logical + ")");
            }
            replicas[expert]++;
        }
        for (int expert = 0; expert < logical; expert++) {
            if (replicas[expert] < 1) {
                throw new IllegalArgumentException("the new layout of layer " + layer.il() + " leaves logical expert "
                        + expert + " with no physical slot; every routed expert needs at least one");
            }
        }
        int max = max(replicas);
        for (Map<Integer, RemapBuffers> rankRemaps : remaps.values()) {
            RemapBuffers buffers = rankRemaps.get(layer.weightLayer());
            if (buffers != null && max > buffers.maxReplicas) {
                throw new IllegalArgumentException("the new layout of layer " + layer.il() + " gives one logical"
                        + " expert " + max + " replica(s), the uploaded map holds at most " + buffers.maxReplicas);
            }
        }
    }

    /** Re-copies one rank's changed bank slots, scale values and remap tables of a layer. */
    private void migrateLayer(TpMoePlan.Layer layer, int rank, int[] current, int[] newLayout, ComputeQueue queue) {
        int base = rank * layer.slotsPerRank();
        for (Map.Entry<String, TpMoePlan.Shard> entry : layer.tensors().entrySet()) {
            if (entry.getValue() != TpMoePlan.Shard.EXPERT_SLICE) {
                continue;
            }
            String name = layer.tensor(entry.getKey());
            Bank bank = banks.get(bankKey(rank, name));
            if (bank == null) {
                throw new IllegalStateException("the bank of " + name + " on rank " + rank + " was not uploaded");
            }
            rewriteBank(name, bank, current, newLayout, base, queue);
        }
        if (layer.expertParallel()) {
            for (String scaleName : expertScaleNames(layer)) {
                Bank bank = scales.get(bankKey(rank, scaleName));
                if (bank != null && bank.logicalSource != null) {
                    rewriteScale(bank, current, newLayout, layer.physicalExperts(), queue);
                }
            }
        }
        rewriteRemap(layer, rank, newLayout, queue);
    }

    /** Re-copies a bank's changed slots from the model's source tensors, in slot order. */
    private void rewriteBank(String name, Bank bank, int[] current, int[] newLayout, int base, ComputeQueue queue) {
        long perSlot = (long) bank.srcRows * bank.rowBytes;
        for (int c = 0; c < bank.buffers.size(); c++) {
            GpuBuffer dst = bank.buffers.get(c);
            for (int j = 0; j < bank.slots[c]; j++) {
                int slot = base + bank.firstSlot[c] + j;
                if (current[slot] == newLayout[slot]) {
                    continue;
                }
                weights.copyRows(name, (long) newLayout[slot] * bank.srcRows, bank.srcRows, dst,
                        (long) j * perSlot, queue);
            }
        }
    }

    /** Rewrites a physical per-expert scale vector's changed slots from its cached logical values. */
    private void rewriteScale(Bank bank, int[] current, int[] newLayout, int physical, ComputeQueue queue) {
        GpuBuffer dst = bank.buffers.get(0);
        for (int slot = 0; slot < physical; slot++) {
            if (current[slot] == newLayout[slot]) {
                continue;
            }
            try (GpuEvent ignored = queue.enqueueWriteBuffer(dst, (long) slot * Float.BYTES, (long) Float.BYTES,
                    new float[]{bank.logicalSource[newLayout[slot]]})) {
                // the write copies at enqueue
            }
        }
    }

    /** Rewrites a rank's logical-to-physical map and replica counts, keeping the uploaded stride. */
    private void rewriteRemap(TpMoePlan.Layer layer, int rank, int[] newLayout, ComputeQueue queue) {
        RemapBuffers buffers = remaps.get(rank).get(layer.weightLayer());
        if (buffers == null) {
            throw new IllegalStateException("the remap buffers of layer " + layer.il() + " on rank " + rank
                    + " were not uploaded");
        }
        int logical = layer.nExpert();
        int stride = buffers.maxReplicas;
        int[] flat = new int[logical * stride];
        int[] counts = new int[logical];
        for (int slot = 0; slot < newLayout.length; slot++) {
            int expert = newLayout[slot];
            flat[expert * stride + counts[expert]++] = slot;
        }
        intBuffer(flat, buffers.logicalToPhysical, queue);
        intBuffer(counts, buffers.replicaCount, queue);
    }

    /** Writes {@code values} whole into an uploaded remap buffer. */
    private static void intBuffer(int[] values, GpuBuffer dst, ComputeQueue queue) {
        ByteBuffer staged = ByteBuffer.allocateDirect(values.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        staged.asIntBuffer().put(values);
        queue.enqueueWriteBuffer(dst, 0L, (long) values.length * Integer.BYTES, staged);
    }

    private GpuWeight upload(TpMoePlan.Layer layer, TpMoePlan.Shard shard, GpuWeightUploader.TensorPlan p, int rank,
                             ComputeBackend backend, ComputeQueue queue) {
        switch (shard) {
            case EXPERT_ROWS:
                return expertRows(layer, p, rank, false, backend, queue);
            case EXPERT_FUSED:
                return expertRows(layer, p, rank, true, backend, queue);
            case EXPERT_DOWN:
                return expertDown(layer, p, rank, backend, queue);
            case EXPERT_SLICE:
                return expertSlice(layer, p, rank, backend, queue);
            case SHARED_ROWS:
                return rows(p, layer.sharedStart(rank), layer.sharedShard(), backend, queue);
            case SHARED_FUSED:
                return fusedRows(layer, p, layer.sharedStart(rank), layer.sharedShard(), backend, queue);
            case SHARED_DOWN:
                return columns(p, layer.sharedStart(rank), layer.sharedShard(), backend, queue);
            case SHARED_HIDDEN_VECTOR:
                return hiddenVector(p, layer.sharedStart(rank), layer.sharedShard(), backend, queue);
            case SHARED_OUTPUT_VECTOR:
                return rank == 0 ? outputVector(p, backend, queue) : null;
            default:
                throw new IllegalStateException("unknown expert shard " + shard + " of " + p.name());
        }
    }

    /**
     * Uploads whole, to every rank, the layer's MoE tensors the split does not shard: the router and its bias, the
     * selection biases, the per-expert scales, the latent projections and the norms the block reads. Every rank runs
     * the same routing over them, so they must be identical on every device.
     */
    private void support(TpMoePlan.Layer layer, int rank, ComputeBackend backend, ComputeQueue queue,
                         Map<String, GpuWeight> out) {
        Set<String> scaleNames = expertScaleNames(layer);
        for (String name : layer.support()) {
            if (out.containsKey(name)) {
                continue;
            }
            GpuWeightUploader.TensorPlan p = weights.plan(name);
            if (p == null || p.problem() != null) {
                continue;
            }
            if (scaleNames.contains(name) && layer.expertParallel()) {
                out.put(name, expertScale(layer, rank, p, backend, queue));
            } else {
                out.put(name, whole(p, backend, queue));
            }
        }
    }

    /** The layer's per-expert scale tensor names (gate, up and down), which a physical bank indexes by slot. */
    private static Set<String> expertScaleNames(TpMoePlan.Layer layer) {
        Set<String> out = new HashSet<>();
        out.add(layer.tensor(TensorNames.scale(TensorNames.FFN_GATE_EXPS)));
        out.add(layer.tensor(TensorNames.scale(TensorNames.FFN_UP_EXPS)));
        out.add(layer.tensor(TensorNames.scale(TensorNames.FFN_DOWN_EXPS)));
        return out;
    }

    /**
     * The per-expert scale vector in physical slot order: {@code out[slot] = scale[logicalOf(slot)]}, uploaded
     * whole to every rank because the expert kernels index it by the global physical id.
     */
    private GpuWeight expertScale(TpMoePlan.Layer layer, int rank, GpuWeightUploader.TensorPlan p,
                                  ComputeBackend backend, ComputeQueue queue) {
        requireF32(p);
        int logical = layer.nExpert();
        int physical = layer.physicalExperts();
        int values = (int) p.info().numElements();
        if (values != logical) {
            throw new UnsupportedOperationException("tensor " + p.name() + " holds " + values + " per-expert"
                    + " values, the layer " + logical + "; a physical expert bank needs one scale per expert");
        }
        ByteBuffer staged = ByteBuffer.allocateDirect(logical * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        read(p, p.info().offset(), logical * Float.BYTES, staged);
        float[] source = new float[logical];
        staged.asFloatBuffer().get(source);
        float[] out = new float[physical];
        for (int slot = 0; slot < physical; slot++) {
            out[slot] = source[layer.logicalOfSlot(slot)];
        }
        return writeFloatVector(layer, rank, p, out, source, backend, queue);
    }

    /** Writes a physical per-expert scale vector into a fresh buffer and records it for a rearrangement. */
    private GpuWeight writeFloatVector(TpMoePlan.Layer layer, int rank, GpuWeightUploader.TensorPlan p,
                                       float[] values, float[] logicalSource, ComputeBackend backend,
                                       ComputeQueue queue) {
        GpuBuffer dst = allocate((long) values.length * Float.BYTES, backend);
        ByteBuffer staged = ByteBuffer.allocateDirect(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        staged.asFloatBuffer().put(values);
        queue.enqueueWriteBuffer(dst, 0L, (long) values.length * Float.BYTES, staged);
        scales.put(bankKey(rank, p.name()), new Bank(layer.weightLayer(), layer.expertParallel(), 1,
                Float.BYTES, Collections.singletonList(dst), new int[]{0}, new int[]{values.length},
                new long[]{0L}, logicalSource));
        return GpuWeight.of(p.name(), GGMLType.F32, new long[]{values.length}, dst);
    }

    /** The whole tensor on this rank, one buffer or row-sharded when it exceeds the binding size. */
    private GpuWeight whole(GpuWeightUploader.TensorPlan p, ComputeBackend backend, ComputeQueue queue) {
        long rows = p.rows();
        long rowBytes = p.sourceRowBytes();
        long perBuffer = Math.max(1L, weights.maxBufferBytes() / Math.max(1L, rowBytes));
        List<GpuBuffer> buffers = new ArrayList<>();
        List<Long> counts = new ArrayList<>();
        for (long done = 0; done < rows; done += perBuffer) {
            long n = Math.min(perBuffer, rows - done);
            GpuBuffer dst = allocate(n * p.deviceRowBytes(), backend);
            weights.copyRows(p.name(), done, n, dst, 0L, queue);
            buffers.add(dst);
            counts.add(n);
        }
        return sharded(p.name(), p.deviceType(), p.ne(), buffers, counts);
    }

    /**
     * The rank's intermediate rows of every expert's gate or up projection, or both segments of a fused projection:
     * {@code [cols, rowsPerExpert, nExpert]} with {@code rowsPerExpert} the rank's width (twice it for a fused
     * projection, whose gate rows come first).
     */
    private GpuWeight expertRows(TpMoePlan.Layer layer, GpuWeightUploader.TensorPlan p, int rank, boolean fused,
                                 ComputeBackend backend, ComputeQueue queue) {
        int nExpert = (int) p.ne()[2];
        int srcRows = (int) p.ne()[1] / (fused ? 2 : 1);
        int dstRows = fused ? 2 * layer.shard() : layer.shard();
        int start = layer.start(rank);
        long rowBytes = p.deviceRowBytes();
        long perExpertBytes = (long) dstRows * rowBytes;
        long chunk = Math.max(1L, weights.maxBufferBytes() / Math.max(1L, perExpertBytes));
        List<GpuBuffer> buffers = new ArrayList<>();
        List<Long> rows = new ArrayList<>();
        for (int e0 = 0; e0 < nExpert; e0 += (int) Math.min(chunk, nExpert - e0)) {
            int count = (int) Math.min(chunk, nExpert - e0);
            GpuBuffer dst = allocate(count * perExpertBytes, backend);
            for (int e = e0; e < e0 + count; e++) {
                long dstBase = (long) (e - e0) * perExpertBytes;
                if (fused) {
                    weights.copyRows(p.name(), (long) e * 2 * srcRows + start, layer.shard(), dst,
                            dstBase, queue);
                    weights.copyRows(p.name(), (long) e * 2 * srcRows + srcRows + start, layer.shard(), dst,
                            dstBase + (long) layer.shard() * rowBytes, queue);
                } else {
                    weights.copyRows(p.name(), (long) e * srcRows + start, layer.shard(), dst, dstBase, queue);
                }
            }
            buffers.add(dst);
            rows.add((long) count * dstRows);
        }
        return sharded(p.name(), p.deviceType(), new long[]{p.ne()[0], dstRows, nExpert}, buffers, rows);
    }

    /**
     * The rank's physical bank of a routed expert tensor (expert parallel): the contiguous physical slots
     * {@code [rank * slots, (rank + 1) * slots)}, slot {@code s} holding logical expert
     * {@link TpMoePlan.Layer#logicalOfSlot(int)}'s rows. The tensor keeps its global physical shape, so the kernels
     * address the rank's range and skip the other ranks' slots.
     */
    private GpuWeight expertSlice(TpMoePlan.Layer layer, GpuWeightUploader.TensorPlan p, int rank,
                                  ComputeBackend backend, ComputeQueue queue) {
        int srcRows = (int) p.ne()[1];
        int physical = layer.physicalExperts();
        int slots = layer.slotsPerRank();
        int base = rank * slots;
        long rowBytes = p.deviceRowBytes();
        long perSlot = (long) srcRows * rowBytes;
        long perBuffer = Math.max(1L, weights.maxBufferBytes() / Math.max(1L, perSlot));
        List<GpuBuffer> buffers = new ArrayList<>();
        List<Long> rows = new ArrayList<>();
        List<Long> starts = new ArrayList<>();
        List<int[]> chunks = new ArrayList<>();
        for (int i = 0; i < slots; i += (int) Math.min(perBuffer, slots - i)) {
            int count = (int) Math.min(perBuffer, slots - i);
            GpuBuffer dst = allocate((long) count * perSlot, backend);
            for (int j = 0; j < count; j++) {
                int slot = base + i + j;
                int logical = layer.logicalOfSlot(slot);
                weights.copyRows(p.name(), (long) logical * srcRows, srcRows, dst, (long) j * perSlot, queue);
            }
            buffers.add(dst);
            rows.add((long) count * srcRows);
            starts.add((long) (base + i) * srcRows);
            chunks.add(new int[]{i, count});
        }
        GpuWeight w = GpuWeight.expertRuns(p.name(), p.deviceType(), new long[]{p.ne()[0], srcRows, physical},
                buffers, toArray(rows), toArray(starts));
        banks.put(bankKey(rank, p.name()), new Bank(layer.weightLayer(), true, srcRows, rowBytes, buffers,
                firsts(chunks), counts(chunks), toArray(starts), null));
        return companions(w);
    }

    private static String bankKey(int rank, String name) {
        return rank + "/" + name;
    }

    private static int[] firsts(List<int[]> chunks) {
        int[] out = new int[chunks.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = chunks.get(i)[0];
        }
        return out;
    }

    private static int[] counts(List<int[]> chunks) {
        int[] out = new int[chunks.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = chunks.get(i)[1];
        }
        return out;
    }

    /** One rank's uploaded physical bank: its buffers and the local slot range each holds. */
    private static final class Bank {
        private final int weightLayer;
        private final boolean expertRows;
        private final int srcRows;
        private final long rowBytes;
        private final List<GpuBuffer> buffers;
        private final int[] firstSlot;
        private final int[] slots;
        private final long[] rowStarts;
        /** A per-expert scale vector's logical source values, null for an expert weight bank. */
        private final float[] logicalSource;

        private Bank(int weightLayer, boolean expertRows, int srcRows, long rowBytes, List<GpuBuffer> buffers,
                     int[] firstSlot, int[] slots, long[] rowStarts, float[] logicalSource) {
            this.weightLayer = weightLayer;
            this.expertRows = expertRows;
            this.srcRows = srcRows;
            this.rowBytes = rowBytes;
            this.buffers = buffers;
            this.firstSlot = firstSlot;
            this.slots = slots;
            this.rowStarts = rowStarts;
            this.logicalSource = logicalSource;
        }
    }

    /** One rank's logical-to-physical mapping buffers of a weight layer. */
    private static final class RemapBuffers {
        private final GpuBuffer logicalToPhysical;
        private final GpuBuffer replicaCount;
        private final int logical;
        private final int physical;
        private final int maxReplicas;

        private RemapBuffers(GpuBuffer logicalToPhysical, GpuBuffer replicaCount, int logical, int physical,
                             int maxReplicas) {
            this.logicalToPhysical = logicalToPhysical;
            this.replicaCount = replicaCount;
            this.logical = logical;
            this.physical = physical;
            this.maxReplicas = maxReplicas;
        }
    }

    private static long[] toArray(List<Long> values) {
        long[] out = new long[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    private static int max(int[] values) {
        int max = 0;
        for (int v : values) {
            max = Math.max(max, v);
        }
        return max;
    }

    /** The rank's intermediate columns of every expert's down projection: {@code [cols, out, nExpert]}. */
    private GpuWeight expertDown(TpMoePlan.Layer layer, GpuWeightUploader.TensorPlan p, int rank,
                                 ComputeBackend backend, ComputeQueue queue) {
        requireRaw(p, "an expert down projection");
        int nExpert = (int) p.ne()[2];
        int out = (int) p.ne()[1];
        int start = layer.start(rank);
        int cols = layer.shard();
        long srcRow = p.sourceRowBytes();
        long srcCol = p.sourceType().getBytesForElements(start);
        long dstRow = p.deviceType().getBytesForElements(cols);
        int bytes = (int) dstRow;
        long perExpertBytes = (long) out * dstRow;
        long chunk = Math.max(1L, weights.maxBufferBytes() / Math.max(1L, perExpertBytes));
        ByteBuffer staged = ByteBuffer.allocateDirect(bytes).order(ByteOrder.LITTLE_ENDIAN);
        List<GpuBuffer> buffers = new ArrayList<>();
        List<Long> rows = new ArrayList<>();
        for (int e0 = 0; e0 < nExpert; e0 += (int) Math.min(chunk, nExpert - e0)) {
            int count = (int) Math.min(chunk, nExpert - e0);
            GpuBuffer dst = allocate(count * perExpertBytes, backend);
            for (int e = e0; e < e0 + count; e++) {
                for (int r = 0; r < out; r++) {
                    staged.clear();
                    read(p, p.info().offset() + ((long) e * out + r) * srcRow + srcCol, bytes, staged);
                    try (GpuEvent ignored = queue.enqueueWriteBuffer(dst,
                            (long) (e - e0) * perExpertBytes + (long) r * dstRow, bytes, staged)) {
                        // the write copies at enqueue
                    }
                }
            }
            buffers.add(dst);
            rows.add((long) count * out);
        }
        return sharded(p.name(), p.deviceType(), new long[]{cols, out, nExpert}, buffers, rows);
    }

    /** A contiguous row range of a 2-D shared-expert projection. */
    private GpuWeight rows(GpuWeightUploader.TensorPlan p, int first, int count, ComputeBackend backend,
                           ComputeQueue queue) {
        long rowBytes = p.deviceRowBytes();
        long perBuffer = Math.max(1L, weights.maxBufferBytes() / Math.max(1L, rowBytes));
        List<GpuBuffer> buffers = new ArrayList<>();
        List<Long> rows = new ArrayList<>();
        for (long done = 0; done < count; done += perBuffer) {
            long n = Math.min(perBuffer, count - done);
            GpuBuffer dst = allocate(n * rowBytes, backend);
            weights.copyRows(p.name(), (long) first + done, n, dst, 0L, queue);
            buffers.add(dst);
            rows.add(n);
        }
        return companions(sharded(p.name(), p.deviceType(), new long[]{p.ne()[0], count}, buffers, rows));
    }

    /** The rank's two halves of a fused shared-expert up projection. */
    private GpuWeight fusedRows(TpMoePlan.Layer layer, GpuWeightUploader.TensorPlan p, int first, int count,
                                ComputeBackend backend, ComputeQueue queue) {
        long rowBytes = p.deviceRowBytes();
        List<GpuBuffer> buffers = new ArrayList<>();
        List<Long> rows = new ArrayList<>();
        GpuBuffer dst = allocate((long) 2 * count * rowBytes, backend);
        weights.copyRows(p.name(), first, count, dst, 0L, queue);
        weights.copyRows(p.name(), (long) layer.nFfShared() + first, count, dst, (long) count * rowBytes, queue);
        buffers.add(dst);
        rows.add((long) 2 * count);
        return companions(sharded(p.name(), p.deviceType(), new long[]{p.ne()[0], 2L * count}, buffers, rows));
    }

    /** The rank's column block of a shared-expert down projection. */
    private GpuWeight columns(GpuWeightUploader.TensorPlan p, int first, int count, ComputeBackend backend,
                              ComputeQueue queue) {
        requireRaw(p, "a shared-expert down projection");
        long rowsTotal = p.ne()[1];
        long srcRow = p.sourceRowBytes();
        long srcCol = p.sourceType().getBytesForElements(first);
        long dstRow = p.deviceType().getBytesForElements(count);
        int bytes = (int) dstRow;
        ByteBuffer staged = ByteBuffer.allocateDirect(bytes).order(ByteOrder.LITTLE_ENDIAN);
        long perBuffer = Math.max(1L, weights.maxBufferBytes() / Math.max(1L, dstRow));
        List<GpuBuffer> buffers = new ArrayList<>();
        List<Long> rows = new ArrayList<>();
        for (long done = 0; done < rowsTotal; done += perBuffer) {
            long n = Math.min(perBuffer, rowsTotal - done);
            GpuBuffer dst = allocate(n * dstRow, backend);
            for (long r = 0; r < n; r++) {
                staged.clear();
                read(p, p.info().offset() + (done + r) * srcRow + srcCol, bytes, staged);
                try (GpuEvent ignored = queue.enqueueWriteBuffer(dst, r * dstRow, bytes, staged)) {
                    // the write copies at enqueue
                }
            }
            buffers.add(dst);
            rows.add(n);
        }
        return companions(sharded(p.name(), p.deviceType(), new long[]{count, rowsTotal}, buffers, rows));
    }

    /** The rank's slice of a shared-expert hidden-width F32 bias. */
    private GpuWeight hiddenVector(GpuWeightUploader.TensorPlan p, int first, int count, ComputeBackend backend,
                                   ComputeQueue queue) {
        requireF32(p);
        GpuBuffer dst = allocate((long) count * Float.BYTES, backend);
        ByteBuffer staged = ByteBuffer.allocateDirect((int) (count * Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
        read(p, p.info().offset() + (long) first * Float.BYTES, (int) (count * Float.BYTES), staged);
        queue.enqueueWriteBuffer(dst, 0L, (long) count * Float.BYTES, staged);
        return GpuWeight.of(p.name(), GGMLType.F32, new long[]{count}, dst);
    }

    /** The whole output-width down bias, on rank 0. */
    private GpuWeight outputVector(GpuWeightUploader.TensorPlan p, ComputeBackend backend, ComputeQueue queue) {
        requireF32(p);
        int count = (int) p.info().numElements();
        GpuBuffer dst = allocate((long) count * Float.BYTES, backend);
        ByteBuffer staged = ByteBuffer.allocateDirect((int) (count * Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
        read(p, p.info().offset(), (int) (count * Float.BYTES), staged);
        queue.enqueueWriteBuffer(dst, 0L, (long) count * Float.BYTES, staged);
        return GpuWeight.of(p.name(), GGMLType.F32, new long[]{count}, dst);
    }

    private void requireF32(GpuWeightUploader.TensorPlan p) {
        if (p.needsDecode() || p.deviceType() != GGMLType.F32) {
            throw new UnsupportedOperationException("tensor " + p.name() + " is " + p.deviceType()
                    + "; a split shared-expert vector needs F32");
        }
    }

    private void requireRaw(GpuWeightUploader.TensorPlan p, String what) {
        if (p.needsDecode()) {
            throw new UnsupportedOperationException("tensor " + p.name() + " is decoded on upload; " + what
                    + " needs a stored type the device reads directly");
        }
    }

    /** Folds the projection's single-element {@code <base>.scale} into the shard, and its Hadamard if any. */
    private GpuWeight companions(GpuWeight w) {
        GpuWeight out = w;
        String name = w.name();
        if (name.endsWith(".weight")) {
            String scale = name.substring(0, name.length() - ".weight".length()) + ".scale";
            GpuWeightUploader.TensorPlan p = weights.plan(scale);
            if (p != null && p.info().numElements() == 1) {
                out = out.withScale(weights.readScalar(scale));
            }
        }
        if (weights.transforms() != null && !weights.transforms().isEmpty()) {
            HadamardSpec hadamard = weights.transforms().hadamard(name);
            if (hadamard != null) {
                out = out.withHadamard(hadamard);
            }
        }
        return out;
    }

    private static GpuWeight sharded(String name, GGMLType type, long[] ne, List<GpuBuffer> buffers,
                                     List<Long> rows) {
        long[] counts = new long[rows.size()];
        for (int i = 0; i < counts.length; i++) {
            counts[i] = rows.get(i);
        }
        return buffers.size() == 1 ? GpuWeight.of(name, type, ne, buffers.get(0))
                : GpuWeight.sharded(name, type, ne, buffers, counts);
    }

    private void read(GpuWeightUploader.TensorPlan p, long byteOffset, int n, ByteBuffer dst) {
        try {
            store.readTensorDirect(byteOffset, n, dst);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + n + " bytes of tensor " + p.name() + " at byte "
                    + byteOffset, e);
        }
    }

    private GpuBuffer allocate(long bytes, ComputeBackend backend) {
        GpuBuffer buffer = backend.allocateBuffer(bytes, true);
        owned.add(buffer);
        return buffer;
    }

    @Override
    public void close() {
        List<GpuBuffer> buffers = new ArrayList<>(owned);
        owned.clear();
        for (GpuBuffer b : buffers) {
            b.close();
        }
    }
}

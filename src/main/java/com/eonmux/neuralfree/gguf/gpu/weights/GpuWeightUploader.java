package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.GpuModelUnsupportedException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.W12F16Mode;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.InputTransform;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.quant.W12Codec;
import com.eonmux.neuralfree.utils.Closeables;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;
import java.util.regex.Pattern;

/**
 * Uploads the tensors of a GGUF model to the device and serves them as {@link GpuWeight}s.
 *
 * <p><b>Per-tensor device type.</b> Every tensor is classified by how the forward path consumes it
 * ({@link TensorUse}) and gets its own device type ({@link TensorPlan#deviceType()}): a matmul, lookup or expert
 * weight keeps its stored type when the backend has the kernels for it ({@link TypePolicy}, checked with
 * {@code LlmKernels.has}) and is otherwise dequantized to F32 on the CPU; tensors consumed by element-wise or
 * special kernels (norm weights, biases, scales, convolution and recurrent parameters, LoRA factors, ...) are
 * always F32; integer tables are uploaded as stored. Decisions are made per tensor, never per layer or per weight
 * index, so models whose quantization varies across layers get each tensor's own kernels.</p>
 *
 * <p><b>Row shards.</b> A tensor larger than {@code min(maxAllocSize, maxStorageBufferRange)} is split into row
 * shards of balanced size, each in its own buffer ({@link GpuWeight#sharded}); tensors of rank 3 or more split
 * only on {@code ne[1]} (expert) boundaries. A tensor is never dropped for its size.</p>
 *
 * <p><b>Experts.</b> Expert tensors ({@code *_exps.weight}, {@code *_chexps.weight}) are uploaded whole and stay
 * resident, sharded on expert boundaries.</p>
 *
 * <p><b>Per-tensor transforms.</b> A scalar {@code <base>.scale} tensor is the output scale {@code w_s} of
 * {@code <base>.weight} (llama-model.cpp:1724-1790, applied at llama-graph.cpp:1643-1645) and is
 * attached with {@link GpuWeight#withScale}; the scale tensor itself stays resident as F32. The input transforms of
 * the model's input transforms ({@link InputTransformLoader.Result}) are attached whole with
 * {@link GpuWeight#withInputTransform(InputTransform, GpuBuffer)} — HADAMARD, ACT_QUANT and PERMUTE, the
 * {@code <base>.input_perm} int tensor uploaded once beside its weight — and the PRISM inverse rotations with
 * {@link GpuWeight#withInverseHadamard}.</p>
 *
 * <p>Uploads run on the given queue in order and copy their host data at enqueue; every upload call ends with the
 * queue finished. Not thread-safe. The uploader owns every buffer it allocates; {@link #close()} releases them.</p>
 */
public final class GpuWeightUploader implements GpuWeights, AutoCloseable {

    private static final Logger LOGGER = LogManager.getLogger(GpuWeightUploader.class);

    /** Default host staging size per upload chunk. */
    public static final int DEFAULT_STAGING_BYTES = 64 * 1024 * 1024;

    private final ComputeBackend backend;
    private final ComputeQueue queue;
    private final String archName;
    private final Config config;
    private final long maxBufferBytes;
    private final Map<String, TensorPlan> plans;
    private final Map<String, GpuWeight> resident = new LinkedHashMap<>();
    private final List<GpuBuffer> owned = new ArrayList<>();
    private final WeightRowCopier copier;
    /** Weights the F16 grid rounding moved, collected when {@code w12F16 = round}; null otherwise. */
    private final W12Rounding w12Rounding;
    /** Tensors a split uploads as per-rank shards, mapped to this device's exact shard bytes. */
    private Map<String, Long> splitShardBytes = Collections.emptyMap();
    private long residentBytes;
    private boolean closed;

    /**
     * Plans every tensor of the model; nothing is uploaded until {@link #uploadResident} or {@link #upload}.
     *
     * @param backend  device the tensors are allocated on
     * @param queue    queue the uploads run on
     * @param tensors  every tensor of the model
     * @param store    tensor data of the model
     * @param archName {@code general.architecture}, used in messages
     * @param config   type policy, expert placement and transforms
     * @throws IllegalStateException when a tensor's shape is invalid for its type; a tensor that cannot be uploaded
     *                               (see {@link TensorPlan#requireUploadable()}) fails only when it is uploaded
     */
    public GpuWeightUploader(ComputeBackend backend, ComputeQueue queue, List<GGUFTensorInfo> tensors,
                             WeightStore store, String archName, Config config) {
        this.backend = Objects.requireNonNull(backend, "backend");
        this.queue = Objects.requireNonNull(queue, "queue");
        Objects.requireNonNull(store, "store");
        this.archName = archName == null || archName.isEmpty() ? "unknown" : archName;
        this.config = Objects.requireNonNull(config, "config");
        Objects.requireNonNull(tensors, "tensors");
        long deviceMax = deviceMaxBufferBytes(backend);
        this.maxBufferBytes = config.maxBufferBytes > 0 ? Math.min(config.maxBufferBytes, deviceMax) : deviceMax;
        this.plans = planAll(tensors);
        this.w12Rounding = config.w12 && config.w12F16 == W12F16Mode.ROUND ? new W12Rounding() : null;
        this.copier = new WeightRowCopier(store, config.stagingBytes, config.w12F16);
    }

    /** A configuration builder with the given type policy. */
    public static Config.Builder config(TypePolicy policy) {
        return new Config.Builder(policy);
    }

    /**
     * Device bytes the tensors {@code placed} accepts of another model file ({@code source}, a drafter) take on
     * {@code backend} when uploaded with the type policy of {@code kernels}: plans only, nothing is allocated. The
     * PRISM transforms of a file attach to its device tensors without changing their plans, so none is applied.
     *
     * @throws IOException when the file's tensor data cannot be mapped
     */
    public static long plannedBytesOf(ComputeBackend backend, ComputeQueue queue, LlmKernels kernels,
                                      ModelSource source, Predicate<GGUFTensorInfo> placed) throws IOException {
        try (WeightStore store = source.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            GpuWeightUploader planner = new GpuWeightUploader(backend, queue, source.getTensors(), store,
                    source.getString("general.architecture"), config(TypePolicy.forKernels(kernels)).build());
            try {
                return planner.plannedBytes(placed);
            } finally {
                planner.close();
            }
        }
    }

    /**
     * The largest single buffer this uploader allocates on {@code backend}: {@code min(maxAllocSize,
     * maxStorageBufferRange)} over the limits the device reports, or {@link Long#MAX_VALUE} when it reports none.
     */
    public static long deviceMaxBufferBytes(ComputeBackend backend) {
        return WeightShards.maxBufferBytes(backend.capabilities());
    }

    // ------------------------------------------------------------------ GpuWeights

    @Override
    public GpuWeight find(String name) {
        return resident.get(name);
    }

    @Override
    public String archName() {
        return archName;
    }

    /**
     * Registers a tensor another uploader allocated on this device, so the model's weights resolve it under
     * {@code name}: a tensor-parallel split uploads this device's shard of a split projection itself and then adopts
     * it, because the model's blocks (the embedding and the LM head) read it through this uploader.
     *
     * @throws IllegalStateException when this uploader already holds a tensor of that name
     */
    public void adopt(String name, GpuWeight weight) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(weight, "weight");
        ensureOpen();
        if (resident.putIfAbsent(name, weight) != null) {
            throw new IllegalStateException("tensor " + name + " is already resident");
        }
        residentBytes = Math.addExact(residentBytes, weight.sizeInBytes());
    }

    // ------------------------------------------------------------------ planning

    /** The upload plan of a tensor, or null when the model has no tensor of that name. */
    public TensorPlan plan(String name) {
        return plans.get(name);
    }

    /** The PRISM input transforms of the model, or null when the load configured none. */
    public InputTransformLoader.Result transforms() {
        return config.transforms;
    }

    /** Every tensor plan, in model order. Unmodifiable. */
    public Map<String, TensorPlan> plans() {
        return Collections.unmodifiableMap(plans);
    }

    /** The plans of the tensors {@code selected} accepts, in model order. Unmodifiable. */
    public List<TensorPlan> plans(Predicate<GGUFTensorInfo> selected) {
        Objects.requireNonNull(selected, "selected");
        List<TensorPlan> out = new ArrayList<>();
        for (TensorPlan p : plans.values()) {
            if (selected.test(p.info())) {
                out.add(p);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /** Device bytes of the tensors {@code selected} accepts, as {@link #uploadResident} allocates them. */
    public long plannedBytes(Predicate<GGUFTensorInfo> selected) {
        long bytes = 0;
        for (TensorPlan p : plans(selected)) {
            bytes = Math.addExact(bytes, deviceBytesOf(p));
        }
        return bytes;
    }

    /**
     * Declares the tensors a tensor-parallel or row split uploads as per-rank shards, mapping each tensor name to the
     * exact device bytes this device's shard takes: {@link #plannedBytes} accounts that figure instead of the whole
     * tensor, which {@code uploadResident} skips when the split uploads it. Set before the placement runs, so the fit
     * is computed over the shard bytes; an uneven split therefore accounts each rank's own share.
     */
    public void splitShards(Map<String, Long> bytesByTensor) {
        Objects.requireNonNull(bytesByTensor, "bytesByTensor");
        this.splitShardBytes = Collections.unmodifiableMap(new LinkedHashMap<>(bytesByTensor));
    }

    /** The device bytes a tensor occupies after the placement: its shard bytes when a split takes it. */
    private long deviceBytesOf(TensorPlan p) {
        Long shard = splitShardBytes.get(p.name());
        return shard == null ? p.deviceBytes() : shard;
    }

    // ------------------------------------------------------------------ row split copies

    /**
     * Enqueues the upload of rows {@code [firstRow, firstRow + nRows)} of a tensor to {@code dst} at byte offset
     * {@code dstOffset}, staging through the uploader's host buffers as {@link #uploadResident} does. The caller
     * finishes {@code q} and calls {@link #syncCopier(ComputeQueue)} when its writes are done. A W12 tensor's rows
     * refuse an off-grid F16 weight under {@code w12F16 = refuse}, and feed the load's rounding report otherwise.
     *
     * @throws IllegalArgumentException when the model has no tensor of that name, or a W12 row range holds an
     *                                 off-grid F16 weight under {@code w12F16 = refuse}
     */
    public void copyRows(String name, long firstRow, long nRows, GpuBuffer dst, long dstOffset, ComputeQueue q) {
        copier.copyRows(requirePlan(name), firstRow, nRows, dst, dstOffset, q, w12Rounding);
    }

    /**
     * The single element of a one-element tensor, decoded from its stored type.
     *
     * @throws IllegalArgumentException when the model has no tensor of that name
     */
    public float readScalar(String name) {
        return copier.readScalar(requirePlan(name));
    }

    /** Declares that {@code q} was finished, so none of the writes enqueued through the host staging are pending. */
    public void syncCopier(ComputeQueue q) {
        copier.synced(q);
    }

    /**
     * The device type of a tensor: its stored type, or F32 when it is dequantized on upload.
     *
     * @throws IllegalArgumentException when the model has no tensor of that name
     */
    public GGMLType deviceType(String name) {
        return requirePlan(name).deviceType();
    }

    /** Largest single buffer this uploader allocates. */
    public long maxBufferBytes() {
        return maxBufferBytes;
    }

    /** Number of tensors whose plan streams as lossless W12 (the {@code neuralfree.weights.w12} target set). */
    public int w12TensorCount() {
        int n = 0;
        for (TensorPlan p : plans.values()) {
            if (p.deviceType() == GGMLType.W12) {
                n++;
            }
        }
        return n;
    }

    // ------------------------------------------------------------------ upload

    /**
     * Uploads every tensor that is not deferred, then finishes the queue. With {@code w12F16 = round}, the load
     * reports once how many weights the grid rounding moved and their largest relative change.
     *
     * @param deferred tensors to leave for a later {@link #upload(String)} or {@code uploadResident} call, for another
     *                 owner (streamed layers), or off the device ({@link WeightPlacement})
     * @throws IllegalArgumentException      when a W12 tensor's weights cannot be stored losslessly (an off-grid F16
     *                                       weight under {@code w12F16 = refuse}), naming the tensor, the element
     *                                       and {@link SpeculativeOptions#PROP_W12_F16}
     * @throws IllegalStateException         when an allocation or upload fails; the message names the tensor and sizes
     * @throws UnsupportedOperationException when a tensor to upload cannot be uploaded
     *                                       ({@link TensorPlan#requireUploadable()}), or the model gives the W12
     *                                       streaming no target
     */
    public void uploadResident(Predicate<GGUFTensorInfo> deferred) {
        Objects.requireNonNull(deferred, "deferred");
        ensureOpen();
        int count = 0;
        int dequantized = 0;
        int sharded = 0;
        long bytes = 0;
        for (TensorPlan p : plans.values()) {
            if (resident.containsKey(p.name()) || deferred.test(p.info())) {
                continue;
            }
            GpuWeight w = uploadPlan(p);
            count++;
            bytes += p.deviceBytes();
            if (p.needsDecode()) {
                dequantized++;
            }
            if (!w.isSingleShard()) {
                sharded++;
            }
        }
        finishQueue();
        attachScales();
        copier.release();
        if (config.w12 && w12TensorCount() == 0) {
            throw new UnsupportedOperationException(SpeculativeOptions.PROP_W12
                    + "=true has no W12 target on " + archName + ": with the output head "
                    + (config.w12Head ? "included" : "excluded") + ", no projection weight is a matmul-consumed"
                    + " BF16/F16 tensor wide enough for " + GGMLType.W12 + " blocks and the backend has no"
                    + " lossless W12 kernels for this model's types; leave the weights in their stored type");
        }
        if (w12Rounding != null) {
            LOGGER.info("{}: {}=round moved {} weights onto the bfloat16 grid, largest relative change {}", archName,
                    SpeculativeOptions.PROP_W12_F16, w12Rounding.movedWeights(),
                    String.format(java.util.Locale.ROOT, "%.4g", w12Rounding.maxRelativeChange()));
        }
        LOGGER.info("{}: uploaded {} tensors ({} MB), {} dequantized to F32, {} split into row shards"
                        + " (largest buffer {} MB)", archName, count, bytes >> 20, dequantized, sharded,
                maxBufferBytes == Long.MAX_VALUE ? "unbounded" : String.valueOf(maxBufferBytes >> 20));
    }

    /**
     * Uploads one tensor now (a no-op when it is already resident), then finishes the queue.
     *
     * @return the resident tensor
     * @throws IllegalArgumentException when the model has no such tensor
     * @throws IllegalStateException    when the allocation or upload fails
     */
    public GpuWeight upload(String name) {
        ensureOpen();
        TensorPlan p = requirePlan(name);
        GpuWeight w = resident.get(name);
        if (w == null) {
            uploadPlan(p);
            finishQueue();
            attachScales();
            copier.release();
            w = resident.get(name);
        }
        return w;
    }

    /** Device bytes of the resident tensors. */
    public long residentBytes() {
        return residentBytes;
    }

    /** Number of resident tensors. */
    public int residentCount() {
        return resident.size();
    }

    // ------------------------------------------------------------------ live updates

    /**
     * Writes updated contents over the resident device buffers of the tensor {@code name}: a live weight update
     * (vLLM's in-process weight transfer). Nothing is reallocated and the shape is fixed, so the next forward pass
     * on this queue serves the new bytes. The write copies its host data at enqueue; the queue is finished before
     * this returns, so the update is on the device ({@code update_weights}'s synchronize).
     *
     * @param name  the tensor's stored name, exactly as the model file holds it
     * @param dtype element type the {@code data} bytes hold: the tensor's stored type (decoded to the device type
     *              the way a load decodes it) or F32 when the tensor is consumed as F32
     * @param ne    ggml shape of the update; must equal the model's
     * @param data  the whole tensor, little-endian, row-major over {@code ne[1..]}
     * @throws IllegalArgumentException      when the model has no tensor of that name, or the shape or dtype does
     *                                       not match
     * @throws UnsupportedOperationException when the tensor is not resident on the device (streamed, host-placed,
     *                                       expert-cached or split), naming it
     * @throws IllegalStateException         when the uploader is closed or a write fails
     */
    public void updateTensor(String name, GGMLType dtype, long[] ne, ByteBuffer data) {
        ensureOpen();
        TensorPlan p = requirePlan(name);
        GpuWeight w = resident.get(p.name());
        if (w == null) {
            throw new UnsupportedOperationException("tensor " + name + " is not resident on the device;"
                    + " a streamed, host-placed, expert-cached or split model takes no live update for it");
        }
        if (!Arrays.equals(p.ne(), ne)) {
            throw new IllegalArgumentException("tensor " + name + " has shape " + Arrays.toString(p.ne())
                    + ", the update carries " + Arrays.toString(ne));
        }
        if (dtype == p.sourceType()) {
            writeUpdate(p, w, data, p.needsDecode());
        } else if (dtype == GGMLType.F32 && (p.deviceType() == GGMLType.F32 || p.deviceType() == GGMLType.W12)) {
            writeUpdate(p, w, data, false);
        } else if (dtype == GGMLType.F32) {
            throw new IllegalArgumentException("tensor " + name + " is consumed as " + p.deviceType()
                    + " on the device; F32 update data would requantize it — send its stored " + p.sourceType()
                    + " bytes instead");
        } else {
            throw new IllegalArgumentException("tensor " + name + " takes update data as its stored "
                    + p.sourceType() + " or F32" + (p.deviceType() == GGMLType.F32 ? "" : " (it is consumed as "
                    + p.deviceType() + ")") + ", got " + dtype);
        }
        finishQueue();
    }

    /**
     * Enqueues the bytes of one updated tensor into its resident shard buffers: raw when the update dtype is the
     * device type, decoded to F32 chunk by chunk when it is the stored type of a tensor the device holds
     * dequantized, and re-encoded losslessly for a W12 tensor.
     */
    private void writeUpdate(TensorPlan p, GpuWeight w, ByteBuffer data, boolean decode) {
        if (p.deviceType() == GGMLType.W12) {
            writeW12Update(p, w, data, decode);
            return;
        }
        long devRowBytes = p.deviceRowBytes();
        long cols = p.cols();
        long chunkBytes = Math.min(config.stagingBytes, (long) Integer.MAX_VALUE - 8);
        long rowsPerChunk = decode
                ? Math.max(1, Math.min(chunkBytes / p.sourceRowBytes(), chunkBytes / (cols * 4L)))
                : Math.max(1, chunkBytes / devRowBytes);
        float[] decoded = decode ? new float[(int) (rowsPerChunk * cols)] : null;
        for (int s = 0; s < w.shardCount(); s++) {
            long shardRow = w.shardRowStart(s);
            long shardRows = w.shardRows(s);
            GpuBuffer buffer = w.shard(s);
            long done = 0;
            while (done < shardRows) {
                long rows = Math.min(rowsPerChunk, shardRows - done);
                long dstOff = done * devRowBytes;
                GpuEvent e;
                if (decode) {
                    int n = (int) (rows * cols);
                    CpuBlockDecoders.decodeElements(p.sourceType(), data,
                            (shardRow + done) * p.sourceRowBytes(), decoded, 0, n);
                    e = queue.enqueueWriteBuffer(buffer, dstOff, n * 4L, decoded);
                } else {
                    long bytes = rows * devRowBytes;
                    ByteBuffer chunk = data.duplicate();
                    chunk.position((int) ((shardRow + done) * devRowBytes));
                    chunk.limit((int) ((shardRow + done) * devRowBytes + bytes));
                    e = queue.enqueueWriteBuffer(buffer, dstOff, bytes, chunk.slice());
                }
                if (e != null) {
                    e.close();
                }
                done += rows;
            }
        }
    }

    /**
     * Re-encodes an updated W12 tensor into its resident shard buffers: each shard's rows are scanned once
     * (stored type when {@code decode}, else F32 floats, rounded onto the bfloat16 grid the records store) to
     * size their new escape tables, then the records are filled and verified bit-exactly chunk by chunk. A shard
     * whose re-encoded record escapes differently than its loaded one and leaves the buffer refuses here,
     * naming the tensor.
     */
    private void writeW12Update(TensorPlan p, GpuWeight w, ByteBuffer data, boolean decode) {
        for (int s = 0; s < w.shardCount(); s++) {
            long shardRow = w.shardRowStart(s);
            long shardRows = w.shardRows(s);
            GpuBuffer buffer = w.shard(s);
            long[] escapes = new long[2];
            copier.scanW12(p, shardRow, shardRows, data, decode, escapes);
            long need = shardRows * p.deviceRowBytes() + escapes[0] * W12Codec.ESCAPE_ENTRY_BYTES
                    + escapes[1] * W12Codec.Q_ESCAPE_ENTRY_BYTES;
            if (buffer.sizeInBytes() < need) {
                throw new UnsupportedOperationException("tensor " + p.name() + ": the re-encoded " + GGMLType.W12
                        + " record of shard " + s + " needs " + need + " bytes, its resident buffer holds "
                        + buffer.sizeInBytes() + " (the update's rows escape differently than the loaded ones);"
                        + " reload the tensor");
            }
            copier.fillW12(p, shardRow, shardRows, buffer, 0L, escapes[0], escapes[1], data, decode, queue, null);
        }
    }

    // ------------------------------------------------------------------ lifecycle

    /** Releases every buffer this uploader allocated and the host staging. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = Closeables.closeAll(owned, null);
        owned.clear();
        resident.clear();
        residentBytes = 0;
        copier.release();
        if (failure != null) {
            throw failure;
        }
    }

    // ------------------------------------------------------------------ internals

    private Map<String, TensorPlan> planAll(List<GGUFTensorInfo> tensors) {
        Map<String, GGUFTensorInfo> byName = new LinkedHashMap<>();
        for (GGUFTensorInfo t : tensors) {
            byName.put(t.name(), t);
        }
        boolean tiedOutput = !byName.containsKey("output.weight") && !byName.containsKey("lm_head.weight");
        Map<String, TensorPlan> out = new LinkedHashMap<>();
        for (GGUFTensorInfo t : byName.values()) {
            TensorUse use = TensorUse.classify(t, tiedOutput);
            GGMLType deviceType = deviceTypeFor(t, use);
            out.put(t.name(), TensorPlan.create(t, use, deviceType, maxBufferBytes));
        }
        return out;
    }

    private GGMLType deviceTypeFor(GGUFTensorInfo t, TensorUse use) {
        GGMLType type = t.type();
        if (w12Eligible(t, use)) {
            return GGMLType.W12;
        }
        switch (use) {
            case INTEGER:
                return type;
            case ELEMENTWISE:
                return GGMLType.F32;
            case LOOKUP_MATMUL:
                return type == GGMLType.F32 || (wholeCodeRuns(type, t)
                        && config.policy.isNative(type, TensorUse.LOOKUP)
                        && config.policy.isNative(type, TensorUse.MATMUL)) ? type : GGMLType.F32;
            default:
                return type == GGMLType.F32 || (wholeCodeRuns(type, t) && config.policy.isNative(type, use))
                        ? type : GGMLType.F32;
        }
    }

    /**
     * Whether the tensor streams as lossless W12 ({@code neuralfree.weights.w12}): a matmul-consumed projection
     * weight the backend has W12 kernels for, stored as BF16 or as F16 held to the lossless contract of
     * {@link SpeculativeOptions#PROP_W12_F16} (see {@link #decodeW12Chunk}), wide enough for whole W12 blocks. The
     * output head streams only with {@code neuralfree.weights.w12.head}.
     */
    private boolean w12Eligible(GGUFTensorInfo t, TensorUse use) {
        if (!config.w12 || use != TensorUse.MATMUL
                || !config.policy.isNative(GGMLType.W12, TensorUse.MATMUL)) {
            return false;
        }
        GGMLType type = t.type();
        if (type != GGMLType.BF16 && type != GGMLType.F16) {
            return false;
        }
        String local = TensorUse.localName(t.name());
        if ((local.equals("output.weight") || local.equals("lm_head.weight")) && !config.w12Head) {
            return false;
        }
        return t.dim(0) % GGMLType.W12.getBlockSize() == 0;
    }

    /**
     * Whether the stored tensor fits the device kernels of its type: the FP8 code rows work on 32-element runs,
     * so ggml's per-element F8_E4M3FN (one byte per element) needs rows that are whole 32-element blocks; other
     * rows decode to F32 on upload. Every other type fits at any row width.
     */
    private static boolean wholeCodeRuns(GGMLType type, GGUFTensorInfo t) {
        return type != GGMLType.F8_E4M3FN || t.dim(0) % GGMLType.F8_E4M3.getBlockSize() == 0;
    }

    private TensorPlan requirePlan(String name) {
        TensorPlan p = plans.get(name);
        if (p == null) {
            throw new IllegalArgumentException(archName + " has no tensor " + name);
        }
        return p;
    }

    /**
     * Wraps one shard's failure with the tensor, its row range and the target device. A typed device failure (the
     * device out of memory) surfaces unchanged, so the model factory still reads it as a device limitation.
     */
    private RuntimeException shardFailure(TensorPlan p, long row, long rows, RuntimeException e) {
        if (e instanceof GpuOutOfMemoryException) {
            return e;
        }
        return new IllegalStateException("upload of tensor " + p.name() + " rows [" + row + ", "
                + (row + rows) + ") to " + backend.capabilities().getDeviceName() + " failed: "
                + e.getMessage(), e);
    }

    /**
     * Allocates and fills every shard of one tensor; on failure releases its buffers. A W12 shard's refusal of an
     * off-grid F16 weight ({@link IllegalArgumentException} naming the tensor, the element and
     * {@link SpeculativeOptions#PROP_W12_F16}) surfaces unwrapped, so the model factory can rethrow it instead of
     * falling back; a typed device failure ({@link GpuOutOfMemoryException}) surfaces as itself, and every other
     * failure is wrapped with the tensor, its row range and the device.
     */
    private GpuWeight uploadPlan(TensorPlan p) {
        p.requireUploadable();
        long[] shardRows = p.shardRows();
        List<GpuBuffer> buffers = new ArrayList<>(shardRows.length);
        try {
            long row = 0;
            for (long rows : shardRows) {
                long bytes = rows * p.deviceRowBytes();
                long[] escapes = new long[]{0L, 0L};
                if (p.deviceType() == GGMLType.W12) {
                    try {
                        copier.scanW12(p, row, rows, null, false, escapes);
                    } catch (UncheckedIOException | IllegalArgumentException | UnsupportedOperationException e) {
                        throw e;
                    } catch (RuntimeException e) {
                        throw shardFailure(p, row, rows, e);
                    }
                    bytes += escapes[0] * W12Codec.ESCAPE_ENTRY_BYTES
                            + escapes[1] * W12Codec.Q_ESCAPE_ENTRY_BYTES;
                }
                GpuBuffer b = allocate(p, bytes);
                buffers.add(b);
                try {
                    if (p.deviceType() == GGMLType.W12) {
                        copier.fillW12(p, row, rows, b, 0L, escapes[0], escapes[1], null, false, queue,
                                w12Rounding);
                    } else {
                        copier.copyRows(p, row, rows, b, 0L, queue, w12Rounding);
                    }
                } catch (UncheckedIOException | IllegalArgumentException | UnsupportedOperationException e) {
                    throw e;
                } catch (RuntimeException e) {
                    throw shardFailure(p, row, rows, e);
                }
                row += rows;
            }
        } catch (RuntimeException e) {
            for (GpuBuffer b : buffers) {
                b.close();
            }
            throw e;
        }
        owned.addAll(buffers);
        GpuWeight w = buffers.size() == 1
                ? GpuWeight.of(p.name(), p.deviceType(), p.ne(), buffers.get(0))
                : GpuWeight.sharded(p.name(), p.deviceType(), p.ne(), buffers, shardRows);
        w = withPrism(w);
        resident.put(p.name(), w);
        residentBytes += p.deviceBytes();
        if (buffers.size() > 1) {
            LOGGER.info("{}: {} ({} {}, {} MB) split into {} row shards of at most {} MB", archName, p.name(),
                    p.sourceType(), Arrays.toString(p.ne()), p.deviceBytes() >> 20, buffers.size(),
                    maxBufferBytes >> 20);
        }
        return w;
    }

    /**
     * The device buffer of one shard. The device being out of memory surfaces as its {@link GpuOutOfMemoryException};
     * every other allocation failure is wrapped with the tensor and the bytes already allocated.
     */
    private GpuBuffer allocate(TensorPlan p, long bytes) {
        try {
            return backend.allocateBuffer(bytes);
        } catch (GpuOutOfMemoryException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IllegalStateException("cannot allocate " + bytes + " bytes for tensor " + p.name() + " ("
                    + p.deviceType() + ", " + backend.getAllocatedBytes() + " bytes already allocated on "
                    + backend.capabilities().getDeviceName() + "): " + e.getMessage(), e);
        }
    }

    /**
     * Attaches the input transforms and inverse rotations of a resident tensor: the weight's full input transform
     * chain (HADAMARD, ACT_QUANT, PERMUTE) with the device int buffer of its {@code .input_perm} tensor uploaded
     * once, and the PRISM inverse of a latent embedding table.
     */
    private GpuWeight withPrism(GpuWeight w) {
        InputTransformLoader.Result transforms = config.transforms;
        if (transforms == null || transforms.isEmpty()) {
            return w;
        }
        InputTransform t = transforms.forward(w.name());
        if (!t.isIdentity()) {
            w = w.withInputTransform(t, t.hasPerm() ? inputPermBuffer(w) : null,
                    t.hasPreScale() ? inputPreScaleBuffer(w) : null);
        }
        HadamardSpec inverse = transforms.inverse().get(w.name());
        if (inverse != null) {
            w = w.withInverseHadamard(inverse);
        }
        return w;
    }

    /**
     * The device int buffer of the {@code .input_perm} tensor of a weight with a PERMUTE part, uploading the tensor
     * now if it is not resident yet. The perm tensor is INTEGER: it uploads as stored, and stays resident so
     * {@link #uploadResident} skips it in its normal pass.
     *
     * @throws IllegalStateException when the model has no {@code .input_perm} tensor for the weight (the loader
     *                               binds only transforms the file provides, so this cannot happen to a
     *                               well-formed model)
     */
    private GpuBuffer inputPermBuffer(GpuWeight w) {
        String name = GpuWeight.permNameOf(w.name());
        GpuWeight perm = resident.get(name);
        if (perm == null) {
            TensorPlan p = plans.get(name);
            if (p == null) {
                throw new IllegalStateException(archName + " has no tensor " + name + " for the input permutation"
                        + " of " + w.name());
            }
            perm = uploadPlan(p);
        }
        return perm.shard(0);
    }

    /**
     * The device float buffer of the {@code .input_pre_scale} tensor of a weight with a PRE_SCALE part, uploading the
     * tensor now if it is not resident yet. It stays resident so {@link #uploadResident} skips it in its normal pass.
     *
     * @throws IllegalStateException when the model has no {@code .input_pre_scale} tensor for the weight (the loader
     *                               binds only transforms the file provides, so this cannot happen to a
     *                               well-formed model)
     */
    private GpuBuffer inputPreScaleBuffer(GpuWeight w) {
        String name = GpuWeight.preScaleNameOf(w.name());
        GpuWeight pre = resident.get(name);
        if (pre == null) {
            TensorPlan p = plans.get(name);
            if (p == null) {
                throw new IllegalStateException(archName + " has no tensor " + name + " for the activation"
                        + " pre-scale of " + w.name());
            }
            pre = uploadPlan(p);
        }
        return pre.shard(0);
    }

    /** Attaches each resident scalar {@code <base>.scale} to its resident non-expert {@code <base>.weight}. */
    private void attachScales() {
        for (Map.Entry<String, GpuWeight> e : new ArrayList<>(resident.entrySet())) {
            GpuWeight w = e.getValue();
            String name = e.getKey();
            if (!name.endsWith(".weight") || w.hasScale()) {
                continue;
            }
            TensorPlan wp = plans.get(name);
            if (wp == null || wp.use() == TensorUse.EXPERT) {
                continue;
            }
            TensorPlan sp = plans.get(name.substring(0, name.length() - ".weight".length()) + ".scale");
            if (sp == null || sp.info().numElements() != 1) {
                continue;
            }
            float s = copier.readScalar(sp);
            if (!Float.isFinite(s)) {
                throw new IllegalStateException("tensor " + sp.name() + " holds a non-finite output scale " + s);
            }
            resident.put(name, w.withScale(s));
        }
    }

    private void finishQueue() {
        try {
            queue.finish();
            copier.synced(queue);
        } catch (GpuOutOfMemoryException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IllegalStateException("weight upload to " + backend.capabilities().getDeviceName()
                    + " failed: " + e.getMessage(), e);
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("weight uploader is closed");
        }
    }

    // ================================================================== nested types

    /** How the forward path consumes a tensor; decides its device type. */
    public enum TensorUse {
        /** Projection weight read by {@code {T}_MatVec} / batch matmul kernels. */
        MATMUL,
        /** Row lookup table read by {@code {T}_EmbeddingLookup} kernels. */
        LOOKUP,
        /** The token embedding of a model whose output head is tied to it: looked up and multiplied. */
        LOOKUP_MATMUL,
        /**
         * Position or token-type table ({@code position_embd}, {@code token_types}) whose rows are added to the
         * token embedding; a row lookup ({@code GET_ROWS}) like {@link #LOOKUP}.
         */
        ADDED_LOOKUP,
        /** Expert tensor {@code [cols, rows, n_expert]} read per expert. */
        EXPERT,
        /** Consumed by element-wise or special kernels: always F32. */
        ELEMENTWISE,
        /** Integer table: uploaded as stored. */
        INTEGER;

        /**
         * Names (after the {@code blk.N.} / {@code enc.blk.N.} / {@code dec.blk.N.} / {@code posnet.N.} /
         * {@code convnext.N.} prefix and the {@code .weight}/{@code .bias} suffix) of tensors that ggml consumes
         * with an op other than {@code MUL_MAT}/{@code MUL_MAT_ID}/{@code GET_ROWS} (llama-arch.cpp:817-1144,
         * {@code LLM_TENSOR_INFOS}), plus the relative-position bias and compressor position tables
         * ({@code attn_rel_b}, {@code *_compressor_ape}) that ggml reads with {@code GET_ROWS}; NeuralFree's kernels
         * for all of them read F32.
         */
        private static final Set<String> ELEMENTWISE_NAMES = new HashSet<>(Arrays.asList(
                "altup_correct_scale", "altup_router_norm", "attn_compressor_ape", "attn_compressor_norm",
                "attn_conv_base", "attn_k_norm", "attn_kv_a_norm", "attn_norm", "attn_norm_2", "attn_output_norm",
                "attn_q_a_norm", "attn_q_norm", "attn_rel_b", "attn_res_score", "attn_resid_alpha", "attn_resid_beta",
                "attn_sinks", "attn_sub_norm", "channel_mix_lerp_k", "channel_mix_lerp_r", "cls.norm", "conv1",
                "conv1d", "conv2", "cross_attn_norm", "cross_attn_norm_kv", "cross_attn_rel_b", "dec.output_norm",
                "decision.field_norm", "decision.hidden_norm", "decision.option_norm", "decision.option_summary_norm",
                "decision.query_norm", "decision.scales", "dw", "enc.output_norm", "exp_probs_b", "exp_probs_b_vl",
                "ffn.act", "ffn_conv_base", "ffn_norm", "ffn_norm_exps", "ffn_post_exp_norm", "ffn_pre_exp_norm",
                "ffn_res_score", "ffn_resid_alpha", "ffn_resid_beta", "ffn_routed_norm", "ffn_sub_norm", "gamma",
                "hc_attn_base", "hc_attn_norm", "hc_attn_scale", "hc_ffn_base", "hc_ffn_norm", "hc_ffn_scale",
                "hrm.z_l_init", "indexer.k_norm", "indexer.q_norm", "indexer_compressor_ape",
                "indexer_compressor_norm", "laurel_post_norm", "layer_output_norm", "layer_output_scale",
                "lilicorr.anchor_norm", "lilicorr.attention_bias", "lilicorr.output_norm", "lilicorr.rank_embedding",
                "lilicorr.slot_embedding", "nextn.enorm", "nextn.hc_head_norm", "nextn.hnorm",
                "nextn.shared_head_norm", "norm", "norm1", "norm2", "output_hc_base", "output_hc_norm",
                "output_hc_scale", "output_norm", "output_res_score", "per_layer_proj_norm", "ple_conv1d",
                "ple_norm_conv", "ple_norm_key", "ple_norm_query", "post_attention_norm", "post_ffw_norm",
                "post_ffw_norm_1", "post_ffw_norm_2", "post_norm", "pre_ffw_norm_2", "rope_factors_long",
                "rope_factors_short", "rope_freqs", "shortconv.conv", "ssm_a", "ssm_b_norm", "ssm_c_norm",
                "ssm_conv1d", "ssm_conv1d_k", "ssm_conv1d_q", "ssm_conv1d_v", "ssm_d", "ssm_dt_norm", "ssm_norm",
                "time_mix_a0", "time_mix_decay", "time_mix_first", "time_mix_k_a", "time_mix_k_k",
                "time_mix_lerp_fused", "time_mix_lerp_g", "time_mix_lerp_k", "time_mix_lerp_r", "time_mix_lerp_v",
                "time_mix_lerp_w", "time_mix_lerp_x", "time_mix_ln", "time_mix_r_k", "time_mix_v0", "time_mix_w0",
                "token_embd_norm"));

        /** Token row lookup tables ({@code GET_ROWS} in llama-arch.cpp {@code LLM_TENSOR_INFOS}). */
        private static final Set<String> LOOKUP_NAMES = new HashSet<>(Arrays.asList(
                "token_embd.weight", "tok_embeddings.weight", "per_layer_token_embd.weight",
                "nextn.embed_tokens.weight"));

        /** Lookup tables whose rows are added to the token embedding (also {@code GET_ROWS}). */
        private static final Set<String> ADDED_LOOKUP_NAMES = new HashSet<>(Arrays.asList(
                "position_embd.weight", "token_types.weight"));

        private static final Pattern PREFIX =
                Pattern.compile("^(?:(?:enc|dec)\\.)?(?:blk|posnet|convnext)\\.\\d+\\.");

        /** The tensor's name without the {@code blk.N.}-style layer prefix. */
        static String localName(String name) {
            return PREFIX.matcher(name).replaceFirst("");
        }

        /** Classifies one tensor; {@code tiedOutput} when the model has no separate output head. */
        static TensorUse classify(GGUFTensorInfo t, boolean tiedOutput) {
            GGMLType type = t.type();
            if (type == GGMLType.I8 || type == GGMLType.I16 || type == GGMLType.I32 || type == GGMLType.I64) {
                return INTEGER;
            }
            String name = t.name();
            String local = localName(name);
            if (WeightShards.isExpert(local, t.rank())) {
                return EXPERT;
            }
            if (LOOKUP_NAMES.contains(local)) {
                boolean tokenEmbd = local.equals("token_embd.weight") || local.equals("tok_embeddings.weight");
                return tokenEmbd && tiedOutput && local.equals(name) ? LOOKUP_MATMUL : LOOKUP;
            }
            if (ADDED_LOOKUP_NAMES.contains(local)) {
                return ADDED_LOOKUP;
            }
            if (t.rank() <= 1 || local.endsWith(".bias") || local.endsWith(".scale")
                    || local.endsWith(".lora_a") || local.endsWith(".lora_b")) {
                return ELEMENTWISE;
            }
            String base = local.endsWith(".weight") ? local.substring(0, local.length() - ".weight".length()) : local;
            return ELEMENTWISE_NAMES.contains(base) ? ELEMENTWISE : MATMUL;
        }
    }

    /** Whether a backend has kernels for a weight type in a given use. */
    @FunctionalInterface
    public interface TypePolicy {
        /** True when tensors of {@code type} in {@code use} are consumed as stored (no CPU dequantization). */
        boolean isNative(GGMLType type, TensorUse use);

        /**
         * The policy of the engine's GPU path: a matmul type needs {@code GpuOps} support and its
         * {@code {T}_MatVec} kernel, a lookup type its embedding-lookup kernel, an expert type its
         * {@code {T}_MatVecId} kernel.
         */
        static TypePolicy forKernels(LlmKernels kernels) {
            Objects.requireNonNull(kernels, "kernels");
            return (type, use) -> {
                switch (use) {
                    case MATMUL:
                        return GpuOps.supportsWeightType(type)
                                && kernels.has(matVecKernel(type));
                    case LOOKUP:
                    case LOOKUP_MATMUL:
                    case ADDED_LOOKUP:
                        return GpuOps.supportsWeightType(type)
                                && kernels.has(lookupKernel(type));
                    case EXPERT:
                        return MoeKernels.supports(type)
                                && kernels.has(MoeKernels.matVecId(type));
                    default:
                        return false;
                }
            };
        }

        /** {@code {T}_MatVec} kernel name of a weight type ({@code Float_MatVec} for F32): the type's GPU family's
         *  row, so types that share a family (the per-element F8_E4M3FN reusing the F8_E4M3 rows) share the name. */
        static String matVecKernel(GGMLType type) {
            String name = GpuOps.matVecKernel(type);
            return name != null ? name : type == GGMLType.F32 ? "Float_MatVec" : type.name() + "_MatVec";
        }

        /** Single-token embedding-lookup kernel name of a weight type, from its GPU family. */
        static String lookupKernel(GGMLType type) {
            String name = GpuOps.lookupKernel(type);
            if (name != null) {
                return name;
            }
            switch (type) {
                case F32:
                    return "LLM_EmbeddingLookup";
                case Q8_0:
                    return "LLM_Q8_0_EmbeddingLookup";
                case MXFP4:
                    return "LLM_MXFP4_EmbeddingLookup";
                default:
                    return type.name() + "_EmbeddingLookup";
            }
        }
    }

    /** Immutable uploader configuration. */
    public static final class Config {
        private final TypePolicy policy;
        private final InputTransformLoader.Result transforms;
        private final int stagingBytes;
        private final long maxBufferBytes;
        private final boolean w12;
        private final boolean w12Head;
        private final W12F16Mode w12F16;

        private Config(Builder b) {
            this.policy = b.policy;
            this.transforms = b.transforms;
            this.stagingBytes = b.stagingBytes;
            this.maxBufferBytes = b.maxBufferBytes;
            this.w12 = b.w12;
            this.w12Head = b.w12Head;
            this.w12F16 = b.w12F16;
        }

        public TypePolicy policy() {
            return policy;
        }

        /** The model's input transforms, or null. */
        public InputTransformLoader.Result transforms() {
            return transforms;
        }

        public int stagingBytes() {
            return stagingBytes;
        }

        /** Largest buffer override in bytes; 0 uses the device limit. */
        public long maxBufferBytes() {
            return maxBufferBytes;
        }

        /** Whether bfloat16-representable projection weights stream as lossless W12 blocks. */
        public boolean w12() {
            return w12;
        }

        /** Whether the output head streams too ({@link #w12()}). */
        public boolean w12Head() {
            return w12Head;
        }

        /** How an F16 tensor's weights meet the bfloat16 grid the W12 rows store. */
        public W12F16Mode w12F16() {
            return w12F16;
        }

        /** Builder of {@link Config}. */
        public static final class Builder {
            private final TypePolicy policy;
            private InputTransformLoader.Result transforms;
            private int stagingBytes = DEFAULT_STAGING_BYTES;
            private long maxBufferBytes;
            private boolean w12;
            private boolean w12Head;
            private W12F16Mode w12F16 = W12F16Mode.DEFAULT;

            Builder(TypePolicy policy) {
                this.policy = Objects.requireNonNull(policy, "policy");
            }

            /**
             * The model's input transforms to attach; null or empty for none. The transforms bind at load
             * ({@link InputTransformLoader#load}), where a malformed one throws
             * {@code com.eonmux.neuralfree.gguf.hparams.GgufFormatException}; a malformed fit throws from
             * {@link GpuWeight#withInputTransform(InputTransform, GpuBuffer)} as the tensor uploads.
             */
            public Builder transforms(InputTransformLoader.Result v) {
                this.transforms = v;
                return this;
            }

            /** Host staging per upload chunk, at least 1 MiB. */
            public Builder stagingBytes(int v) {
                if (v < (1 << 20)) {
                    throw new IllegalArgumentException("staging must be at least 1 MiB, got " + v);
                }
                this.stagingBytes = v;
                return this;
            }

            /** Caps every buffer below the device limit; 0 (the default) uses the device limit. */
            public Builder maxBufferBytes(long v) {
                if (v < 0) {
                    throw new IllegalArgumentException("maxBufferBytes must be non-negative, got " + v);
                }
                this.maxBufferBytes = v;
                return this;
            }

            /** Streams bfloat16-representable projection weights as lossless W12 blocks (needs the backend's
             * W12 kernels; an F16 tensor whose values are not bfloat16-representable refuses at upload unless
             * {@link #w12F16(W12F16Mode)} says {@code round}). */
            public Builder w12(boolean v) {
                this.w12 = v;
                return this;
            }

            /** Streams the output head too; only effective with {@link #w12(boolean)}. */
            public Builder w12Head(boolean v) {
                this.w12Head = v;
                return this;
            }

            /**
             * How an F16 tensor's weights meet the bfloat16 grid the W12 rows store ({@code refuse}, the
             * default, refuses an off-grid weight by name; {@code round} rounds it there lossily and reports
             * the moved weights at load).
             *
             * @throws NullPointerException when {@code v} is null
             */
            public Builder w12F16(W12F16Mode v) {
                this.w12F16 = Objects.requireNonNull(v, "w12F16");
                return this;
            }

            public Config build() {
                if (w12Head && !w12) {
                    throw new IllegalArgumentException("w12Head needs w12, which streams the target's weights"
                            + " it applies to");
                }
                return new Config(this);
            }
        }
    }

    /** Upload plan of one tensor: use, device type and row shards. Immutable. */
    public static final class TensorPlan {
        private final GGUFTensorInfo info;
        private final TensorUse use;
        private final GGMLType deviceType;
        private final long[] ne;
        private final long rows;
        private final long cols;
        private final long sourceRowBytes;
        private final long deviceRowBytes;
        private final long[] shardRows;
        private final String problem;

        private TensorPlan(GGUFTensorInfo info, TensorUse use, GGMLType deviceType, long[] ne, long rows, long cols,
                           long sourceRowBytes, long deviceRowBytes, long[] shardRows, String problem) {
            this.info = info;
            this.use = use;
            this.deviceType = deviceType;
            this.ne = ne;
            this.rows = rows;
            this.cols = cols;
            this.sourceRowBytes = sourceRowBytes;
            this.deviceRowBytes = deviceRowBytes;
            this.shardRows = shardRows;
            this.problem = problem;
        }

        static TensorPlan create(GGUFTensorInfo info, TensorUse use, GGMLType deviceType, long maxBufferBytes) {
            GGMLType src = info.type();
            long[] ne = info.dims().length == 0 ? new long[]{1} : info.dims().clone();
            for (long d : ne) {
                if (d <= 0) {
                    throw new IllegalStateException("tensor " + info.name() + " has a non-positive dimension "
                            + Arrays.toString(ne));
                }
            }
            if (src.getBlockSize() <= 0 || src.getTypeSize() <= 0 || ne[0] % src.getBlockSize() != 0) {
                throw new IllegalStateException("tensor " + info.name() + " (" + src + ") has row width " + ne[0]
                        + ", not a whole number of " + src + " blocks");
            }
            if (deviceType == GGMLType.W12 && ne[0] % GGMLType.W12.getBlockSize() != 0) {
                throw new IllegalStateException("tensor " + info.name() + " has row width " + ne[0]
                        + ", not a whole number of " + GGMLType.W12.getBlockSize() + "-element "
                        + GGMLType.W12 + " blocks");
            }
            long cols = ne[0];
            long rows = 1;
            for (int i = 1; i < ne.length; i++) {
                rows = Math.multiplyExact(rows, ne[i]);
            }
            long srcRowBytes = src.getBytesForElements(cols);
            final long devRowBytes;
            if (deviceType == GGMLType.W12) {
                if (cols > Integer.MAX_VALUE - 8) {
                    throw new IllegalStateException("tensor " + info.name() + " has row width " + cols
                            + ", beyond the " + GGMLType.W12 + " row-record range");
                }
                devRowBytes = W12Codec.rowBytes((int) cols);
            } else {
                devRowBytes = deviceType.getBytesForElements(cols);
            }
            long unitRows = ne.length >= 3 ? ne[1] : 1;
            String problem = null;
            long[] shards = null;
            if (deviceType != src && !CpuBlockDecoders.supports(src)) {
                problem = "tensor " + info.name() + " has type " + src
                        + ", which the backend has no kernels for and the CPU cannot dequantize";
            } else {
                long unitBytes = Math.multiplyExact(unitRows, devRowBytes);
                if (unitBytes > maxBufferBytes) {
                    problem = "tensor " + info.name() + ": " + (unitRows > 1 ? "one expert" : "one row") + " of "
                            + unitBytes + " bytes exceeds the device's largest buffer of " + maxBufferBytes
                            + " bytes";
                } else {
                    shards = WeightShards.shardRows(rows, unitRows, unitBytes, maxBufferBytes);
                }
            }
            return new TensorPlan(info, use, deviceType, ne, rows, cols, srcRowBytes, devRowBytes, shards, problem);
        }

        /**
         * Throws when this tensor cannot be uploaded: its type needs a dequantization the CPU decoders lack, or one
         * row (one expert for rank-3 tensors) exceeds the device's largest buffer.
         */
        public void requireUploadable() {
            if (problem != null) {
                throw new GpuModelUnsupportedException(problem);
            }
        }

        /** Why the tensor cannot be uploaded, or null. */
        public String problem() {
            return problem;
        }

        public String name() {
            return info.name();
        }

        public GGUFTensorInfo info() {
            return info;
        }

        public TensorUse use() {
            return use;
        }

        /** Stored type in the GGUF file. */
        public GGMLType sourceType() {
            return info.type();
        }

        /** Type of the device copy: the stored type, or F32 when dequantized on upload. */
        public GGMLType deviceType() {
            return deviceType;
        }

        /** True when the device copy is decoded to F32 on the CPU. */
        public boolean needsDecode() {
            return deviceType != info.type();
        }

        /** ggml shape (copy). */
        public long[] ne() {
            return ne.clone();
        }

        public long rows() {
            return rows;
        }

        public long cols() {
            return cols;
        }

        public long sourceRowBytes() {
            return sourceRowBytes;
        }

        public long deviceRowBytes() {
            return deviceRowBytes;
        }

        /** Device bytes of the whole tensor. */
        public long deviceBytes() {
            return rows * deviceRowBytes;
        }

        /** Rows per shard (copy); one entry when the tensor fits one buffer, empty when it cannot be uploaded. */
        public long[] shardRows() {
            return shardRows == null ? new long[0] : shardRows.clone();
        }

        public int shardCount() {
            return shardRows == null ? 0 : shardRows.length;
        }

        @Override
        public String toString() {
            return info.name() + " " + info.type() + (needsDecode() ? "->" + deviceType : "") + " "
                    + Arrays.toString(ne) + " " + use + " shards=" + shardCount();
        }
    }
}
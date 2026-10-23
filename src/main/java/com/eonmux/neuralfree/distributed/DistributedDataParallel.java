package com.eonmux.neuralfree.distributed;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.optim.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Data-parallel training of one module replica per process: each replica runs its forward pass and its loss on its
 * own shard, and the parameter gradients are averaged across the processes ({@link ReduceOp#MEAN} all-reduce through
 * the {@link DistributedBackend}) before the optimizer step, as PyTorch's {@code DistributedDataParallel} does.
 *
 * <p><b>State.</b> Construction overwrites every rank's parameters and persistent buffers ({@link Module#stateDict()})
 * with rank 0's, so the replicas start identical whatever their initialisation. With {@code broadcastBuffers} (the
 * default, PyTorch's {@code broadcast_buffers=True}) every {@link #forward} in training mode first overwrites the
 * persistent buffers (running statistics) with rank 0's. Each of these is one broadcast of the flattened values.</p>
 *
 * <p><b>Steps.</b> {@link #forward} records in the caller's grad mode, so the local loss computed from its result is
 * backpropagated to the replica's parameters. A step is either {@link #backward(Tensor)} (backward and all-reduce)
 * followed by the optimizer's own {@code step()}, or the loss's {@code backward()} followed by {@link #step(Optimizer)}
 * (all-reduce and step). The gradient all-reduce writes gradients, never parameters, and runs under
 * {@link ComputationGraph#noGrad()}; so does the optimizer step.</p>
 *
 * <p><b>Buckets.</b> The parameters are grouped, in reverse registration order (the order a backward pass produces
 * their gradients), into buckets of about {@code bucketSizeMb} megabytes, or one bucket per parameter without
 * bucketing. Each bucket is reduced by one all-reduce of its trainable parameters' gradients, flattened, followed by
 * one usage flag per parameter. A trainable parameter without a gradient on this rank (its backward pass did not
 * reach it, as with an expert no token was routed to) contributes zeros, so every rank issues the same collectives
 * whatever its data, as PyTorch's {@code find_unused_parameters} does: after the reduction it holds the mean
 * gradient when some rank produced one, and stays without a gradient when no rank did.</p>
 */
public final class DistributedDataParallel implements AutoCloseable {
    private final Module module;
    private final DistributedBackend backend;
    private final boolean broadcastBuffers;
    private final AtomicBoolean closed;
    private final List<List<Parameter>> buckets;
    /** The state dict paths of the module's persistent buffers, in state dict order. */
    private final List<String> bufferPaths;

    public DistributedDataParallel(Module module, DistributedBackend backend) {
        this(module, backend, true, 25);
    }

    public DistributedDataParallel(Module module, DistributedBackend backend, boolean gradientBucketing, int bucketSizeMb) {
        this(module, backend, gradientBucketing, bucketSizeMb, true);
    }

    /**
     * Wraps {@code module} and overwrites its parameters and persistent buffers with rank 0's (one broadcast, which
     * every rank's construction joins).
     *
     * @throws IllegalArgumentException when {@code module} or {@code backend} is null, or {@code bucketSizeMb} is not
     *                                  positive while bucketing is enabled
     */
    public DistributedDataParallel(Module module, DistributedBackend backend, boolean gradientBucketing,
                                   int bucketSizeMb, boolean broadcastBuffers) {
        if (module == null) {
            throw new IllegalArgumentException("Module cannot be null");
        }
        if (backend == null) {
            throw new IllegalArgumentException("Backend cannot be null");
        }
        if (gradientBucketing && bucketSizeMb < 1) {
            throw new IllegalArgumentException("Bucket size must be at least 1 MB, got " + bucketSizeMb);
        }

        this.module = module;
        this.backend = backend;
        this.broadcastBuffers = broadcastBuffers;
        this.closed = new AtomicBoolean(false);
        this.buckets = gradientBucketing ? bucketsOf(module.parameters(), bucketSizeMb * 1024L * 1024L)
            : singleBuckets(module.parameters());

        Map<String, Object> state = module.stateDict();
        this.bufferPaths = PersistentBuffers.paths(module, state);
        broadcastFromRankZero(state);
    }

    /** {@code params} in reverse order, grouped into buckets of at most {@code bucketBytes} (at least one each). */
    private static List<List<Parameter>> bucketsOf(List<Parameter> params, long bucketBytes) {
        List<List<Parameter>> result = new ArrayList<>();
        List<Parameter> current = new ArrayList<>();
        long currentBytes = 0;
        for (int i = params.size() - 1; i >= 0; i--) {
            Parameter param = params.get(i);
            long paramBytes = param.byteSize();
            if (currentBytes + paramBytes > bucketBytes && !current.isEmpty()) {
                result.add(Collections.unmodifiableList(current));
                current = new ArrayList<>();
                currentBytes = 0;
            }
            current.add(param);
            currentBytes += paramBytes;
        }
        if (!current.isEmpty()) {
            result.add(Collections.unmodifiableList(current));
        }
        return Collections.unmodifiableList(result);
    }

    /** One bucket per parameter, in reverse order. */
    private static List<List<Parameter>> singleBuckets(List<Parameter> params) {
        List<List<Parameter>> result = new ArrayList<>();
        for (int i = params.size() - 1; i >= 0; i--) {
            result.add(Collections.singletonList(params.get(i)));
        }
        return Collections.unmodifiableList(result);
    }

    /**
     * The replica's output for its shard, recorded in the caller's grad mode; in training mode with
     * {@code broadcastBuffers}, the persistent buffers first take rank 0's values (see the class description).
     */
    public Tensor forward(Tensor input) {
        ensureOpen();
        if (broadcastBuffers && module.isTraining() && !bufferPaths.isEmpty() && backend.worldSize() > 1) {
            broadcastFromRankZero(PersistentBuffers.read(module, bufferPaths));
        }
        return module.forward(input);
    }

    /**
     * Backpropagates the local {@code loss} ({@link Tensor#backward()}, which releases its graph) and then averages
     * the parameter gradients across the processes ({@link #synchronizeGradients()}).
     *
     * @throws IllegalStateException when {@code loss} does not require grad
     */
    public void backward(Tensor loss) {
        ensureOpen();
        Objects.requireNonNull(loss, "Loss cannot be null");
        loss.backward();
        synchronizeGradients();
    }

    /**
     * Averages the parameter gradients across the processes after the caller's own backward pass; the same as
     * {@link #synchronizeGradients()}.
     */
    public void backward() {
        ensureOpen();
        synchronizeGradients();
    }

    /**
     * All-reduces ({@link ReduceOp#MEAN}) the gradients of the trainable parameters, one collective per bucket, in
     * place and without recording (see the class description). Every rank must call it at the same point, with the
     * same parameters trainable.
     */
    public void synchronizeGradients() {
        ensureOpen();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (List<Parameter> bucket : buckets) {
                reduce(bucket);
            }
        }
    }

    /**
     * One all-reduce of the bucket's trainable parameters: their gradients flattened (zeros for a parameter without
     * one), then one usage flag per parameter (1 when it has a gradient). The mean of a flag is positive exactly when
     * some rank produced that gradient. Gradients of every floating-point dtype are exchanged as FLOAT32
     * ({@link CollectiveTensors#read}); a FLOAT32 gradient takes the mean in place, any other gradient, and a gradient
     * only another rank produced, is replaced by a new one in its own (or its parameter's) dtype.
     */
    private void reduce(List<Parameter> bucket) {
        List<Parameter> members = new ArrayList<>();
        for (Parameter param : bucket) {
            if (param.requiresGrad()) {
                members.add(param);
            }
        }
        if (members.isEmpty()) {
            return;
        }
        int[] offsets = new int[members.size() + 1];
        for (int i = 0; i < members.size(); i++) {
            offsets[i + 1] = Math.addExact(offsets[i], Math.toIntExact(members.get(i).numElements()));
        }
        int flags = offsets[members.size()];
        float[] flat = new float[Math.addExact(flags, members.size())];
        for (int i = 0; i < members.size(); i++) {
            Tensor grad = members.get(i).grad();
            if (grad != null) {
                float[] values = CollectiveTensors.read(grad);
                System.arraycopy(values, 0, flat, offsets[i], values.length);
                flat[flags + i] = 1f;
            }
        }
        float[] reduced;
        try (Tensor carrier = Tensor.fromArray(members.get(0).backend(), flat)) {
            backend.allReduce(carrier, ReduceOp.MEAN);
            reduced = carrier.toFloatArray();
        }
        for (int i = 0; i < members.size(); i++) {
            Parameter param = members.get(i);
            Tensor grad = param.grad();
            if (grad == null && reduced[flags + i] == 0f) {
                continue;
            }
            float[] mean = Arrays.copyOfRange(reduced, offsets[i], offsets[i + 1]);
            if (grad != null && grad.dtype() == DType.FLOAT32) {
                CollectiveTensors.writeBack(grad, mean);
            } else {
                param.setGrad(CollectiveTensors.shapedLike(grad != null ? grad : param.tensor(), mean));
            }
        }
    }

    /**
     * Overwrites the module state dict entries in {@code values} (row-major {@code float[]} values keyed by path) with
     * rank 0's: one broadcast of the flattened values, loaded with {@link Module#loadStateDict(Map, boolean)} on the
     * other ranks.
     */
    private void broadcastFromRankZero(Map<String, Object> values) {
        if (backend.worldSize() == 1 || values.isEmpty()) {
            return;
        }
        int total = 0;
        for (Object value : values.values()) {
            total = Math.addExact(total, ((float[]) value).length);
        }
        float[] flat = new float[total];
        int offset = 0;
        for (Object value : values.values()) {
            float[] entry = (float[]) value;
            System.arraycopy(entry, 0, flat, offset, entry.length);
            offset += entry.length;
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor carrier = Tensor.fromArray(stateBackend(), flat)) {
            backend.broadcast(carrier, 0);
            if (backend.rank() == 0) {
                return;
            }
            float[] received = carrier.toFloatArray();
            Map<String, Object> loaded = new LinkedHashMap<>();
            offset = 0;
            for (Map.Entry<String, Object> entry : values.entrySet()) {
                int length = ((float[]) entry.getValue()).length;
                loaded.put(entry.getKey(), Arrays.copyOfRange(received, offset, offset + length));
                offset += length;
            }
            module.loadStateDict(loaded, false);
        }
    }

    /** The backend of the module's parameters, or of its first buffer when it has none. */
    private ComputeBackend stateBackend() {
        ComputeBackend device = module.backend();
        if (device != null) {
            return device;
        }
        for (Tensor buffer : module.namedBuffers().values()) {
            return buffer.backend();
        }
        throw new IllegalStateException(module.moduleName() + " holds no tensor to synchronize");
    }

    /** Averages the gradients across the processes, then steps {@code optimizer} without recording. */
    public void step(Optimizer optimizer) {
        ensureOpen();
        Objects.requireNonNull(optimizer, "Optimizer cannot be null");
        synchronizeGradients();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            optimizer.step();
        }
    }

    public void zeroGrad() {
        ensureOpen();
        module.zeroGrad();
    }

    public Module module() {
        ensureOpen();
        return module;
    }

    public DistributedBackend backend() {
        ensureOpen();
        return backend;
    }

    public int rank() {
        ensureOpen();
        return backend.rank();
    }

    public int worldSize() {
        ensureOpen();
        return backend.worldSize();
    }

    public boolean isDistributed() {
        ensureOpen();
        return backend.worldSize() > 1;
    }

    public Map<String, Object> stateDict() {
        ensureOpen();
        return module.stateDict();
    }

    /** {@link Module#loadStateDict(Map)} on the wrapped module (non-strict). */
    public void loadStateDict(Map<String, Object> stateDict) {
        ensureOpen();
        module.loadStateDict(stateDict);
    }

    /** {@link Module#loadStateDict(Map, boolean)} on the wrapped module. */
    public void loadStateDict(Map<String, Object> stateDict, boolean strict) {
        ensureOpen();
        module.loadStateDict(stateDict, strict);
    }

    public void train() {
        ensureOpen();
        module.train();
    }

    public void eval() {
        ensureOpen();
        module.eval();
    }

    public long parameterCount() {
        ensureOpen();
        return module.parameterCount();
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("DistributedDataParallel has been closed");
        }
    }

    /** Marks this wrapper closed; it does not own the module or the backend. */
    @Override
    public void close() {
        closed.set(true);
    }
}

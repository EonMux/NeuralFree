package com.eonmux.neuralfree.distributed;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.List;
import java.util.Objects;

public interface DistributedBackend extends AutoCloseable {
    int rank();
    int worldSize();
    void allReduce(Tensor tensor, ReduceOp op);

    /**
     * Reduces host {@code values} across the ranks with {@code op} and returns the result, a new array of the same
     * length: the collective of {@link #allReduce(Tensor, ReduceOp)} for values that are already on the host. A
     * backend whose transport is the host overrides it to exchange the array directly; this default stages the values
     * in a tensor on {@code staging} and reduces that.
     *
     * @throws IllegalArgumentException as {@link #allReduce(Tensor, ReduceOp)} throws for the ranks' calls
     */
    default float[] allReduce(float[] values, ReduceOp op, ComputeBackend staging) {
        Objects.requireNonNull(values, "values cannot be null");
        Objects.requireNonNull(staging, "staging backend cannot be null");
        try (Tensor carrier = Tensor.fromArray(staging, values)) {
            allReduce(carrier, op);
            return carrier.toFloatArray();
        }
    }

    void broadcast(Tensor tensor, int srcRank);
    void allGather(List<Tensor> output, Tensor input);
    void barrier();
    boolean isInitialized();
}

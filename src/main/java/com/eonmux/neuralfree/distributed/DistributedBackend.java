package com.eonmux.neuralfree.distributed;

import com.eonmux.neuralfree.tensor.Tensor;

import java.util.List;

public interface DistributedBackend extends AutoCloseable {
    int rank();
    int worldSize();
    void allReduce(Tensor tensor, ReduceOp op);
    void broadcast(Tensor tensor, int srcRank);
    void allGather(List<Tensor> output, Tensor input);
    void barrier();
    boolean isInitialized();
}

package com.eonmux.neuralfree.distributed;

import org.junit.Test;
import static org.junit.Assert.*;

import java.util.List;

public class DistributedBackendTest {

    @Test
    public void testReduceOpEnum() {
        assertEquals(5, ReduceOp.values().length);
        assertNotNull(ReduceOp.SUM);
        assertNotNull(ReduceOp.MEAN);
        assertNotNull(ReduceOp.MAX);
        assertNotNull(ReduceOp.MIN);
        assertNotNull(ReduceOp.PRODUCT);
    }

    @Test
    public void testDistributedBackendInterfaceMethods() throws Exception {
        DistributedBackend mockBackend = new MockDistributedBackend(0, 2);
        
        assertEquals(0, mockBackend.rank());
        assertEquals(2, mockBackend.worldSize());
        assertTrue(mockBackend.isInitialized());
        
        mockBackend.close();
        assertFalse(mockBackend.isInitialized());
    }

    @Test
    public void testDistributedBackendBarrier() throws Exception {
        DistributedBackend backend = new MockDistributedBackend(0, 2);
        
        backend.barrier();
        
        backend.close();
    }

    private static class MockDistributedBackend implements DistributedBackend {
        private final int rank;
        private final int worldSize;
        private boolean closed = false;

        MockDistributedBackend(int rank, int worldSize) {
            this.rank = rank;
            this.worldSize = worldSize;
        }

        @Override
        public int rank() { return rank; }

        @Override
        public int worldSize() { return worldSize; }

        @Override
        public void allReduce(com.eonmux.neuralfree.tensor.Tensor tensor, ReduceOp op) {
        }

        @Override
        public void broadcast(com.eonmux.neuralfree.tensor.Tensor tensor, int srcRank) {
        }

        @Override
        public void allGather(List<com.eonmux.neuralfree.tensor.Tensor> output, com.eonmux.neuralfree.tensor.Tensor input) {
        }

        @Override
        public void barrier() {
        }

        @Override
        public boolean isInitialized() { return !closed; }

        @Override
        public void close() { closed = true; }
    }
}

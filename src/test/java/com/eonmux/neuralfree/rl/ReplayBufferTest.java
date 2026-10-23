package com.eonmux.neuralfree.rl;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.*;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

public class ReplayBufferTest {

    @Test
    public void testReplayBufferCreation() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(100, 4, backend);

        assertEquals(100, buffer.capacity());
        assertEquals(0, buffer.size());
        assertTrue(buffer.isEmpty());
        assertFalse(buffer.isFull());
        assertFalse(buffer.usesPrioritizedReplay());
    }

    @Test
    public void testReplayBufferBuilder() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer.Builder()
            .capacity(50)
            .stateSize(8)
            .backend(backend)
            .usePrioritizedReplay(true)
            .alpha(0.6f)
            .beta(0.4f)
            .betaIncrement(0.001f)
            .epsilon(1e-6f)
            .build();

        assertEquals(50, buffer.capacity());
        assertTrue(buffer.usesPrioritizedReplay());
        assertEquals(0.4f, buffer.beta(), 0.0f);
    }

    @Test
    public void testReplayBufferAdd() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(10, 4, backend);

        float[] state = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] nextState = {5.0f, 6.0f, 7.0f, 8.0f};

        buffer.add(state, 0, 1.0f, nextState, false);

        assertEquals(1, buffer.size());
        assertFalse(buffer.isEmpty());
    }

    @Test
    public void testReplayBufferAddTransition() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(10, 4, backend);

        Transition t = new Transition(new float[]{1, 2, 3, 4}, 1, 0.5f, new float[]{5, 6, 7, 8}, true);
        buffer.add(t);

        assertEquals(1, buffer.size());
    }

    @Test
    public void continuousActionsAreReplayedAsGiven() {
        ReplayBuffer buffer = new ReplayBuffer(1, 2, new MockComputeBackend());
        float[] action = {0.6f, -0.2f};
        buffer.add(new float[]{1, 2}, action, 0.5f, new float[]{3, 4}, false);
        action[0] = 9f;

        ReplayBuffer.SampleBatch batch = buffer.sample(2, new Random(7));
        for (int j = 0; j < 2; j++) {
            assertArrayEquals("the stored copy", new float[]{0.6f, -0.2f}, batch.continuousActions()[j], 0f);
            assertEquals("no discrete action", Transition.NO_DISCRETE_ACTION, batch.actions()[j]);
        }

        buffer.add(new Transition(new float[]{1, 2}, 1, 0.5f, new float[]{3, 4}, true));
        ReplayBuffer.SampleBatch discrete = buffer.sample(1, new Random(7));
        assertNull("a discrete transition has no continuous action", discrete.continuousActions()[0]);
        assertEquals(1, discrete.actions()[0]);
    }

    /** An environment that reuses its observation arrays must not change the transitions already stored. */
    @Test
    public void storedStatesAreCopiesOfTheCallersArrays() {
        ReplayBuffer buffer = new ReplayBuffer(4, 2, new MockComputeBackend());
        float[] observation = new float[2];
        float[] next = new float[2];
        for (int i = 0; i < 2; i++) {
            observation[0] = i;
            observation[1] = 10 + i;
            next[0] = 20 + i;
            next[1] = 30 + i;
            buffer.add(observation, i, 0.5f, next, false);
        }
        Transition reused = new Transition(observation, 1, 0.5f, next, true);
        buffer.add(reused);
        observation[0] = -1f;
        next[0] = -1f;

        ReplayBuffer.SampleBatch batch = buffer.sample(3, new Random(3));
        for (int j = 0; j < batch.batchSize(); j++) {
            int action = batch.actions()[j];
            boolean last = batch.dones()[j];
            float[] expectedState = last ? new float[]{1, 11} : new float[]{action, 10 + action};
            float[] expectedNext = last ? new float[]{21, 31} : new float[]{20 + action, 30 + action};
            assertArrayEquals("stored state " + j, expectedState, batch.states()[j], 0f);
            assertArrayEquals("stored next state " + j, expectedNext, batch.nextStates()[j], 0f);
        }
    }

    @Test
    public void transitionsKeepTheirContinuousAction() {
        Transition t = new Transition(new float[]{1}, new float[]{0.25f, 0.5f}, 1f, new float[]{2}, false, 1f);
        assertArrayEquals(new float[]{0.25f, 0.5f}, t.withPriority(3f).continuousAction(), 0f);
        assertEquals(t, new Transition(new float[]{1}, new float[]{0.25f, 0.5f}, 1f, new float[]{2}, false, 1f));
        assertNotEquals(t, new Transition(new float[]{1}, new float[]{0.25f, 0.75f}, 1f, new float[]{2}, false, 1f));
        assertThrows(IllegalArgumentException.class,
            () -> new Transition(new float[]{1}, new float[0], 1f, new float[]{2}, false, 1f));
    }

    @Test
    public void testReplayBufferCircularBehavior() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(5, 2, backend);

        for (int i = 0; i < 10; i++) {
            buffer.add(new float[]{i, i + 1}, i, i, new float[]{i + 2, i + 3}, false);
        }

        assertEquals(5, buffer.size());
        assertTrue(buffer.isFull());
    }

    @Test
    public void testReplayBufferSample() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(100, 4, backend);

        for (int i = 0; i < 50; i++) {
            buffer.add(new float[]{i, i + 1, i + 2, i + 3}, i % 4, i * 0.1f, 
                      new float[]{i + 4, i + 5, i + 6, i + 7}, i == 49);
        }

        ReplayBuffer.SampleBatch batch = buffer.sample(32, new Random(42));

        assertEquals(32, batch.batchSize());
        assertEquals(32, batch.states().length);
        assertEquals(32, batch.actions().length);
        assertEquals(32, batch.rewards().length);
        assertEquals(32, batch.nextStates().length);
        assertEquals(32, batch.dones().length);
        assertEquals(32, batch.indices().length);
        assertEquals(32, batch.importanceWeights().length);
    }

    @Test
    public void testReplayBufferSampleFlattened() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(100, 4, backend);

        for (int i = 0; i < 20; i++) {
            buffer.add(new float[]{i, i, i, i}, 0, 0, new float[]{0, 0, 0, 0}, false);
        }

        ReplayBuffer.SampleBatch batch = buffer.sample(8, new Random(42));

        float[] flatStates = batch.flattenedStates();
        float[] flatNextStates = batch.flattenedNextStates();

        assertEquals(8 * 4, flatStates.length);
        assertEquals(8 * 4, flatNextStates.length);
    }

    @Test(expected = IllegalStateException.class)
    public void testReplayBufferSampleInsufficientData() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(100, 4, backend);

        buffer.add(new float[]{1, 2, 3, 4}, 0, 0, new float[]{0, 0, 0, 0}, false);

        buffer.sample(10);
    }

    @Test
    public void testReplayBufferClear() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(100, 4, backend);

        for (int i = 0; i < 50; i++) {
            buffer.add(new float[]{i, i, i, i}, 0, 0, new float[4], false);
        }

        assertEquals(50, buffer.size());
        buffer.clear();
        assertEquals(0, buffer.size());
        assertTrue(buffer.isEmpty());
    }

    @Test
    public void testReplayBufferUpdatePriorities() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer.Builder()
            .capacity(100)
            .stateSize(4)
            .backend(backend)
            .usePrioritizedReplay(true)
            .build();

        for (int i = 0; i < 10; i++) {
            buffer.add(new float[]{i, i, i, i}, 0, 0, new float[4], false);
        }

        buffer.updatePriorities(new int[]{0, 1, 2}, new float[]{10.0f, 20.0f, 5.0f});

        // Proportional sampling: index 1 (priority 20) is drawn more often than index 2 (priority 5), which is drawn
        // more often than index 3 (still at the initial priority 1).
        int[] draws = new int[10];
        Random random = new Random(42);
        for (int round = 0; round < 400; round++) {
            for (int index : buffer.sample(10, random).indices()) {
                draws[index]++;
            }
        }
        assertTrue("priority 20 vs 5: " + draws[1] + " vs " + draws[2], draws[1] > draws[2]);
        assertTrue("priority 5 vs 1: " + draws[2] + " vs " + draws[3], draws[2] > draws[3]);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testReplayBufferInvalidCapacity() {
        new ReplayBuffer(0, 4, new MockComputeBackend());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testReplayBufferInvalidStateSize() {
        new ReplayBuffer(100, 0, new MockComputeBackend());
    }

    @Test(expected = NullPointerException.class)
    public void testReplayBufferNullBackend() {
        new ReplayBuffer(100, 4, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testReplayBufferInvalidStateSizeOnAdd() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(100, 4, backend);

        buffer.add(new float[]{1, 2}, 0, 0, new float[4], false);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testReplayBufferInvalidNextStateSizeOnAdd() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(100, 4, backend);

        buffer.add(new float[]{1, 2, 3, 4}, 0, 0, new float[]{1, 2}, false);
    }

    @Test
    public void testReplayBufferClose() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(100, 4, backend);

        buffer.close();
    }

    @Test(expected = IllegalStateException.class)
    public void testReplayBufferAddAfterClose() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(100, 4, backend);
        buffer.close();

        buffer.add(new float[]{1, 2, 3, 4}, 0, 0, new float[4], false);
    }

    @Test
    public void testReplayBufferToString() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(100, 4, backend);

        String str = buffer.toString();
        assertTrue(str.contains("capacity=100"));
        assertTrue(str.contains("size=0"));
    }

    @Test
    public void testSampleBatchToString() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(100, 4, backend);

        for (int i = 0; i < 20; i++) {
            buffer.add(new float[]{i, i, i, i}, 0, 0, new float[4], false);
        }

        ReplayBuffer.SampleBatch batch = buffer.sample(8);
        String str = batch.toString();
        assertTrue(str.contains("batchSize=8"));
    }

    @Test
    public void testSampleBatchRowsAreCopiesOfStoredTransitions() {
        MockComputeBackend backend = new MockComputeBackend();
        ReplayBuffer buffer = new ReplayBuffer(4, 2, backend);
        buffer.add(new float[]{1f, 2f}, new float[]{0.5f, -0.5f}, 1f, new float[]{3f, 4f}, false);

        ReplayBuffer.SampleBatch first = buffer.sample(1, new Random(7));
        first.states()[0][0] = 100f;
        first.nextStates()[0][0] = 200f;
        first.continuousActions()[0][0] = 300f;

        ReplayBuffer.SampleBatch second = buffer.sample(1, new Random(7));
        assertArrayEquals(new float[]{1f, 2f}, second.states()[0], 0.0f);
        assertArrayEquals(new float[]{3f, 4f}, second.nextStates()[0], 0.0f);
        assertArrayEquals(new float[]{0.5f, -0.5f}, second.continuousActions()[0], 0.0f);
        assertNotSame(first.states()[0], second.states()[0]);
    }

    private static class MockComputeBackend implements ComputeBackend {
        private final MockComputeQueue computeQueue = new MockComputeQueue();
        private final MockComputeQueue transferQueue = new MockComputeQueue();
        private volatile boolean closed = false;

        @Override
        public BackendType backendType() {
            return BackendType.OPENCL;
        }

        @Override
        public DeviceCapabilities capabilities() {
            return new DeviceCapabilities.Builder()
                    .deviceName("MockDevice")
                    .totalVramBytes(1024 * 1024 * 1024)
                    .computeUnits(8)
                    .maxClockFrequencyMhz(1000)
                    .maxWorkGroupSize(256)
                    .supportsAsyncTransfer(true)
                    .driverVersion("1.0")
                    .build();
        }

        @Override
        public ComputeQueue computeQueue() {
            return computeQueue;
        }

        @Override
        public ComputeQueue transferQueue() {
            return transferQueue;
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes) {
            return new MockGpuBuffer(sizeInBytes);
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            return new MockGpuBuffer(sizeInBytes);
        }

        @Override
        public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
            return new MockKernelHandle(kernelName);
        }

        /** Declines kernel artifacts so tensor ops take their CPU path. */
        @Override
        public boolean supportsArtifact(KernelArtifact.Kind kind) {
            return false;
        }

        @Override
        public KernelHandle loadKernel(KernelArtifact artifact) {
            if (artifact == null) {
                throw new IllegalArgumentException("kernel artifact must not be null");
            }
            throw new UnsupportedKernelException("mock backend declines " + artifact.kind()
                    + " artifact for kernel " + artifact.kernelName());
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockComputeQueue implements ComputeQueue {
        @Override
        public QueueType getQueueType() {
            return QueueType.TRANSFER;
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, byte[] srcData, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, float[] srcData, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, byte[] dstData, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, float[] dstData, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public void flush() {
        }

        @Override
        public void finish() {
        }

        @Override
        public void close() {
        }
    }

    private static class MockGpuBuffer implements GpuBuffer {
        private final long sizeInBytes;
        private volatile boolean closed = false;

        MockGpuBuffer(long sizeInBytes) {
            this.sizeInBytes = sizeInBytes;
        }

        @Override
        public long sizeInBytes() {
            return sizeInBytes;
        }

        @Override
        public long bufferId() {
            return System.identityHashCode(this);
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockGpuEvent implements GpuEvent {
        private volatile boolean complete = true;
        private volatile boolean closed = false;

        @Override
        public void awaitCompletion() throws InterruptedException {
            complete = true;
        }

        @Override
        public boolean isComplete() {
            return complete;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockKernelHandle implements KernelHandle {
        private final String kernelName;
        private volatile boolean closed = false;

        MockKernelHandle(String kernelName) {
            this.kernelName = kernelName;
        }

        @Override
        public String kernelName() {
            return kernelName;
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
